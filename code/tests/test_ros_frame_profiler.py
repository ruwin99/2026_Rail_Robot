"""Bounded profiler persistence and ROS adapter contract tests."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch

from test_ros_vision import JETSON_CODE, ROS_DOUBLES

sys.path.insert(0, str(JETSON_CODE))


ROS_PROFILER_DOUBLES = ROS_DOUBLES.replace(
    "module('rail_robot_interfaces.msg', CameraFrame=FrameMessage, DetectionSample=Sample, ModelTiming=Sample,",
    "module('rail_robot_interfaces.msg', CameraFrame=FrameMessage, FrameTrace=Sample, DetectionSample=Sample, ModelTiming=Sample,",
)


class RosFrameProfilerTests(unittest.TestCase):
    def profiler_module(self):
        try:
            return importlib.import_module("node_runtime.frame_profiler")
        except ImportError as exc:
            self.fail(f"FrameProfiler implementation is missing: {exc}")

    def test_tegrastats_parser_extracts_available_resource_metrics(self):
        module = self.profiler_module()
        sample = module.parse_tegrastats_line(
            "RAM 1234/7852MB CPU [4%@1420,off,30%@1420] "
            "GR3D_FREQ 45%@918 CPU@52C GPU@49C SOC@47C "
            "VDD_IN 7000mW/8000mW VDD_CPU_GPU_CV 3000mW/4000mW",
            123456,
        )
        self.assertEqual(sample["monotonic_ns"], 123456)
        self.assertAlmostEqual(sample["cpu_usage_percent"], 17.0)
        self.assertEqual(sample["gpu_usage_percent"], 45.0)
        self.assertEqual(sample["ram_used_mb"], 1234.0)
        self.assertEqual(sample["ram_total_mb"], 7852.0)
        self.assertEqual(sample["thermal_c"]["CPU"], 52.0)
        self.assertEqual(sample["thermal_c"]["GPU"], 49.0)
        self.assertEqual(sample["thermal_c"]["SOC"], 47.0)
        self.assertEqual(sample["power_rails_mw"]["VDD_IN"], 7000.0)
        self.assertEqual(sample["power_rails_mw"]["VDD_CPU_GPU_CV"], 3000.0)

    def test_tegrastats_parser_keeps_partial_and_empty_metrics_null(self):
        module = self.profiler_module()
        partial = module.parse_tegrastats_line("CPU [5%@1420] RAM 512/4096MB", 77)
        self.assertEqual(partial["cpu_usage_percent"], 5.0)
        self.assertEqual(partial["ram_used_mb"], 512.0)
        self.assertIsNone(partial["gpu_usage_percent"])
        self.assertIsNone(partial["thermal_c"]["CPU"])
        self.assertIsNone(partial["power_rails_mw"]["VDD_IN"])
        empty = module.parse_tegrastats_line("", 78)
        self.assertEqual(empty["monotonic_ns"], 78)
        self.assertIsNone(empty["cpu_usage_percent"])
        self.assertIsNone(empty["gpu_usage_percent"])
        self.assertIsNone(empty["ram_used_mb"])
        self.assertIsNone(empty["ram_total_mb"])
        self.assertIsNone(empty["thermal_c"]["CPU"])
        self.assertIsNone(empty["power_rails_mw"]["VDD_IN"])

    def test_full_trace_queue_drops_without_blocking_counts_and_flushes(self):
        module = self.profiler_module()
        entered = threading.Event()
        release = threading.Event()
        original_write = module.FrameProfiler._write_jsonl_row

        def blocked_write(profiler, stream, row):
            if not entered.is_set():
                entered.set()
                release.wait(2.0)
            return original_write(stream, row)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module.FrameProfiler, "_write_jsonl_row", blocked_write):
                profiler = module.FrameProfiler(directory)
                started = time.perf_counter()
                for index in range(600):
                    profiler.submit({"record_type": "probe", "output_sequence": index})
                submit_seconds = time.perf_counter() - started
                self.assertTrue(entered.wait(1.0))
                self.assertEqual(profiler.queue_capacity, 512)
                self.assertLessEqual(profiler.queue_depth, 512)
                self.assertEqual(
                    profiler.accepted_rows + profiler.recorder_dropped_rows, 600
                )
                self.assertGreater(profiler.recorder_dropped_rows, 0)
                self.assertLess(submit_seconds, 1.0)
                accepted = profiler.accepted_rows
                dropped = profiler.recorder_dropped_rows
                release.set()
                self.assertTrue(profiler.close(5.0))
            rows = [json.loads(line) for line in profiler.jsonl_path.read_text().splitlines()]
            summary = json.loads(profiler.summary_path.read_text())
        self.assertEqual(len(rows), accepted)
        self.assertTrue(all(row["record_type"] == "probe" for row in rows))
        self.assertEqual(summary["accepted_rows"], accepted)
        self.assertEqual(summary["written_rows"], accepted)
        self.assertEqual(summary["recorder_dropped_rows"], dropped)
        self.assertEqual(summary["queue_capacity"], 512)
        self.assertLessEqual(summary["queue_high_water_mark"], 512)
        self.assertEqual(summary["queue_depth_at_shutdown"], 0)

    def test_writer_close_wait_is_bounded_and_later_flushes(self):
        module = self.profiler_module()
        entered = threading.Event()
        release = threading.Event()
        original_write = module.FrameProfiler._write_jsonl_row

        def blocked_write(profiler, stream, row):
            entered.set()
            release.wait(2.0)
            return original_write(stream, row)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module.FrameProfiler, "_write_jsonl_row", blocked_write):
                profiler = module.FrameProfiler(directory, queue_capacity=4)
                self.assertTrue(profiler.submit({"record_type": "one", "value": 1}))
                self.assertTrue(entered.wait(1.0))
                for value in range(2, 5):
                    self.assertTrue(profiler.submit({"record_type": "one", "value": value}))
                started = time.perf_counter()
                self.assertFalse(profiler.close(0.02))
                self.assertLess(time.perf_counter() - started, 0.5)
                self.assertFalse(profiler.submit({"record_type": "late", "value": 5}))
                release.set()
                self.assertTrue(profiler.close(3.0))
            rows = [json.loads(line) for line in profiler.jsonl_path.read_text().splitlines()]
            summary = json.loads(profiler.summary_path.read_text())
        self.assertEqual([row["value"] for row in rows], [1, 2, 3, 4])
        self.assertEqual(summary["written_rows"], 4)
        self.assertEqual(summary["recorder_dropped_rows"], 0)

    def test_summary_deduplicates_physical_runs_but_preserves_logical_events(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            profiler = module.FrameProfiler(directory, session_id="aggregate")
            def add_trace(sequence, base_ns):
                profiler.submit({
                    "record_type": "frame_trace",
                    "trace": {
                        "output_sequence": sequence,
                        "front_stream_id": "front-a",
                        "front_camera_sequence": sequence,
                        "front_capture_complete_monotonic_ns": base_ns - 60_000_000,
                        "side_stream_id": "side-a",
                        "side_camera_sequence": sequence + 20,
                        "side_capture_complete_monotonic_ns": base_ns - 55_000_000,
                        "compose_start_monotonic_ns": base_ns,
                        "compose_complete_monotonic_ns": base_ns + 20_000_000,
                        "jpeg_complete_monotonic_ns": base_ns + 30_000_000,
                        "gui_publish_start_monotonic_ns": base_ns + 31_000_000,
                        "gui_publish_complete_monotonic_ns": base_ns + 36_000_000,
                    },
                })
            add_trace(7, 10_000_000_000)
            add_trace(8, 11_000_000_000)
            add_trace(8, 11_000_000_000)  # exact repeated publication is kept but counted once
            profiler.submit({"record_type": "camera_fps", "camera_role": "front",
                             "fps_source": "camera_publisher_reported", "reported_fps": 15.0})
            profiler.submit({"record_type": "camera_fps", "camera_role": "front",
                             "fps_source": "camera_publisher_reported", "reported_fps": 17.0})
            timing = {
                "record_type": "model_timing", "model_id": "rust",
                "execution_id": "shared-call-1", "processing_start_monotonic_ns": 10_000_000_000,
                "processing_end_monotonic_ns": 10_010_000_000, "processing_ms": 10.0,
            }
            profiler.submit(timing)
            profiler.submit({**timing, "model_id": "crack"})
            profiler.submit({
                "record_type": "model_timing", "model_id": "rust", "execution_id": "call-2",
                "processing_start_monotonic_ns": 20_000_000_000,
                "processing_end_monotonic_ns": 20_010_000_000, "processing_ms": 10.0,
            })
            self.assertTrue(profiler.close(3.0))
            rows = [json.loads(line) for line in profiler.jsonl_path.read_text().splitlines()]
            summary = json.loads(profiler.summary_path.read_text())
        timings = [row for row in rows if row["record_type"] == "model_timing"]
        self.assertEqual(len(timings), 3)
        self.assertEqual([row["execution_id"] for row in timings],
                         ["shared-call-1", "shared-call-1", "call-2"])
        physical = summary["model_completed_runs"]
        self.assertEqual(physical["unique_execution_id_count"], 2)
        self.assertTrue(physical["available"])
        self.assertAlmostEqual(physical["fps"], 1e9 / 10_000_000_000)
        logical = summary["logical_model_result_event_rates"]
        self.assertEqual(logical["metric_name"], "logical_model_result_event_rate")
        self.assertEqual(logical["by_model"]["rust"]["event_count"], 2)
        self.assertEqual(logical["by_model"]["crack"]["event_count"], 1)
        self.assertEqual(summary["gui_fps"]["unique_output_count"], 2)
        self.assertEqual(summary["gui_fps"]["unique_output_sequence_value_count"], 2)
        self.assertAlmostEqual(summary["gui_fps"]["fps"], 1.0)
        self.assertEqual(summary["camera_capture_fps"]["front"]["source"],
                         "camera_publisher_reported")
        self.assertAlmostEqual(summary["camera_capture_fps"]["front"]["mean_reported_fps"], 16.0)
        stages = summary["stage_latency_ms"]
        self.assertEqual(stages["compose"]["sample_count"], 3)
        self.assertEqual(stages["compose"]["p50"], 20.0)
        self.assertEqual(stages["compose_complete_to_jpeg_complete"]["p95"], 10.0)
        self.assertNotIn("jpeg_encode", stages)
        self.assertIn("not a codec-only benchmark",
                      summary["fps_definitions"]["compose_complete_to_jpeg_complete"])
        self.assertEqual(stages["front_capture_to_gui_publish"]["sample_count"], 3)
        self.assertEqual(stages["model_processing"]["sample_count"], 2)
        self.assertEqual(stages["model_result_event_processing"]["sample_count"], 3)

    def test_gui_sequence_reset_creates_a_new_output_epoch(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            profiler = module.FrameProfiler(directory)
            for publish_end in (10_000_000_000, 20_000_000_000, 20_000_000_000):
                profiler.submit({
                    "record_type": "frame_trace",
                    "trace": {
                        "output_sequence": 1,
                        "gui_publish_complete_monotonic_ns": publish_end,
                    },
                })
            self.assertTrue(profiler.close(3.0))
            rows = [json.loads(line) for line in profiler.jsonl_path.read_text().splitlines()]
            summary = json.loads(profiler.summary_path.read_text())
        self.assertEqual(len([row for row in rows if row["record_type"] == "frame_trace"]), 3)
        self.assertEqual(summary["gui_fps"]["unique_output_count"], 2)
        self.assertEqual(summary["gui_fps"]["unique_output_sequence_value_count"], 1)
        self.assertEqual(summary["gui_fps"]["identity_fields"], [
            "output_sequence", "gui_publish_complete_monotonic_ns"
        ])
        self.assertAlmostEqual(summary["gui_fps"]["fps"], 0.1)

    def test_cumulative_gui_render_drop_counter_uses_reset_aware_deltas(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            profiler = module.FrameProfiler(directory)
            for sequence, drops in enumerate((5, 7, 9, 1, 2), 1):
                profiler.submit({"record_type": "frame_trace", "trace": {
                    "output_sequence": sequence,
                    "gui_publish_complete_monotonic_ns": sequence * 1_000_000_000,
                    "gui_render_drop_count": drops,
                }})
            self.assertTrue(profiler.close(3.0))
            summary = json.loads(profiler.summary_path.read_text())
        totals = summary["frame_trace_quality_totals"]
        self.assertEqual(totals["gui_render_drop_count"], 6)
        self.assertEqual(totals["gui_render_drop_counter_reset_count"], 1)
        self.assertIn("first observed value is a baseline",
                      summary["frame_trace_quality_definitions"]["gui_render_drop_count"])

    def test_missing_execution_id_marks_physical_run_rate_unavailable(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            profiler = module.FrameProfiler(directory)
            profiler.submit({"record_type": "model_timing", "model_id": "rust",
                             "execution_id": "", "processing_start_monotonic_ns": 10,
                             "processing_end_monotonic_ns": 20, "processing_ms": 0.00001})
            self.assertTrue(profiler.close(3.0))
            summary = json.loads(profiler.summary_path.read_text())
        physical = summary["model_completed_runs"]
        self.assertFalse(physical["available"])
        self.assertIsNone(physical["fps"])
        self.assertEqual(physical["unavailable_reason"], "missing_execution_id")
        self.assertIsNone(summary["resource_metrics"]["gpu_usage_percent"]["p95"])
        self.assertIsNone(summary["resource_metrics"]["power_rails_mw.VDD_IN"]["mean"])

    def test_missing_physical_completion_time_marks_run_rate_unavailable(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            profiler = module.FrameProfiler(directory)
            profiler.submit({"record_type": "model_timing", "model_id": "rust",
                             "execution_id": "one", "processing_end_monotonic_ns": 10})
            profiler.submit({"record_type": "model_timing", "model_id": "rust",
                             "execution_id": "two"})
            self.assertTrue(profiler.close(3.0))
            summary = json.loads(profiler.summary_path.read_text())
        physical = summary["model_completed_runs"]
        self.assertFalse(physical["available"])
        self.assertIsNone(physical["fps"])
        self.assertEqual(physical["unavailable_reason"], "missing_completion_timestamp")

    def test_logical_event_rate_is_null_when_any_event_has_no_completion_time(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            profiler = module.FrameProfiler(directory)
            profiler.submit({"record_type": "model_timing", "model_id": "rust",
                             "execution_id": "one", "processing_start_monotonic_ns": 10,
                             "processing_end_monotonic_ns": 20})
            profiler.submit({"record_type": "model_timing", "model_id": "rust",
                             "execution_id": "two"})
            self.assertTrue(profiler.close(3.0))
            summary = json.loads(profiler.summary_path.read_text())
        result = summary["logical_model_result_event_rates"]["by_model"]["rust"]
        self.assertEqual(result["event_count"], 2)
        self.assertEqual(result["timed_event_count"], 1)
        self.assertIsNone(result["logical_result_event_rate_fps"])

    def test_model_and_resource_summary_keys_are_bounded_and_exclusions_counted(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            profiler = module.FrameProfiler(directory, session_id="cardinality")
            for index in range(100):
                profiler.submit({
                    "record_type": "model_timing", "model_id": f"model-{index}",
                    "execution_id": f"run-{index}",
                    "processing_start_monotonic_ns": index * 10 + 1,
                    "processing_end_monotonic_ns": index * 10 + 2,
                })
            profiler.submit({"record_type": "resource_sample", "sample": {
                "cpu_usage_percent": 10, "gpu_usage_percent": 20,
                "ram_used_mb": 100, "ram_total_mb": 1000,
                "thermal_c": {"CPU": 40, "GPU": 41, "SOC": 42},
                "power_rails_mw": {f"VDD_RAIL_{index}": index for index in range(100)},
            }})
            self.assertTrue(profiler.close(3.0))
            summary = json.loads(profiler.summary_path.read_text())
        model_rates = summary["logical_model_result_event_rates"]["by_model"]
        resource_metrics = summary["resource_metrics"]
        limits = summary["summary_key_limits"]
        self.assertLessEqual(len(model_rates), limits["model_key_capacity"])
        self.assertLessEqual(len(resource_metrics), limits["resource_metric_key_capacity"])
        self.assertEqual(limits["model_key_capacity"], 64)
        self.assertEqual(limits["excluded_model_event_rows"], 36)
        self.assertEqual(limits["resource_metric_key_capacity"], 64)
        self.assertEqual(limits["excluded_resource_metric_samples"], 45)

    def test_reused_requested_and_default_session_ids_never_overwrite_runs(self):
        module = self.profiler_module()
        with tempfile.TemporaryDirectory() as directory:
            explicit_one = module.FrameProfiler(directory, session_id="same-session")
            explicit_two = module.FrameProfiler(directory, session_id="same-session")
            default_one = module.FrameProfiler(directory)
            default_two = module.FrameProfiler(directory)
            for profiler, value in ((explicit_one, "first"), (explicit_two, "second"),
                                    (default_one, "default-1"), (default_two, "default-2")):
                profiler.submit({"record_type": "sentinel", "value": value})
            self.assertEqual(len({profiler.session_id for profiler in
                                  (explicit_one, explicit_two, default_one, default_two)}), 4)
            self.assertEqual(len({profiler.jsonl_path for profiler in
                                  (explicit_one, explicit_two, default_one, default_two)}), 4)
            for profiler in (explicit_one, explicit_two, default_one, default_two):
                self.assertTrue(profiler.close(3.0))
            files = [json.loads(path.read_text().splitlines()[0])
                     for path in Path(directory).glob("*.jsonl")]
        self.assertEqual(sorted(row["value"] for row in files),
                         ["default-1", "default-2", "first", "second"])

    def test_exclusive_file_creation_reports_collision_without_truncating_existing_output(self):
        module = self.profiler_module()
        fixed_uuid = type("FixedUUID", (), {"hex": "fixed-run-id"})()
        with tempfile.TemporaryDirectory() as directory:
            existing_jsonl = Path(directory) / "frame_profile_collision_fixed-run-id.jsonl"
            existing_jsonl.write_text("preserve this prior run\n")
            with patch.object(module.uuid, "uuid4", return_value=fixed_uuid):
                profiler = module.FrameProfiler(directory, session_id="collision")
                self.assertFalse(profiler.close(3.0))
            self.assertIn("FileExistsError", profiler.writer_error)
            self.assertEqual(existing_jsonl.read_text(), "preserve this prior run\n")

    def test_concurrent_submit_and_close_accounts_for_every_attempt(self):
        module = self.profiler_module()
        entered = threading.Event()
        release = threading.Event()
        original_write = module.FrameProfiler._write_jsonl_row

        def blocked_write(profiler, stream, row):
            if not entered.is_set():
                entered.set()
                release.wait(3.0)
            return original_write(stream, row)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module.FrameProfiler, "_write_jsonl_row", blocked_write):
                profiler = module.FrameProfiler(directory, queue_capacity=8)
                self.assertTrue(profiler.submit({"record_type": "seed"}))
                self.assertTrue(entered.wait(1.0))

                def produce(prefix):
                    for index in range(250):
                        profiler.submit({"record_type": "concurrent", "key": f"{prefix}-{index}"})

                with ThreadPoolExecutor(max_workers=3) as pool:
                    producers = [pool.submit(produce, prefix) for prefix in ("a", "b")]
                    closer = pool.submit(profiler.close, 5.0)
                    for task in producers:
                        task.result(timeout=2.0)
                    self.assertTrue(profiler._stopping.wait(1.0))
                    release.set()
                    self.assertTrue(closer.result(timeout=4.0))
            rows = [json.loads(line) for line in profiler.jsonl_path.read_text().splitlines()]
            summary = json.loads(profiler.summary_path.read_text())
        recorder = summary["recorder"]
        self.assertEqual(
            recorder["accepted_rows"] + recorder["recorder_dropped_rows"]
            + recorder["rejected_after_close"], 501,
        )
        self.assertEqual(len(rows), recorder["written_rows"])
        self.assertEqual(recorder["written_rows"], recorder["accepted_rows"])

    def run_ros_contract(self, source):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(JETSON_CODE), environment.get("PYTHONPATH", ""))
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c",
             ROS_PROFILER_DOUBLES + textwrap.dedent(source)],
            env=environment, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_node_uses_raw_metric_topics_and_preserves_same_execution_id_rows(self):
        self.run_ros_contract('''
            import json, tempfile
            from pathlib import Path
            from types import SimpleNamespace as NS
            from node_runtime.ros.frame_profiler_node import FrameProfilerNode

            with tempfile.TemporaryDirectory() as directory:
                node=FrameProfilerNode(output_directory_override=directory)
                topics=[subscription[0] for subscription in node.subscriptions]
                assert set(topics)=={
                    '/rail_robot/debug/frame_trace','debug/vision/timing',
                    'debug/camera/front/fps','debug/camera/side/fps'}
                assert not any('image' in topic for topic in topics)
                trace=NS(output_sequence=1,front_stream_id='f',front_frame_id='fid-front',
                    front_camera_sequence=4,front_capture_complete_monotonic_ns=100,
                    side_stream_id='s',side_frame_id='fid-side',side_camera_sequence=6,
                    side_capture_complete_monotonic_ns=110,
                    model_camera_roles=['side','side'],model_ids=['rust','crack'],
                    model_stream_ids=['s','s'],model_source_sequences=[6,6],
                    model_start_monotonic_ns=[120,120],model_end_monotonic_ns=[130,130],
                    model_stale=[False,False],compose_start_monotonic_ns=140,
                    compose_complete_monotonic_ns=150,jpeg_complete_monotonic_ns=160,
                    gui_publish_start_monotonic_ns=170,gui_publish_complete_monotonic_ns=180,
                    missing_camera_count=0,stale_model_count=0,duplicate_frame_count=0,
                    source_sequence_gap_count=0,gui_render_drop_count=0,
                    header=NS(frame_id='gui',stamp=NS(sec=3,nanosec=4)))
                node._on_frame_trace(trace)
                for model in ('rust','crack'):
                    node._on_model_timing(NS(header=trace.header,model=model,
                        execution_id='shared-physical-run',camera_role='side',stream_id='s',
                        sequence=6,processing_start_monotonic_ns=120,
                        processing_end_monotonic_ns=130,processing_ms=0.00001))
                node._on_camera_fps('front',NS(data=15.0))
                node._on_camera_fps('side',NS(data=14.0))
                node._on_camera_fps('side',NS(data=float('nan')))
                node.prepare_shutdown()
                assert node.destroy_node()
                jsonl=next(Path(directory).glob('*.jsonl'))
                rows=[json.loads(line) for line in jsonl.read_text().splitlines()]
                summary=json.loads(next(Path(directory).glob('*summary.json')).read_text())
                timing_rows=[row for row in rows if row['record_type']=='model_timing']
                fps_rows=[row for row in rows if row['record_type']=='camera_fps']
                assert fps_rows[-1]['reported_fps'] is None
                assert len(timing_rows)==2
                assert [row['execution_id'] for row in timing_rows]==[
                    'shared-physical-run','shared-physical-run']
                assert summary['model_completed_runs']['unique_execution_id_count']==1
                assert summary['model_completed_runs']['fps'] is None
                assert summary['camera_capture_fps']['front']['source']=='camera_publisher_reported'
        ''')

    def test_missing_tegrastats_is_recorded_as_unavailable_without_failing_node(self):
        self.run_ros_contract('''
            import json, tempfile
            from pathlib import Path
            from node_runtime.ros.frame_profiler_node import FrameProfilerNode

            with tempfile.TemporaryDirectory() as directory:
                node=FrameProfilerNode(output_directory_override=directory,
                    enable_tegrastats_override=True,tegrastats_lookup=lambda name:None)
                assert node._tegrastats_status['state']=='unavailable'
                node.prepare_shutdown()
                assert node.destroy_node()
                rows=[json.loads(line) for line in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
                status=next(row for row in rows if row['record_type']=='resource_sampler_status')
                assert status['state']=='unavailable'
                summary=json.loads(next(Path(directory).glob('*summary.json')).read_text())
                assert summary['resource_sampler']['state']=='unavailable'
        ''')

    def test_tegrastats_read_exception_transitions_running_status_to_unavailable(self):
        self.run_ros_contract('''
            import json, tempfile
            from pathlib import Path
            from node_runtime.ros.frame_profiler_node import FrameProfilerNode

            class BrokenOutput:
                def __iter__(self): return self
                def __next__(self): raise OSError('synthetic pipe failure')
                def close(self): pass
            class Process:
                stdout=BrokenOutput()
                def terminate(self): pass
                def wait(self,timeout=None): pass

            with tempfile.TemporaryDirectory() as directory:
                node=FrameProfilerNode(output_directory_override=directory,
                    enable_tegrastats_override=False)
                node._sampler_process=Process()
                node._tegrastats_status={'state':'running','source':'tegrastats'}
                node._read_tegrastats()
                assert node._tegrastats_status['state']=='unavailable'
                assert 'synthetic pipe failure' in node._tegrastats_status['reason']
                node.prepare_shutdown()
                assert node.destroy_node()
                rows=[json.loads(line) for line in next(Path(directory).glob('*.jsonl')).read_text().splitlines()]
                statuses=[row for row in rows if row['record_type']=='resource_sampler_status']
                assert any(row.get('reason','').find('synthetic pipe failure')>=0 for row in statuses)
                summary=json.loads(next(Path(directory).glob('*summary.json')).read_text())
                assert summary['resource_sampler']['state']=='unavailable'
                assert 'synthetic pipe failure' in summary['resource_sampler']['reason']
        ''')

    def test_profiler_node_registration_is_in_entrypoint_and_install_roles(self):
        root = Path(__file__).resolve().parents[1]
        entrypoints = (root / "jetson_code/node_runtime/ros/entrypoints.py").read_text(encoding="utf-8")
        cmake = (root / "ros2/rail_robot_nodes/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn(
            '"frame_profiler_node": ("frame_profiler_node", "FrameProfilerNode")',
            entrypoints,
        )
        self.assertRegex(cmake, r"foreach\(ROLE[^\n]*frame_profiler[^\n]*\)")

if __name__ == "__main__":
    unittest.main()
