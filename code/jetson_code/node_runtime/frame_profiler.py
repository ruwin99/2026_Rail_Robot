"""Bounded JSONL recorder and run-level summaries for frame provenance telemetry."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import queue
import re
import threading
import time
import uuid
from pathlib import Path


DEFAULT_QUEUE_CAPACITY = 512
_MAX_TRACKED_IDS = 200_000
_MAX_METRIC_SAMPLES = 8_192
_MAX_MODEL_SUMMARY_KEYS = 64
_MAX_RESOURCE_SUMMARY_KEYS = 64
_RESOURCE_METRICS = (
    "cpu_usage_percent", "gpu_usage_percent", "ram_used_mb", "ram_total_mb",
    "thermal_c.CPU", "thermal_c.GPU", "thermal_c.SOC",
    "power_rails_mw.VDD_IN", "power_rails_mw.VDD_CPU_GPU_CV",
)

_NUMBER = r"(-?\d+(?:\.\d+)?)"


def parse_tegrastats_line(line: str, monotonic_ns: int) -> dict:
    """Parse the common Jetson tegrastats fields; absent fields remain None."""
    sample = {
        "monotonic_ns": int(monotonic_ns),
        "cpu_usage_percent": None,
        "gpu_usage_percent": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "thermal_c": {"CPU": None, "GPU": None, "SOC": None},
        "power_rails_mw": {"VDD_IN": None, "VDD_CPU_GPU_CV": None},
    }
    text = line if isinstance(line, str) else ""
    cpu = re.search(r"\bCPU\s*\[([^\]]*)\]", text, re.IGNORECASE)
    if cpu:
        values = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*%@", cpu.group(1))]
        if values:
            sample["cpu_usage_percent"] = sum(values) / len(values)

    gpu = re.search(
        r"\b(?:GR3D_FREQ|GPU)\s*[: ]*?(\d+(?:\.\d+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    if gpu:
        sample["gpu_usage_percent"] = float(gpu.group(1))

    ram = re.search(
        r"\bRAM\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*MB",
        text,
        re.IGNORECASE,
    )
    if ram:
        sample["ram_used_mb"] = float(ram.group(1))
        sample["ram_total_mb"] = float(ram.group(2))

    for name, value in re.findall(
        r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)\s*C\b",
        text,
        re.IGNORECASE,
    ):
        sample["thermal_c"][name.upper()] = float(value)

    for name, value in re.findall(
        r"\b([A-Za-z][A-Za-z0-9_]*)\s+(-?\d+(?:\.\d+)?)\s*mW(?:\s*/\s*-?\d+(?:\.\d+)?\s*mW)?",
        text,
        re.IGNORECASE,
    ):
        sample["power_rails_mw"][name.upper()] = float(value)

    return sample


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return float(ordered[index])


class _RunSummary:
    """Bounded online run metrics; raw event rows remain the source of truth."""

    STAGES = (
        "compose",
        "compose_complete_to_jpeg_complete",
        "gui_publish",
        "front_capture_to_compose",
        "front_capture_to_gui_publish",
        "side_capture_to_compose",
        "side_capture_to_gui_publish",
        "model_processing",
        "model_result_event_processing",
    )

    def __init__(self):
        self.stage_samples = {name: deque(maxlen=_MAX_METRIC_SAMPLES) for name in self.STAGES}
        self.stage_observed = defaultdict(int)
        self._resource_metric_names = set(_RESOURCE_METRICS)
        self._resource_samples = {
            name: deque(maxlen=_MAX_METRIC_SAMPLES) for name in _RESOURCE_METRICS
        }
        self._resource_observed = defaultdict(int)
        self.excluded_resource_metric_samples = 0
        self._gui_outputs = {}
        self.gui_output_tracking_truncated = False
        self._execution_ids = set()
        self.execution_tracking_truncated = False
        self.missing_execution_id_events = 0
        self.model_timing_events = 0
        self._execution_completion_ids = set()
        self._execution_stage_ids = set()
        self._physical_completion_ns = deque(maxlen=_MAX_TRACKED_IDS)
        self._model_event_counts = {}
        self._model_completion_ranges = {}
        self.excluded_model_event_rows = 0
        self.camera_fps = {
            "front": {"sample_count": 0, "sum": 0.0, "latest": None},
            "side": {"sample_count": 0, "sum": 0.0, "latest": None},
        }
        self.trace_events = 0
        self.resource_events = 0
        self.trace_quality_totals = {
            "missing_camera_count": 0,
            "stale_model_count": 0,
            "duplicate_frame_count": 0,
            "source_sequence_gap_count": 0,
            "gui_render_drop_count": 0,
            "gui_render_drop_counter_reset_count": 0,
        }
        self._last_gui_render_drop_counter = None

    def _add_stage(self, name, start_ns, end_ns):
        if not isinstance(start_ns, int) or not isinstance(end_ns, int):
            return
        if start_ns <= 0 or end_ns < start_ns:
            return
        self.stage_observed[name] += 1
        self.stage_samples[name].append((end_ns - start_ns) / 1_000_000.0)

    def _observe_trace(self, row):
        trace = row.get("trace", row)
        if not isinstance(trace, dict):
            return
        self.trace_events += 1
        sequence = trace.get("output_sequence")
        publish_end = trace.get("gui_publish_complete_monotonic_ns")
        if isinstance(sequence, int) and sequence > 0 and isinstance(publish_end, int) and publish_end > 0:
            # sequence restarts with DisplayNode; publisher-call monotonic time distinguishes epochs.
            output_identity = (sequence, publish_end)
            if output_identity not in self._gui_outputs:
                if len(self._gui_outputs) < _MAX_TRACKED_IDS:
                    self._gui_outputs[output_identity] = publish_end
                else:
                    self.gui_output_tracking_truncated = True
        for field in self.trace_quality_totals:
            if field in ("gui_render_drop_count", "gui_render_drop_counter_reset_count"):
                continue
            value = trace.get(field)
            if isinstance(value, int) and value >= 0:
                self.trace_quality_totals[field] += value

        # FrameTrace carries a cumulative render-skip counter, not a per-frame delta.
        current_drops = trace.get("gui_render_drop_count")
        if isinstance(current_drops, int) and current_drops >= 0:
            previous_drops = self._last_gui_render_drop_counter
            if previous_drops is not None:
                if current_drops >= previous_drops:
                    self.trace_quality_totals["gui_render_drop_count"] += current_drops - previous_drops
                else:
                    # A counter decrease starts a new DisplayNode counter epoch.
                    self.trace_quality_totals["gui_render_drop_count"] += current_drops
                    self.trace_quality_totals["gui_render_drop_counter_reset_count"] += 1
            self._last_gui_render_drop_counter = current_drops

        compose_start = trace.get("compose_start_monotonic_ns")
        compose_end = trace.get("compose_complete_monotonic_ns")
        jpeg_end = trace.get("jpeg_complete_monotonic_ns")
        publish_start = trace.get("gui_publish_start_monotonic_ns")
        self._add_stage("compose", compose_start, compose_end)
        self._add_stage("compose_complete_to_jpeg_complete", compose_end, jpeg_end)
        self._add_stage("gui_publish", publish_start, publish_end)
        for role in ("front", "side"):
            capture = trace.get(f"{role}_capture_complete_monotonic_ns")
            if not isinstance(capture, int) or capture <= 0:
                continue
            self._add_stage(f"{role}_capture_to_compose", capture, compose_start)
            self._add_stage(f"{role}_capture_to_gui_publish", capture, publish_end)

    def _observe_timing(self, row):
        self.model_timing_events += 1
        model = str(row.get("model_id") or row.get("model") or "unknown")
        tracked_model = model in self._model_event_counts
        if not tracked_model and len(self._model_event_counts) < _MAX_MODEL_SUMMARY_KEYS:
            self._model_event_counts[model] = 0
            self._model_completion_ranges[model] = [0, None, None]
            tracked_model = True
        if tracked_model:
            self._model_event_counts[model] += 1
        else:
            self.excluded_model_event_rows += 1
        start = row.get("processing_start_monotonic_ns")
        end = row.get("processing_end_monotonic_ns")
        valid_stage = (
            isinstance(start, int) and isinstance(end, int)
            and start > 0 and end >= start
        )
        if valid_stage:
            if tracked_model:
                completion = self._model_completion_ranges[model]
                completion[0] += 1
                completion[1] = end if completion[1] is None else min(completion[1], end)
                completion[2] = end if completion[2] is None else max(completion[2], end)
            self._add_stage("model_result_event_processing", start, end)

        execution_id = row.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id.strip():
            self.missing_execution_id_events += 1
            return
        if execution_id in self._execution_ids:
            if (execution_id not in self._execution_completion_ids
                    and isinstance(end, int) and end > 0):
                self._execution_completion_ids.add(execution_id)
                self._physical_completion_ns.append(end)
            if valid_stage and execution_id not in self._execution_stage_ids:
                self._execution_stage_ids.add(execution_id)
                self._add_stage("model_processing", start, end)
            return
        if len(self._execution_ids) >= _MAX_TRACKED_IDS:
            self.execution_tracking_truncated = True
            return
        self._execution_ids.add(execution_id)
        if isinstance(end, int) and end > 0:
            self._execution_completion_ids.add(execution_id)
            self._physical_completion_ns.append(end)
        if valid_stage:
            self._execution_stage_ids.add(execution_id)
            self._add_stage("model_processing", start, end)

    def _observe_camera_fps(self, row):
        role = row.get("camera_role")
        value = _finite_number(row.get("reported_fps"))
        if role not in self.camera_fps or value is None or value < 0:
            return
        bucket = self.camera_fps[role]
        bucket["sample_count"] += 1
        bucket["sum"] += value
        bucket["latest"] = value

    def _observe_resource(self, row):
        self.resource_events += 1
        values = row.get("sample", row)
        if not isinstance(values, dict):
            return
        for name in ("cpu_usage_percent", "gpu_usage_percent", "ram_used_mb", "ram_total_mb"):
            self._add_resource_metric(name, values.get(name))
        for group in ("thermal_c", "power_rails_mw"):
            for metric, value in (values.get(group) or {}).items():
                self._add_resource_metric(f"{group}.{metric}", value)

    def _add_resource_metric(self, name, value):
        number = _finite_number(value)
        if number is None:
            return
        if name not in self._resource_metric_names:
            if len(self._resource_metric_names) >= _MAX_RESOURCE_SUMMARY_KEYS:
                self.excluded_resource_metric_samples += 1
                return
            self._resource_metric_names.add(name)
            self._resource_samples[name] = deque(maxlen=_MAX_METRIC_SAMPLES)
        self._resource_observed[name] += 1
        self._resource_samples[name].append(number)

    @staticmethod
    def _rate(count, times_ns):
        valid = [value for value in times_ns if isinstance(value, int) and value > 0]
        if count < 2 or len(valid) < 2 or max(valid) <= min(valid):
            return None
        return (count - 1) * 1_000_000_000.0 / (max(valid) - min(valid))

    def observe(self, row):
        if not isinstance(row, dict):
            return
        record_type = row.get("record_type")
        if record_type == "frame_trace":
            self._observe_trace(row)
        elif record_type == "model_timing":
            self._observe_timing(row)
        elif record_type == "camera_fps":
            self._observe_camera_fps(row)
        elif record_type == "resource_sample":
            self._observe_resource(row)

    def build(self, *, session_id, started_ns, ended_ns, queue_capacity,
              accepted_rows, written_rows, dropped_rows, rejected_after_close,
              queue_high_water_mark, metadata, writer_error):
        physical_reason = None
        physical_fps = None
        if self.missing_execution_id_events:
            physical_reason = "missing_execution_id"
        elif self.execution_tracking_truncated:
            physical_reason = "execution_id_tracking_limit"
        elif len(self._execution_completion_ids) != len(self._execution_ids):
            physical_reason = "missing_completion_timestamp"
        elif len(self._execution_ids) < 2:
            physical_reason = "insufficient_unique_execution_ids"
        else:
            physical_fps = self._rate(len(self._execution_ids), self._physical_completion_ns)
            if physical_fps is None:
                physical_reason = "insufficient_completion_time_span"

        physical = {
            "metric_name": "physical_detector_run_fps",
            "unique_execution_id_count": len(self._execution_ids),
            "missing_execution_id_event_count": self.missing_execution_id_events,
            "available": physical_fps is not None,
            "fps": physical_fps,
            "unavailable_reason": physical_reason,
        }

        model_rates = {}
        for model, count in sorted(self._model_event_counts.items()):
            timed_count, first_end, last_end = self._model_completion_ranges[model]
            rate = None
            if timed_count == count and count >= 2 and last_end > first_end:
                rate = (count - 1) * 1_000_000_000.0 / (last_end - first_end)
            model_rates[model] = {
                "event_count": count,
                "timed_event_count": timed_count,
                "logical_result_event_rate_fps": rate,
            }

        gui_times = list(self._gui_outputs.values())
        gui_fps = None if self.gui_output_tracking_truncated else self._rate(
            len(self._gui_outputs), gui_times
        )
        stage_summary = {}
        for name in self.STAGES:
            values = list(self.stage_samples[name])
            stage_summary[name] = {
                "sample_count": self.stage_observed[name],
                "sample_window_size": len(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
            }

        camera_summary = {}
        for role, bucket in self.camera_fps.items():
            count = bucket["sample_count"]
            camera_summary[role] = {
                "source": "camera_publisher_reported",
                "sample_count": count,
                "mean_reported_fps": bucket["sum"] / count if count else None,
                "latest_reported_fps": bucket["latest"],
            }

        resource_summary = {}
        for name in sorted(self._resource_metric_names):
            values_deque = self._resource_samples[name]
            values = list(values_deque)
            resource_summary[name] = {
                "source": "tegrastats",
                "sample_count": self._resource_observed[name],
                "sample_window_size": len(values),
                "mean": sum(values) / len(values) if values else None,
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "max": max(values) if values else None,
            }

        elapsed = max(0.0, (ended_ns - started_ns) / 1_000_000_000.0)
        sampler = metadata.get("resource_sampler", {"state": "disabled", "source": "tegrastats"})
        return {
            "schema_version": 1,
            "session_id": session_id,
            "requested_session_id": metadata.get("requested_session_id"),
            "started_monotonic_ns": started_ns,
            "ended_monotonic_ns": ended_ns,
            "elapsed_seconds": elapsed,
            "recorded_trace_events": self.trace_events,
            "recorded_model_timing_events": self.model_timing_events,
            "recorded_resource_events": self.resource_events,
            "fps_definitions": {
                "camera_capture_fps": (
                    "Mean of received Float64 camera-publisher FPS values; not recomputed from frames."
                ),
                "model_completed_runs": (
                    "Physical detector invocations counted by unique ModelTiming.execution_id; "
                    "rate is (unique runs - 1) divided by first-to-last unique completion interval."
                ),
                "model_processing": (
                    "Physical-call processing latency; one stage sample per execution_id."
                ),
                "model_result_event_processing": (
                    "Logical ModelTiming-row processing latency; multitask rows can share one physical call."
                ),
                "compose_complete_to_jpeg_complete": (
                    "Elapsed time from composition completion until JPEG completion; includes intervening "
                    "output preparation and is not a codec-only benchmark."
                ),
                "logical_model_result_event_rates": (
                    "Per-model ModelTiming result-event rate; these logical rows are not physical runs "
                    "and must not be summed as inference FPS."
                ),
                "gui_fps": (
                    "Unique (FrameTrace.output_sequence, gui_publish_complete_monotonic_ns) identity count; "
                    "the monotonic publish time distinguishes DisplayNode sequence resets. Rate is "
                    "(unique outputs - 1) divided by first-to-last publisher-call completion interval."
                ),
            },
            "camera_capture_fps": camera_summary,
            "model_completed_runs": physical,
            "logical_model_result_event_rates": {
                "metric_name": "logical_model_result_event_rate",
                "by_model": model_rates,
            },
            "gui_fps": {
                "metric_name": "unique_gui_output_identity_fps",
                "identity_fields": ["output_sequence", "gui_publish_complete_monotonic_ns"],
                "unique_output_count": len(self._gui_outputs),
                "unique_output_sequence_value_count": len({key[0] for key in self._gui_outputs}),
                "available": gui_fps is not None,
                "fps": gui_fps,
                "unavailable_reason": (
                    "sequence_tracking_limit" if self.gui_output_tracking_truncated
                    else "insufficient_unique_outputs_or_time_span" if gui_fps is None else None
                ),
            },
            "stage_latency_ms": stage_summary,
            "frame_trace_quality_totals": self.trace_quality_totals,
            "frame_trace_quality_definitions": {
                "gui_render_drop_count": (
                    "Reset-aware sum of observed deltas from DisplayNode's cumulative counter. "
                    "The first observed value is a baseline and is excluded; after a decrease, "
                    "the new value is counted as the first delta of a reset epoch."
                ),
                "gui_render_drop_counter_reset_count": (
                    "Number of decreases in the cumulative GUI render-drop counter observed during this run."
                ),
            },
            "resource_sampler": sampler,
            "resource_metrics": resource_summary,
            "summary_key_limits": {
                "model_key_capacity": _MAX_MODEL_SUMMARY_KEYS,
                "excluded_model_event_rows": self.excluded_model_event_rows,
                "resource_metric_key_capacity": _MAX_RESOURCE_SUMMARY_KEYS,
                "excluded_resource_metric_samples": self.excluded_resource_metric_samples,
            },
            "recorder": {
                "queue_capacity": queue_capacity,
                "queue_high_water_mark": queue_high_water_mark,
                "queue_depth_at_shutdown": 0,
                "accepted_rows": accepted_rows,
                "written_rows": written_rows,
                "recorder_dropped_rows": dropped_rows,
                "rejected_after_close": rejected_after_close,
                "writer_error": writer_error,
            },
            "accepted_rows": accepted_rows,
            "written_rows": written_rows,
            "recorder_dropped_rows": dropped_rows,
            "queue_capacity": queue_capacity,
            "queue_high_water_mark": queue_high_water_mark,
            "queue_depth_at_shutdown": 0,
        }


class FrameProfiler:
    """Nonblocking producer API backed by one bounded queue and one file-owning writer."""

    def __init__(self, output_directory, queue_capacity=DEFAULT_QUEUE_CAPACITY,
                 session_id=None, metadata=None):
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int):
            raise TypeError("queue_capacity must be an integer")
        if queue_capacity < 1 or queue_capacity > DEFAULT_QUEUE_CAPACITY:
            raise ValueError(f"queue_capacity must be between 1 and {DEFAULT_QUEUE_CAPACITY}")
        self.output_directory = Path(output_directory)
        requested_id = "run" if session_id is None else session_id
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(requested_id)).strip("_")[:80]
        self.requested_session_id = safe_id or "run"
        self.session_id = f"{self.requested_session_id}_{uuid.uuid4().hex}"
        self.jsonl_path = self.output_directory / f"frame_profile_{self.session_id}.jsonl"
        self.summary_path = self.output_directory / f"frame_profile_{self.session_id}_summary.json"
        self.queue_capacity = queue_capacity
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._state_lock = threading.Lock()
        self._accepting = True
        self._stopping = threading.Event()
        self._started_ns = time.monotonic_ns()
        self._metadata = deepcopy(metadata or {"resource_sampler": {
            "state": "disabled", "source": "tegrastats"
        }})
        self._metadata["requested_session_id"] = self.requested_session_id
        self._accepted_rows = 0
        self._dropped_rows = 0
        self._rejected_after_close = 0
        self._queue_high_water_mark = 0
        self._written_rows = 0
        self._writer_error = None
        self._summary = _RunSummary()
        self._writer = threading.Thread(
            target=self._writer_main,
            name=f"frame-profiler-{self.session_id[:8]}",
            daemon=True,
        )
        self._writer.start()

    @property
    def queue_depth(self):
        return self._queue.qsize()

    @property
    def accepted_rows(self):
        with self._state_lock:
            return self._accepted_rows

    @property
    def recorder_dropped_rows(self):
        with self._state_lock:
            return self._dropped_rows

    @property
    def queue_high_water_mark(self):
        with self._state_lock:
            return self._queue_high_water_mark

    @property
    def writer_alive(self):
        return self._writer.is_alive()

    @property
    def writer_error(self):
        return self._writer_error

    def set_metadata(self, key, value):
        with self._state_lock:
            self._metadata[str(key)] = deepcopy(value)

    def submit(self, trace_row):
        """Copy and enqueue a JSON-compatible row without waiting for disk I/O."""
        if not isinstance(trace_row, dict):
            raise TypeError("trace_row must be a dict")
        row = deepcopy(trace_row)
        row.setdefault("recorder_received_monotonic_ns", time.monotonic_ns())
        with self._state_lock:
            if not self._accepting:
                self._rejected_after_close += 1
                return False
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                self._dropped_rows += 1
                return False
            self._accepted_rows += 1
            self._queue_high_water_mark = max(self._queue_high_water_mark, self._queue.qsize())
            return True

    @staticmethod
    def _write_jsonl_row(stream, row):
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        stream.write("\n")

    def _writer_main(self):
        self._writer_error = None
        try:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            # Exclusive creation means even a run-ID collision can never truncate prior data.
            with self.jsonl_path.open("x", encoding="utf-8", newline="\n") as jsonl:
                while not self._stopping.is_set() or not self._queue.empty():
                    try:
                        row = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    try:
                        self._write_jsonl_row(jsonl, row)
                        self._summary.observe(row)
                        self._written_rows += 1
                    except Exception as exc:
                        self._writer_error = f"{type(exc).__name__}: {exc}"
                    finally:
                        self._queue.task_done()
            ended_ns = time.monotonic_ns()
            with self._state_lock:
                metadata = deepcopy(self._metadata)
                accepted = self._accepted_rows
                dropped = self._dropped_rows
                rejected = self._rejected_after_close
                high_water = self._queue_high_water_mark
                written = self._written_rows
                writer_error = self._writer_error
            summary = self._summary.build(
                session_id=self.session_id,
                started_ns=self._started_ns,
                ended_ns=ended_ns,
                queue_capacity=self.queue_capacity,
                accepted_rows=accepted,
                written_rows=written,
                dropped_rows=dropped,
                rejected_after_close=rejected,
                queue_high_water_mark=high_water,
                metadata=metadata,
                writer_error=writer_error,
            )
            with self.summary_path.open("x", encoding="utf-8", newline="\n") as summary_file:
                json.dump(summary, summary_file, ensure_ascii=False, indent=2, allow_nan=False)
                summary_file.write("\n")
        except Exception as exc:
            self._writer_error = f"{type(exc).__name__}: {exc}"

    def close(self, timeout_seconds=5.0):
        """Stop accepting rows and wait at most timeout_seconds for the writer to flush."""
        try:
            timeout = max(0.0, float(timeout_seconds))
        except (TypeError, ValueError, OverflowError):
            timeout = 0.0
        with self._state_lock:
            self._accepting = False
            self._stopping.set()
        self._writer.join(timeout=timeout)
        return not self._writer.is_alive() and self._writer_error is None
