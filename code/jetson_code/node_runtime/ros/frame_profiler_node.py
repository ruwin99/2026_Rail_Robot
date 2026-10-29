"""Record frame provenance and resource metrics without subscribing to images."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rail_robot_interfaces.msg import FrameTrace, ModelTiming
from std_msgs.msg import Float64

from node_runtime.frame_profiler import FrameProfiler, parse_tegrastats_line
from .common import qos_events, qos_sensor


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _stamp_fields(stamp):
    try:
        return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}
    except (AttributeError, TypeError, ValueError):
        return None


class FrameProfilerNode(Node):
    """Persist raw trace/timing/FPS events and optional tegrastats samples."""

    def __init__(self, *, output_directory_override=None,
                 enable_tegrastats_override=None, tegrastats_lookup=shutil.which,
                 node_name="rail_frame_profiler", **kwargs):
        super().__init__(node_name, **kwargs)
        descriptor = ParameterDescriptor(read_only=True)
        default_directory = str(Path.home() / "Desktop" / "rail_robot" / "code" / "outputs" / "frame_profiles")
        output_directory = output_directory_override
        if output_directory is None:
            output_directory = self.declare_parameter(
                "output_directory", default_directory, descriptor=descriptor
            ).value
        session_id = self.declare_parameter(
            "session_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            descriptor=descriptor,
        ).value
        queue_capacity = self.declare_parameter(
            "queue_capacity", 512, descriptor=descriptor
        ).value
        enabled = enable_tegrastats_override
        if enabled is None:
            enabled = self.declare_parameter(
                "enable_tegrastats", True, descriptor=descriptor
            ).value
        self._profiler = FrameProfiler(
            os.path.expandvars(os.path.expanduser(str(output_directory))),
            queue_capacity=queue_capacity,
            session_id=str(session_id),
            metadata={"resource_sampler": {
                "state": "disabled", "source": "tegrastats",
                "scope": "whole-Jetson reported rails; not model-only power",
            }},
        )
        self._closed = False
        self._sampler_process = None
        self._sampler_thread = None
        self._sampler_stop = threading.Event()
        self._sampler_status_lock = threading.Lock()
        self._tegrastats_lookup = tegrastats_lookup
        self._tegrastats_status = {"state": "disabled", "source": "tegrastats"}

        self.create_subscription(
            FrameTrace, "/rail_robot/debug/frame_trace", self._on_frame_trace, qos_sensor
        )
        self.create_subscription(
            ModelTiming, "debug/vision/timing", self._on_model_timing, qos_events
        )
        for role in ("front", "side"):
            self.create_subscription(
                Float64, f"debug/camera/{role}/fps",
                lambda message, role=role: self._on_camera_fps(role, message), qos_sensor,
            )
        if enabled:
            self._start_tegrastats()
        self._profiler.set_metadata("resource_sampler", dict(self._tegrastats_status))
        self._profiler.submit({
            "record_type": "resource_sampler_status", **self._tegrastats_status,
        })

    @property
    def output_directory(self):
        return self._profiler.output_directory

    @property
    def jsonl_path(self):
        return self._profiler.jsonl_path

    @property
    def summary_path(self):
        return self._profiler.summary_path

    def _on_frame_trace(self, message):
        """Copy only the compact provenance/timing payload, never image bytes."""
        roles = list(message.model_camera_roles)
        models = list(message.model_ids)
        streams = list(message.model_stream_ids)
        sequences = list(message.model_source_sequences)
        starts = list(message.model_start_monotonic_ns)
        ends = list(message.model_end_monotonic_ns)
        stale = list(message.model_stale)
        count = min(map(len, (roles, models, streams, sequences, starts, ends, stale)), default=0)
        trace = {
            "output_sequence": int(message.output_sequence),
            "front_stream_id": str(message.front_stream_id),
            "front_frame_id": str(message.front_frame_id),
            "front_camera_sequence": int(message.front_camera_sequence),
            "front_capture_complete_monotonic_ns": int(message.front_capture_complete_monotonic_ns),
            "side_stream_id": str(message.side_stream_id),
            "side_frame_id": str(message.side_frame_id),
            "side_camera_sequence": int(message.side_camera_sequence),
            "side_capture_complete_monotonic_ns": int(message.side_capture_complete_monotonic_ns),
            "models": [
                {"camera_role": roles[index], "model_id": models[index],
                 "stream_id": streams[index], "source_sequence": int(sequences[index]),
                 "start_monotonic_ns": int(starts[index]), "end_monotonic_ns": int(ends[index]),
                 "stale": bool(stale[index])}
                for index in range(count)
            ],
            "compose_start_monotonic_ns": int(message.compose_start_monotonic_ns),
            "compose_complete_monotonic_ns": int(message.compose_complete_monotonic_ns),
            "jpeg_complete_monotonic_ns": int(message.jpeg_complete_monotonic_ns),
            "gui_publish_start_monotonic_ns": int(message.gui_publish_start_monotonic_ns),
            "gui_publish_complete_monotonic_ns": int(message.gui_publish_complete_monotonic_ns),
            "missing_camera_count": int(message.missing_camera_count),
            "stale_model_count": int(message.stale_model_count),
            "duplicate_frame_count": int(message.duplicate_frame_count),
            "source_sequence_gap_count": int(message.source_sequence_gap_count),
            "gui_render_drop_count": int(message.gui_render_drop_count),
        }
        self._profiler.submit({
            "record_type": "frame_trace",
            "header_stamp": _stamp_fields(message.header.stamp),
            "gui_frame_id": str(message.header.frame_id),
            "trace": trace,
        })

    def _on_model_timing(self, message):
        self._profiler.submit({
            "record_type": "model_timing",
            "header_stamp": _stamp_fields(message.header.stamp),
            "source_stamp": _stamp_fields(getattr(message, "source_stamp", None)),
            "camera_role": str(message.camera_role),
            "stream_id": str(message.stream_id),
            "sequence": int(message.sequence),
            "model_id": str(message.model),
            "execution_id": str(message.execution_id),
            "processing_start_monotonic_ns": int(message.processing_start_monotonic_ns),
            "processing_end_monotonic_ns": int(message.processing_end_monotonic_ns),
            "processing_ms": _finite_float(message.processing_ms),
        })

    def _on_camera_fps(self, role, message):
        self._profiler.submit({
            "record_type": "camera_fps", "camera_role": role,
            "fps_source": "camera_publisher_reported", "reported_fps": _finite_float(message.data),
            "received_monotonic_ns": time.monotonic_ns(),
        })

    def _start_tegrastats(self):
        try:
            executable = self._tegrastats_lookup("tegrastats")
        except Exception as exc:
            self._tegrastats_status = {
                "state": "unavailable", "source": "tegrastats",
                "reason": f"lookup_failed: {type(exc).__name__}: {exc}",
            }
            return
        if not executable:
            self._tegrastats_status = {
                "state": "unavailable", "source": "tegrastats",
                "reason": "executable_not_found",
            }
            return
        try:
            self._sampler_process = subprocess.Popen(
                [executable, "--interval", "1000"], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            self._tegrastats_status = {
                "state": "running", "source": "tegrastats", "interval_ms": 1000,
                "scope": "whole-Jetson reported rails; not model-only power",
            }
            self._sampler_thread = threading.Thread(
                target=self._read_tegrastats, name="rail-tegrastats-reader", daemon=True
            )
            self._sampler_thread.start()
        except Exception as exc:
            self._sampler_process = None
            self._tegrastats_status = {
                "state": "unavailable", "source": "tegrastats",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def _read_tegrastats(self):
        process = self._sampler_process
        if process is None:
            return
        if process.stdout is None:
            self._mark_tegrastats_unavailable("stdout_unavailable")
            return
        try:
            for line in process.stdout:
                if self._sampler_stop.is_set():
                    break
                sample = parse_tegrastats_line(line, time.monotonic_ns())
                self._profiler.submit({"record_type": "resource_sample", "sample": sample})
            if not self._sampler_stop.is_set():
                return_code = getattr(process, "returncode", None)
                self._mark_tegrastats_unavailable(
                    "process_exited_or_output_closed" if return_code is None
                    else f"process_exited_with_code_{return_code}"
                )
        except Exception as exc:
            if not self._sampler_stop.is_set():
                self._mark_tegrastats_unavailable(f"stdout_read_failed: {type(exc).__name__}: {exc}")
                self.get_logger().warning(f"tegrastats sampling stopped: {exc}")

    def _mark_tegrastats_unavailable(self, reason):
        with self._sampler_status_lock:
            if self._sampler_stop.is_set() or self._tegrastats_status.get("state") != "running":
                return
            self._tegrastats_status = {
                "state": "unavailable", "source": "tegrastats", "reason": str(reason),
            }
            status = dict(self._tegrastats_status)
        self._profiler.set_metadata("resource_sampler", status)
        self._profiler.submit({"record_type": "resource_sampler_status", **status})

    def _stop_tegrastats(self):
        self._sampler_stop.set()
        process = self._sampler_process
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=0.5)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=0.5)
                except Exception:
                    pass
            try:
                if process.stdout is not None:
                    process.stdout.close()
            except Exception:
                pass
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=0.5)
        with self._sampler_status_lock:
            if self._tegrastats_status.get("state") == "running":
                self._tegrastats_status = dict(self._tegrastats_status, state="stopped")
            status = dict(self._tegrastats_status)
        self._profiler.set_metadata("resource_sampler", status)

    def prepare_shutdown(self):
        if self._closed:
            return
        self._closed = True
        self._stop_tegrastats()
        self._profiler.set_metadata("resource_sampler", dict(self._tegrastats_status))
        if not self._profiler.close(timeout_seconds=2.0):
            detail = self._profiler.writer_error or "writer did not exit before its shutdown deadline"
            self.get_logger().error(f"Frame profiler output was not finalized: {detail}")

    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
