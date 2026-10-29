"""On-demand full-frame capture inspection with an independent responsive GUI."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import queue
import threading
import time
from uuid import uuid4

import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rail_robot_interfaces.msg import CameraFrame as CameraFrameMessage, CaptureDebugResult
from std_msgs.msg import Empty
from std_srvs.srv import Trigger

from ..camera import realtime_camera_frame_is_fresh
from ..data import CameraFrame
from ..display import (
    annotate_foreign_capture_results,
    annotate_front_capture_results,
    open_latest_frame_display,
)
from ..inspection import prioritize_capture_crack_over_rust
from ..vision import suppress_front_obstacles_overlapping_cracks
from .capture_debug_models import (
    create_capture_debug_detectors,
    validate_capture_crack_result,
    validate_primary_result,
)
from .common import qos_events, qos_sensor, stamp_seconds
from .deployment import close_detectors
from .images import encode_image, message_to_frame


DEBUG_DEFAULTS = {
    "capture_role": "front",
    "show_gui": True,
    "capture_output_directory": "~/rail_robot_data/debug_captures",
}


def _parameter(node, name):
    if node.has_parameter(name):
        return node.get_parameter(name).value
    return node.declare_parameter(
        name, DEBUG_DEFAULTS[name], descriptor=ParameterDescriptor(read_only=True)
    ).value


def _save_jpeg(path, frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode capture JPEG.")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            data = encoded.tobytes()
            if stream.write(data) != len(data):
                raise OSError("Incomplete capture JPEG write.")
    except FileExistsError:
        raise
    except Exception:
        path.unlink(missing_ok=True)
        raise


class CaptureDebugNode(Node):
    """Snapshot one fresh original; one worker owns all inference and model close."""

    def __init__(self, primary_detector=None, crack_detector=None, *,
                 capture_role=None, show_gui=None, output_directory=None,
                 display_factory=open_latest_frame_display,
                 node_name="capture_debug", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self._lock = threading.RLock()
        self._closed = self._stopping = self._busy = self._quit = False
        self._stop = threading.Event()
        self._jobs, self._completed = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self._worker = self._display = None
        self._detectors = (primary_detector, crack_detector)
        self._latest = self._latest_header = self._latest_key = None
        self._captured_key = self._capture_view = None
        self._status = "Waiting for a fresh camera frame. S: capture, Q/Esc: quit"
        try:
            self._role = _parameter(self, "capture_role") if capture_role is None else capture_role
            if self._role not in ("side", "front", "foreign"):
                raise ValueError("capture_role must be front, side, or foreign.")
            self._camera_role = "side" if self._role == "side" else "top"
            show_gui = _parameter(self, "show_gui") if show_gui is None else show_gui
            if not isinstance(show_gui, bool):
                raise ValueError("show_gui must be a boolean.")
            output_directory = _parameter(self, "capture_output_directory") if output_directory is None else output_directory
            if not str(output_directory).strip():
                raise ValueError("capture_output_directory must not be empty.")
            self._output = Path(output_directory).expanduser().resolve() / self._role
            if primary_detector is None:
                if crack_detector is not None:
                    raise ValueError("Inject both capture detectors, or neither.")
                self._detectors = create_capture_debug_detectors(self, self._role)
            elif self._role == "foreign":
                if crack_detector is not None:
                    raise ValueError("Foreign capture must not inject a crack detector.")
                self._detectors = (primary_detector, None)
            elif crack_detector is None:
                raise ValueError("Inject both capture detectors, or neither.")
            self._results = self.create_publisher(CaptureDebugResult, f"debug/{self._role}/result", qos_events)
            self._camera_subscription = self.create_subscription(
                CameraFrameMessage, f"camera/{self._camera_role}", self._receive, qos_sensor)
            self._trigger_subscription = self.create_subscription(
                Empty, f"debug/{self._role}/capture", self._trigger, qos_events)
            self._capture_service = self.create_service(Trigger, f"debug/{self._role}/capture", self._capture)
            if show_gui:
                self._display = display_factory()
            self._timer = self.create_timer(1.0 / 30.0, self._tick)
            self._worker = threading.Thread(target=self._work, name=f"{self._role}-capture", daemon=True)
            self._worker.start()
        except Exception:
            self.destroy_node()
            raise

    def _receive(self, message):
        with self._lock:
            if self._closed or self._stopping or message.camera_role != self._camera_role:
                return
            try:
                if not message.stream_id:
                    raise ValueError("Camera stream ID is missing.")
                key = (message.stream_id, message.sequence, stamp_seconds(message.header.stamp))
                if self._latest_key is not None:
                    old_stream, old_sequence, old_stamp = self._latest_key
                    if key[2] <= old_stamp or (key[0] == old_stream and key[1] <= old_sequence):
                        return
                frame = message_to_frame(message, self.get_clock().now().nanoseconds / 1e9, time.monotonic())
                self._latest, self._latest_key = frame, key
                self._latest_header = deepcopy(message.header)
            except (ValueError, TypeError) as exc:
                self.get_logger().warning(f"Capture input rejected: {exc}")

    def readiness(self):
        with self._lock:
            if self._closed or self._stopping:
                return False, "Capture debug is stopping."
            if self._busy:
                return False, "A capture is already being analyzed."
            if self._latest is None or not realtime_camera_frame_is_fresh(self._latest):
                return False, "No fresh camera original is available."
            return True, f"Fresh {self._role.upper()} original available."

    def _accept_capture(self):
        with self._lock:
            ready, reason = self.readiness()
            if not ready:
                self._status = reason
                return False, reason
            if self._captured_key == self._latest_key:
                self._status = "This camera sequence was already captured; wait for a new frame."
                return False, self._status
            capture_id = uuid4().hex
            frame = CameraFrame(self._latest.frame.copy(), self._latest.sequence, self._latest.read_completed_at)
            self._busy = True
            self._captured_key = self._latest_key
            self._capture_view = frame.frame
            self._status = f"Capturing {self._role} seq {frame.sequence}; analysis running..."
            self._jobs.put_nowait((capture_id, frame, deepcopy(self._latest_header), self._latest_key))
            return True, f"Accepted {capture_id}; completion: debug/{self._role}/result"

    def _capture(self, request, response):
        del request
        response.success, response.message = self._accept_capture()
        return response

    def _trigger(self, message):
        del message
        success, detail = self._accept_capture()
        (self.get_logger().info if success else self.get_logger().warning)(detail)

    def _analyze(self, job):
        capture_id, frame, header, key = job
        result = CaptureDebugResult()
        result.header = header
        result.capture_role, result.camera_role = self._role, self._camera_role
        result.stream_id, result.sequence, result.capture_id = key[0], frame.sequence, capture_id
        started = time.perf_counter()
        try:
            captured_at = datetime.fromtimestamp(key[2], tz=timezone.utc)
            basename = f"{self._role}_{captured_at:%Y%m%d_%H%M%S_%f}_{frame.sequence}_{capture_id[:8]}"
            raw_path = self._output / f"{basename}_raw.jpg"
            _save_jpeg(raw_path, frame.frame)
            result.raw_path = str(raw_path)
            if self._stop.is_set():
                raise RuntimeError("Capture cancelled during shutdown.")
            start = time.perf_counter()
            primary = self._detectors[0].detect(frame.frame)
            result.primary_seconds = time.perf_counter() - start
            result.primary_method = primary.method
            validate_primary_result(primary, self._role)
            if self._role == "foreign":
                result.obstacle_count = len(primary.detections)
                annotated = annotate_foreign_capture_results(frame.frame, primary)
            else:
                if self._stop.is_set():
                    raise RuntimeError("Capture cancelled during shutdown.")
                start = time.perf_counter()
                crack = self._detectors[1].detect(frame.frame)
                result.crack_seconds = time.perf_counter() - start
                result.crack_method = crack.method
                result.crack_ratio = validate_capture_crack_result(crack)
                if self._role == "front":
                    primary = suppress_front_obstacles_overlapping_cracks(primary, crack.mask)
                    result.obstacle_count = len(primary.detections)
                    annotated = annotate_front_capture_results(frame.frame, primary, crack)
                else:
                    from rust_detector import annotate
                    from crack_detector import annotate_cracks
                    primary = prioritize_capture_crack_over_rust(primary, crack)
                    result.rust_ratio = float(primary.rust_ratio)
                    annotated = annotate_cracks(annotate(frame.frame, primary), crack, None)
            if self._stop.is_set():
                raise RuntimeError("Capture cancelled during shutdown.")
            annotated_path = self._output / f"{basename}_annotated.jpg"
            _save_jpeg(annotated_path, annotated)
            result.annotated_path = str(annotated_path)
            result.annotated = encode_image(annotated, header)
            result.success = True
            view = annotated
        except Exception as exc:
            result.success, result.error = False, str(exc)
            view = frame.frame
        result.total_seconds = time.perf_counter() - started
        return result, view

    def _work(self):
        try:
            while not self._stop.is_set():
                try:
                    job = self._jobs.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._stop.is_set():
                    break
                outcome = self._analyze(job)
                if not self._stop.is_set():
                    self._completed.put_nowait(outcome)
        finally:
            # The worker retains exclusive model ownership even if shutdown
            # times out waiting for a driver; never free an in-use CUDA context.
            close_detectors(self._detectors)

    def _tick(self):
        with self._lock:
            if self._closed or self._stopping:
                return
            try:
                result, view = self._completed.get_nowait()
            except queue.Empty:
                pass
            else:
                self._busy = False
                self._capture_view = view
                self._results.publish(result)
                if result.success and self._role == "foreign":
                    self._status = (
                        f"seq {result.sequence}: foreign YOLO {result.primary_seconds * 1000:.1f} ms, "
                        f"detections {result.obstacle_count}; saved. S: capture"
                    )
                elif result.success:
                    self._status = (
                        f"seq {result.sequence}: primary {result.primary_seconds * 1000:.1f} ms, "
                        f"crack {result.crack_seconds * 1000:.1f} ms; saved. S: capture"
                    )
                else:
                    self._status = f"seq {result.sequence} failed: {result.error}"
                (self.get_logger().info if result.success else self.get_logger().error)(self._status)
            if self._display is None:
                return
            if self._display.check_status():
                self._quit = True
                return
            key = self._display.poll_key()
            if key in (27, ord("q"), ord("Q")):
                self._quit = True
                return
            if key in (ord("s"), ord("S")):
                self._accept_capture()
            self._display.submit(self._compose_view())

    def _compose_view(self):
        # Captured results have their own tile and never overlay a newer live frame.
        live = self._latest.frame if self._latest is not None else np.zeros((720, 1280, 3), np.uint8)
        captured = self._capture_view if self._capture_view is not None else np.zeros_like(live)
        canvas = np.zeros((440, 1280, 3), np.uint8)
        canvas[45:405, :640] = cv2.resize(live, (640, 360))
        canvas[45:405, 640:] = cv2.resize(captured, (640, 360))
        sequence = self._latest.sequence if self._latest is not None else "-"
        captured_sequence = self._captured_key[1] if self._captured_key is not None else "-"
        for label, position in ((f"{self._role.upper()} LIVE seq {sequence}", (12, 28)),
                                (f"CAPTURE / RESULT seq {captured_sequence}", (652, 28)),
                                (self._status[:155], (12, 430))):
            cv2.putText(canvas, label, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        return canvas

    def shutdown_requested(self):
        return self._quit

    def prepare_shutdown(self):
        with self._lock:
            self._stopping = True
            self._stop.set()

    def destroy_node(self):
        with self._lock:
            if self._closed:
                return True
            self._closed = self._stopping = True
            self._stop.set()
        if self._worker is not None and self._worker.ident is not None:
            self._worker.join(timeout=5.0)
            if self._worker.is_alive():
                self.get_logger().error("Capture worker is still in inference; model cleanup deferred to its exit.")
        else:
            close_detectors(self._detectors)
        if self._display is not None:
            try:
                self._display.close()
            except Exception as exc:
                self.get_logger().error(f"Capture display shutdown failed: {exc}")
        return super().destroy_node()
