"""Preview core: HighGUI ownership and image annotations.

Persistent summary values are for display only and never become control input.
"""
from __future__ import annotations

import os
import queue
import threading
import time

import cv2
import numpy as np

from crack_detector import CRACK_COLOR_BGR, annotate_cracks
from obstacle_detector import ObstacleDetectionResult
from rust_detector import CLASS_COLORS_BGR

from .data import RealtimeInferenceOutcome
from .settings import *
from .vision import crack_stop_roi_bounds, obstacle_control_roi_bounds, water_control_roi_bounds

__all__ = [
    "LatestFrameDisplay",
    "open_latest_frame_display",
    "graphical_display_environment_available",
    "annotate_realtime_control_results",
    "annotate_top_realtime_results",
    "annotate_front_capture_results",
    "draw_roi_guide",
    "draw_realtime_roi_guide",
    "stack_dual_camera_displays",
    "draw_realtime_result_summary",
    "draw_realtime_test_statistics",
    "format_realtime_test_statistics",
    "RealtimeDisplayCache"
]


class LatestFrameDisplay:
    """Present only the newest submitted image from one HighGUI owner thread."""

    def __init__(self, window_name: str = WINDOW_NAME) -> None:
        self._window_name = window_name
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._latest: np.ndarray | None = None
        self._keys: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._error: Exception | None = None
        self._exit_requested = False
        self._closed = False
        self._dropped_frames = 0
        self._presented_frames = 0
        self._first_presented_at: float | None = None
        self._last_presented_at: float | None = None
        self._thread = threading.Thread(
            target=self._display_loop,
            name="latest-frame-highgui",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_frames(self) -> int:
        with self._condition:
            return self._dropped_frames

    def submit(self, frame: np.ndarray) -> bool:
        """Overwrite any not-yet-presented image without blocking control."""

        if not isinstance(frame, np.ndarray):
            raise TypeError("Display frame must be a NumPy array.")
        with self._condition:
            self._raise_if_failed_locked()
            if self._exit_requested or self._closed:
                return False
            if self._latest is not None:
                self._dropped_frames += 1
            self._latest = frame
            self._condition.notify_all()
            return True

    def check_status(self) -> bool:
        """Raise a UI failure or report an operator/window close request."""

        with self._condition:
            self._raise_if_failed_locked()
            return self._exit_requested

    def poll_key(self) -> int | None:
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return None

    def completed_statistics(self) -> tuple[int, float]:
        """Return completed display intervals and their wall-clock span."""

        with self._condition:
            if (
                self._presented_frames <= 1
                or self._first_presented_at is None
                or self._last_presented_at is None
            ):
                return 0, 0.0
            return (
                self._presented_frames - 1,
                max(0.0, self._last_presented_at - self._first_presented_at),
            )

    def close(self) -> None:
        self._stop_requested.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(LATEST_DISPLAY_CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("HighGUI display thread did not stop cleanly.")
        with self._condition:
            self._raise_if_failed_locked()

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"HighGUI display failed: {self._error}") from self._error

    def _display_loop(self) -> None:
        shown_once = False
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            while not self._stop_requested.is_set():
                with self._condition:
                    if self._latest is None:
                        self._condition.wait(DISPLAY_EVENT_POLL_SECONDS)
                    frame = self._latest
                    self._latest = None
                if frame is not None:
                    cv2.imshow(self._window_name, frame)
                    shown_once = True
                key = cv2.waitKey(1) & 0xFF
                completed_at = time.perf_counter()
                if frame is not None:
                    with self._condition:
                        self._presented_frames += 1
                        if self._first_presented_at is None:
                            self._first_presented_at = completed_at
                        self._last_presented_at = completed_at
                if key in (ord("q"), 27):
                    with self._condition:
                        self._exit_requested = True
                        self._condition.notify_all()
                    break
                if key not in (0, 255):
                    self._keys.put(key)
                if shown_once and cv2.getWindowProperty(
                    self._window_name,
                    cv2.WND_PROP_VISIBLE,
                ) == 0.0:
                    with self._condition:
                        self._exit_requested = True
                        self._condition.notify_all()
                    break
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                with self._condition:
                    if self._error is None:
                        self._error = exc
            with self._condition:
                self._closed = True
                self._condition.notify_all()


def open_latest_frame_display() -> LatestFrameDisplay:
    return LatestFrameDisplay()


def graphical_display_environment_available(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether this process has an X11 or Wayland display target."""

    values = os.environ if environment is None else environment
    return any(
        str(values.get(name, "")).strip()
        for name in ("DISPLAY", "WAYLAND_DISPLAY")
    )


def annotate_realtime_control_results(
    frame: np.ndarray,
    rust_result,
    crack_result=None,
    *,
    copy_frame: bool = True,
) -> np.ndarray:
    """Map cropped realtime results back onto original-frame coordinates."""

    water_top, water_bottom = water_control_roi_bounds(frame)
    crack_top, crack_bottom = crack_stop_roi_bounds(frame)
    full_shape = frame.shape[:2]
    water_shape = (water_bottom - water_top, full_shape[1])
    crack_shape = (crack_bottom - crack_top, full_shape[1])

    output = frame.copy() if copy_frame else frame
    if rust_result is not None:
        rust_mask = getattr(rust_result, "mask", None)
        if not isinstance(rust_mask, np.ndarray) or rust_mask.shape != water_shape:
            raise ValueError(
                "Realtime rust mask shape does not match the water-control ROI."
            )
        rust_class_map = getattr(rust_result, "class_map", None)
        rust_overlay = output.copy()
        rust_overlay_roi = rust_overlay[water_top:water_bottom, :]
        if rust_class_map is not None:
            if (
                not isinstance(rust_class_map, np.ndarray)
                or rust_class_map.shape != water_shape
            ):
                raise ValueError(
                    "Realtime rust class-map shape does not match the "
                    "water-control ROI."
                )
            for class_id, color in CLASS_COLORS_BGR.items():
                rust_overlay_roi[rust_class_map == class_id] = color
        else:
            rust_overlay_roi[rust_mask > 0] = (0, 95, 255)
        output = cv2.addWeighted(rust_overlay, 0.35, output, 0.65, 0)
        rust_box_color = (60, 180, 75)
        for class_name, class_id in (("Fair", 1), ("Poor", 2), ("Severe", 3)):
            if rust_result.class_ratios.get(class_name, 0.0) > 0.0:
                rust_box_color = CLASS_COLORS_BGR[class_id]
        for x, y, width, height in rust_result.boxes:
            cv2.rectangle(
                output[water_top:water_bottom, :],
                (x, y),
                (min(x + width, FRAME_WIDTH - 1), min(y + height, water_shape[0] - 1)),
                rust_box_color,
                2,
            )

    if crack_result is None:
        return output
    crack_mask = getattr(crack_result, "mask", None)
    if not isinstance(crack_mask, np.ndarray) or crack_mask.shape != crack_shape:
        raise ValueError(
            "Realtime crack mask shape does not match the crack-stop ROI."
        )
    crack_overlay = output.copy()
    crack_overlay[crack_top:crack_bottom, :][crack_mask > 0] = CRACK_COLOR_BGR
    output = cv2.addWeighted(crack_overlay, 0.4, output, 0.6, 0)
    crack_box_color = (0, 0, 255) if crack_result.detected else (60, 180, 75)
    for x, y, width, height in crack_result.boxes:
        cv2.rectangle(
            output[crack_top:crack_bottom, :],
            (x, y),
            (min(x + width, FRAME_WIDTH - 1), min(y + height, crack_shape[0] - 1)),
            crack_box_color,
            2,
        )
    return output


def annotate_top_realtime_results(
    frame: np.ndarray,
    obstacle_result: ObstacleDetectionResult | None,
    crack_result,
) -> np.ndarray:
    """Draw obstacle boxes first so an overlapping crack remains visible."""

    output = frame.copy()
    if obstacle_result is not None:
        # Draw into the ROI view so even the rectangle stroke stays inside it.
        roi_top, roi_bottom = obstacle_control_roi_bounds(output)
        obstacle_view = output[roi_top:roi_bottom, :]
        for detection in obstacle_result.detections:
            left, top, right, bottom = detection.box_xyxy
            left, right = max(0.0, left), min(float(FRAME_WIDTH), right)
            top, bottom = max(float(roi_top), top), min(float(roi_bottom), bottom)
            if right <= left or bottom <= top:
                continue
            x1, y1, x2, y2 = (
                int(round(value)) for value in (left, top, right, bottom)
            )
            cv2.rectangle(
                obstacle_view,
                (x1, y1 - roi_top),
                (min(x2, FRAME_WIDTH - 1), min(y2, roi_bottom - 1) - roi_top),
                (0, 220, 255),
                2,
            )
    return annotate_realtime_control_results(
        output,
        None,
        crack_result,
        copy_frame=False,
    )


def annotate_front_capture_results(
    frame: np.ndarray,
    obstacle_result: ObstacleDetectionResult,
    crack_result,
) -> np.ndarray:
    """Draw FRONT/TOP obstacle boxes, then the full-frame capture crack mask."""

    output = frame.copy()
    for detection in obstacle_result.detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.box_xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 255), 2)
    return annotate_cracks(output, crack_result, None)


def annotate_foreign_capture_results(
    frame: np.ndarray,
    obstacle_result: ObstacleDetectionResult,
) -> np.ndarray:
    """Draw only foreign-matter YOLO boxes and confidence labels on a copy."""

    output = frame.copy()
    color = (0, 220, 255)
    for detection in obstacle_result.detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.box_xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"FOREIGN {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        label_bottom = label_top + text_height + baseline + 6
        label_right = min(output.shape[1] - 1, x1 + text_width + 6)
        cv2.rectangle(output, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            output,
            label,
            (x1 + 3, label_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def draw_roi_guide(frame, roi_display=None):
    """Show that capture inference uses the complete native camera frame."""

    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        raise ValueError("Capture guide requires the native 1280x720 BGR frame.")
    display = frame.copy() if roi_display is None else roi_display.copy()
    if display.shape != frame.shape:
        raise ValueError(
            "Capture result shape must match the full camera frame: "
            f"expected {frame.shape}, got {display.shape}."
        )

    guide_color = (0, 255, 255)
    cv2.rectangle(
        display,
        (1, 1),
        (FRAME_WIDTH - 2, FRAME_HEIGHT - 2),
        guide_color,
        2,
    )
    cv2.putText(
        display,
        "CAPTURE ROI FULL FRAME 1280x720",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        guide_color,
        2,
        cv2.LINE_AA,
    )

    return display


def draw_realtime_roi_guide(
    frame,
    annotated_frame=None,
    *,
    primary_roi_label: str = "RUST",
):
    """Show full-frame realtime results and fixed ROI boundaries."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("Realtime camera frame must be an HxWx3 image.")
    water_top, water_bottom = water_control_roi_bounds(frame)
    if primary_roi_label.startswith("OBSTACLE"):
        water_top, water_bottom = obstacle_control_roi_bounds(frame)
    crack_top, crack_bottom = crack_stop_roi_bounds(frame)
    display = frame.copy()
    if annotated_frame is not None:
        if annotated_frame.shape != frame.shape:
            raise ValueError(
                "Realtime annotated frame shape does not match the camera frame: "
                f"expected {frame.shape}, got {annotated_frame.shape}."
            )
        display = annotated_frame.copy()

    frame_width = display.shape[1]
    water_guide_color = (255, 220, 0)
    crack_guide_color = (255, 0, 255)
    cv2.line(
        display,
        (0, water_top),
        (frame_width - 1, water_top),
        water_guide_color,
        2,
    )
    cv2.putText(
        display,
        f"{primary_roi_label} ROI FULL WIDTH (Y={water_top}:{water_bottom})",
        (12, water_top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        water_guide_color,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        display,
        (0, crack_top),
        (frame_width - 1, crack_top),
        crack_guide_color,
        2,
    )
    cv2.putText(
        display,
        f"CRACK ROI FULL WIDTH (Y={crack_top}:{crack_bottom})",
        (12, crack_top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        crack_guide_color,
        2,
        cv2.LINE_AA,
    )
    return display


def stack_dual_camera_displays(
    side_display: np.ndarray,
    top_display: np.ndarray,
) -> np.ndarray:
    """Place native SIDE and TOP camera displays side by side."""

    expected_shape = (FRAME_HEIGHT, FRAME_WIDTH, 3)
    if side_display.shape != expected_shape or top_display.shape != expected_shape:
        raise ValueError(
            "Dual-camera displays must both be native 1280x720 BGR frames."
        )
    return np.ascontiguousarray(
        np.hstack(
            (
                cv2.resize(side_display, (640, 360)),
                cv2.resize(top_display, (640, 360)),
            )
        )
    )


def draw_realtime_result_summary(
    frame: np.ndarray,
    rust_result,
    crack_result,
    *,
    crack_bypassed: bool = False,
) -> np.ndarray:
    """Draw the persistent UI-only result summary in the bottom-right corner."""

    output = frame.copy()
    if rust_result is None:
        rust_label = "RUST WAITING"
    else:
        fair = rust_result.class_ratios.get("Fair", 0.0) * 100.0
        poor = rust_result.class_ratios.get("Poor", 0.0) * 100.0
        severe = rust_result.class_ratios.get("Severe", 0.0) * 100.0
        rust_label = (
            f"RUST {rust_result.rust_ratio * 100.0:.2f}%  "
            f"F {fair:.2f}  P {poor:.2f}  S {severe:.2f}"
        )
    if crack_bypassed:
        crack_label = "CRACK BYPASS(TEST)"
    elif crack_result is None:
        crack_label = "CRACK WAITING"
    else:
        crack_state = "DETECTED" if crack_result.detected else "CLEAR"
        crack_label = (
            f"CRACK {crack_state}  {crack_result.crack_ratio * 100.0:.3f}%"
        )

    labels = (rust_label, crack_label)
    font_scale = 0.56
    thickness = 2
    sizes = [
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        for label in labels
    ]
    max_width = max(size[0][0] for size in sizes)
    max_height = max(size[0][1] for size in sizes)
    max_baseline = max(size[1] for size in sizes)
    line_step = max_height + max_baseline + 8
    margin = 12
    padding = 8
    label_x = max(margin + padding, output.shape[1] - margin - max_width)
    second_baseline_y = output.shape[0] - margin - padding
    first_baseline_y = second_baseline_y - line_step
    box_left = max(0, label_x - padding)
    box_top = max(0, first_baseline_y - max_height - padding)
    box_right = min(output.shape[1] - 1, output.shape[1] - margin + padding)
    box_bottom = min(
        output.shape[0] - 1,
        second_baseline_y + max_baseline + padding,
    )
    cv2.rectangle(
        output,
        (box_left, box_top),
        (box_right, box_bottom),
        (20, 35, 50),
        -1,
    )
    for label, baseline_y, color in (
        (rust_label, first_baseline_y, (0, 180, 255)),
        (crack_label, second_baseline_y, (0, 0, 255)),
    ):
        cv2.putText(
            output,
            label,
            (label_x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return output


def draw_realtime_test_statistics(frame, statistics_label: str):
    """Overlay control and completed display-loop timing on a preview frame."""

    output = frame.copy()
    label_lines = statistics_label.split(" | ")
    label_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        for line in label_lines
    ]
    label_width = max(size[0][0] for size in label_sizes)
    label_height = max(size[0][1] for size in label_sizes)
    label_baseline = max(size[1] for size in label_sizes)
    line_step = label_height + label_baseline + 8
    label_x = 12
    label_y = max(
        label_height + 12,
        output.shape[0] - 20 - line_step * (len(label_lines) - 1),
    )
    label_right = min(output.shape[1] - 1, label_x + label_width + 16)
    cv2.rectangle(
        output,
        (label_x - 8, label_y - label_height - 8),
        (
            label_right,
            label_y
            + line_step * (len(label_lines) - 1)
            + label_baseline
            + 8,
        ),
        (20, 35, 50),
        -1,
    )
    for line_index, line in enumerate(label_lines):
        cv2.putText(
            output,
            line,
            (label_x, label_y + line_step * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 220, 90),
            2,
            cv2.LINE_AA,
        )
    return output


def format_realtime_test_statistics(
    frame_count: int,
    decision_span_seconds: float,
    total_control_latency_seconds: float,
    total_rust_seconds: float,
    total_crack_seconds: float,
    *,
    app_frame_count: int = 0,
    total_app_seconds: float = 0.0,
    multitask: bool = False,
) -> str:
    """Return truthful control-path and completed display-loop statistics."""

    if frame_count <= 0:
        raise ValueError("Realtime test statistics require at least one frame.")
    control_ms = total_control_latency_seconds * 1000.0 / frame_count
    rust_ms = total_rust_seconds * 1000.0 / frame_count
    crack_ms = total_crack_seconds * 1000.0 / frame_count
    control_fps = (
        (frame_count - 1) / decision_span_seconds
        if frame_count > 1 and decision_span_seconds > 0.0
        else None
    )
    if app_frame_count > 0:
        app_ms = total_app_seconds * 1000.0 / app_frame_count
        app_fps = (
            app_frame_count / total_app_seconds if total_app_seconds > 0.0 else 0.0
        )
        app_label = f"APP={app_fps:.1f} FPS {app_ms:.1f} ms"
    else:
        app_label = "APP=warming up"
    control_rate = (
        "CONTROL=warming up"
        if control_fps is None
        else f"CONTROL={control_fps:.1f} FPS"
    )
    control_label = f"{control_rate} latency={control_ms:.1f} ms"
    if multitask:
        return (
            f"{control_label} | {app_label} | "
            f"multitask={rust_ms:.1f} ms frames={frame_count}"
        )
    return (
        f"{control_label} | {app_label} | "
        f"rust={rust_ms:.1f} ms crack={crack_ms:.1f} ms frames={frame_count}"
    )


class RealtimeDisplayCache:
    """Persist numeric UI-summary results; never use them for control/overlays."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rust_result = None
        self.crack_result = None

    def update(self, outcome: RealtimeInferenceOutcome | None) -> None:
        if outcome is None:
            return
        if outcome.display_rust_result is not None:
            self.rust_result = outcome.display_rust_result
        if outcome.display_crack_result is not None:
            self.crack_result = outcome.display_crack_result
