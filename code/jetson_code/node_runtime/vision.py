"""Realtime inference core: fixed ROIs, model scheduling, and result expiry.

Detector instances are supplied by the coordinator so shared TensorRT models keep
one execution owner. No camera, UART, report writer, or GUI is created here.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from obstacle_detector import ObstacleDetection, ObstacleDetectionResult

from .camera import realtime_camera_frame_is_fresh
from .data import CameraFrame, RealtimeInferenceOutcome, TimedRealtimeResult, TopRealtimeInferenceOutcome
from .settings import *

if TYPE_CHECKING:
    from hrsegnet_crack_detector import HrSegNetCrackDetector
    from obstacle_detector import ObstacleDetector

__all__ = [
    "water_control_roi_bounds",
    "obstacle_control_roi_bounds",
    "crack_stop_roi_bounds",
    "extract_realtime_control_rois",
    "detect_realtime_control",
    "AlternatingTopRealtimeInference",
    "AlternatingRealtimeInference",
    "dual_realtime_inputs_are_fresh",
    "realtime_role_inputs_are_fresh",
    "clip_realtime_obstacles_to_roi",
    "suppress_front_obstacles_overlapping_cracks"
]


def water_control_roi_bounds(image: np.ndarray) -> tuple[int, int]:
    """Return the bottom third used for rust inference and water control."""

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Water control requires a two- or three-dimensional image.")
    if image.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            "Realtime control requires the original "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} camera frame; received "
            f"{image.shape[1]}x{image.shape[0]}."
        )
    return REALTIME_RUST_ROI_TOP, REALTIME_RUST_ROI_BOTTOM


def obstacle_control_roi_bounds(image: np.ndarray) -> tuple[int, int]:
    """Return the bottom third used for realtime obstacle decisions."""

    water_control_roi_bounds(image)  # Validate the shared native frame size.
    return REALTIME_OBSTACLE_ROI_TOP, REALTIME_OBSTACLE_ROI_BOTTOM


def crack_stop_roi_bounds(image: np.ndarray) -> tuple[int, int]:
    """Return the full-width strip ending at the rust ROI's lower boundary."""

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Crack stop control requires a two- or three-dimensional image.")
    if image.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            "Realtime control requires the original "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} camera frame; received "
            f"{image.shape[1]}x{image.shape[0]}."
        )
    return REALTIME_CRACK_ROI_TOP, REALTIME_CRACK_ROI_BOTTOM


def extract_realtime_control_rois(frame: np.ndarray):
    """Crop fixed full-width rust and crack inputs from the original frame."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("Realtime camera frame must be an HxWx3 image.")
    water_top, water_bottom = water_control_roi_bounds(frame)
    crack_top, crack_bottom = crack_stop_roi_bounds(frame)
    return (
        frame[water_top:water_bottom, :],
        frame[crack_top:crack_bottom, :],
    )


def detect_realtime_control(
    water_roi: np.ndarray,
    crack_roi: np.ndarray,
    student_detector,
    realtime_crack_detector,
    *,
    multitask_enabled: bool,
    hybrid_enabled: bool,
):
    """Select rust and crack results without mixing detector roles."""

    if hybrid_enabled:
        rust_result = student_detector.detect(water_roi)
        _unused_multitask_rust, crack_result = realtime_crack_detector.detect(
            water_roi
        )
        return rust_result, crack_result
    if multitask_enabled:
        return student_detector.detect(water_roi)
    rust_result = student_detector.detect(water_roi)
    crack_result = (
        None
        if realtime_crack_detector is None
        else realtime_crack_detector.detect(crack_roi)
    )
    return rust_result, crack_result


class AlternatingTopRealtimeInference:
    """Alternate full-frame obstacle and crack ROI inference for the top camera."""

    def __init__(
        self,
        obstacle_detector: ObstacleDetector,
        crack_detector: HrSegNetCrackDetector,
        *,
        max_result_age_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        self.obstacle_detector = obstacle_detector
        self.crack_detector = crack_detector
        self.max_result_age_seconds = max_result_age_seconds
        self.crack_inference_count = 0
        self.reset()

    def reset(self) -> None:
        # Offset the shared heavy crack model from the side camera: side starts
        # with rust while top starts with crack.
        self._next_role = "crack"
        self._obstacle: TimedRealtimeResult | None = None
        self._crack: TimedRealtimeResult | None = None

    def expire_cache(self, *, now: float | None = None) -> None:
        """Drop only expired results while preserving the next model phase."""

        if now is None:
            now = time.monotonic()
        if self._obstacle is not None and not self._is_fresh(self._obstacle, now):
            self._obstacle = None
        if self._crack is not None and not self._is_fresh(self._crack, now):
            self._crack = None

    def process(
        self,
        camera_frame: CameraFrame,
        obstacle_frame: np.ndarray,
        crack_roi: np.ndarray,
    ) -> TopRealtimeInferenceOutcome:
        obstacle_seconds = 0.0
        crack_seconds = 0.0
        obstacle_start_ns = obstacle_end_ns = 0
        crack_start_ns = crack_end_ns = 0
        obstacle_execution_id = crack_execution_id = ""
        display_obstacle_result = None
        display_crack_result = None
        if self._next_role == "obstacle":
            obstacle_execution_id = uuid4().hex
            obstacle_start_ns = time.monotonic_ns()
            obstacle_result = self.obstacle_detector.detect(obstacle_frame)
            obstacle_end_ns = time.monotonic_ns()
            obstacle_seconds = (obstacle_end_ns - obstacle_start_ns) / 1e9
            obstacle_result = clip_realtime_obstacles_to_roi(obstacle_result)
            self._obstacle = TimedRealtimeResult(
                obstacle_result,
                camera_frame.read_completed_at,
            )
            display_obstacle_result = obstacle_result
            self._next_role = "crack"
        else:
            crack_execution_id = uuid4().hex
            crack_start_ns = time.monotonic_ns()
            crack_result = self.crack_detector.detect(crack_roi)
            crack_end_ns = time.monotonic_ns()
            crack_seconds = (crack_end_ns - crack_start_ns) / 1e9
            self.crack_inference_count += 1
            self._crack = TimedRealtimeResult(
                crack_result,
                camera_frame.read_completed_at,
            )
            display_crack_result = crack_result
            self._next_role = "obstacle"

        now = time.monotonic()
        ready = self._is_fresh(self._obstacle, now) and self._is_fresh(
            self._crack, now
        )
        return TopRealtimeInferenceOutcome(
            None if self._obstacle is None else self._obstacle.result,
            None if self._crack is None else self._crack.result,
            display_obstacle_result,
            display_crack_result,
            obstacle_seconds,
            crack_seconds,
            ready,
            camera_frame.read_completed_at,
            obstacle_start_ns,
            obstacle_end_ns,
            crack_start_ns,
            crack_end_ns,
            obstacle_execution_id,
            crack_execution_id,
        )

    def _is_fresh(self, timed_result: TimedRealtimeResult | None, now: float) -> bool:
        return (
            timed_result is not None
            and 0.0
            <= now - timed_result.frame_read_completed_at
            <= self.max_result_age_seconds
        )

    def remaining_fresh_seconds(self, *, now: float | None = None) -> float | None:
        required = (self._obstacle, self._crack)
        if any(result is None for result in required):
            return None
        if now is None:
            now = time.monotonic()
        oldest = min(
            result.frame_read_completed_at for result in required if result is not None
        )
        return max(0.0, oldest + self.max_result_age_seconds - now)


class AlternatingRealtimeInference:
    """Alternate separate models while rejecting missing or stale cached results."""

    def __init__(
        self,
        student_detector,
        crack_detector,
        *,
        multitask_enabled: bool,
        hybrid_enabled: bool,
        no_crack: bool = False,
        max_result_age_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        self.student_detector = student_detector
        self.crack_detector = crack_detector
        self.multitask_enabled = multitask_enabled
        self.hybrid_enabled = hybrid_enabled
        self.no_crack = no_crack
        self.max_result_age_seconds = max_result_age_seconds
        self.crack_inference_count = 0
        self.reset()

    def reset(self) -> None:
        self._next_role = "rust"
        self._rust: TimedRealtimeResult | None = None
        self._crack: TimedRealtimeResult | None = None

    def expire_cache(self, *, now: float | None = None) -> None:
        """Drop only expired results while preserving the next model phase."""

        if now is None:
            now = time.monotonic()
        if self._rust is not None and not self._is_fresh(self._rust, now):
            self._rust = None
        if self._crack is not None and not self._is_fresh(self._crack, now):
            self._crack = None

    def process(
        self,
        camera_frame: CameraFrame,
        water_roi: np.ndarray,
        crack_roi: np.ndarray,
    ) -> RealtimeInferenceOutcome:
        rust_seconds = 0.0
        crack_seconds = 0.0
        rust_start_ns = rust_end_ns = 0
        crack_start_ns = crack_end_ns = 0
        rust_execution_id = crack_execution_id = ""
        display_rust_result = None
        display_crack_result = None

        if self.multitask_enabled and not self.hybrid_enabled:
            rust_execution_id = crack_execution_id = uuid4().hex
            rust_start_ns = time.monotonic_ns()
            rust_result, crack_result = self.student_detector.detect(water_roi)
            rust_end_ns = time.monotonic_ns()
            rust_seconds = (rust_end_ns - rust_start_ns) / 1e9
            crack_start_ns, crack_end_ns = rust_start_ns, rust_end_ns
            crack_seconds = rust_seconds
            self.crack_inference_count += 1
            self._rust = TimedRealtimeResult(
                rust_result,
                camera_frame.read_completed_at,
            )
            self._crack = TimedRealtimeResult(
                crack_result,
                camera_frame.read_completed_at,
            )
            display_rust_result = rust_result
            display_crack_result = crack_result
        elif self.crack_detector is None:
            rust_execution_id = uuid4().hex
            rust_start_ns = time.monotonic_ns()
            rust_result = self.student_detector.detect(water_roi)
            rust_end_ns = time.monotonic_ns()
            rust_seconds = (rust_end_ns - rust_start_ns) / 1e9
            self._rust = TimedRealtimeResult(
                rust_result,
                camera_frame.read_completed_at,
            )
            display_rust_result = rust_result
        elif self._next_role == "rust":
            rust_execution_id = uuid4().hex
            rust_start_ns = time.monotonic_ns()
            rust_result = self.student_detector.detect(water_roi)
            rust_end_ns = time.monotonic_ns()
            rust_seconds = (rust_end_ns - rust_start_ns) / 1e9
            self._rust = TimedRealtimeResult(
                rust_result,
                camera_frame.read_completed_at,
            )
            display_rust_result = rust_result
            self._next_role = "crack"
        else:
            crack_execution_id = uuid4().hex
            crack_start_ns = time.monotonic_ns()
            if self.hybrid_enabled:
                _unused_rust, crack_result = self.crack_detector.detect(water_roi)
            else:
                crack_result = self.crack_detector.detect(crack_roi)
            crack_end_ns = time.monotonic_ns()
            crack_seconds = (crack_end_ns - crack_start_ns) / 1e9
            self.crack_inference_count += 1
            self._crack = TimedRealtimeResult(
                crack_result,
                camera_frame.read_completed_at,
            )
            display_crack_result = crack_result
            self._next_role = "rust"

        now = time.monotonic()
        ready = self._is_fresh(self._rust, now) and (
            self.no_crack or self._is_fresh(self._crack, now)
        )
        return RealtimeInferenceOutcome(
            None if self._rust is None else self._rust.result,
            None if self._crack is None else self._crack.result,
            display_rust_result,
            display_crack_result,
            rust_seconds,
            crack_seconds,
            ready,
            camera_frame.read_completed_at,
            rust_start_ns,
            rust_end_ns,
            crack_start_ns,
            crack_end_ns,
            rust_execution_id,
            crack_execution_id,
        )

    def _is_fresh(self, timed_result: TimedRealtimeResult | None, now: float) -> bool:
        return (
            timed_result is not None
            and 0.0
            <= now - timed_result.frame_read_completed_at
            <= self.max_result_age_seconds
        )

    def remaining_fresh_seconds(self, *, now: float | None = None) -> float | None:
        """Return the oldest required cache TTL, or None until caches exist."""

        required = [self._rust]
        if not self.no_crack:
            required.append(self._crack)
        if any(result is None for result in required):
            return None
        if now is None:
            now = time.monotonic()
        oldest_read_completed_at = min(
            result.frame_read_completed_at
            for result in required
            if result is not None
        )
        return max(
            0.0,
            oldest_read_completed_at + self.max_result_age_seconds - now,
        )


def dual_realtime_inputs_are_fresh(
    side_frame: CameraFrame,
    top_frame: CameraFrame,
    side_inference: AlternatingRealtimeInference,
    top_inference: AlternatingTopRealtimeInference,
    *,
    now: float | None = None,
) -> bool:
    """Reject stale inputs while allowing initial alternating-cache warm-up."""

    if now is None:
        now = time.monotonic()
    return (
        realtime_role_inputs_are_fresh(side_frame, side_inference, now=now)
        and realtime_role_inputs_are_fresh(top_frame, top_inference, now=now)
    )


def realtime_role_inputs_are_fresh(
    camera_frame: CameraFrame | None,
    inference: AlternatingRealtimeInference | AlternatingTopRealtimeInference,
    *,
    now: float | None = None,
) -> bool:
    """Validate one camera/inference pair without coupling the other role."""

    if camera_frame is None:
        return False
    if now is None:
        now = time.monotonic()
    inference.expire_cache(now=now)
    return realtime_camera_frame_is_fresh(camera_frame, now=now)


def clip_realtime_obstacles_to_roi(
    result: ObstacleDetectionResult,
) -> ObstacleDetectionResult:
    """Intersect validated full-frame boxes with the realtime control ROI."""

    detections = []
    for detection in result.detections:
        x1, y1, x2, y2 = detection.box_xyxy
        x1, x2 = max(0.0, x1), min(float(FRAME_WIDTH), x2)
        y1 = max(float(REALTIME_OBSTACLE_ROI_TOP), y1)
        y2 = min(float(REALTIME_OBSTACLE_ROI_BOTTOM), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            ObstacleDetection(
                box_xyxy=(x1, y1, x2, y2),
                confidence=detection.confidence,
                class_id=detection.class_id,
            )
        )
    return ObstacleDetectionResult(
        detections=tuple(detections),
        method=result.method,
        confidence_threshold=result.confidence_threshold,
        control_roi_detected=bool(detections),
    )


def suppress_front_obstacles_overlapping_cracks(
    obstacle_result: ObstacleDetectionResult,
    crack_mask: np.ndarray,
    *,
    overlap_ratio_threshold: float = FRONT_CAPTURE_OBSTACLE_CRACK_OVERLAP_RATIO,
) -> ObstacleDetectionResult:
    """Drop FRONT obstacle boxes whose area is at least 1% crack pixels."""

    if not isinstance(obstacle_result, ObstacleDetectionResult):
        raise ValueError("FRONT obstacle result is invalid.")
    if not isinstance(crack_mask, np.ndarray) or crack_mask.ndim != 2:
        raise ValueError("FRONT crack mask must be a 2D array.")
    if (
        not np.isfinite(overlap_ratio_threshold)
        or not 0.0 <= overlap_ratio_threshold <= 1.0
    ):
        raise ValueError("FRONT obstacle/crack overlap threshold must be 0 through 1.")

    mask_height, mask_width = crack_mask.shape
    kept_detections = []
    for detection in obstacle_result.detections:
        x1, y1, x2, y2 = detection.box_xyxy
        left = int(np.clip(np.floor(x1), 0, mask_width))
        top = int(np.clip(np.floor(y1), 0, mask_height))
        right = int(np.clip(np.ceil(x2), 0, mask_width))
        bottom = int(np.clip(np.ceil(y2), 0, mask_height))
        box_area = (right - left) * (bottom - top)
        overlap_ratio = (
            float(np.count_nonzero(crack_mask[top:bottom, left:right]))
            / float(box_area)
            if box_area > 0
            else 0.0
        )
        if overlap_ratio < overlap_ratio_threshold:
            kept_detections.append(detection)

    kept_detections = tuple(kept_detections)
    return ObstacleDetectionResult(
        detections=kept_detections,
        method=obstacle_result.method,
        confidence_threshold=obstacle_result.confidence_threshold,
        control_roi_detected=any(
            detection.box_xyxy[1] < REALTIME_OBSTACLE_ROI_BOTTOM
            and detection.box_xyxy[3] > REALTIME_OBSTACLE_ROI_TOP
            for detection in kept_detections
        ),
    )
