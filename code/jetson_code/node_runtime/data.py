"""Value objects passed between the runtime's node responsibilities.

read_completed_at is the local monotonic camera-read completion time, not sensor
exposure or a ROS/network timestamp. ROS adapters must translate clock domains.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .settings import SIDE_CAMERA_ROLE

if TYPE_CHECKING:
    import numpy as np
    from obstacle_detector import ObstacleDetectionResult

@dataclass(frozen=True)
class CaptureAnalysisTask:
    raw_capture_path: Path
    phase: str
    phase_sequence: int | None
    trigger: str
    captured_at: datetime
    camera_role: str = SIDE_CAMERA_ROLE
    frame_read_completed_at: float | None = None


@dataclass(frozen=True)
class CameraFrame:
    frame: np.ndarray
    sequence: int
    read_completed_at: float
    read_completed_monotonic_ns: int = 0


@dataclass(frozen=True)
class DualRealtimeFrameRead:
    side_frame: CameraFrame | None
    top_frame: CameraFrame | None
    side_cache_expired: bool
    top_cache_expired: bool
    side_error: str | None = None
    top_error: str | None = None
    side_waiting: bool = False
    top_waiting: bool = False


@dataclass(frozen=True)
class TimedRealtimeResult:
    result: object
    frame_read_completed_at: float


@dataclass(frozen=True)
class RealtimeInferenceOutcome:
    rust_result: object | None
    crack_result: object | None
    display_rust_result: object | None
    display_crack_result: object | None
    rust_seconds: float
    crack_seconds: float
    ready: bool
    frame_read_completed_at: float | None = None
    rust_start_monotonic_ns: int = 0
    rust_end_monotonic_ns: int = 0
    crack_start_monotonic_ns: int = 0
    crack_end_monotonic_ns: int = 0
    rust_execution_id: str = ""
    crack_execution_id: str = ""


@dataclass(frozen=True)
class TopRealtimeInferenceOutcome:
    obstacle_result: ObstacleDetectionResult | None
    crack_result: object | None
    display_obstacle_result: ObstacleDetectionResult | None
    display_crack_result: object | None
    obstacle_seconds: float
    crack_seconds: float
    ready: bool
    frame_read_completed_at: float | None = None
    obstacle_start_monotonic_ns: int = 0
    obstacle_end_monotonic_ns: int = 0
    crack_start_monotonic_ns: int = 0
    crack_end_monotonic_ns: int = 0
    obstacle_execution_id: str = ""
    crack_execution_id: str = ""



