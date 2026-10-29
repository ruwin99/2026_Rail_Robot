"""Camera acquisition core: newest-frame ownership and capture freshness.

This module owns VideoCapture; it does not control actuators or run inference.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

from .data import CameraFrame
from .settings import *

__all__ = [
    "LatestFrameCamera",
    "open_latest_frame_camera",
    "validate_distinct_camera_devices",
    "realtime_camera_frame_is_fresh",
    "read_stopped_capture_frame",
    "read_stopped_capture_pair"
]


class LatestFrameCamera:
    """Drain one VideoCapture in a thread and expose only its newest frame."""

    def __init__(self, capture) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._latest: CameraFrame | None = None
        self._error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="latest-camera-frame-reader",
            daemon=True,
        )
        self._thread.start()

    @property
    def latest_sequence(self) -> int:
        with self._condition:
            return 0 if self._latest is None else self._latest.sequence

    def read_latest(
        self,
        *,
        after_sequence: int | None = None,
        timeout: float = LATEST_FRAME_TIMEOUT_SECONDS,
    ) -> CameraFrame:
        """Return a new-enough snapshot without replaying queued old frames."""

        deadline = time.monotonic() + timeout
        minimum_sequence = 0 if after_sequence is None else after_sequence + 1
        while True:
            with self._condition:
                latest = self._latest
                if (
                    latest is not None
                    and latest.sequence >= minimum_sequence
                ):
                    return latest
                if self._error is not None:
                    raise RuntimeError(f"Camera reader failed: {self._error}") from self._error
                if self._closed:
                    raise RuntimeError("Camera reader is closed.")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"No fresh camera frame arrived within {timeout:.3f} seconds."
                    )
                self._condition.wait(remaining)

    def close(self) -> None:
        if self._closed and not self._thread.is_alive():
            if self._error is not None:
                raise RuntimeError(f"Camera reader failed: {self._error}") from self._error
            return
        self._stop_requested.set()
        self._thread.join(LATEST_FRAME_CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("Camera reader thread did not stop cleanly.")
        if self._error is not None:
            raise RuntimeError(f"Camera reader failed: {self._error}") from self._error

    def release(self) -> None:
        self.close()

    def _reader_loop(self) -> None:
        sequence = 0
        try:
            while not self._stop_requested.is_set():
                frame_ok, frame = self._capture.read()
                read_completed_monotonic_ns = time.monotonic_ns()
                read_completed_at = time.monotonic()
                if not frame_ok or frame is None:
                    if not self._stop_requested.is_set():
                        raise RuntimeError("Could not read a camera frame.")
                    break
                sequence += 1
                with self._condition:
                    self._latest = CameraFrame(
                        frame, sequence, read_completed_at, read_completed_monotonic_ns
                    )
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            try:
                self._capture.release()
            except Exception as exc:
                with self._condition:
                    if self._error is None:
                        self._error = exc
            with self._condition:
                self._closed = True
                self._condition.notify_all()


def open_latest_frame_camera(camera_source: int | str) -> LatestFrameCamera:
    """Configure VideoCapture, then transfer all reads/releases to one thread."""

    capture = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera source {camera_source!r}.")
    except Exception:
        capture.release()
        raise
    return LatestFrameCamera(capture)


def validate_distinct_camera_devices(side_device: str, top_device: str) -> None:
    """Reject missing or aliased dual-camera device paths before UART is opened."""

    side = Path(side_device).expanduser()
    top = Path(top_device).expanduser()
    if not side.exists():
        raise ValueError(f"Side camera device was not found: {side}")
    if not top.exists():
        raise ValueError(f"Top camera device was not found: {top}")
    try:
        side_resolved = side.resolve(strict=True)
        top_resolved = top.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Could not resolve dual-camera device paths: {exc}") from exc
    if side_resolved == top_resolved:
        raise ValueError(
            "Side and top camera device paths resolve to the same video node: "
            f"{side_resolved}"
        )


def realtime_camera_frame_is_fresh(
    camera_frame: CameraFrame,
    *,
    now: float | None = None,
) -> bool:
    if now is None:
        now = time.monotonic()
    age = now - camera_frame.read_completed_at
    return 0.0 <= age <= REALTIME_INPUT_MAX_AGE_SECONDS


def read_stopped_capture_frame(
    camera: LatestFrameCamera,
    *,
    after_sequence: int | None = None,
) -> CameraFrame:
    """Wait for a frame completed after the post-stop settling deadline."""

    time.sleep(CAPTURE_SETTLE_SECONDS)
    baseline_sequence = camera.latest_sequence
    if after_sequence is not None:
        baseline_sequence = max(baseline_sequence, after_sequence)
    return camera.read_latest(
        after_sequence=baseline_sequence,
        timeout=LATEST_FRAME_TIMEOUT_SECONDS + CAPTURE_SETTLE_SECONDS,
    )


def read_stopped_capture_pair(
    side_camera: LatestFrameCamera,
    top_camera: LatestFrameCamera,
) -> tuple[CameraFrame, CameraFrame]:
    """Settle once, then obtain one new side/top frame under one deadline."""

    time.sleep(CAPTURE_SETTLE_SECONDS)
    side_baseline = side_camera.latest_sequence
    top_baseline = top_camera.latest_sequence
    deadline = time.monotonic() + LATEST_FRAME_TIMEOUT_SECONDS
    side_frame = side_camera.read_latest(
        after_sequence=side_baseline,
        timeout=max(0.0, deadline - time.monotonic()),
    )
    top_frame = top_camera.read_latest(
        after_sequence=top_baseline,
        timeout=max(0.0, deadline - time.monotonic()),
    )
    return side_frame, top_frame

