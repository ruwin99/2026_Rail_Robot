"""Inspection core: durable raw captures, bounded analysis, and reports.

Capture acceptance follows successful persistence and atomic queue submission.
The mission/UART owner sends acknowledgements; this module never owns the UART.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from crack_detector import annotate_cracks
from dashboard_export import export_capture_record, export_top_crack_record
from firebase_uploader import upload_dashboard_manifest_from_environment
from hrsegnet_crack_detector import CAPTURE_DETECTOR_PREFIX as HRSEGNET_CAPTURE_CRACK_DETECTOR_PREFIX
from inspection_report import INITIAL_PHASE, MANUAL_PHASE, RESCAN_PHASE, InspectionWorkbook
from rust_detector import LetterboxTransform, annotate, result_from_class_map

from .data import CaptureAnalysisTask
from .settings import *

if TYPE_CHECKING:
    from hrsegnet_crack_detector import HrSegNetCrackDetector
    from rust_detector import RustDetector

__all__ = [
    "firebase_upload_is_required",
    "db_test_manual_capture_is_enabled",
    "publish_dashboard_manifest",
    "validate_db_test_dashboard_manifest",
    "dashboard_manifest_for_publish",
    "save_capture",
    "queue_capture_for_analysis",
    "prepare_capture_analysis_task",
    "queue_dual_capture_for_analysis",
    "prioritize_capture_crack_over_rust",
    "capture_and_analyze",
    "capture_top_crack_only",
    "CaptureAnalysisWorker"
]


def firebase_upload_is_required(
    environment: dict[str, str] | None = None,
) -> bool:
    values = os.environ if environment is None else environment
    return str(values.get(FIREBASE_UPLOAD_REQUIRED_ENV, "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def db_test_manual_capture_is_enabled(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether the dedicated interactive initial+rescan DB test is active."""

    values = os.environ if environment is None else environment
    return str(values.get(DB_TEST_MANUAL_CAPTURE_ENV, "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def publish_dashboard_manifest(
    dashboard_manifest_path: Path | None,
    *,
    firebase_upload_safe: bool,
    upload_required: bool,
) -> bool:
    """Publish one finalized run and report whether a required upload succeeded."""

    if dashboard_manifest_path is None:
        if upload_required:
            print(
                "Firebase upload required, but no finalized dashboard manifest "
                "is available.",
                file=sys.stderr,
            )
            return False
        return True

    if not firebase_upload_safe:
        print(
            "Firebase upload skipped because actuator/UART safe shutdown "
            "was not confirmed; local dashboard files remain available.",
            file=sys.stderr,
        )
        return not upload_required

    try:
        firebase_result = upload_dashboard_manifest_from_environment(
            dashboard_manifest_path
        )
    except Exception as exc:
        print(
            "Firebase upload failed; local dashboard JSON and images "
            f"remain available for retry: {exc}",
            file=sys.stderr,
        )
        return not upload_required

    if firebase_result is None:
        print(
            "Firebase upload: disabled; finalized dashboard files remain local."
        )
        return not upload_required

    print(
        "Firebase upload complete: "
        f"run_id={firebase_result.run_id}, "
        f"artifacts={firebase_result.artifact_count}, "
        f"bytes={firebase_result.uploaded_bytes}, "
        f"document={firebase_result.firestore_document}"
    )
    return True


def validate_db_test_dashboard_manifest(manifest_path: Path) -> None:
    """Require the complete initial+rescan capture matrix before a DB-test upload."""

    try:
        document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read finalized DB test manifest: {exc}") from exc
    captures = document.get("captures")
    summaries = document.get("run_summaries")
    runs = document.get("runs")
    if not isinstance(captures, list):
        raise RuntimeError("Finalized DB test manifest has no capture list.")
    expected = {
        (phase, sequence, camera_role)
        for phase in (INITIAL_PHASE, RESCAN_PHASE)
        for sequence in range(1, AUTOMATIC_RAIL_SECTION_TARGET + 1)
        for camera_role in (SIDE_CAMERA_ROLE, TOP_CAMERA_ROLE)
    }
    actual = [
        (
            capture.get("phase"),
            capture.get("phase_sequence"),
            capture.get("camera_role"),
        )
        for capture in captures
    ]
    if len(actual) != len(expected) or set(actual) != expected:
        # 구간 수는 RAIL_ROBOT_SECTION_TARGET(기본 6, rail_mission_test 2)를 따른다.
        raise RuntimeError(
            f"Finalized DB test manifest must contain exactly {len(expected)} unique SIDE/TOP "
            f"captures covering initial 1-{AUTOMATIC_RAIL_SECTION_TARGET} and rescan "
            f"1-{AUTOMATIC_RAIL_SECTION_TARGET}; found {len(actual)}."
        )
    if (
        not isinstance(summaries, list)
        or len(summaries) != 1
        or summaries[0].get("summary_complete") is not True
    ):
        raise RuntimeError("Finalized DB test manifest summary is incomplete.")
    if (
        not isinstance(runs, list)
        or len(runs) != 1
        or runs[0].get("status") != "complete"
    ):
        raise RuntimeError("Finalized DB test manifest run is not complete.")


def dashboard_manifest_for_publish(
    manifest_path: Path | None,
    *,
    db_test_manual_capture: bool,
    db_test_publish_allowed: bool,
) -> Path | None:
    """Suppress DB-test publication until all completion checks have passed."""

    if db_test_manual_capture and not db_test_publish_allowed:
        return None
    return manifest_path


def save_capture(
    frame,
    phase: str,
    phase_sequence: int | None = None,
    output_directory: Path = CAPTURE_DIRECTORY,
    captured_at: datetime | None = None,
) -> Path:
    prefixes = {
        INITIAL_PHASE: "initial",
        RESCAN_PHASE: "rescan",
        MANUAL_PHASE: "manual",
    }
    if phase not in prefixes:
        raise ValueError(f"Unknown capture phase: {phase}")
    if phase in (INITIAL_PHASE, RESCAN_PHASE):
        if (
            not isinstance(phase_sequence, int)
            or isinstance(phase_sequence, bool)
            or not 1 <= phase_sequence <= AUTOMATIC_RAIL_SECTION_TARGET
        ):
            raise ValueError(
                "Automatic rail-section sequence must be from 1 through "
                f"{AUTOMATIC_RAIL_SECTION_TARGET}."
            )
        capture_name = f"{prefixes[phase]}_{phase_sequence:02d}"
    else:
        capture_name = prefixes[phase]

    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        encoded_ok, encoded = cv2.imencode(".jpg", frame)
    except cv2.error as exc:
        raise RuntimeError("Could not encode the camera frame as JPEG.") from exc
    if not encoded_ok:
        raise RuntimeError("Could not encode the camera frame as JPEG.")

    captured_at = captured_at or datetime.now()
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S")
    sequence = 0
    while True:
        suffix = "" if sequence == 0 else f"_{sequence:03d}"
        capture_path = output_directory / f"{capture_name}_{timestamp}{suffix}.jpg"
        try:
            with capture_path.open("xb") as capture_file:
                encoded_bytes = encoded.tobytes()
                written = capture_file.write(encoded_bytes)
                if written != len(encoded_bytes):
                    raise OSError(
                        "Could not write the complete JPEG capture "
                        f"({written}/{len(encoded_bytes)} bytes)."
                    )
            return capture_path
        except FileExistsError:
            sequence += 1


def queue_capture_for_analysis(
    frame,
    phase: str,
    phase_sequence: int | None,
    trigger: str,
    analysis_worker,
    output_directory: Path = CAPTURE_DIRECTORY,
    *,
    camera_role: str | None = None,
    captured_at: datetime | None = None,
    frame_read_completed_at: float | None = None,
) -> CaptureAnalysisTask:
    """Persist an original frame, then enqueue its deferred analysis."""

    task = prepare_capture_analysis_task(
        frame,
        phase,
        phase_sequence,
        trigger,
        output_directory,
        camera_role=camera_role,
        captured_at=captured_at,
        frame_read_completed_at=frame_read_completed_at,
    )
    try:
        analysis_worker.submit(task)
    except Exception as exc:
        raise RuntimeError(
            "Capture was saved but could not be queued for analysis; "
            f"no acknowledgement will be sent. Raw capture: {task.raw_capture_path}. "
            f"Queue error: {exc}"
        ) from exc
    print(f"Raw capture saved and queued: {task.raw_capture_path}")
    return task


def prepare_capture_analysis_task(
    frame,
    phase: str,
    phase_sequence: int | None,
    trigger: str,
    output_directory: Path = CAPTURE_DIRECTORY,
    *,
    camera_role: str | None = None,
    captured_at: datetime | None = None,
    frame_read_completed_at: float | None = None,
) -> CaptureAnalysisTask:
    """Persist one raw frame and build a task without enqueuing it yet."""

    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        shape = getattr(frame, "shape", None)
        raise ValueError(
            "Capture frame must be the native 1280x720 BGR camera frame; "
            f"got {shape}."
        )
    if camera_role is not None and camera_role not in (
        SIDE_CAMERA_ROLE,
        TOP_CAMERA_ROLE,
    ):
        raise ValueError(f"Unknown capture camera role: {camera_role}")
    task_camera_role = SIDE_CAMERA_ROLE if camera_role is None else camera_role
    captured_at = captured_at or datetime.now()
    raw_directory = output_directory / "raw"
    if camera_role is not None:
        raw_directory /= task_camera_role
    raw_capture_path = save_capture(
        frame,
        phase,
        phase_sequence,
        raw_directory,
        captured_at,
    )
    task = CaptureAnalysisTask(
        raw_capture_path=raw_capture_path,
        phase=phase,
        phase_sequence=phase_sequence,
        trigger=trigger,
        captured_at=captured_at,
        camera_role=task_camera_role,
        frame_read_completed_at=frame_read_completed_at,
    )
    return task


def queue_dual_capture_for_analysis(
    side_frame,
    top_frame,
    phase: str,
    phase_sequence: int,
    trigger: str,
    analysis_worker,
    *,
    captured_at: datetime,
    side_frame_read_completed_at: float | None,
    top_frame_read_completed_at: float | None,
    output_directory: Path = CAPTURE_DIRECTORY,
) -> tuple[CaptureAnalysisTask, CaptureAnalysisTask]:
    """Persist and atomically enqueue the SIDE/TOP tasks for one capture."""

    side_task = prepare_capture_analysis_task(
        side_frame,
        phase,
        phase_sequence,
        trigger,
        output_directory,
        camera_role=SIDE_CAMERA_ROLE,
        captured_at=captured_at,
        frame_read_completed_at=side_frame_read_completed_at,
    )
    top_task = prepare_capture_analysis_task(
        top_frame,
        phase,
        phase_sequence,
        trigger,
        output_directory,
        camera_role=TOP_CAMERA_ROLE,
        captured_at=captured_at,
        frame_read_completed_at=top_frame_read_completed_at,
    )
    try:
        analysis_worker.submit_batch((side_task, top_task))
    except Exception as exc:
        raise RuntimeError(
            "Dual capture was saved but its SIDE/TOP pair could not be queued "
            "atomically; no acknowledgement/count increment will occur. Raw "
            f"captures: {side_task.raw_capture_path}, {top_task.raw_capture_path}. "
            f"Queue error: {exc}"
        ) from exc
    print(f"Raw capture saved and queued: {side_task.raw_capture_path}")
    print(f"Raw capture saved and queued: {top_task.raw_capture_path}")
    return side_task, top_task


def prioritize_capture_crack_over_rust(result, crack_result):
    """Return a capture-only rust result with crack-overlap pixels set to Good."""

    class_map = result.class_map
    if class_map is None:
        raise ValueError("Capture rust result does not contain a class map.")
    if class_map.shape != crack_result.mask.shape:
        raise ValueError(
            "Capture rust and crack masks must have the same shape; "
            f"got {class_map.shape} and {crack_result.mask.shape}."
        )

    corrected_class_map = class_map.copy()
    corrected_class_map[crack_result.mask != 0] = 0
    height, width = corrected_class_map.shape
    corrected = result_from_class_map(
        corrected_class_map,
        LetterboxTransform(
            original_height=height,
            original_width=width,
            resized_height=height,
            resized_width=width,
            top=0,
            left=0,
        ),
        result.method,
    )
    corrected.status = result.status
    return corrected


def capture_and_analyze(
    frame,
    phase: str,
    phase_sequence: int | None,
    trigger: str,
    detector: RustDetector,
    workbook: InspectionWorkbook,
    output_directory: Path = CAPTURE_DIRECTORY,
    *,
    crack_detector: HrSegNetCrackDetector | None = None,
    crack_zones: set[int] | None = None,
    raw_capture_path: Path | None = None,
    captured_at: datetime | None = None,
):
    captured_at = captured_at or datetime.now()
    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        print(
            "Capture frame must be the native 1280x720 BGR camera frame.",
            file=sys.stderr,
        )
        return None
    try:
        result = detector.detect(frame)
    except RuntimeError as exc:
        print(f"Capture inference failed: {exc}", file=sys.stderr)
        return None

    if result.status != "ready":
        print(
            f"Capture inference did not complete synchronously: {result.status}",
            file=sys.stderr,
        )
        return None

    if not result.method.startswith(TEACHER_DETECTOR_PREFIX):
        print(
            "Capture result was not produced by the TensorRT teacher: "
            f"{result.method}",
            file=sys.stderr,
        )
        return None

    zone_number = (
        phase_sequence if phase in (INITIAL_PHASE, RESCAN_PHASE) else None
    )
    crack_status = "disabled"
    crack_method = None
    crack_detected = None
    crack_pixels = None
    crack_inspected_pixels = None
    crack_ratio = None
    crack_result = None
    if crack_detector is not None:
        try:
            crack_result = crack_detector.detect(frame)
        except RuntimeError as exc:
            print(f"Crack capture inference failed: {exc}", file=sys.stderr)
            return None
        if crack_result.status != "ready":
            print(
                "Crack capture inference did not complete synchronously: "
                f"{crack_result.status}",
                file=sys.stderr,
            )
            return None
        if not crack_result.method.startswith(HRSEGNET_CAPTURE_CRACK_DETECTOR_PREFIX):
            print(
                "Crack result was not produced by the capture HrSegNet TensorRT "
                f"detector: {crack_result.method}",
                file=sys.stderr,
            )
            return None
        if crack_result.mask.shape != frame.shape[:2]:
            print(
                "Crack inference mask shape does not match the capture frame.",
                file=sys.stderr,
            )
            return None
        crack_inspected_pixels = int(crack_result.mask.size)
        if crack_inspected_pixels <= 0:
            print("Crack inference returned an empty mask.", file=sys.stderr)
            return None
        crack_pixels = int(np.count_nonzero(crack_result.mask))
        crack_ratio = crack_pixels / crack_inspected_pixels
        crack_detected = crack_pixels > 0
        crack_status = "ready"
        crack_method = crack_result.method
        try:
            result = prioritize_capture_crack_over_rust(result, crack_result)
        except ValueError as exc:
            print(f"Could not apply capture crack priority: {exc}", file=sys.stderr)
            return None
        if crack_detected and zone_number is not None and crack_zones is not None:
            crack_zones.add(zone_number)

    inspected_pixels = int(result.mask.size)
    if inspected_pixels <= 0:
        print("Capture inference returned an empty mask.", file=sys.stderr)
        return None
    rust_pixels = int((result.mask != 0).sum())
    rust_ratio = rust_pixels / inspected_pixels
    display = annotate(frame, result)
    if crack_detector is not None:
        display = annotate_cracks(display, crack_result, zone_number)

    try:
        capture_path = save_capture(
            display,
            phase,
            phase_sequence,
            output_directory,
            captured_at,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not save segmented capture: {exc}", file=sys.stderr)
        return None

    try:
        overall_ratio = workbook.append_capture(
            capture_path=capture_path,
            phase=phase,
            trigger=trigger,
            detector=result.method,
            rust_pixels=rust_pixels,
            inspected_pixels=inspected_pixels,
            crack_status=crack_status,
            crack_detector=crack_method,
            crack_detected=crack_detected,
            crack_pixels=crack_pixels,
            crack_inspected_pixels=crack_inspected_pixels,
            captured_at=captured_at,
        )
    except Exception as exc:
        print(f"Could not record capture in {workbook.path}: {exc}", file=sys.stderr)
        return None

    if raw_capture_path is not None:
        try:
            dashboard_manifest = export_capture_record(
                frame=frame,
                rust_result=result,
                crack_result=crack_result,
                workbook=workbook,
                capture_path=capture_path,
                raw_capture_path=raw_capture_path,
                phase=phase,
                phase_sequence=phase_sequence,
                trigger=trigger,
                captured_at=captured_at,
                output_directory=output_directory,
            )
            print(f"Dashboard export updated: {dashboard_manifest}")
        except Exception as exc:
            print(
                "Dashboard export failed; XLSX and captured images remain valid: "
                f"{exc}",
                file=sys.stderr,
            )
            if firebase_upload_is_required():
                raise RuntimeError(
                    "Dashboard export is required for the Firebase upload."
                ) from exc

    notice = (
        f"captured rust={rust_ratio * 100:.2f}% "
        f"phase_overall={overall_ratio * 100:.2f}%"
    )
    if crack_status == "ready":
        crack_state = "yes" if crack_detected else "no"
        notice += f" crack={crack_state} ({crack_ratio * 100:.3f}%)"
        if zone_number is not None:
            notice += f" rail_section={zone_number}"
    cv2.putText(
        display,
        notice,
        (24, max(36, display.shape[0] - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (60, 220, 90),
        2,
        cv2.LINE_AA,
    )
    print(f"Segmented capture saved: {capture_path}")
    print(notice)
    if crack_detected and zone_number is not None:
        print(
            f"크랙 후보 감지: {zone_number}번 레일 구간 "
            f"(Rail Section 인덱스 {zone_number - 1})"
        )
    return display


def capture_top_crack_only(
    frame: np.ndarray,
    task: CaptureAnalysisTask,
    crack_detector: HrSegNetCrackDetector | None,
    workbook: InspectionWorkbook,
    output_directory: Path = CAPTURE_DIRECTORY,
    *,
    obstacle_detector=None,
):
    """Analyze and report the top-camera capture without rust inference."""

    if crack_detector is None:
        print("Top capture requires the full-frame HrSegNet detector.", file=sys.stderr)
        return None
    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        print(
            "Top capture frame must be the native 1280x720 BGR camera frame.",
            file=sys.stderr,
        )
        return None
    try:
        crack_result = crack_detector.detect(frame)
    except RuntimeError as exc:
        print(f"Top crack capture inference failed: {exc}", file=sys.stderr)
        return None
    if (
        crack_result.status != "ready"
        or not crack_result.method.startswith(HRSEGNET_CAPTURE_CRACK_DETECTOR_PREFIX)
        or crack_result.mask.shape != frame.shape[:2]
    ):
        print("Top crack capture result failed its runtime contract.", file=sys.stderr)
        return None

    zone_number = (
        task.phase_sequence
        if task.phase in (INITIAL_PHASE, RESCAN_PHASE)
        else None
    )
    display = annotate_cracks(frame, crack_result, zone_number)
    obstacle_result = None
    obstacle_provenance = None
    if obstacle_detector is not None:
        from .vision import suppress_front_obstacles_overlapping_cracks
        from dashboard_export import obstacle_bbox_measurement
        obstacle_result = suppress_front_obstacles_overlapping_cracks(
            obstacle_detector.detect(frame), crack_result.mask,
        )
        obstacle_measurement, display = obstacle_bbox_measurement(display, obstacle_result)
        obstacle_provenance = {
            "model_filename": Path(obstacle_detector.engine_path).name,
            "model_sha256": obstacle_detector.engine_sha256,
            "input_contract": "images FP32 [1,3,736,1280]",
            "output_contract": "output0 FP32 [1,300,6]",
            "preprocessing": "BGR->RGB /255; pad top/bottom 8 pixels with 114",
        }
    notice = (
        "top crack="
        f"{'yes' if crack_result.detected else 'no'} "
        f"({crack_result.crack_ratio * 100:.3f}%)"
    )
    cv2.putText(
        display,
        notice,
        (24, max(36, display.shape[0] - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if crack_result.detected else (60, 220, 90),
        2,
        cv2.LINE_AA,
    )
    try:
        capture_path = save_capture(
            display,
            task.phase,
            task.phase_sequence,
            output_directory / "top_crack",
            task.captured_at,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not save top crack capture: {exc}", file=sys.stderr)
        return None
    crack_pixels = int(np.count_nonzero(crack_result.mask))
    crack_inspected_pixels = int(crack_result.mask.size)
    try:
        workbook.append_top_crack_capture(
            raw_capture_path=task.raw_capture_path,
            capture_path=capture_path,
            phase=task.phase,
            phase_sequence=task.phase_sequence,
            trigger=task.trigger,
            crack_detector=crack_result.method,
            crack_detected=crack_pixels > 0,
            crack_pixels=crack_pixels,
            crack_inspected_pixels=crack_inspected_pixels,
            captured_at=task.captured_at,
        )
    except Exception as exc:
        print(
            f"Could not record top crack capture in {workbook.path}: {exc}",
            file=sys.stderr,
        )
        return None
    if obstacle_result is not None:
        workbook.append_obstacle_capture(
            phase=task.phase, phase_sequence=task.phase_sequence,
            captured_at=task.captured_at, capture_path=capture_path,
            measurement=obstacle_measurement,
        )
    try:
        dashboard_manifest = export_top_crack_record(
            frame=frame,
            crack_result=crack_result,
            workbook=workbook,
            capture_path=capture_path,
            raw_capture_path=task.raw_capture_path,
            phase=task.phase,
            phase_sequence=task.phase_sequence,
            trigger=task.trigger,
            captured_at=task.captured_at,
            output_directory=output_directory,
            obstacle_result=obstacle_result,
            obstacle_provenance=obstacle_provenance,
        )
        print(f"Dashboard TOP export updated: {dashboard_manifest}")
    except Exception as exc:
        print(
            "Dashboard TOP export failed; XLSX and captured images remain valid: "
            f"{exc}",
            file=sys.stderr,
        )
        if firebase_upload_is_required():
            raise RuntimeError(
                "Dashboard TOP export is required for the Firebase upload."
            ) from exc
    print(f"Top crack capture saved: {capture_path}")
    print(notice)
    return display


class CaptureAnalysisWorker:
    """Run capture inference and XLSX writes on one FIFO worker thread."""

    _STOP = object()

    def __init__(
        self,
        detector: RustDetector,
        workbook: InspectionWorkbook,
        output_directory: Path = CAPTURE_DIRECTORY,
        *,
        crack_detector: HrSegNetCrackDetector | None = None,
        obstacle_detector_factory=None,
        queue_capacity: int = CAPTURE_ANALYSIS_QUEUE_CAPACITY,
        live_capture_observer=None,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("Capture analysis queue capacity must be positive.")
        self.detector = detector
        self.workbook = workbook
        self.output_directory = output_directory
        self.crack_detector = crack_detector
        self._obstacle_detector_factory = obstacle_detector_factory
        self._obstacle_detector = None
        self._live_capture_observer = live_capture_observer
        self._tasks: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._thread = threading.Thread(
            target=self._run,
            name="capture-analysis",
            # A native TensorRT call cannot be cancelled safely from Python.  Keep
            # the normal path joined below, but do not let a hung CUDA worker keep
            # the UART mission process alive after the bounded fail-safe timeout.
            # The raw JPEG is saved before a task is submitted, so the input is
            # still recoverable if this exceptional path is taken.
            daemon=True,
        )
        self._state_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._failure: str | None = None
        self._skipped_raw_paths: list[Path] = []
        self._pending_tasks = 0
        self._idle = threading.Event()
        self._idle.set()
        self._started = False
        self._stopped = False
        self._stop_requested = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Capture analysis worker was already started.")
        self._thread.start()
        self._started = True

    def submit(self, task: CaptureAnalysisTask) -> None:
        self.submit_batch((task,))

    def submit_batch(self, tasks: tuple[CaptureAnalysisTask, ...]) -> None:
        """Enqueue one logical capture batch without accepting only a prefix."""

        if not tasks:
            raise ValueError("Capture analysis batch must not be empty.")
        for task in tasks:
            if not isinstance(task, CaptureAnalysisTask):
                raise TypeError("Capture analysis worker requires a capture task.")
        with self._submit_lock:
            self._submit_batch_locked(tasks)

    def _submit_batch_locked(self, tasks: tuple[CaptureAnalysisTask, ...]) -> None:
        if not self._started:
            raise RuntimeError("Capture analysis worker is not running.")
        if self._stopped:
            raise RuntimeError("Capture analysis worker is already stopped.")
        failure = self.failure_message()
        if failure is not None:
            raise RuntimeError(failure)
        available = self._tasks.maxsize - self._tasks.qsize()
        if available < len(tasks):
            if len(tasks) == 1:
                location = f"raw capture remains at {tasks[0].raw_capture_path}."
            else:
                location = (
                    "raw capture batch remains at "
                    + ", ".join(str(item.raw_capture_path) for item in tasks)
                    + "."
                )
            raise RuntimeError(
                "Capture analysis queue is full "
                f"({self._tasks.maxsize}); {location}"
            )
        with self._state_lock:
            self._pending_tasks += len(tasks)
            self._idle.clear()
        for item in tasks:
            self._notify_live("captured", item)
            # The submit lock serializes producers and the capacity preflight
            # guarantees every member fits. The consumer can only free space.
            self._tasks.put_nowait(item)

    def _notify_live(self, method, *args):
        if self._live_capture_observer is None:
            return
        try:
            getattr(self._live_capture_observer, method)(*args)
        except Exception as exc:
            # Cloud delivery is presentation-only. Preserve inspection and
            # retain the original capture even if local outbox persistence fails.
            print(f"Live capture {method} notification failed ({type(exc).__name__}); "
                  f"original remains at {args[0].raw_capture_path}", file=sys.stderr)

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0.0:
            raise ValueError("Capture analysis idle timeout must not be negative.")
        if not self._started:
            return True
        return self._idle.wait(timeout)

    def failure_message(self) -> str | None:
        with self._state_lock:
            if self._failure is None:
                return None
            message = self._failure
            if self._skipped_raw_paths:
                skipped = ", ".join(map(str, self._skipped_raw_paths))
                message += (
                    f" Skipped {len(self._skipped_raw_paths)} queued raw capture(s) "
                    f"after the failure: {skipped}."
                )
            return message

    def shutdown(
        self,
        timeout: float = CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS,
    ) -> str | None:
        if timeout < 0.0:
            raise ValueError("Capture analysis shutdown timeout must not be negative.")
        if self._started and not self._stopped:
            deadline = time.monotonic() + timeout
            if not self._stop_requested:
                try:
                    self._tasks.put(self._STOP, timeout=timeout)
                    self._stop_requested = True
                except queue.Full:
                    self._record_worker_failure(
                        "Capture analysis worker queue did not accept its stop "
                        f"request within {timeout:.1f} seconds."
                    )
            remaining = max(0.0, deadline - time.monotonic())
            self._thread.join(remaining)
            if self._thread.is_alive():
                self._record_worker_failure(
                    "Capture analysis worker did not stop within "
                    f"{timeout:.1f} seconds. Raw captures remain recoverable; "
                    "the mission process is stopping."
                )
            else:
                self._stopped = True
                if self._obstacle_detector is not None:
                    self._obstacle_detector.close()
                    self._obstacle_detector = None
        return self.failure_message()

    def _record_worker_failure(self, message: str) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = message

    def _record_failure(self, task: CaptureAnalysisTask, exc: Exception) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = (
                    "Background capture analysis failed for "
                    f"{task.raw_capture_path}: {exc}"
                )
            else:
                self._skipped_raw_paths.append(task.raw_capture_path)

    def _finish_task(self) -> None:
        with self._state_lock:
            self._pending_tasks -= 1
            if self._pending_tasks < 0:
                raise RuntimeError("Capture analysis pending task count became negative.")
            if self._pending_tasks == 0:
                self._idle.set()

    def _run(self) -> None:
        while True:
            queued_item = self._tasks.get()
            is_task = queued_item is not self._STOP
            try:
                if queued_item is self._STOP:
                    return
                task = queued_item
                assert isinstance(task, CaptureAnalysisTask)
                if self.failure_message() is not None:
                    with self._state_lock:
                        self._skipped_raw_paths.append(task.raw_capture_path)
                    continue
                try:
                    frame = cv2.imread(str(task.raw_capture_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError("Could not read the saved raw JPEG.")
                    if task.camera_role == TOP_CAMERA_ROLE:
                        if self._obstacle_detector is None and self._obstacle_detector_factory is not None:
                            self._obstacle_detector = self._obstacle_detector_factory()
                        analyzed = capture_top_crack_only(
                            frame,
                            task,
                            self.crack_detector,
                            self.workbook,
                            self.output_directory,
                            obstacle_detector=self._obstacle_detector,
                        )
                    else:
                        analyzed = capture_and_analyze(
                            frame,
                            task.phase,
                            task.phase_sequence,
                            task.trigger,
                            self.detector,
                            self.workbook,
                            self.output_directory,
                            crack_detector=self.crack_detector,
                            raw_capture_path=task.raw_capture_path,
                            captured_at=task.captured_at,
                        )
                    if analyzed is None:
                        raise RuntimeError(
                            "Capture inference, analyzed-image save, or report "
                            "write did not complete."
                        )
                    self._notify_live("processed", task, analyzed)
                except Exception as exc:
                    self._record_failure(task, exc)
            finally:
                self._tasks.task_done()
                if is_task:
                    self._finish_task()
