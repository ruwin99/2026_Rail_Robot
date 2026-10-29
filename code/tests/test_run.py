from __future__ import annotations

import hashlib
import io
import queue
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np


serial_stub = types.ModuleType("serial")
serial_stub.SerialException = OSError
serial_stub.Serial = object
sys.modules.setdefault("serial", serial_stub)

JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from inspection_report import INITIAL_PHASE, RESCAN_PHASE  # noqa: E402
from crack_detector import CrackDetectionResult  # noqa: E402
import run as run_module  # noqa: E402
from node_runtime import display as display_module, inspection as inspection_module  # noqa: E402
from run import (  # noqa: E402
    CAPTURE_TRIGGER,
    CAPTURE_OK_COMMAND,
    CLEANER_OFF_COMMAND,
    CLEANER_ON_COMMAND,
    CLEANER_PWM_25_COMMAND,
    CRACK_CONTROL_STOP_RATIO,
    FRONT_CAPTURE_OBSTACLE_CRACK_OVERLAP_RATIO,
    FRONT_ACTUATOR_COMMANDS,
    FRONT_CLEANER_PWM_25_COMMAND,
    FRONT_PUMP_ON_COMMAND,
    PUMP_OFF_COMMAND,
    PUMP_ON_COMMAND,
    SIDE_ACTUATOR_COMMANDS,
    SIDE_CLEANER_PWM_25_COMMAND,
    SIDE_PUMP_ON_COMMAND,
    AlternatingRealtimeInference,
    AlternatingTopRealtimeInference,
    CameraFrame,
    CaptureAnalysisTask,
    CaptureAnalysisWorker,
    CleanerController,
    ControlOffDiagnostic,
    DualControlOffTransitionLogger,
    DualTimingLogger,
    DualRealtimeFrameRead,
    FrontControlDecision,
    FrontControlHistory,
    LatestFrameCamera,
    LatestFrameDisplay,
    PumpController,
    RealtimeActuatorArbiter,
    RealtimeActuatorDecisionHold,
    RealtimeControlDecision,
    RealtimeControlHistory,
    RealtimeDisplayCache,
    RealtimeInferenceOutcome,
    annotate_realtime_control_results,
    annotate_top_realtime_results,
    capture_and_analyze,
    capture_top_crack_only,
    cleaning_blocked_by_crack,
    crack_stop_roi_bounds,
    draw_realtime_roi_guide,
    draw_realtime_result_summary,
    draw_roi_guide,
    dual_realtime_inputs_are_fresh,
    detect_realtime_control,
    extract_realtime_control_rois,
    format_realtime_test_statistics,
    force_cleaning_safe_off,
    force_cleaning_pairs_safe_off,
    maintain_waiting_role_control,
    obstacle_detected_in_control_roi,
    parse_args,
    prepare_realtime_transition,
    prioritize_capture_crack_over_rust,
    queue_capture_for_analysis,
    read_realtime_camera_frame,
    read_stopped_capture_pair,
    read_stopped_capture_frame,
    realtime_camera_frame_is_fresh,
    send_capture_ok,
    select_control_off_reason,
    stack_dual_camera_displays,
    suppress_front_obstacles_overlapping_cracks,
    update_cleaning_actuators,
    validate_engine,
    validate_tf32_runtime_environment,
    water_control_blocked_by_rust,
    water_control_roi_bounds,
)
from obstacle_detector import (  # noqa: E402
    DETECTOR_PREFIX as OBSTACLE_DETECTOR_PREFIX,
    ObstacleDetection,
    ObstacleDetectionResult,
)
from rust_detector import DetectionResult  # noqa: E402


class _FakeDetector:
    def __init__(self, result: DetectionResult) -> None:
        self.result = result
        self.frames: list[np.ndarray] = []
        self.thread_ids: list[int] = []

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self.thread_ids.append(threading.get_ident())
        self.frames.append(frame.copy())
        return self.result


class _FakeCrackDetector:
    def __init__(self, result: CrackDetectionResult) -> None:
        self.result = result
        self.frames: list[np.ndarray] = []

    def detect(self, frame: np.ndarray) -> CrackDetectionResult:
        self.frames.append(frame.copy())
        return self.result


class _FakeWorkbook:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.captures: list[dict[str, object]] = []
        self.top_crack_captures: list[dict[str, object]] = []
        self.thread_ids: list[int] = []

    def append_capture(self, **capture: object) -> float:
        self.thread_ids.append(threading.get_ident())
        self.captures.append(capture)
        return 0.5

    def append_top_crack_capture(self, **capture: object) -> float:
        self.thread_ids.append(threading.get_ident())
        self.top_crack_captures.append(capture)
        return 0.0


class _FakeUart:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, command: bytes) -> int:
        self.writes.append(command)
        return len(command)


class _FakeLatestCamera:
    def __init__(self, frames: tuple[np.ndarray, ...]) -> None:
        self._frames = iter(frames)
        self.latest_sequence = 0
        self.release_calls = 0

    def read_latest(self, **_kwargs: object) -> CameraFrame:
        frame = next(self._frames)
        self.latest_sequence += 1
        return CameraFrame(frame, self.latest_sequence, time.monotonic())

    def release(self) -> None:
        self.release_calls += 1


class _FakeDisplay:
    def __init__(self, keys: tuple[int, ...] = ()) -> None:
        self._keys = list(keys)
        self._pending_keys: list[int] = []
        self.frames: list[np.ndarray] = []
        self.close_calls = 0
        self.exit_requested = False

    def check_status(self) -> bool:
        return self.exit_requested

    def submit(self, frame: np.ndarray) -> bool:
        self.frames.append(frame)
        if self._keys:
            key = self._keys.pop(0)
            if key in (ord("q"), 27):
                self.exit_requested = True
            elif key not in (0, 255):
                self._pending_keys.append(key)
        return not self.exit_requested

    def poll_key(self) -> int | None:
        return self._pending_keys.pop(0) if self._pending_keys else None

    def completed_statistics(self) -> tuple[int, float]:
        intervals = max(0, len(self.frames) - 1)
        return intervals, float(intervals) * 0.05

    def close(self) -> None:
        self.close_calls += 1


class LatestFrameCameraTests(unittest.TestCase):
    class _RepeatingCapture:
        def __init__(self) -> None:
            self.read_thread_ids: list[int] = []
            self.release_thread_ids: list[int] = []
            self.released = threading.Event()
            self.value = 0

        def read(self) -> tuple[bool, np.ndarray]:
            time.sleep(0.002)
            self.read_thread_ids.append(threading.get_ident())
            self.value += 1
            return True, np.full((2, 2, 3), self.value % 255, dtype=np.uint8)

        def release(self) -> None:
            self.release_thread_ids.append(threading.get_ident())
            self.released.set()

    def test_drops_old_frames_and_capture_has_one_thread_owner(self) -> None:
        capture = self._RepeatingCapture()
        camera = LatestFrameCamera(capture)
        try:
            first = camera.read_latest(timeout=0.2)
            deadline = time.monotonic() + 0.2
            while camera.latest_sequence < first.sequence + 4:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.002)
            latest = camera.read_latest(timeout=0.2)
            self.assertGreaterEqual(latest.sequence, first.sequence + 4)
            self.assertEqual(int(latest.frame[0, 0, 0]), latest.sequence % 255)
        finally:
            camera.close()

        self.assertTrue(capture.released.is_set())
        self.assertEqual(len(set(capture.read_thread_ids)), 1)
        self.assertEqual(capture.release_thread_ids, [capture.read_thread_ids[0]])
        self.assertNotEqual(capture.read_thread_ids[0], threading.get_ident())

    def test_timeout_and_reader_error_are_propagated(self) -> None:
        class SlowCapture:
            def read(self):
                time.sleep(0.05)
                return True, np.zeros((2, 2, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        slow_camera = LatestFrameCamera(SlowCapture())
        with self.assertRaises(TimeoutError):
            slow_camera.read_latest(timeout=0.005)
        slow_camera.close()

        class FailedCapture:
            def read(self):
                return False, None

            def release(self) -> None:
                pass

        failed_camera = LatestFrameCamera(FailedCapture())
        with self.assertRaisesRegex(RuntimeError, "Camera reader failed"):
            failed_camera.read_latest(timeout=0.2)
        with self.assertRaisesRegex(RuntimeError, "Camera reader failed"):
            failed_camera.close()

    def test_stopped_capture_waits_then_requires_another_sequence(self) -> None:
        returned = CameraFrame(np.zeros((2, 2, 3), dtype=np.uint8), 8, 10.1)
        camera = Mock()
        camera.latest_sequence = 7
        camera.read_latest.return_value = returned

        with patch.object(run_module.time, "sleep") as sleep:
            actual = read_stopped_capture_frame(camera, after_sequence=5)

        self.assertIs(actual, returned)
        sleep.assert_called_once_with(run_module.CAPTURE_SETTLE_SECONDS)
        camera.read_latest.assert_called_once_with(
            after_sequence=7,
            timeout=(
                run_module.LATEST_FRAME_TIMEOUT_SECONDS
                + run_module.CAPTURE_SETTLE_SECONDS
            ),
        )


class LatestFrameDisplayTests(unittest.TestCase):
    def test_ignores_unsupported_visibility_and_exits_when_window_is_hidden(
        self,
    ) -> None:
        shown_values: list[int] = []
        second_frame_shown = threading.Event()
        unsupported_seen_twice = threading.Event()
        allow_window_close = threading.Event()
        unsupported_checks = 0

        def show(_window: str, frame: np.ndarray) -> None:
            shown_values.append(int(frame[0, 0, 0]))
            if len(shown_values) == 2:
                second_frame_shown.set()

        def visible(_window: str, _property: int) -> float:
            nonlocal unsupported_checks
            if allow_window_close.is_set():
                return 0.0
            unsupported_checks += 1
            if unsupported_checks >= 2:
                unsupported_seen_twice.set()
            return -1.0

        with (
            patch.object(run_module.cv2, "namedWindow"),
            patch.object(run_module.cv2, "imshow", side_effect=show),
            patch.object(run_module.cv2, "waitKey", return_value=-1),
            patch.object(run_module.cv2, "getWindowProperty", side_effect=visible),
            patch.object(run_module.cv2, "destroyAllWindows"),
        ):
            display = LatestFrameDisplay("visibility-window")
            try:
                self.assertTrue(display.submit(np.full((2, 2, 3), 1, np.uint8)))
                deadline = time.monotonic() + 0.5
                while len(shown_values) < 1:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.002)
                self.assertFalse(display.check_status())

                self.assertTrue(display.submit(np.full((2, 2, 3), 2, np.uint8)))
                self.assertTrue(second_frame_shown.wait(0.5))
                self.assertTrue(unsupported_seen_twice.wait(0.5))
                self.assertFalse(display.check_status())

                allow_window_close.set()
                deadline = time.monotonic() + 0.5
                while not display.check_status():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.002)
            finally:
                display.close()

        self.assertEqual(shown_values[:2], [1, 2])

    def test_drops_pending_frames_and_highgui_has_one_thread_owner(self) -> None:
        entered_first_show = threading.Event()
        release_first_show = threading.Event()
        shown_values: list[int] = []
        owner_thread_ids: list[int] = []

        def record_owner(*_args: object, **_kwargs: object) -> None:
            owner_thread_ids.append(threading.get_ident())

        def record_key(_delay: int) -> int:
            owner_thread_ids.append(threading.get_ident())
            return -1

        def record_visible(*_args: object) -> float:
            owner_thread_ids.append(threading.get_ident())
            return 1.0

        def show(_window: str, frame: np.ndarray) -> None:
            owner_thread_ids.append(threading.get_ident())
            shown_values.append(int(frame[0, 0, 0]))
            if len(shown_values) == 1:
                entered_first_show.set()
                self.assertTrue(release_first_show.wait(0.5))

        with (
            patch.object(run_module.cv2, "namedWindow", side_effect=record_owner),
            patch.object(run_module.cv2, "imshow", side_effect=show),
            patch.object(run_module.cv2, "waitKey", side_effect=record_key),
            patch.object(run_module.cv2, "getWindowProperty", side_effect=record_visible),
            patch.object(run_module.cv2, "destroyAllWindows", side_effect=record_owner),
        ):
            display = LatestFrameDisplay("test-window")
            try:
                self.assertTrue(display.submit(np.full((2, 2, 3), 1, np.uint8)))
                self.assertTrue(entered_first_show.wait(0.5))
                self.assertTrue(display.submit(np.full((2, 2, 3), 2, np.uint8)))
                self.assertTrue(display.submit(np.full((2, 2, 3), 3, np.uint8)))
                release_first_show.set()
                deadline = time.monotonic() + 0.5
                while len(shown_values) < 2:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.002)
            finally:
                release_first_show.set()
                display.close()

        self.assertEqual(shown_values[:2], [1, 3])
        self.assertEqual(display.dropped_frames, 1)
        self.assertEqual(len(set(owner_thread_ids)), 1)
        self.assertNotEqual(owner_thread_ids[0], threading.get_ident())

    def test_ui_exception_is_propagated_and_close_has_a_timeout(self) -> None:
        show_entered = threading.Event()
        release_show = threading.Event()

        def blocked_show(_window: str, _frame: np.ndarray) -> None:
            show_entered.set()
            release_show.wait(0.5)

        with (
            patch.object(run_module.cv2, "namedWindow"),
            patch.object(run_module.cv2, "imshow", side_effect=blocked_show),
            patch.object(run_module.cv2, "waitKey", return_value=-1),
            patch.object(run_module.cv2, "getWindowProperty", return_value=1.0),
            patch.object(run_module.cv2, "destroyAllWindows"),
            patch.object(display_module, "LATEST_DISPLAY_CLOSE_TIMEOUT_SECONDS", 0.01),
        ):
            display = LatestFrameDisplay("blocked-window")
            display.submit(np.zeros((2, 2, 3), dtype=np.uint8))
            self.assertTrue(show_entered.wait(0.5))
            with self.assertRaisesRegex(RuntimeError, "did not stop cleanly"):
                display.close()
            release_show.set()
            deadline = time.monotonic() + 0.5
            while display._thread.is_alive():
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.002)
            display.close()

        with (
            patch.object(run_module.cv2, "namedWindow"),
            patch.object(run_module.cv2, "imshow", side_effect=RuntimeError("GTK failed")),
            patch.object(run_module.cv2, "waitKey", return_value=-1),
            patch.object(run_module.cv2, "destroyAllWindows"),
        ):
            failed = LatestFrameDisplay("failed-window")
            failed.submit(np.zeros((2, 2, 3), dtype=np.uint8))
            deadline = time.monotonic() + 0.5
            while True:
                try:
                    failed.check_status()
                except RuntimeError as exc:
                    self.assertIn("GTK failed", str(exc))
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.002)
            with self.assertRaisesRegex(RuntimeError, "GTK failed"):
                failed.close()


class DualControlOffTransitionLoggerTests(unittest.TestCase):
    def test_selects_every_reason_with_safety_priority(self) -> None:
        base = dict(
            camera_error=False,
            new_crack_blocked=False,
            new_rust_grade=None,
            new_rust_forced_off=False,
            rolling_history_hazard=False,
            had_valid_decision=None,
        )

        def reason(**overrides) -> str:
            return select_control_off_reason(**(base | overrides))

        self.assertEqual(
            reason(
                camera_error=True,
                new_crack_blocked=True,
                new_rust_grade=3,
                new_rust_forced_off=True,
                rolling_history_hazard=True,
                had_valid_decision=True,
            ),
            "camera_error",
        )
        self.assertEqual(
            reason(
                new_crack_blocked=True,
                new_rust_grade=3,
                new_rust_forced_off=True,
                rolling_history_hazard=True,
                had_valid_decision=True,
            ),
            "new_crack",
        )
        self.assertEqual(
            reason(
                new_rust_grade=2,
                new_rust_forced_off=True,
                rolling_history_hazard=True,
                had_valid_decision=True,
            ),
            "new_rust_grade",
        )
        self.assertEqual(
            reason(rolling_history_hazard=True, had_valid_decision=True),
            "rolling_hazard",
        )
        self.assertEqual(
            reason(
                had_valid_decision=True,
                loss_deadline_was_set=True,
            ),
            "history_hold_expired",
        )
        self.assertEqual(
            reason(
                had_valid_decision=True,
                watchdog_expired=True,
            ),
            "decision_watchdog_expired",
        )
        self.assertEqual(
            reason(had_valid_decision=False),
            "startup_or_no_valid_decision",
        )
        self.assertEqual(reason(), "unknown")

    def test_emits_once_per_independent_on_to_full_off_transition(self) -> None:
        logger = DualControlOffTransitionLogger()
        front = ControlOffDiagnostic(
            reason="new_crack",
            monotonic_seconds=10.0,
            history_ready=False,
            frame_sequence=21,
            frame_age_ms=35.25,
            new_crack_pct=0.0501,
            new_crack_blocked=True,
            history_crack_pct=0.0501,
            loss_age_ms=100.0,
            hold_remaining_ms=1100.0,
        )
        side = ControlOffDiagnostic(
            reason="new_rust_grade",
            monotonic_seconds=10.1,
            history_ready=False,
            frame_sequence=22,
            new_rust_grade=2,
        )

        self.assertEqual(
            logger.observe((12.5, True, 25.0, False), front=front, side=side),
            (),
        )
        front_lines = logger.observe(
            (0.0, False, 25.0, False),
            front=front,
            side=side,
        )
        self.assertEqual(len(front_lines), 1)
        self.assertIn("role=front reason=new_crack", front_lines[0])
        self.assertIn("new_crack_pct=0.0501", front_lines[0])
        self.assertEqual(
            logger.observe(
                (0.0, False, 25.0, False),
                front=front,
                side=side,
            ),
            (),
        )
        side_lines = logger.observe(
            (0.0, False, 0.0, False),
            front=front,
            side=side,
        )
        self.assertEqual(len(side_lines), 1)
        self.assertIn("role=side reason=new_rust_grade", side_lines[0])
        self.assertEqual(
            logger.observe((12.5, True, 0.0, False), front=front, side=side),
            (),
        )
        self.assertEqual(
            len(
                logger.observe(
                    (0.0, False, 0.0, False),
                    front=front,
                    side=side,
                )
            ),
            1,
        )

    def test_forced_runtime_off_logs_each_active_role_once_with_safe_detail(
        self,
    ) -> None:
        logger = DualControlOffTransitionLogger()
        lines = logger.observe_forced_off(
            (12.5, True, 25.0, False),
            reason="uart_error",
            detail="UART failed\nwhile writing",
            now=12.0,
        )

        self.assertEqual(len(lines), 2)
        self.assertTrue(all("reason=uart_error" in line for line in lines))
        self.assertTrue(
            all("detail='UART failed while writing'" in line for line in lines)
        )
        self.assertTrue(all("\n" not in line for line in lines))
        self.assertEqual(
            logger.observe_forced_off(
                (0.0, False, 0.0, False),
                reason="runtime_error",
                detail="already off",
                now=12.1,
            ),
            (),
        )

    def test_classifies_uart_runtime_errors_without_hiding_detail(self) -> None:
        self.assertEqual(
            run_module.classify_runtime_control_off(
                RuntimeError("Realtime UART arbiter failed:\nserial disconnected")
            ),
            (
                "uart_error",
                "Realtime UART arbiter failed: serial disconnected",
            ),
        )
        self.assertEqual(
            run_module.classify_runtime_control_off(ValueError("bad mask")),
            ("runtime_error", "bad mask"),
        )


class DualTimingLoggerTests(unittest.TestCase):
    def test_excludes_warmup_and_reports_five_second_window(self) -> None:
        logger = DualTimingLogger(
            warmup_seconds=10.0,
            report_interval_seconds=5.0,
            started_at=0.0,
        )
        side_rust = SimpleNamespace(rust_seconds=0.010, crack_seconds=0.0)
        side_crack = SimpleNamespace(rust_seconds=0.0, crack_seconds=0.020)
        top_crack = SimpleNamespace(obstacle_seconds=0.0, crack_seconds=0.030)
        top_obstacle = SimpleNamespace(
            obstacle_seconds=0.040,
            crack_seconds=0.0,
        )

        warmup_summary = logger.record(
            loop_seconds=9.0,
            side_update_seconds=8.0,
            top_update_seconds=7.0,
            side_outcome=side_rust,
            top_outcome=top_crack,
            side_frame_age_seconds=6.0,
            top_frame_age_seconds=5.0,
            side_stale=True,
            top_stale=True,
            side_ready=True,
            front_ready=True,
            now=9.0,
        )
        self.assertIsNone(warmup_summary)

        first_summary = logger.record(
            loop_seconds=0.100,
            side_update_seconds=0.030,
            top_update_seconds=0.040,
            side_outcome=side_rust,
            top_outcome=top_crack,
            side_frame_age_seconds=0.110,
            top_frame_age_seconds=0.120,
            side_stale=False,
            top_stale=False,
            side_ready=True,
            front_ready=True,
            now=10.5,
        )
        self.assertIsNone(first_summary)
        second_summary = logger.record(
            loop_seconds=0.200,
            side_update_seconds=0.050,
            top_update_seconds=0.060,
            side_outcome=side_crack,
            top_outcome=top_obstacle,
            side_frame_age_seconds=0.210,
            top_frame_age_seconds=0.220,
            side_stale=True,
            top_stale=False,
            side_ready=False,
            front_ready=True,
            now=12.0,
        )
        self.assertIsNone(second_summary)
        summary = logger.record(
            loop_seconds=0.300,
            side_update_seconds=None,
            top_update_seconds=None,
            side_outcome=None,
            top_outcome=None,
            side_frame_age_seconds=None,
            top_frame_age_seconds=None,
            side_stale=False,
            top_stale=True,
            side_ready=False,
            front_ready=False,
            now=15.0,
        )

        self.assertIsNotNone(summary)
        self.assertIn("[DUAL_TIMING] window=5.0s", summary)
        self.assertIn("loop_ms(n=3,mean=200.0", summary)
        self.assertIn("side_update_ms(n=2,mean=40.0", summary)
        self.assertIn("top_update_ms(n=2,mean=50.0", summary)
        self.assertIn("side_rust_ms(n=1,mean=10.0", summary)
        self.assertIn("side_crack_ms(n=1,mean=20.0", summary)
        self.assertIn("top_obstacle_ms(n=1,mean=40.0", summary)
        self.assertIn("top_crack_ms(n=1,mean=30.0", summary)
        self.assertIn("side_frame_age_ms(n=2,mean=160.0", summary)
        self.assertIn("top_frame_age_ms(n=2,mean=170.0", summary)
        self.assertIn("stale(side=1,top=1)", summary)
        self.assertIn("ready_to_off(side=1,front=1)", summary)


class AlternatingRealtimeInferenceTests(unittest.TestCase):
    @staticmethod
    def _rust_result() -> DetectionResult:
        return DetectionResult(
            mask=np.zeros((240, 1280), dtype=np.uint8),
            boxes=[],
            rust_ratio=0.0,
            method="rust/fake",
            class_map=np.zeros((240, 1280), dtype=np.uint8),
            class_ratios={},
        )

    @staticmethod
    def _crack_result(*, detected: bool = False) -> CrackDetectionResult:
        return CrackDetectionResult(
            mask=np.zeros((180, 1280), dtype=np.uint8),
            boxes=[],
            crack_pixels=int(detected),
            inspected_pixels=180 * 1280,
            crack_ratio=0.0,
            detected=detected,
            method="crack/fake",
            probability_threshold=0.5,
        )

    def test_waits_for_both_results_then_alternates_models(self) -> None:
        rust_detector = Mock()
        rust_detector.detect.return_value = self._rust_result()
        crack_detector = Mock()
        crack_detector.detect.return_value = self._crack_result()
        scheduler = AlternatingRealtimeInference(
            rust_detector,
            crack_detector,
            multitask_enabled=False,
            hybrid_enabled=False,
        )
        water_roi = np.zeros((240, 1280, 3), dtype=np.uint8)
        crack_roi = np.zeros((180, 1280, 3), dtype=np.uint8)

        now = time.monotonic()
        first = scheduler.process(CameraFrame(water_roi, 1, now), water_roi, crack_roi)
        second = scheduler.process(CameraFrame(water_roi, 2, now), water_roi, crack_roi)

        self.assertFalse(first.ready)
        self.assertTrue(second.ready)
        self.assertIsNotNone(first.display_rust_result)
        self.assertIsNone(first.display_crack_result)
        self.assertIsNone(second.display_rust_result)
        self.assertIsNotNone(second.display_crack_result)
        rust_detector.detect.assert_called_once_with(water_roi)
        crack_detector.detect.assert_called_once_with(crack_roi)
        self.assertEqual(scheduler.crack_inference_count, 1)
        scheduler.reset()
        self.assertEqual(scheduler.crack_inference_count, 1)

    def test_old_cached_result_keeps_control_fail_closed(self) -> None:
        rust_detector = Mock()
        rust_detector.detect.return_value = self._rust_result()
        crack_detector = Mock()
        crack_detector.detect.return_value = self._crack_result()
        scheduler = AlternatingRealtimeInference(
            rust_detector,
            crack_detector,
            multitask_enabled=False,
            hybrid_enabled=False,
            max_result_age_seconds=0.2,
        )
        water_roi = np.zeros((240, 1280, 3), dtype=np.uint8)
        crack_roi = np.zeros((180, 1280, 3), dtype=np.uint8)

        with patch.object(run_module.time, "monotonic", return_value=10.0):
            scheduler.process(CameraFrame(water_roi, 1, 10.0), water_roi, crack_roi)
        with patch.object(run_module.time, "monotonic", return_value=10.25):
            outcome = scheduler.process(
                CameraFrame(water_roi, 2, 10.25),
                water_roi,
                crack_roi,
            )

        self.assertFalse(outcome.ready)

    def test_stale_frame_and_safe_off_stop_active_outputs(self) -> None:
        frame = CameraFrame(
            np.zeros((2, 2, 3), dtype=np.uint8),
            1,
            1.0,
        )
        self.assertFalse(realtime_camera_frame_is_fresh(frame, now=1.21))

        uart = _FakeUart()
        cleaner = CleanerController(uart)
        pump = PumpController(uart)
        cleaner.force_on()
        pump.update(True)
        force_cleaning_safe_off(cleaner, pump)
        self.assertFalse(cleaner.is_on)
        self.assertFalse(pump.is_on)
        self.assertEqual(uart.writes[-2:], [CLEANER_OFF_COMMAND, PUMP_OFF_COMMAND])

    def test_dual_input_freshness_is_independent_of_cached_result_ttl(self) -> None:
        side_frame = CameraFrame(np.zeros((2, 2, 3), dtype=np.uint8), 1, 10.0)
        top_frame = CameraFrame(np.zeros((2, 2, 3), dtype=np.uint8), 2, 10.0)
        side_inference = Mock()
        top_inference = Mock()
        side_inference.remaining_fresh_seconds.return_value = 0.01
        top_inference.remaining_fresh_seconds.return_value = 0.02

        self.assertTrue(
            dual_realtime_inputs_are_fresh(
                side_frame,
                top_frame,
                side_inference,
                top_inference,
                now=10.19,
            )
        )
        side_inference.remaining_fresh_seconds.return_value = 0.0
        self.assertTrue(
            dual_realtime_inputs_are_fresh(
                side_frame,
                top_frame,
                side_inference,
                top_inference,
                now=10.20,
            )
        )

    def test_dual_initial_missing_caches_are_warmup_not_expiry(self) -> None:
        side_frame = CameraFrame(np.zeros((2, 2, 3), dtype=np.uint8), 1, 10.0)
        top_frame = CameraFrame(np.zeros((2, 2, 3), dtype=np.uint8), 2, 10.0)
        side_inference = Mock()
        top_inference = Mock()
        side_inference.remaining_fresh_seconds.return_value = None
        top_inference.remaining_fresh_seconds.return_value = None

        self.assertTrue(
            dual_realtime_inputs_are_fresh(
                side_frame,
                top_frame,
                side_inference,
                top_inference,
                now=10.19,
            )
        )

        side_inference.remaining_fresh_seconds.return_value = 0.0
        self.assertTrue(
            dual_realtime_inputs_are_fresh(
                side_frame,
                top_frame,
                side_inference,
                top_inference,
                now=10.20,
            )
        )

    def test_camera_wait_stops_outputs_at_cache_expiry_not_full_timeout(self) -> None:
        rust_detector = Mock()
        rust_detector.detect.return_value = self._rust_result()
        crack_detector = Mock()
        crack_detector.detect.return_value = self._crack_result()
        scheduler = AlternatingRealtimeInference(
            rust_detector,
            crack_detector,
            multitask_enabled=False,
            hybrid_enabled=False,
            max_result_age_seconds=0.2,
        )
        water_roi = np.zeros((240, 1280, 3), dtype=np.uint8)
        crack_roi = np.zeros((180, 1280, 3), dtype=np.uint8)
        with patch.object(run_module.time, "monotonic", return_value=10.0):
            scheduler.process(CameraFrame(water_roi, 1, 10.0), water_roi, crack_roi)
            scheduler.process(CameraFrame(water_roi, 2, 10.0), water_roi, crack_roi)

        uart = _FakeUart()
        cleaner = CleanerController(uart)
        pump = PumpController(uart)
        cleaner.force_on()
        pump.update(True)
        clock = [10.19]
        waits: list[float] = []
        fresh_frame = CameraFrame(water_roi, 3, 10.2)

        def read_latest(**kwargs: object) -> CameraFrame:
            timeout = float(kwargs["timeout"])
            waits.append(timeout)
            clock[0] += timeout
            if len(waits) == 1:
                raise TimeoutError("cache-expiry boundary")
            return fresh_frame

        camera = Mock()
        camera.read_latest.side_effect = read_latest
        with patch.object(
            run_module.time,
            "monotonic",
            side_effect=lambda: clock[0],
        ):
            actual, cache_expired = read_realtime_camera_frame(
                camera,
                after_sequence=2,
                inference=scheduler,
                cleaner_controller=cleaner,
                pump_controller=pump,
            )

        self.assertIs(actual, fresh_frame)
        self.assertTrue(cache_expired)
        self.assertAlmostEqual(waits[0], 0.01, places=6)
        self.assertAlmostEqual(sum(waits), 0.2, places=6)
        self.assertEqual(uart.writes[-2:], [CLEANER_OFF_COMMAND, PUMP_OFF_COMMAND])

    def test_ui_summary_cache_keeps_both_results_while_models_alternate(self) -> None:
        rust_result = self._rust_result()
        crack_result = self._crack_result()
        cache = RealtimeDisplayCache()
        cache.update(
            RealtimeInferenceOutcome(
                rust_result,
                None,
                rust_result,
                None,
                0.01,
                0.0,
                False,
            )
        )
        self.assertIs(cache.rust_result, rust_result)
        self.assertIsNone(cache.crack_result)

        cache.update(
            RealtimeInferenceOutcome(
                rust_result,
                crack_result,
                None,
                crack_result,
                0.0,
                0.02,
                True,
            )
        )
        self.assertIs(cache.rust_result, rust_result)
        self.assertIs(cache.crack_result, crack_result)


class CaptureQueueTests(unittest.TestCase):
    def test_dual_batch_capacity_failure_enqueues_neither_task(self) -> None:
        worker = run_module.CaptureAnalysisWorker(
            Mock(), Mock(), queue_capacity=1
        )
        worker.start()
        tasks = tuple(
            run_module.CaptureAnalysisTask(
                raw_capture_path=Path(f"raw-{role}.jpg"),
                phase=INITIAL_PHASE,
                phase_sequence=1,
                trigger=CAPTURE_TRIGGER,
                captured_at=datetime.now(),
                camera_role=role,
            )
            for role in (
                run_module.SIDE_CAMERA_ROLE,
                run_module.TOP_CAMERA_ROLE,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "raw capture batch remains"):
            worker.submit_batch(tasks)
        self.assertEqual(worker._tasks.qsize(), 0)
        self.assertTrue(worker.wait_until_idle(0.1))
        self.assertIsNone(worker.shutdown())

    def test_raw_capture_is_saved_before_worker_submission(self) -> None:
        events: list[str] = []

        class RecordingWorker:
            def submit(self, task: CaptureAnalysisTask) -> None:
                self.assert_raw_exists(task)
                events.append("queued")

            @staticmethod
            def assert_raw_exists(task: CaptureAnalysisTask) -> None:
                if not task.raw_capture_path.is_file():
                    raise AssertionError("raw capture was not saved before submit")

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "captures"
            task = queue_capture_for_analysis(
                frame,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                RecordingWorker(),
                output_directory,
            )

            self.assertEqual(events, ["queued"])
            self.assertEqual(task.raw_capture_path.parent, output_directory / "raw")
            self.assertTrue(task.raw_capture_path.is_file())
            self.assertEqual(task.trigger, "CAMERA_CAPTURE")

    def test_queue_rejection_keeps_raw_file_and_does_not_send_ack(self) -> None:
        class RejectingWorker:
            def submit(self, task: CaptureAnalysisTask) -> None:
                raise RuntimeError(
                    f"queue is full; raw capture remains at {task.raw_capture_path}"
                )

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        uart = _FakeUart()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "captures"
            with self.assertRaisesRegex(RuntimeError, "no acknowledgement"):
                queue_capture_for_analysis(
                    frame,
                    INITIAL_PHASE,
                    1,
                    CAPTURE_TRIGGER,
                    RejectingWorker(),
                    output_directory,
                )

            self.assertEqual(uart.writes, [])
            self.assertEqual(len(list((output_directory / "raw").glob("*.jpg"))), 1)

    def test_capture_ack_requires_complete_uart_write(self) -> None:
        uart = _FakeUart()
        send_capture_ok(uart)
        self.assertEqual(uart.writes, [CAPTURE_OK_COMMAND])

        class ShortWriteUart(_FakeUart):
            def write(self, command: bytes) -> int:
                self.writes.append(command)
                return len(command) - 1

        with self.assertRaisesRegex(RuntimeError, "complete CAPTURE_OK"):
            send_capture_ok(ShortWriteUart())

    def test_worker_runs_inference_and_workbook_on_its_single_thread(self) -> None:
        class_map = np.zeros((720, 1280), dtype=np.uint8)
        detector = _FakeDetector(
            DetectionResult(
                mask=np.zeros_like(class_map),
                boxes=[],
                rust_ratio=0.0,
                method="deeplabv3plus-tensorrt/teacher/fake",
                class_map=class_map,
                class_ratios={},
            )
        )
        main_thread_id = threading.get_ident()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "captures"
            workbook = _FakeWorkbook(Path(temporary_directory) / "report.xlsx")
            worker = CaptureAnalysisWorker(
                detector,
                workbook,
                output_directory,
            )
            worker.start()
            queue_capture_for_analysis(
                frame,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                worker,
                output_directory,
            )
            worker.wait_until_idle()
            failure = worker.shutdown()

            self.assertIsNone(failure)
            self.assertEqual(len(workbook.captures), 1)
            self.assertEqual(detector.thread_ids, workbook.thread_ids)
            self.assertNotEqual(detector.thread_ids, [main_thread_id])

    def test_worker_failure_is_visible_and_rejects_later_tasks(self) -> None:
        detector = object()
        with tempfile.TemporaryDirectory() as temporary_directory:
            workbook = _FakeWorkbook(Path(temporary_directory) / "report.xlsx")
            worker = CaptureAnalysisWorker(detector, workbook)
            worker.start()
            missing_path = Path(temporary_directory) / "missing.jpg"
            failed_task = CaptureAnalysisTask(
                missing_path,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                datetime.now(),
            )
            worker.submit(failed_task)
            worker.wait_until_idle()

            failure = worker.failure_message()
            self.assertIsNotNone(failure)
            self.assertIn(str(missing_path), failure)
            with self.assertRaisesRegex(RuntimeError, "Background capture analysis"):
                worker.submit(
                    CaptureAnalysisTask(
                        Path(temporary_directory) / "later.jpg",
                        INITIAL_PHASE,
                        2,
                        CAPTURE_TRIGGER,
                        datetime.now(),
                    )
                )
            self.assertEqual(worker.shutdown(), failure)

    def test_bounded_queue_reports_overflow_with_raw_path(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocking_analysis(*_args: object, **_kwargs: object) -> np.ndarray:
            started.set()
            if not release.wait(timeout=2.0):
                raise RuntimeError("test did not release worker")
            return np.zeros((10, 10, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.jpg"
            self.assertTrue(
                cv2.imwrite(
                    str(raw_path),
                    np.zeros((10, 10, 3), dtype=np.uint8),
                )
            )
            worker = CaptureAnalysisWorker(
                object(),
                _FakeWorkbook(Path(temporary_directory) / "report.xlsx"),
                queue_capacity=1,
            )
            tasks = [
                CaptureAnalysisTask(
                    raw_path,
                    RESCAN_PHASE,
                    sequence,
                    CAPTURE_TRIGGER,
                    datetime.now(),
                )
                for sequence in (1, 2, 3)
            ]
            with patch("node_runtime.inspection.capture_and_analyze", side_effect=blocking_analysis):
                worker.start()
                worker_failure = None
                try:
                    worker.submit(tasks[0])
                    self.assertTrue(started.wait(timeout=1.0))
                    worker.submit(tasks[1])
                    with self.assertRaisesRegex(RuntimeError, "raw[.]jpg"):
                        worker.submit(tasks[2])
                finally:
                    release.set()
                    worker_failure = worker.shutdown()
                self.assertIsNone(worker_failure)

    def test_wait_until_idle_times_out_for_hung_capture_analysis(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocking_analysis(*_args: object, **_kwargs: object) -> np.ndarray:
            started.set()
            release.wait(timeout=2.0)
            return np.zeros((10, 10, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.jpg"
            self.assertTrue(cv2.imwrite(str(raw_path), np.zeros((10, 10, 3), dtype=np.uint8)))
            worker = CaptureAnalysisWorker(
                object(),
                _FakeWorkbook(Path(temporary_directory) / "report.xlsx"),
            )
            task = CaptureAnalysisTask(
                raw_path,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                datetime.now(),
            )
            with patch("node_runtime.inspection.capture_and_analyze", side_effect=blocking_analysis):
                worker.start()
                worker.submit(task)
                self.assertTrue(started.wait(timeout=1.0))
                self.assertFalse(worker.wait_until_idle(timeout=0.01))
                release.set()
                self.assertTrue(worker.wait_until_idle(timeout=1.0))
                self.assertIsNone(worker.shutdown())

    def test_shutdown_is_bounded_when_capture_analysis_is_hung(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocking_analysis(*_args: object, **_kwargs: object) -> np.ndarray:
            started.set()
            release.wait(timeout=2.0)
            return np.zeros((10, 10, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "raw.jpg"
            self.assertTrue(cv2.imwrite(str(raw_path), np.zeros((10, 10, 3), dtype=np.uint8)))
            worker = CaptureAnalysisWorker(
                object(),
                _FakeWorkbook(Path(temporary_directory) / "report.xlsx"),
            )
            task = CaptureAnalysisTask(
                raw_path,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                datetime.now(),
            )
            with patch("node_runtime.inspection.capture_and_analyze", side_effect=blocking_analysis):
                worker.start()
                worker.submit(task)
                self.assertTrue(started.wait(timeout=1.0))
                started_at = time.monotonic()
                failure = worker.shutdown(timeout=0.01)
                elapsed = time.monotonic() - started_at

                self.assertLess(elapsed, 0.2)
                self.assertIn("did not stop within 0.0 seconds", failure)
                self.assertTrue(worker._thread.daemon)
                release.set()
                worker._thread.join(timeout=1.0)
                self.assertFalse(worker._thread.is_alive())

    def test_realtime_transition_orders_outputs_off_before_bounded_drain(self) -> None:
        events: list[object] = []

        class Controller:
            def __init__(self, name: str) -> None:
                self.name = name

            def force_off(self) -> None:
                events.append(f"{self.name}-off")

        class Worker:
            def wait_until_idle(self, timeout: float) -> bool:
                events.append(("wait", timeout))
                return True

            def failure_message(self) -> None:
                events.append("failure-check")
                return None

        prepare_realtime_transition(Controller("cleaner"), Controller("pump"), Worker())
        events.append("realtime-reset")

        self.assertEqual(events[0:2], ["cleaner-off", "pump-off"])
        self.assertEqual(events[2][0], "wait")
        self.assertGreater(events[2][1], 0.0)
        self.assertEqual(events[3], "failure-check")
        self.assertEqual(events[4], "realtime-reset")

    def test_realtime_transition_timeout_keeps_no_uart_controllers_off(self) -> None:
        cleaner = CleanerController(None)
        pump = PumpController(None)
        cleaner.is_on = True
        pump.is_on = True
        worker = Mock()
        worker.wait_until_idle.return_value = False

        with self.assertRaisesRegex(RuntimeError, "did not become idle"):
            prepare_realtime_transition(cleaner, pump, worker)

        self.assertFalse(cleaner.is_on)
        self.assertFalse(pump.is_on)
        worker.wait_until_idle.assert_called_once_with(
            run_module.CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS
        )
        worker.failure_message.assert_not_called()


class RealtimeActuatorArbiterTests(unittest.TestCase):
    def test_heartbeats_during_blocking_inference_and_expires_only_one_role(
        self,
    ) -> None:
        class ThreadRecordingUart(_FakeUart):
            def __init__(self) -> None:
                super().__init__()
                self.thread_ids: list[int] = []
                self.lock = threading.Lock()

            def write(self, command: bytes) -> int:
                with self.lock:
                    self.thread_ids.append(threading.get_ident())
                    self.writes.append(command)
                return len(command)

        uart = ThreadRecordingUart()
        arbiter = RealtimeActuatorArbiter(uart, heartbeat_seconds=0.02)
        arbiter.start()
        now = time.monotonic()
        arbiter.publish(
            "front",
            FrontControlDecision(25.0, True, True, 0.0),
            valid_until=now + 0.06,
        )
        arbiter.publish(
            "side",
            RealtimeControlDecision(12.5, True, 0, 0.0),
            valid_until=now + 0.80,
        )

        # This sleep represents a blocking TensorRT inference in the main thread.
        time.sleep(0.46)
        self.assertEqual(arbiter.desired_state(), (0.0, False, 12.5, True))
        arbiter.close()

        self.assertGreaterEqual(
            uart.writes.count(SIDE_ACTUATOR_COMMANDS.cleaner_pwm_12_5),
            2,
        )
        self.assertIn(FRONT_ACTUATOR_COMMANDS.cleaner_off, uart.writes)
        self.assertEqual(len(set(uart.thread_ids)), 1)
        self.assertNotEqual(uart.thread_ids[0], threading.get_ident())


class RealtimeActuatorDecisionHoldTests(unittest.TestCase):
    def test_loss_starts_deadline_and_repeated_maintain_does_not_extend_it(self) -> None:
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        decision = RealtimeControlDecision(12.5, True, 0, 0.0)
        hold.accept(decision, now=9.5)

        self.assertEqual(hold.valid_until, 10.7)
        self.assertIs(hold.begin_loss(now=10.0), decision)
        self.assertEqual(hold.valid_until, 11.2)
        self.assertIs(hold.begin_loss(now=11.199), decision)
        self.assertEqual(hold.valid_until, 11.2)
        self.assertIsNone(hold.begin_loss(now=11.201))
        self.assertIsNone(hold.valid_until)

    def test_recovery_resets_loss_and_subsequent_loss_gets_new_deadline(self) -> None:
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        first = RealtimeControlDecision(12.5, True, 0, 0.0)
        second = RealtimeControlDecision(25.0, False, 1, 0.0)
        hold.accept(first, now=9.5)
        self.assertIs(hold.begin_loss(now=10.0), first)
        self.assertEqual(hold.valid_until, 11.2)

        hold.accept(second, now=10.5)
        self.assertEqual(hold.valid_until, 11.7)
        self.assertIs(hold.current(now=11.0), second)
        self.assertIs(hold.begin_loss(now=11.0), second)
        self.assertEqual(hold.valid_until, 12.2)

    def test_direct_control_heartbeats_hold_then_stops_at_deadline(self) -> None:
        uart = _FakeUart()
        cleaner = CleanerController(uart, SIDE_ACTUATOR_COMMANDS)
        pump = PumpController(uart, SIDE_ACTUATOR_COMMANDS)
        history = Mock()
        history.ready = False
        inference = Mock()
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        hold.accept(
            RealtimeControlDecision(12.5, True, 0, 0.0),
            now=9.5,
        )

        with patch("run.time.monotonic", return_value=10.0):
            maintained = maintain_waiting_role_control(
                cleaner,
                pump,
                history=history,
                inference=inference,
                role="side",
                hold=hold,
            )
        self.assertTrue(maintained)
        self.assertTrue(cleaner.is_on)
        self.assertTrue(pump.is_on)
        self.assertEqual(hold.valid_until, 11.2)

        with patch("run.time.monotonic", return_value=11.201):
            maintained = maintain_waiting_role_control(
                cleaner,
                pump,
                history=history,
                inference=inference,
                role="side",
                hold=hold,
            )
        self.assertFalse(maintained)
        self.assertFalse(cleaner.is_on)
        self.assertFalse(pump.is_on)

    def test_arbiter_heartbeat_uses_loss_deadline(self) -> None:
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        decision = FrontControlDecision(25.0, True, True, 0.0)
        hold.accept(decision, now=9.5)
        arbiter = Mock()
        history = Mock()
        history.ready = False

        with patch("run.time.monotonic", return_value=10.0):
            self.assertTrue(
                maintain_waiting_role_control(
                    CleanerController(None, FRONT_ACTUATOR_COMMANDS),
                    PumpController(None, FRONT_ACTUATOR_COMMANDS),
                    history=history,
                    inference=Mock(),
                    role="front",
                    hold=hold,
                    arbiter=arbiter,
                )
            )
        arbiter.publish.assert_called_once_with(
            "front",
            decision,
            valid_until=11.2,
        )

    def test_valid_publish_starts_finite_main_stall_watchdog(self) -> None:
        decision = RealtimeControlDecision(12.5, True, 0, 0.0)
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        history = Mock(valid_until=10.1)
        arbiter = Mock()

        with patch("run.time.monotonic", return_value=10.0):
            run_module.publish_realtime_actuator_decision(
                CleanerController(None, SIDE_ACTUATOR_COMMANDS),
                PumpController(None, SIDE_ACTUATOR_COMMANDS),
                role="side",
                history=history,
                decision=decision,
                hold=hold,
                arbiter=arbiter,
            )

        arbiter.publish.assert_called_once_with(
            "side",
            decision,
            valid_until=11.2,
        )
        self.assertEqual(hold.valid_until, 11.2)

    def test_startup_without_previous_decision_stays_off(self) -> None:
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        cleaner = CleanerController(None, SIDE_ACTUATOR_COMMANDS)
        pump = PumpController(None, SIDE_ACTUATOR_COMMANDS)
        history = Mock()
        history.ready = False
        self.assertFalse(
            maintain_waiting_role_control(
                cleaner,
                pump,
                history=history,
                inference=Mock(),
                role="side",
                hold=hold,
            )
        )
        self.assertFalse(cleaner.is_on)
        self.assertFalse(pump.is_on)

    def test_waiting_while_history_is_still_ready_does_not_start_loss(self) -> None:
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        decision = RealtimeControlDecision(12.5, True, 0, 0.0)
        hold.accept(decision, now=10.0)
        history = Mock()
        history.ready = True
        arbiter = Mock()

        with patch("run.time.monotonic", return_value=10.0):
            self.assertTrue(
                maintain_waiting_role_control(
                    CleanerController(None, SIDE_ACTUATOR_COMMANDS),
                    PumpController(None, SIDE_ACTUATOR_COMMANDS),
                    history=history,
                    inference=Mock(),
                    role="side",
                    hold=hold,
                    arbiter=arbiter,
                )
            )
        self.assertEqual(hold.valid_until, 11.2)
        arbiter.publish.assert_called_once_with(
            "side",
            decision,
            valid_until=11.2,
        )

    def test_main_stall_expires_last_publish_without_detected_loss(self) -> None:
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        decision = RealtimeControlDecision(12.5, True, 0, 0.0)
        hold.accept(decision, now=10.0)

        self.assertIs(hold.current(now=11.199), decision)
        self.assertIsNone(hold.current(now=11.201))
        self.assertIsNone(hold.valid_until)

    def test_diagnostic_snapshot_distinguishes_watchdog_from_detected_loss(
        self,
    ) -> None:
        decision = RealtimeControlDecision(12.5, True, 0, 0.0)
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        hold.accept(decision, now=10.0)

        watchdog = hold.diagnostic_snapshot(now=11.201)
        self.assertFalse(watchdog.loss_deadline_was_set)
        self.assertTrue(watchdog.watchdog_expired)
        self.assertEqual(watchdog.valid_until, 11.2)

        hold.accept(decision, now=20.0)
        hold.begin_loss(now=20.2)
        detected_loss = hold.diagnostic_snapshot(now=21.401)
        self.assertTrue(detected_loss.loss_deadline_was_set)
        self.assertFalse(detected_loss.watchdog_expired)
        self.assertEqual(detected_loss.loss_started_at, 20.2)
        self.assertAlmostEqual(detected_loss.valid_until, 21.4)


class CleanerControllerTests(unittest.TestCase):
    def test_front_and_side_controllers_use_role_specific_uart_bytes(self) -> None:
        uart = _FakeUart()
        front_cleaner = CleanerController(uart, FRONT_ACTUATOR_COMMANDS)
        front_pump = PumpController(uart, FRONT_ACTUATOR_COMMANDS)
        side_cleaner = CleanerController(uart, SIDE_ACTUATOR_COMMANDS)
        side_pump = PumpController(uart, SIDE_ACTUATOR_COMMANDS)

        with patch("run.time.monotonic", return_value=1.0):
            front_cleaner.set_duty_percent(25.0)
            front_pump.update(True)
            side_cleaner.set_duty_percent(25.0)
            side_pump.update(True)

        self.assertEqual(
            uart.writes,
            [
                FRONT_CLEANER_PWM_25_COMMAND,
                FRONT_PUMP_ON_COMMAND,
                SIDE_CLEANER_PWM_25_COMMAND,
                SIDE_PUMP_ON_COMMAND,
            ],
        )
        front_cleaner.force_off()
        self.assertFalse(front_cleaner.is_on)
        self.assertTrue(side_cleaner.is_on)

    def test_shared_safe_off_writes_all_false_state_controllers_after_failure(
        self,
    ) -> None:
        class FirstWriteFailsUart(_FakeUart):
            def write(self, command: bytes) -> int:
                self.writes.append(command)
                if len(self.writes) == 1:
                    raise OSError("shared serial failed")
                return len(command)

        uart = FirstWriteFailsUart()
        controllers = (
            CleanerController(uart, FRONT_ACTUATOR_COMMANDS),
            PumpController(uart, FRONT_ACTUATOR_COMMANDS),
            CleanerController(uart, SIDE_ACTUATOR_COMMANDS),
            PumpController(uart, SIDE_ACTUATOR_COMMANDS),
        )
        with self.assertRaisesRegex(RuntimeError, "shared serial failed"):
            force_cleaning_pairs_safe_off(
                (controllers[0], controllers[1]),
                (controllers[2], controllers[3]),
            )

        self.assertTrue(all(not controller.is_on for controller in controllers))
        self.assertEqual(len(uart.writes), 4)

    def test_applies_both_approved_pwm_setpoints_and_stops_immediately(self) -> None:
        uart = _FakeUart()
        controller = CleanerController(uart)
        controller.force_off()

        with patch("run.time.monotonic", side_effect=[1.0, 2.0]):
            controller.set_duty_percent(12.5)
            controller.set_duty_percent(25.0)
            controller.set_duty_percent(0.0)

        self.assertFalse(controller.is_on)
        self.assertEqual(controller.duty_percent, 0.0)
        self.assertEqual(
            uart.writes,
            [
                CLEANER_OFF_COMMAND,
                CLEANER_ON_COMMAND,
                CLEANER_PWM_25_COMMAND,
                CLEANER_OFF_COMMAND,
            ],
        )

    def test_active_pwm_command_is_repeated_as_heartbeat(self) -> None:
        uart = _FakeUart()
        controller = CleanerController(uart)

        with patch("run.time.monotonic", side_effect=[1.0, 1.1, 1.21]):
            controller.set_duty_percent(25.0)
            controller.set_duty_percent(25.0)
            controller.set_duty_percent(25.0)

        self.assertTrue(controller.is_on)
        self.assertEqual(controller.duty_percent, 25.0)
        self.assertEqual(uart.writes, [CLEANER_PWM_25_COMMAND] * 2)

    def test_force_off_resets_duty_without_uart(self) -> None:
        controller = CleanerController(None)

        with patch("run.time.monotonic", return_value=0.0):
            controller.set_duty_percent(25.0)
        self.assertTrue(controller.is_on)
        controller.force_off()
        self.assertFalse(controller.is_on)
        self.assertEqual(controller.duty_percent, 0.0)

    def test_rejects_unapproved_pwm_setpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "0, 12.5, or 25"):
            CleanerController(None).set_duty_percent(20.0)

    def test_partial_pwm_write_does_not_mark_cleaner_active(self) -> None:
        class ShortWriteUart(_FakeUart):
            def write(self, command: bytes) -> int:
                self.writes.append(command)
                return len(command) - 1

        controller = CleanerController(ShortWriteUart())
        with self.assertRaisesRegex(RuntimeError, "complete CLEANER_PWM_25"):
            controller.set_duty_percent(25.0)
        self.assertFalse(controller.is_on)
        self.assertEqual(controller.duty_percent, 0.0)


class PumpControllerTests(unittest.TestCase):
    def test_first_clean_observation_starts_water_immediately(self) -> None:
        uart = _FakeUart()
        controller = PumpController(uart)
        controller.force_off()

        with patch("run.time.monotonic", return_value=1.0):
            controller.update(True)

        self.assertTrue(controller.is_on)
        self.assertEqual(uart.writes, [PUMP_OFF_COMMAND, PUMP_ON_COMMAND])

    def test_stops_and_restarts_immediately(self) -> None:
        uart = _FakeUart()
        controller = PumpController(uart)
        controller.force_off()

        with patch("run.time.monotonic", side_effect=[0.0, 0.1, 0.2]):
            controller.update(True)
            controller.update(False)
            controller.update(True)

        self.assertTrue(controller.is_on)
        self.assertEqual(
            uart.writes,
            [
                PUMP_OFF_COMMAND,
                PUMP_ON_COMMAND,
                PUMP_OFF_COMMAND,
                PUMP_ON_COMMAND,
            ],
        )

    def test_active_pump_command_is_repeated_as_heartbeat(self) -> None:
        uart = _FakeUart()
        controller = PumpController(uart)
        with patch("run.time.monotonic", side_effect=[1.0, 1.1, 1.21]):
            controller.update(True)
            controller.update(True)
            controller.update(True)
        self.assertEqual(uart.writes, [PUMP_ON_COMMAND, PUMP_ON_COMMAND])

    def test_partial_pump_write_does_not_mark_pump_active(self) -> None:
        class ShortWriteUart(_FakeUart):
            def write(self, command: bytes) -> int:
                self.writes.append(command)
                return len(command) - 1

        controller = PumpController(ShortWriteUart())
        with self.assertRaisesRegex(RuntimeError, "complete PUMP_ON"):
            controller.update(True)
        self.assertFalse(controller.is_on)


class RealtimeCleaningControlTests(unittest.TestCase):
    @staticmethod
    def _crack_result(mask: np.ndarray) -> CrackDetectionResult:
        crack_pixels = int(np.count_nonzero(mask))
        return CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=crack_pixels,
            inspected_pixels=mask.size,
            crack_ratio=crack_pixels / mask.size,
            detected=crack_pixels > 0,
            method="bgcrack-tensorrt/realtime/trt-10.3.0/cuda:0",
            probability_threshold=0.5,
        )

    @staticmethod
    def _rust_result(grade: int) -> DetectionResult:
        class_map = np.full((240, 1280), grade, dtype=np.uint8)
        return DetectionResult(
            mask=(class_map > 0).astype(np.uint8),
            boxes=[],
            rust_ratio=float(np.count_nonzero(class_map)) / class_map.size,
            method="test-rust",
            class_map=class_map,
        )

    @staticmethod
    def _inference_outcome(*, rust=None, crack=None, ready: bool = True):
        return RealtimeInferenceOutcome(
            rust_result=rust,
            crack_result=crack,
            display_rust_result=rust,
            display_crack_result=crack,
            rust_seconds=0.0,
            crack_seconds=0.0,
            ready=ready,
        )

    def test_four_result_windows_apply_requested_control_table(self) -> None:
        history = RealtimeControlHistory()
        clear = self._crack_result(np.zeros((180, 1280), dtype=np.uint8))
        for _ in range(4):
            history.update(self._inference_outcome(rust=self._rust_result(0)))
            history.update(self._inference_outcome(crack=clear))
        self.assertTrue(history.ready)
        self.assertEqual(
            history.decision(),
            RealtimeControlDecision(12.5, True, 0, 0.0),
        )

        history.update(self._inference_outcome(rust=self._rust_result(1)))
        self.assertEqual(history.decision().cleaner_duty_percent, 25.0)
        self.assertFalse(history.decision().pump_on)

        history.update(self._inference_outcome(rust=self._rust_result(2)))
        self.assertEqual(history.decision().cleaner_duty_percent, 0.0)
        self.assertFalse(history.decision().pump_on)

    def test_crack_exceeds_point_zero_five_percent_and_holds_four_results(self) -> None:
        history = RealtimeControlHistory()
        self.assertEqual(CRACK_CONTROL_STOP_RATIO, 0.0005)
        pixels_at_limit = int(180 * 1280 * CRACK_CONTROL_STOP_RATIO)
        self.assertEqual(pixels_at_limit, 115)
        at_limit = np.zeros((180, 1280), dtype=np.uint8)
        at_limit.flat[:pixels_at_limit] = 255
        over_limit = at_limit.copy()
        over_limit.flat[pixels_at_limit] = 255
        for _ in range(4):
            history.update(self._inference_outcome(rust=self._rust_result(0)))
            history.update(self._inference_outcome(crack=self._crack_result(at_limit)))
        self.assertEqual(history.decision().cleaner_duty_percent, 12.5)
        self.assertTrue(history.decision().pump_on)

        history.update(self._inference_outcome(crack=self._crack_result(over_limit)))
        self.assertEqual(history.decision().cleaner_duty_percent, 0.0)
        self.assertFalse(history.decision().pump_on)
        for _ in range(3):
            history.update(self._inference_outcome(crack=self._crack_result(at_limit)))
            self.assertEqual(history.decision().cleaner_duty_percent, 0.0)
        history.update(self._inference_outcome(crack=self._crack_result(at_limit)))
        self.assertEqual(history.decision().cleaner_duty_percent, 12.5)

    def test_incomplete_or_not_ready_history_is_fail_closed(self) -> None:
        history = RealtimeControlHistory()
        history.update(self._inference_outcome(rust=self._rust_result(0)))
        self.assertFalse(history.ready)
        with self.assertRaisesRegex(RuntimeError, "Four fresh"):
            history.decision()
        clear = self._crack_result(np.zeros((180, 1280), dtype=np.uint8))
        for _ in range(4):
            history.update(self._inference_outcome(rust=self._rust_result(0)))
            history.update(self._inference_outcome(crack=clear))
        self.assertTrue(history.ready)
        history.update(self._inference_outcome(ready=False))
        self.assertFalse(history.ready)

    def test_decision_commands_pwm_and_pump(self) -> None:
        uart = _FakeUart()
        cleaner = CleanerController(uart)
        pump = PumpController(uart)
        cleaner.force_off()
        pump.force_off()
        with patch("run.time.monotonic", side_effect=[1.0, 1.0, 2.0]):
            update_cleaning_actuators(
                cleaner,
                pump,
                decision=RealtimeControlDecision(25.0, False, 1, 0.0),
            )
            update_cleaning_actuators(
                cleaner,
                pump,
                decision=RealtimeControlDecision(0.0, False, 2, 0.0),
            )
        self.assertEqual(
            uart.writes,
            [
                CLEANER_OFF_COMMAND,
                PUMP_OFF_COMMAND,
                CLEANER_PWM_25_COMMAND,
                CLEANER_OFF_COMMAND,
            ],
        )

    def test_new_side_hazard_stops_hold_before_history_rebuilds(self) -> None:
        clear = self._crack_result(np.zeros((180, 1280), dtype=np.uint8))
        over_limit = np.zeros((180, 1280), dtype=np.uint8)
        over_limit.flat[:116] = 255

        self.assertFalse(
            run_module.side_outcome_requires_immediate_off(
                self._inference_outcome(rust=self._rust_result(0)),
                history_ready=False,
            )
        )
        self.assertTrue(
            run_module.side_outcome_requires_immediate_off(
                self._inference_outcome(rust=self._rust_result(1)),
                history_ready=False,
            )
        )
        self.assertFalse(
            run_module.side_outcome_requires_immediate_off(
                self._inference_outcome(rust=self._rust_result(1)),
                history_ready=True,
            )
        )
        self.assertTrue(
            run_module.side_outcome_requires_immediate_off(
                self._inference_outcome(rust=self._rust_result(2)),
                history_ready=True,
            )
        )
        self.assertFalse(
            run_module.side_outcome_requires_immediate_off(
                self._inference_outcome(crack=clear),
                history_ready=False,
            )
        )
        self.assertTrue(
            run_module.side_outcome_requires_immediate_off(
                self._inference_outcome(
                    crack=self._crack_result(over_limit)
                ),
                history_ready=False,
            )
        )

    def test_off_diagnostic_uses_only_new_display_results(self) -> None:
        over_limit = np.zeros((180, 1280), dtype=np.uint8)
        over_limit.flat[:116] = 255
        cached_crack = self._crack_result(over_limit)
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        hold.accept(RealtimeControlDecision(12.5, True, 0, 0.0), now=9.0)
        hold.begin_loss(now=9.1)
        hold_snapshot = hold.diagnostic_snapshot(now=10.0)
        cached_only = RealtimeInferenceOutcome(
            rust_result=None,
            crack_result=cached_crack,
            display_rust_result=None,
            display_crack_result=None,
            rust_seconds=0.0,
            crack_seconds=0.0,
            ready=False,
        )
        diagnostic = run_module.build_control_off_diagnostic(
            role="side",
            outcome=cached_only,
            history_decision=None,
            history_ready=False,
            camera_error=False,
            camera_error_detail=None,
            hold_snapshot=hold_snapshot,
            hold=hold,
            frame=None,
            now=10.0,
        )
        self.assertEqual(diagnostic.reason, "history_hold_expired")
        self.assertIsNone(diagnostic.new_crack_pct)
        self.assertFalse(diagnostic.new_crack_blocked)
        self.assertTrue(diagnostic.pre_loss_deadline_set)
        self.assertFalse(diagnostic.watchdog_expired_at_off)

        with_new_crack = replace(
            cached_only,
            display_crack_result=cached_crack,
        )
        diagnostic = run_module.build_control_off_diagnostic(
            role="side",
            outcome=with_new_crack,
            history_decision=None,
            history_ready=False,
            camera_error=False,
            camera_error_detail=None,
            hold_snapshot=hold_snapshot,
            hold=hold,
            frame=None,
            now=10.0,
        )
        self.assertEqual(diagnostic.reason, "new_crack")
        self.assertAlmostEqual(
            diagnostic.new_crack_pct,
            (116 / (180 * 1280)) * 100,
        )

        diagnostic = run_module.build_control_off_diagnostic(
            role="side",
            outcome=None,
            history_decision=None,
            history_ready=False,
            camera_error=True,
            camera_error_detail="side camera\nUSB disconnected",
            hold_snapshot=hold_snapshot,
            hold=hold,
            frame=None,
            now=10.0,
        )
        self.assertEqual(diagnostic.reason, "camera_error")
        self.assertEqual(
            diagnostic.detail,
            "side camera USB disconnected",
        )

    def test_off_diagnostic_closes_snapshot_to_expiry_race(self) -> None:
        hold = RealtimeActuatorDecisionHold(hold_seconds=1.2)
        hold.accept(RealtimeControlDecision(12.5, True, 0, 0.0), now=10.0)
        snapshot = hold.diagnostic_snapshot(now=11.19)
        self.assertFalse(snapshot.watchdog_expired)
        self.assertIsNone(hold.current(now=11.201))

        diagnostic = run_module.build_control_off_diagnostic(
            role="side",
            outcome=None,
            history_decision=None,
            history_ready=False,
            camera_error=False,
            camera_error_detail=None,
            hold_snapshot=snapshot,
            hold=hold,
            frame=None,
            now=11.201,
        )

        self.assertEqual(diagnostic.reason, "decision_watchdog_expired")
        self.assertTrue(diagnostic.watchdog_expired_at_off)


class FrontCameraCleaningControlTests(unittest.TestCase):
    @staticmethod
    def _crack_result(crack_pixels: int) -> CrackDetectionResult:
        mask = np.zeros((180, 1280), dtype=np.uint8)
        mask.flat[:crack_pixels] = 255
        return CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=crack_pixels,
            inspected_pixels=mask.size,
            crack_ratio=crack_pixels / mask.size,
            detected=crack_pixels > 0,
            method="hrsegnet-b32-tensorrt/realtime/crack/test/logit-margin",
            probability_threshold=0.55,
        )

    @classmethod
    def _fill(
        cls,
        history: FrontControlHistory,
        *,
        crack_pixels: int = 0,
        foreign: bool = False,
    ) -> None:
        for _ in range(4):
            history.update_crack(cls._crack_result(crack_pixels))
            history.update_foreign(foreign)

    def test_clear_cracks_keep_pump_on_and_foreign_changes_only_cleaner(self) -> None:
        history = FrontControlHistory()
        self._fill(history)
        self.assertEqual(
            history.decision(),
            FrontControlDecision(12.5, True, False, 0.0),
        )

        history.update_foreign(True)
        self.assertEqual(history.decision().cleaner_duty_percent, 25.0)
        self.assertTrue(history.decision().pump_on)
        for _ in range(3):
            history.update_foreign(False)
            self.assertEqual(history.decision().cleaner_duty_percent, 25.0)
            self.assertTrue(history.decision().pump_on)
        history.update_foreign(False)
        self.assertEqual(history.decision().cleaner_duty_percent, 12.5)
        self.assertTrue(history.decision().pump_on)

    def test_front_crack_over_limit_stops_front_outputs(self) -> None:
        history = FrontControlHistory()
        self._fill(history, crack_pixels=116, foreign=True)
        decision = history.decision()
        self.assertEqual(decision.cleaner_duty_percent, 0.0)
        self.assertFalse(decision.pump_on)

        at_limit = FrontControlHistory()
        self._fill(at_limit, crack_pixels=115, foreign=True)
        self.assertEqual(at_limit.decision().cleaner_duty_percent, 25.0)
        self.assertTrue(at_limit.decision().pump_on)

    def test_new_front_crack_stops_hold_before_history_rebuilds(self) -> None:
        safe = run_module.TopRealtimeInferenceOutcome(
            None,
            self._crack_result(115),
            None,
            self._crack_result(115),
            0.0,
            0.0,
            False,
        )
        blocked = run_module.TopRealtimeInferenceOutcome(
            None,
            self._crack_result(116),
            None,
            self._crack_result(116),
            0.0,
            0.0,
            False,
        )
        self.assertFalse(run_module.front_outcome_requires_immediate_off(safe))
        self.assertTrue(run_module.front_outcome_requires_immediate_off(blocked))

    def test_crack_has_absolute_priority_over_all_obstacle_combinations(self) -> None:
        for crack_blocked in (False, True):
            for foreign in (False, True):
                with self.subTest(crack_blocked=crack_blocked, foreign=foreign):
                    history = FrontControlHistory()
                    self._fill(
                        history,
                        crack_pixels=116 if crack_blocked else 0,
                        foreign=foreign,
                    )
                    decision = history.decision()
                    if crack_blocked:
                        self.assertEqual(decision.cleaner_duty_percent, 0.0)
                        self.assertFalse(decision.pump_on)
                    else:
                        self.assertEqual(
                            decision.cleaner_duty_percent,
                            25.0 if foreign else 12.5,
                        )
                        self.assertTrue(decision.pump_on)

    def test_both_front_histories_are_required(self) -> None:
        history = FrontControlHistory()
        for _ in range(4):
            history.update_crack(self._crack_result(0))
        self.assertFalse(history.ready)
        with self.assertRaisesRegex(RuntimeError, "Four fresh top-crack"):
            history.decision()

    def test_invalid_result_resets_every_history(self) -> None:
        history = FrontControlHistory()
        self._fill(history)
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            history.update_foreign(None)  # type: ignore[arg-type]
        self.assertFalse(history.ready)

        self._fill(history)
        invalid = self._crack_result(0)
        invalid.mask = np.zeros((127, 1280), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "shape"):
            history.update_crack(invalid)
        self.assertFalse(history.ready)

        self._fill(history)
        invalid = self._crack_result(0)
        invalid.crack_ratio = None
        with self.assertRaisesRegex(ValueError, "not numeric"):
            history.update_crack(invalid)
        self.assertFalse(history.ready)

    def test_front_obstacle_and_side_rust_decisions_remain_independent(self) -> None:
        front = FrontControlHistory()
        side = RealtimeControlHistory()
        clear = self._crack_result(0)
        side_rust = RealtimeCleaningControlTests._rust_result(0)
        for _ in range(4):
            front.update_crack(clear)
            front.update_foreign(True)
            side.update(
                RealtimeCleaningControlTests._inference_outcome(rust=side_rust)
            )
            side.update(
                RealtimeCleaningControlTests._inference_outcome(crack=clear)
            )

        self.assertEqual(front.decision().cleaner_duty_percent, 25.0)
        self.assertTrue(front.decision().pump_on)
        self.assertEqual(side.decision().cleaner_duty_percent, 12.5)
        self.assertTrue(side.decision().pump_on)


class DualCameraRuntimeTests(unittest.TestCase):
    @staticmethod
    def _obstacle_result(*boxes) -> ObstacleDetectionResult:
        control_roi_detected = any(
            box[1] < run_module.REALTIME_RUST_ROI_BOTTOM and box[3] > run_module.REALTIME_OBSTACLE_ROI_TOP
            for box in boxes
        )
        return ObstacleDetectionResult(
            detections=tuple(
                ObstacleDetection(tuple(box), 0.9, 0) for box in boxes
            ),
            method=f"{OBSTACLE_DETECTOR_PREFIX}test",
            confidence_threshold=0.424,
            control_roi_detected=control_roi_detected,
        )

    def test_side_camera_failure_stops_only_side_pair(self) -> None:
        side_uart = _FakeUart()
        front_uart = _FakeUart()
        side_cleaner = CleanerController(side_uart, SIDE_ACTUATOR_COMMANDS)
        side_pump = PumpController(side_uart, SIDE_ACTUATOR_COMMANDS)
        front_cleaner = CleanerController(front_uart, FRONT_ACTUATOR_COMMANDS)
        front_pump = PumpController(front_uart, FRONT_ACTUATOR_COMMANDS)
        with patch("run.time.monotonic", return_value=1.0):
            side_cleaner.set_duty_percent(12.5)
            side_pump.update(True)
            front_cleaner.set_duty_percent(25.0)
            front_pump.update(True)
        top_frame = CameraFrame(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            2,
            1.0,
        )

        side_camera = Mock()
        side_camera.read_latest.side_effect = RuntimeError("side camera failed")
        top_camera = Mock()
        top_camera.read_latest.return_value = top_frame
        side_inference = Mock()
        side_inference.remaining_fresh_seconds.return_value = None
        top_inference = Mock()
        top_inference.remaining_fresh_seconds.return_value = None

        result = run_module.read_dual_realtime_camera_frames(
            side_camera,
            top_camera,
            after_side_sequence=1,
            after_top_sequence=1,
            side_inference=side_inference,
            top_inference=top_inference,
            side_cleaner_controller=side_cleaner,
            side_pump_controller=side_pump,
            front_cleaner_controller=front_cleaner,
            front_pump_controller=front_pump,
        )

        self.assertIsInstance(result, DualRealtimeFrameRead)
        self.assertIsNone(result.side_frame)
        self.assertIs(result.top_frame, top_frame)
        self.assertTrue(result.side_cache_expired)
        self.assertFalse(result.top_cache_expired)
        self.assertFalse(side_cleaner.is_on)
        self.assertFalse(side_pump.is_on)
        self.assertTrue(front_cleaner.is_on)
        self.assertTrue(front_pump.is_on)

    def test_top_cache_expiry_defers_front_off_to_command_hold(self) -> None:
        frame = CameraFrame(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            2,
            time.monotonic(),
        )
        side_camera = Mock()
        side_camera.read_latest.return_value = frame
        top_camera = Mock()
        top_camera.read_latest.side_effect = TimeoutError("top waiting")
        side_inference = Mock()
        side_inference.remaining_fresh_seconds.return_value = None
        top_inference = Mock()
        top_inference.remaining_fresh_seconds.return_value = 0.0
        arbiter = Mock()

        result = run_module.read_dual_realtime_camera_frames(
            side_camera,
            top_camera,
            after_side_sequence=1,
            after_top_sequence=1,
            side_inference=side_inference,
            top_inference=top_inference,
            side_cleaner_controller=CleanerController(None, SIDE_ACTUATOR_COMMANDS),
            side_pump_controller=PumpController(None, SIDE_ACTUATOR_COMMANDS),
            front_cleaner_controller=CleanerController(
                None,
                FRONT_ACTUATOR_COMMANDS,
            ),
            front_pump_controller=PumpController(None, FRONT_ACTUATOR_COMMANDS),
            actuator_arbiter=arbiter,
        )

        self.assertTrue(result.top_cache_expired)
        arbiter.clear.assert_not_called()

    def test_side_wait_uses_short_poll_and_preserves_top_ttl(self) -> None:
        clock = [10.0]
        side_camera = Mock()

        def side_read_latest(**kwargs: object) -> CameraFrame:
            timeout = float(kwargs["timeout"])
            self.assertLessEqual(
                timeout,
                run_module.DUAL_CAMERA_READ_POLL_SECONDS,
            )
            clock[0] += timeout
            raise TimeoutError("side has no new frame yet")

        side_camera.read_latest.side_effect = side_read_latest
        top_frame = CameraFrame(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            2,
            10.0 + run_module.DUAL_CAMERA_READ_POLL_SECONDS,
        )
        top_camera = Mock()
        top_camera.read_latest.return_value = top_frame
        side_inference = Mock()
        side_inference.remaining_fresh_seconds.return_value = 0.15
        top_inference = Mock()
        top_inference.remaining_fresh_seconds.return_value = 0.15
        side_cleaner = CleanerController(None, SIDE_ACTUATOR_COMMANDS)
        side_pump = PumpController(None, SIDE_ACTUATOR_COMMANDS)
        front_cleaner = CleanerController(None, FRONT_ACTUATOR_COMMANDS)
        front_pump = PumpController(None, FRONT_ACTUATOR_COMMANDS)
        front_cleaner.is_on = True
        front_cleaner.duty_percent = 25.0
        front_pump.is_on = True

        with patch.object(run_module.time, "monotonic", side_effect=lambda: clock[0]):
            result = run_module.read_dual_realtime_camera_frames(
                side_camera,
                top_camera,
                after_side_sequence=1,
                after_top_sequence=1,
                side_inference=side_inference,
                top_inference=top_inference,
                side_cleaner_controller=side_cleaner,
                side_pump_controller=side_pump,
                front_cleaner_controller=front_cleaner,
                front_pump_controller=front_pump,
            )

        self.assertAlmostEqual(
            clock[0],
            10.0 + run_module.DUAL_CAMERA_READ_POLL_SECONDS,
            places=6,
        )
        self.assertIsNone(result.side_frame)
        self.assertTrue(result.side_waiting)
        self.assertIs(result.top_frame, top_frame)
        self.assertFalse(result.top_cache_expired)
        self.assertTrue(front_cleaner.is_on)
        self.assertTrue(front_pump.is_on)

    def test_waiting_role_preserves_history_and_sends_heartbeats(self) -> None:
        uart = _FakeUart()
        cleaner = CleanerController(uart, FRONT_ACTUATOR_COMMANDS)
        pump = PumpController(uart, FRONT_ACTUATOR_COMMANDS)
        history = Mock()
        history.ready = True
        history.decision.return_value = FrontControlDecision(
            25.0,
            True,
            True,
            0.0,
        )
        inference = Mock()
        inference.remaining_fresh_seconds.return_value = 0.1

        with patch(
            "run.time.monotonic",
            side_effect=[1.0, 1.0, 1.21, 1.21],
        ):
            update_cleaning_actuators(
                cleaner,
                pump,
                decision=history.decision(),
            )
            maintained = maintain_waiting_role_control(
                cleaner,
                pump,
                history=history,
                inference=inference,
            )

        self.assertTrue(maintained)
        self.assertEqual(
            uart.writes,
            [
                FRONT_CLEANER_PWM_25_COMMAND,
                FRONT_PUMP_ON_COMMAND,
                FRONT_CLEANER_PWM_25_COMMAND,
                FRONT_PUMP_ON_COMMAND,
            ],
        )
        history.reset.assert_not_called()

    @staticmethod
    def _crack_result() -> CrackDetectionResult:
        mask = np.zeros((180, 1280), dtype=np.uint8)
        return CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=0,
            inspected_pixels=mask.size,
            crack_ratio=0.0,
            detected=False,
            method="hrsegnet-b32-tensorrt/realtime/crack/test/logit-margin",
            probability_threshold=0.55,
        )

    def test_obstacle_control_uses_box_intersection_with_bottom_240_rows(self) -> None:
        self.assertTrue(
            obstacle_detected_in_control_roi(
                self._obstacle_result((10.0, 479.0, 30.0, 500.0))
            )
        )
        self.assertFalse(
            obstacle_detected_in_control_roi(
                self._obstacle_result((10.0, 400.0, 30.0, 480.0))
            )
        )

    def test_obstacle_control_rejects_non_boolean_gpu_flag(self) -> None:
        result = self._obstacle_result()
        object.__setattr__(result, "control_roi_detected", np.bool_(False))
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            obstacle_detected_in_control_roi(result)

    def test_top_scheduler_alternates_obstacle_and_crack_rois(self) -> None:
        obstacle_detector = Mock()
        obstacle_detector.detect.return_value = self._obstacle_result()
        crack_detector = Mock()
        crack_detector.detect.return_value = self._crack_result()
        scheduler = AlternatingTopRealtimeInference(
            obstacle_detector,
            crack_detector,
        )
        frame = CameraFrame(
            np.zeros((720, 1280, 3), dtype=np.uint8),
            1,
            time.monotonic(),
        )
        obstacle_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        crack_roi = np.zeros((180, 1280, 3), dtype=np.uint8)

        first = scheduler.process(frame, obstacle_frame, crack_roi)
        second = scheduler.process(frame, obstacle_frame, crack_roi)

        obstacle_detector.detect.assert_called_once_with(obstacle_frame)
        crack_detector.detect.assert_called_once_with(crack_roi)
        self.assertIsNone(first.display_obstacle_result)
        self.assertIsNotNone(first.display_crack_result)
        self.assertIsNotNone(second.display_obstacle_result)
        self.assertTrue(second.ready)
        self.assertEqual(scheduler.crack_inference_count, 1)
        scheduler.reset()
        self.assertEqual(scheduler.crack_inference_count, 1)

    def test_top_preview_draws_crack_after_obstacle_boxes(self) -> None:
        result = self._obstacle_result((10.0, 500.0, 30.0, 520.0))
        crack = self._crack_result()
        events: list[str] = []

        with patch("run.cv2.rectangle", side_effect=lambda *_args: events.append("box")), patch(
            "node_runtime.display.annotate_realtime_control_results",
            side_effect=lambda image, _rust, _crack, **_kwargs: (
                events.append("crack") or image
            ),
        ):
            annotate_top_realtime_results(
                np.zeros((720, 1280, 3), dtype=np.uint8),
                result,
                crack,
            )

        self.assertEqual(events, ["box", "crack"])

    def test_dual_camera_stale_cycle_still_submits_preview(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        warmup_frames = [
            CameraFrame(frame.copy(), 1, time.monotonic()),
            CameraFrame(frame.copy(), 1, time.monotonic()),
        ]
        stale_frames = DualRealtimeFrameRead(
            side_frame=CameraFrame(frame.copy(), 2, 0.0),
            top_frame=CameraFrame(frame.copy(), 2, 0.0),
            side_cache_expired=False,
            top_cache_expired=False,
        )
        side_camera = Mock()
        side_camera.read_latest.return_value = warmup_frames[0]
        top_camera = Mock()
        top_camera.read_latest.return_value = warmup_frames[1]
        student = Mock(method="rust/gpu-argmax")
        crack = Mock(method="hrsegnet/realtime/logit-margin")
        obstacle = Mock(method="yolo26n/gpu-preprocess/gpu-compact")
        display = Mock()
        display.check_status.return_value = False
        display.submit.return_value = False
        args = SimpleNamespace(
            side_camera_device="side",
            top_camera_device="top",
            realtime_hrsegnet_crack_engine="crack.plan",
            realtime_hrsegnet_crack_engine_sha256="b" * 64,
            realtime_hrsegnet_crack_probability_threshold=0.5,
            realtime_hrsegnet_crack_min_component_pixels=20,
            obstacle_engine="obstacle.plan",
            obstacle_engine_sha256="c" * 64,
            obstacle_confidence_threshold=0.3,
            headless=False,
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch("run.validate_distinct_camera_devices")
            )
            stack.enter_context(patch("run.validate_tf32_runtime_environment"))
            stack.enter_context(
                patch(
                    "run.configured_student_engine",
                    return_value=("student.plan", "a" * 64, False),
                )
            )
            stack.enter_context(
                patch(
                    "run.validate_engine",
                    side_effect=[
                        (Path("student.plan"), "a" * 64),
                        (Path("crack.plan"), "b" * 64),
                        (Path("obstacle.plan"), "c" * 64),
                    ],
                )
            )
            stack.enter_context(patch("run.RustDetector", return_value=student))
            stack.enter_context(
                patch("run.HrSegNetCrackDetector", return_value=crack)
            )
            stack.enter_context(
                patch("run.ObstacleDetector", return_value=obstacle)
            )
            stack.enter_context(
                patch(
                    "run.open_latest_frame_camera",
                    side_effect=[side_camera, top_camera],
                )
            )
            stack.enter_context(
                patch("run.open_latest_frame_display", return_value=display)
            )
            stack.enter_context(
                patch(
                    "run.read_dual_realtime_camera_frames",
                    return_value=stale_frames,
                )
            )
            stack.enter_context(
                patch("node_runtime.vision.realtime_camera_frame_is_fresh", return_value=False)
            )
            stack.enter_context(
                patch(
                    "run.annotate_realtime_control_results",
                    side_effect=lambda image, *_args, **_kwargs: image.copy(),
                )
            )

            exit_code = run_module.run_dual_camera_realtime_test(args)

        self.assertEqual(exit_code, 0)
        display.submit.assert_called_once()

    def test_dual_camera_uart_test_handshakes_then_stops_all_on_error(self) -> None:
        events: list[tuple[str, object]] = []

        class ReadyUart:
            def __init__(self, *_args, **_kwargs) -> None:
                self.pending = bytearray()
                events.append(("uart-open", None))

            @property
            def in_waiting(self) -> int:
                return len(self.pending)

            def reset_input_buffer(self) -> None:
                events.append(("reset", None))
                self.pending.clear()

            def write(self, command: bytes) -> int:
                events.append(("write", command))
                if command == run_module.ACTUATOR_TEST_START_COMMAND:
                    self.pending.extend(b"ACTUATOR_TEST_READY\r\n")
                return len(command)

            def read(self, count: int) -> bytes:
                data = bytes(self.pending[:count])
                del self.pending[:count]
                events.append(("read", data))
                return data

            def close(self) -> None:
                events.append(("close", None))

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        side_camera = Mock()
        side_camera.read_latest.return_value = CameraFrame(
            frame.copy(), 1, time.monotonic()
        )
        top_camera = Mock()
        top_camera.read_latest.return_value = CameraFrame(
            frame.copy(), 1, time.monotonic()
        )
        student = Mock(method="rust/gpu-argmax")
        crack = Mock(method="hrsegnet/realtime/logit-margin")
        obstacle = Mock(method="yolo26n/gpu-preprocess/gpu-compact")
        student.detect.side_effect = lambda _frame: events.append(
            ("warmup", "rust")
        )
        crack.detect.side_effect = lambda _frame: events.append(
            ("warmup", "crack")
        )
        obstacle.detect.side_effect = lambda _frame: events.append(
            ("warmup", "obstacle")
        )
        args = SimpleNamespace(
            side_camera_device="side",
            top_camera_device="top",
            realtime_hrsegnet_crack_engine="crack.plan",
            realtime_hrsegnet_crack_engine_sha256="b" * 64,
            realtime_hrsegnet_crack_probability_threshold=0.5,
            realtime_hrsegnet_crack_min_component_pixels=20,
            obstacle_engine="obstacle.plan",
            obstacle_engine_sha256="c" * 64,
            obstacle_confidence_threshold=0.3,
            realtime_test_uart=True,
            serial_port="/dev/ttyACM0",
            baud_rate=115200,
            headless=True,
        )

        with ExitStack() as stack:
            stack.enter_context(patch("run.validate_distinct_camera_devices"))
            stack.enter_context(patch("run.validate_tf32_runtime_environment"))
            stack.enter_context(
                patch(
                    "run.configured_student_engine",
                    return_value=("student.plan", "a" * 64, False),
                )
            )
            stack.enter_context(
                patch(
                    "run.validate_engine",
                    side_effect=[
                        (Path("student.plan"), "a" * 64),
                        (Path("crack.plan"), "b" * 64),
                        (Path("obstacle.plan"), "c" * 64),
                    ],
                )
            )
            stack.enter_context(patch("run.serial.Serial", ReadyUart))
            stack.enter_context(patch("run.RustDetector", return_value=student))
            stack.enter_context(
                patch("run.HrSegNetCrackDetector", return_value=crack)
            )
            stack.enter_context(
                patch("run.ObstacleDetector", return_value=obstacle)
            )
            stack.enter_context(
                patch(
                    "run.open_latest_frame_camera",
                    side_effect=[side_camera, top_camera],
                )
            )
            stack.enter_context(
                patch(
                    "run.read_dual_realtime_camera_frames",
                    side_effect=RuntimeError("stop test loop"),
                )
            )
            stack.enter_context(
                patch(
                    "run.current_dual_actuator_state",
                    return_value=(12.5, True, 12.5, True),
                )
            )
            stdout = stack.enter_context(redirect_stdout(io.StringIO()))
            stderr = stack.enter_context(redirect_stderr(io.StringIO()))

            exit_code = run_module.run_dual_camera_realtime_test(args)

        self.assertEqual(exit_code, 1)
        writes = [value for event, value in events if event == "write"]
        self.assertEqual(writes[0], run_module.ACTUATOR_TEST_START_COMMAND)
        uart_open_index = events.index(("uart-open", None))
        warmup_indices = [
            index for index, event in enumerate(events) if event[0] == "warmup"
        ]
        self.assertEqual(len(warmup_indices), 4)
        self.assertLess(max(warmup_indices), uart_open_index)
        ready_read_index = next(
            index
            for index, event in enumerate(events)
            if event[0] == "read" and b"ACTUATOR_TEST_READY" in event[1]
        )
        first_off_index = next(
            index
            for index, event in enumerate(events)
            if event == ("write", run_module.FRONT_CLEANER_OFF_COMMAND)
        )
        self.assertLess(ready_read_index, first_off_index)
        self.assertNotIn(run_module.FRONT_CLEANER_PWM_25_COMMAND, writes)
        self.assertNotIn(run_module.FRONT_PUMP_ON_COMMAND, writes)
        self.assertNotIn(run_module.SIDE_CLEANER_PWM_25_COMMAND, writes)
        self.assertNotIn(run_module.SIDE_PUMP_ON_COMMAND, writes)
        self.assertEqual(writes[-1], run_module.ACTUATOR_TEST_STOP_COMMAND)
        self.assertEqual(events[-1], ("close", None))
        self.assertIn("UART TX: ACTUATOR_TEST_START", stdout.getvalue())
        self.assertIn("UART RX: ACTUATOR_TEST_READY", stdout.getvalue())
        self.assertIn(
            "Dual-camera realtime test: UART:/dev/ttyACM0",
            stdout.getvalue(),
        )
        self.assertIn("stop test loop", stderr.getvalue())
        off_lines = [
            line
            for line in stderr.getvalue().splitlines()
            if line.startswith("[CONTROL_OFF]")
        ]
        self.assertEqual(len(off_lines), 2)
        self.assertTrue(all("reason=runtime_error" in line for line in off_lines))
        self.assertTrue(any("role=front" in line for line in off_lines))
        self.assertTrue(any("role=side" in line for line in off_lines))

    def test_uart_actuator_test_timeout_sends_no_on_or_pwm_command(self) -> None:
        uart = Mock()
        uart.in_waiting = 0
        uart.write.side_effect = lambda command: len(command)

        with patch("run.time.monotonic", side_effect=[0.0, 2.0]), patch(
            "run.time.sleep"
        ) as sleep, self.assertRaisesRegex(TimeoutError, "were not enabled"):
            run_module.enter_uart_actuator_test(uart, timeout_seconds=1.0)

        uart.reset_input_buffer.assert_called_once_with()
        uart.write.assert_called_once_with(run_module.ACTUATOR_TEST_START_COMMAND)
        sleep.assert_not_called()

    def test_stopped_capture_pair_settles_once_and_uses_one_deadline(self) -> None:
        side = Mock()
        side.latest_sequence = 7
        side.read_latest.return_value = CameraFrame(
            np.zeros((720, 1280, 3), dtype=np.uint8), 8, 10.0
        )
        top = Mock()
        top.latest_sequence = 11
        top.read_latest.return_value = CameraFrame(
            np.zeros((720, 1280, 3), dtype=np.uint8), 12, 10.0
        )
        with patch("run.time.sleep") as sleep, patch(
            "run.time.monotonic", side_effect=[1.0, 1.01, 1.02]
        ):
            side_frame, top_frame = read_stopped_capture_pair(side, top)
        sleep.assert_called_once_with(run_module.CAPTURE_SETTLE_SECONDS)
        self.assertEqual(side_frame.sequence, 8)
        self.assertEqual(top_frame.sequence, 12)
        self.assertEqual(side.read_latest.call_args.kwargs["after_sequence"], 7)
        self.assertEqual(top.read_latest.call_args.kwargs["after_sequence"], 11)

    def test_capture_worker_routes_top_task_to_crack_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_path = Path(temporary_directory) / "top.jpg"
            raw_path.write_bytes(b"jpeg")
            worker = CaptureAnalysisWorker(Mock(), Mock(), crack_detector=Mock())
            task = CaptureAnalysisTask(
                raw_capture_path=raw_path,
                phase=INITIAL_PHASE,
                phase_sequence=1,
                trigger=CAPTURE_TRIGGER,
                captured_at=datetime.now(),
                camera_role=run_module.TOP_CAMERA_ROLE,
            )
            worker.start()
            with patch("run.cv2.imread", return_value=np.zeros((720, 1280, 3), dtype=np.uint8)), patch(
                "node_runtime.inspection.capture_top_crack_only",
                return_value=np.zeros((720, 1280, 3), dtype=np.uint8),
            ) as top_only, patch("node_runtime.inspection.capture_and_analyze") as side_analysis:
                worker.submit(task)
                self.assertTrue(worker.wait_until_idle(1.0))
                self.assertIsNone(worker.shutdown())
            top_only.assert_called_once()
            self.assertIs(top_only.call_args.args[3], worker.workbook)
            side_analysis.assert_not_called()


class DualDebugBenchmarkTests(unittest.TestCase):
    def test_summary_reports_all_requested_stage_names_and_sample_counts(self) -> None:
        samples = run_module.DualDebugBenchmarkSamples()
        timing = run_module.HrSegNetStageTiming(
            trt_inference_ms=1.0,
            gpu_logit_margin_threshold_ms=2.0,
            candidate_mask_d2h_ms=3.0,
            cpu_8ccl_filter_ms=4.0,
            detector_total_ms=5.0,
        )
        samples.record_hrseg(timing)
        samples.record_rust(6.0)
        samples.record_yolo(7.0)

        summary = samples.summary(3.0)

        self.assertIn("window_s=3.000", summary)
        self.assertEqual(len(summary.splitlines()), 1)
        for name, mean in (
            ("trt_inference_ms", 1.0),
            ("gpu_logit_margin_threshold_ms", 2.0),
            ("candidate_mask_d2h_ms", 3.0),
            ("cpu_8ccl_filter_ms", 4.0),
            ("detector_total_ms", 5.0),
            ("rust_total_ms", 6.0),
            ("yolo_total_ms", 7.0),
        ):
            self.assertIn(f"{name}(mean={mean:.3f},n=1)", summary)

    def test_benchmark_warms_up_then_runs_roles_without_uart_or_actuators(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        side_camera = Mock()
        top_camera = Mock()
        side_camera.read_latest.return_value = CameraFrame(frame, 1, 1.0)
        top_camera.read_latest.return_value = CameraFrame(frame, 1, 1.0)
        timing = run_module.HrSegNetStageTiming(1.0, 2.0, 3.0, 4.0, 5.0)
        rust = Mock(method="rust")
        crack = Mock(method="hrseg", last_stage_timing=timing)
        yolo = Mock(method="yolo")
        args = SimpleNamespace(
            side_camera_device="side",
            top_camera_device="top",
            realtime_hrsegnet_crack_engine="crack.plan",
            realtime_hrsegnet_crack_engine_sha256="b" * 64,
            realtime_hrsegnet_crack_probability_threshold=0.55,
            realtime_hrsegnet_crack_min_component_pixels=20,
            obstacle_engine="obstacle.plan",
            obstacle_engine_sha256="c" * 64,
            obstacle_confidence_threshold=0.3,
        )
        stdout = io.StringIO()

        with (
            patch.object(run_module, "validate_distinct_camera_devices"),
            patch.object(run_module, "validate_tf32_runtime_environment"),
            patch.object(
                run_module,
                "configured_student_engine",
                return_value=("rust.plan", "a" * 64, False),
            ),
            patch.object(
                run_module,
                "validate_engine",
                side_effect=[
                    (Path("rust.plan"), "a" * 64),
                    (Path("crack.plan"), "b" * 64),
                    (Path("obstacle.plan"), "c" * 64),
                ],
            ),
            patch.object(run_module, "RustDetector", return_value=rust),
            patch.object(run_module, "HrSegNetCrackDetector", return_value=crack) as crack_factory,
            patch.object(run_module, "ObstacleDetector", return_value=yolo),
            patch.object(run_module, "CleanerController") as cleaner_factory,
            patch.object(run_module, "PumpController") as pump_factory,
            patch.object(
                run_module, "publish_realtime_actuator_decision"
            ) as publish_actuator,
            patch.object(
                run_module, "DualControlOffTransitionLogger"
            ) as control_log_factory,
            patch.object(
                run_module,
                "open_latest_frame_camera",
                side_effect=[side_camera, top_camera],
            ),
            patch.object(run_module.serial, "Serial") as serial_factory,
            patch.object(
                run_module.time,
                "monotonic",
                side_effect=[100.0, 100.0, 103.1, 103.1],
            ),
            patch.object(
                run_module.time,
                "perf_counter",
                side_effect=[1.0, 1.01, 2.0, 2.02],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.run_dual_debug_benchmark(args)

        self.assertEqual(exit_code, 0)
        serial_factory.assert_not_called()
        cleaner_factory.assert_not_called()
        pump_factory.assert_not_called()
        publish_actuator.assert_not_called()
        control_log_factory.assert_not_called()
        self.assertEqual(rust.detect.call_count, 2)
        self.assertEqual(yolo.detect.call_count, 2)
        self.assertEqual(crack.detect.call_count, 4)
        crack_factory.assert_called_once_with(
            Path("crack.plan"),
            "b" * 64,
            0.55,
            20,
            role="realtime",
            stage_timing=True,
        )
        output = stdout.getvalue()
        self.assertEqual(len(output.splitlines()), 1)
        self.assertIn("[DUAL_DEBUG_BENCHMARK]", output)
        self.assertIn("rust_total_ms(mean=10.000,n=1)", output)
        self.assertIn("yolo_total_ms(mean=20.000,n=1)", output)
        self.assertNotIn("actuator", output.lower())


class RustDebugBenchmarkTests(unittest.TestCase):
    def test_summary_reports_all_rust_stage_names_and_sample_counts(self) -> None:
        samples = run_module.RustDebugBenchmarkSamples()
        samples.record(
            run_module.RustStageTiming(
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            )
        )

        summary = samples.summary(3.0)

        self.assertEqual(len(summary.splitlines()), 1)
        self.assertIn("[RUST_DEBUG_BENCHMARK]", summary)
        for name, mean in (
            ("rust_preprocess_cpu_ms", 1.0),
            ("rust_input_u8_h2d_ms", 2.0),
            ("rust_gpu_preprocess_ms", 3.0),
            ("rust_trt_inference_ms", 4.0),
            ("rust_gpu_argmax_ms", 5.0),
            ("rust_class_map_d2h_ms", 6.0),
            ("rust_cpu_postprocess_ms", 7.0),
            ("rust_total_ms", 8.0),
        ):
            self.assertIn(f"{name}(mean={mean:.3f},n=1)", summary)

    def test_rust_benchmark_warms_up_without_uart_or_actuators(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        side_camera = Mock()
        side_camera.read_latest.return_value = CameraFrame(frame, 1, 1.0)
        timing = run_module.RustStageTiming(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
        )
        rust = Mock(method="rust", last_stage_timing=timing)
        args = SimpleNamespace(
            side_camera_device="side",
            top_camera_device="top",
        )
        stdout = io.StringIO()

        with (
            patch.object(run_module, "validate_distinct_camera_devices"),
            patch.object(run_module, "validate_tf32_runtime_environment"),
            patch.object(
                run_module,
                "configured_student_engine",
                return_value=("rust.plan", "a" * 64, False),
            ),
            patch.object(
                run_module,
                "validate_engine",
                return_value=(Path("rust.plan"), "a" * 64),
            ),
            patch.object(run_module, "RustDetector", return_value=rust) as rust_factory,
            patch.object(run_module, "HrSegNetCrackDetector") as crack_factory,
            patch.object(run_module, "ObstacleDetector") as yolo_factory,
            patch.object(run_module, "CleanerController") as cleaner_factory,
            patch.object(run_module, "PumpController") as pump_factory,
            patch.object(run_module, "publish_realtime_actuator_decision") as publish_actuator,
            patch.object(run_module.serial, "Serial") as serial_factory,
            patch.object(
                run_module,
                "open_latest_frame_camera",
                return_value=side_camera,
            ),
            patch.object(
                run_module.time,
                "monotonic",
                side_effect=[
                    100.0,
                    100.0,
                    100.4,
                    100.8,
                    101.0,
                    200.0,
                    200.0,
                    203.1,
                    203.1,
                ],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.run_dual_rust_debug_benchmark(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_module.RUST_DEBUG_WARMUP_SECONDS, 1.0)
        rust_factory.assert_called_once_with(
            Path("rust.plan"),
            run_module.STUDENT_PROFILE,
            "a" * 64,
            gpu_argmax=True,
            stage_timing=True,
        )
        self.assertEqual(rust.detect.call_count, 4)
        self.assertEqual(side_camera.read_latest.call_count, 4)
        crack_factory.assert_not_called()
        yolo_factory.assert_not_called()
        serial_factory.assert_not_called()
        cleaner_factory.assert_not_called()
        pump_factory.assert_not_called()
        publish_actuator.assert_not_called()
        output = stdout.getvalue()
        self.assertEqual(len(output.splitlines()), 1)
        self.assertEqual(output.count(",n=1)"), 8)
        self.assertIn("rust_total_ms(mean=8.000,n=1)", output)
        self.assertNotIn("actuator", output.lower())


class YoloDebugBenchmarkTests(unittest.TestCase):
    def test_summary_reports_all_yolo_stage_names_and_sample_counts(self) -> None:
        samples = run_module.YoloDebugBenchmarkSamples()
        samples.record(
            run_module.ObstacleStageTiming(
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            )
        )

        summary = samples.summary(3.0)

        self.assertEqual(len(summary.splitlines()), 1)
        self.assertIn("[YOLO_DEBUG_BENCHMARK]", summary)
        for name, mean in (
            ("yolo_preprocess_cpu_ms", 1.0),
            ("yolo_input_u8_h2d_ms", 2.0),
            ("yolo_gpu_preprocess_ms", 3.0),
            ("yolo_trt_inference_ms", 4.0),
            ("yolo_gpu_postprocess_ms", 5.0),
            ("yolo_compact_d2h_ms", 6.0),
            ("yolo_cpu_result_ms", 7.0),
            ("yolo_total_ms", 8.0),
        ):
            self.assertIn(f"{name}(mean={mean:.3f},n=1)", summary)

    def test_top_camera_benchmark_repeats_windows_without_uart_or_actuators(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        top_camera = Mock()
        top_camera.read_latest.side_effect = [
            CameraFrame(frame, 1, 1.0),
            CameraFrame(frame, 2, 2.0),
            KeyboardInterrupt(),
        ]
        timing = run_module.ObstacleStageTiming(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
        )
        yolo = Mock(method="yolo", last_stage_timing=timing)
        args = SimpleNamespace(
            top_camera_device="top",
            obstacle_engine=Path("obstacle.plan"),
            obstacle_engine_sha256="c" * 64,
            obstacle_confidence_threshold=0.424,
        )
        stdout = io.StringIO()

        with (
            patch.object(run_module, "validate_tf32_runtime_environment"),
            patch.object(
                run_module,
                "validate_engine",
                return_value=(Path("obstacle.plan"), "c" * 64),
            ),
            patch.object(
                run_module, "ObstacleDetector", return_value=yolo
            ) as yolo_factory,
            patch.object(run_module, "HrSegNetCrackDetector") as crack_factory,
            patch.object(run_module, "CleanerController") as cleaner_factory,
            patch.object(run_module, "PumpController") as pump_factory,
            patch.object(run_module.serial, "Serial") as serial_factory,
            patch.object(
                run_module,
                "open_latest_frame_camera",
                return_value=top_camera,
            ),
            patch.object(
                run_module.time,
                "monotonic",
                side_effect=[0.0, 0.0, 1.1, 10.0, 10.0, 13.1, 13.2, 20.0, 20.0],
            ),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.run_yolo_debug_benchmark(args)

        self.assertEqual(exit_code, 0)
        yolo_factory.assert_called_once_with(
            Path("obstacle.plan"),
            "c" * 64,
            0.424,
            stage_timing=True,
        )
        self.assertEqual(yolo.detect.call_count, 2)
        top_camera.release.assert_called_once_with()
        crack_factory.assert_not_called()
        serial_factory.assert_not_called()
        cleaner_factory.assert_not_called()
        pump_factory.assert_not_called()
        output = stdout.getvalue()
        self.assertEqual(len(output.splitlines()), 1)
        self.assertEqual(output.count(",n=1)"), 8)
        self.assertIn("yolo_total_ms(mean=8.000,n=1)", output)
        self.assertNotIn("actuator", output.lower())


class EngineValidationTests(unittest.TestCase):
    def test_accepts_single_top_camera_yolo_debug_configuration(self) -> None:
        argv = [
            "run.py",
            "--yolo-debug-benchmark",
            "--top-camera-device",
            "/dev/top",
            "--obstacle-engine",
            "obstacle.plan",
            "--obstacle-engine-sha256",
            "c" * 64,
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertTrue(args.yolo_debug_benchmark)
        self.assertFalse(args.dual_camera)
        self.assertEqual(args.top_camera_device, "/dev/top")
        self.assertEqual(args.obstacle_engine, Path("obstacle.plan"))

    def test_uart_realtime_test_requires_realtime_and_dual_camera(self) -> None:
        invalid_argument_sets = (
            ["--realtime-test-uart"],
            ["--realtime-test", "--realtime-test-uart"],
            [
                "--realtime-test",
                "--realtime-test-uart",
                "--no-uart",
            ],
        )
        for arguments in invalid_argument_sets:
            with self.subTest(arguments=arguments), patch.object(
                sys,
                "argv",
                ["run.py", *arguments],
            ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_dual_camera_realtime_contract_requires_both_devices_and_obstacle_pin(self) -> None:
        base = [
            "run.py",
            "--realtime-test",
            "--student-engine",
            "rust.plan",
            "--student-engine-sha256",
            "b" * 64,
            "--realtime-hrsegnet-crack-engine",
            "crack.plan",
            "--realtime-hrsegnet-crack-engine-sha256",
            "c" * 64,
        ]
        with patch.object(
            sys,
            "argv",
            base + ["--side-camera-device", "/dev/side"],
        ), self.assertRaises(SystemExit):
            parse_args()

        with patch.object(
            sys,
            "argv",
            base
            + [
                "--side-camera-device",
                "/dev/side",
                "--top-camera-device",
                "/dev/top",
                "--obstacle-engine",
                "obstacle.plan",
            ],
        ), self.assertRaises(SystemExit):
            parse_args()

        with patch.object(
            sys,
            "argv",
            base
            + [
                "--side-camera-device",
                "/dev/side",
                "--top-camera-device",
                "/dev/top",
                "--obstacle-engine",
                "obstacle.plan",
                "--obstacle-engine-sha256",
                "d" * 64,
            ],
        ):
            args = parse_args()
        self.assertTrue(args.dual_camera)
        self.assertEqual(args.obstacle_confidence_threshold, 0.424)

    def test_accepts_explicit_dual_camera_uart_realtime_test(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--realtime-test",
                "--realtime-test-uart",
                "--student-engine",
                "rust.plan",
                "--student-engine-sha256",
                "b" * 64,
                "--realtime-hrsegnet-crack-engine",
                "crack.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "c" * 64,
                "--side-camera-device",
                "/dev/side",
                "--top-camera-device",
                "/dev/top",
                "--obstacle-engine",
                "obstacle.plan",
                "--obstacle-engine-sha256",
                "d" * 64,
            ],
        ):
            args = parse_args()

        self.assertTrue(args.realtime_test_uart)
        self.assertTrue(args.dual_camera)

    def test_dual_timing_requires_and_accepts_dual_realtime_test(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run.py", "--dual-timing", "--no-crack", "--no-uart"],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()

        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--realtime-test",
                "--dual-timing",
                "--student-engine",
                "rust.plan",
                "--student-engine-sha256",
                "b" * 64,
                "--realtime-hrsegnet-crack-engine",
                "crack.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "c" * 64,
                "--side-camera-device",
                "/dev/side",
                "--top-camera-device",
                "/dev/top",
                "--obstacle-engine",
                "obstacle.plan",
                "--obstacle-engine-sha256",
                "d" * 64,
            ],
        ):
            args = parse_args()

        self.assertTrue(args.dual_timing)
        self.assertTrue(args.dual_camera)

    def test_automatically_selects_headless_without_display_environment(self) -> None:
        with patch.dict(run_module.os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            ["run.py", "--no-crack", "--no-uart"],
        ):
            args = parse_args()

        self.assertTrue(args.headless)
        self.assertTrue(args.headless_auto)

    def test_display_environment_and_explicit_headless_precedence(self) -> None:
        for variable in ("DISPLAY", "WAYLAND_DISPLAY"):
            with self.subTest(variable=variable), patch.dict(
                run_module.os.environ,
                {variable: ":test"},
                clear=True,
            ), patch.object(
                sys,
                "argv",
                ["run.py", "--no-crack", "--no-uart"],
            ):
                args = parse_args()
                self.assertFalse(args.headless)
                self.assertFalse(args.headless_auto)

        with patch.dict(
            run_module.os.environ,
            {"DISPLAY": ":test"},
            clear=True,
        ), patch.object(
            sys,
            "argv",
            ["run.py", "--no-crack", "--no-uart", "--headless"],
        ):
            args = parse_args()
        self.assertTrue(args.headless)
        self.assertFalse(args.headless_auto)

    def test_firebase_upload_required_environment_flag(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value):
                self.assertTrue(
                    run_module.firebase_upload_is_required(
                        {"RAIL_ROBOT_FIREBASE_UPLOAD_REQUIRED": value}
                    )
                )
        for value in ("", "0", "false", "unexpected"):
            with self.subTest(value=value):
                self.assertFalse(
                    run_module.firebase_upload_is_required(
                        {"RAIL_ROBOT_FIREBASE_UPLOAD_REQUIRED": value}
                    )
                )

    def test_db_test_manual_capture_environment_flag(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value):
                self.assertTrue(
                    run_module.db_test_manual_capture_is_enabled(
                        {"RAIL_ROBOT_DB_TEST_MANUAL_CAPTURE": value}
                    )
                )
        for value in ("", "0", "false", "unexpected"):
            with self.subTest(value=value):
                self.assertFalse(
                    run_module.db_test_manual_capture_is_enabled(
                        {"RAIL_ROBOT_DB_TEST_MANUAL_CAPTURE": value}
                    )
                )

    def test_db_test_manifest_requires_exact_complete_dual_camera_matrix(self) -> None:
        captures = [
            {
                "phase": phase,
                "phase_sequence": sequence,
                "camera_role": role,
            }
            for phase in (INITIAL_PHASE, RESCAN_PHASE)
            for sequence in range(1, run_module.AUTOMATIC_RAIL_SECTION_TARGET + 1)
            for role in (run_module.SIDE_CAMERA_ROLE, run_module.TOP_CAMERA_ROLE)
        ]
        document = {
            "captures": captures,
            "run_summaries": [{"summary_complete": True}],
            "runs": [{"status": "complete"}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "run.json"
            manifest.write_text(run_module.json.dumps(document), encoding="utf-8")
            run_module.validate_db_test_dashboard_manifest(manifest)

            document["captures"] = captures[:-1]
            manifest.write_text(run_module.json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, f"exactly {4 * run_module.AUTOMATIC_RAIL_SECTION_TARGET} unique"
            ):
                run_module.validate_db_test_dashboard_manifest(manifest)

    def test_db_test_manifest_publish_is_suppressed_until_validated(self) -> None:
        manifest = Path("outputs/dashboard/runs/test.json")
        self.assertIsNone(
            run_module.dashboard_manifest_for_publish(
                manifest,
                db_test_manual_capture=True,
                db_test_publish_allowed=False,
            )
        )
        self.assertEqual(
            run_module.dashboard_manifest_for_publish(
                manifest,
                db_test_manual_capture=True,
                db_test_publish_allowed=True,
            ),
            manifest,
        )

    def test_db_test_capture_phase_maps_every_initial_then_rescan_section(self) -> None:
        target = run_module.AUTOMATIC_RAIL_SECTION_TARGET
        self.assertEqual(target, 6)
        actual = [
            run_module.db_test_capture_phase(min(total, target), max(0, total - target))
            for total in range(2 * target)
        ]
        self.assertEqual(
            actual,
            [(INITIAL_PHASE, sequence) for sequence in range(1, target + 1)]
            + [(RESCAN_PHASE, sequence) for sequence in range(1, target + 1)],
        )
        with self.assertRaisesRegex(RuntimeError, f"all {2 * target} captures"):
            run_module.db_test_capture_phase(target, target)
    def test_required_firebase_upload_rejects_every_missing_publish_path(self) -> None:
        manifest = Path("outputs/dashboard/runs/test.json")
        cases = (
            (None, True, object()),
            (manifest, False, object()),
            (manifest, True, None),
            (manifest, True, RuntimeError("network unavailable")),
        )
        for manifest_path, upload_safe, upload_outcome in cases:
            with self.subTest(
                manifest_path=manifest_path,
                upload_safe=upload_safe,
                upload_outcome=upload_outcome,
            ):
                uploader = Mock()
                if isinstance(upload_outcome, Exception):
                    uploader.side_effect = upload_outcome
                else:
                    uploader.return_value = upload_outcome
                with patch.object(
                    inspection_module,
                    "upload_dashboard_manifest_from_environment",
                    uploader,
                ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    published = run_module.publish_dashboard_manifest(
                        manifest_path,
                        firebase_upload_safe=upload_safe,
                        upload_required=True,
                    )

                self.assertFalse(published)
                if manifest_path is None or not upload_safe:
                    uploader.assert_not_called()
                else:
                    uploader.assert_called_once_with(manifest)

    def test_optional_firebase_failures_preserve_previous_success_contract(self) -> None:
        manifest = Path("outputs/dashboard/runs/test.json")
        for upload_outcome in (None, RuntimeError("network unavailable")):
            with self.subTest(upload_outcome=upload_outcome):
                uploader = Mock()
                if isinstance(upload_outcome, Exception):
                    uploader.side_effect = upload_outcome
                else:
                    uploader.return_value = upload_outcome
                with patch.object(
                    inspection_module,
                    "upload_dashboard_manifest_from_environment",
                    uploader,
                ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    published = run_module.publish_dashboard_manifest(
                        manifest,
                        firebase_upload_safe=True,
                        upload_required=False,
                    )

                self.assertTrue(published)

    def test_required_firebase_upload_reports_success(self) -> None:
        manifest = Path("outputs/dashboard/runs/test.json")
        upload_result = SimpleNamespace(
            run_id="run_test",
            artifact_count=4,
            uploaded_bytes=1024,
            firestore_document="inspection_exports/run_test",
        )
        stdout = io.StringIO()
        with patch.object(
            inspection_module,
            "upload_dashboard_manifest_from_environment",
            return_value=upload_result,
        ) as uploader, redirect_stdout(stdout):
            published = run_module.publish_dashboard_manifest(
                manifest,
                firebase_upload_safe=True,
                upload_required=True,
            )

        self.assertTrue(published)
        uploader.assert_called_once_with(manifest)
        self.assertIn("Firebase upload complete: run_id=run_test", stdout.getvalue())

    def test_capture_test_rejects_headless_before_runtime_initialization(self) -> None:
        with patch.dict(run_module.os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--capture-test",
                "--capture-hrsegnet-crack-engine",
                "capture-crack.plan",
                "--capture-hrsegnet-crack-engine-sha256",
                "c" * 64,
            ],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_accepts_pinned_hrsegnet_for_realtime_test(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--realtime-test",
                "--realtime-hrsegnet-crack-engine",
                "hrsegnet.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "7" * 64,
                "--realtime-hrsegnet-crack-probability-threshold",
                "0.73",
                "--realtime-hrsegnet-crack-min-component-pixels",
                "31",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.realtime_hrsegnet_crack_engine, Path("hrsegnet.plan"))
        self.assertEqual(args.realtime_hrsegnet_crack_engine_sha256, "7" * 64)
        self.assertEqual(args.realtime_hrsegnet_crack_probability_threshold, 0.73)
        self.assertEqual(args.realtime_hrsegnet_crack_min_component_pixels, 31)
        self.assertIsNone(args.realtime_crack_engine)
        self.assertIsNone(args.realtime_multitask_engine)

    def test_accepts_hrsegnet_for_normal_operation_with_capture_model(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--capture-hrsegnet-crack-engine",
                "capture.plan",
                "--capture-hrsegnet-crack-engine-sha256",
                "c" * 64,
                "--realtime-hrsegnet-crack-engine",
                "hrsegnet.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "7" * 64,
            ],
        ):
            args = parse_args()

        self.assertFalse(args.realtime_test)
        self.assertEqual(
            args.capture_hrsegnet_crack_engine,
            Path("capture.plan"),
        )
        self.assertEqual(args.capture_hrsegnet_crack_probability_threshold, 0.55)
        self.assertEqual(args.realtime_hrsegnet_crack_engine, Path("hrsegnet.plan"))
        self.assertEqual(args.realtime_hrsegnet_crack_probability_threshold, 0.55)
        self.assertEqual(args.realtime_hrsegnet_crack_min_component_pixels, 20)

    def test_rejects_hrsegnet_with_conflicting_crack_modes(self) -> None:
        invalid_options = (
            [
                "--capture-test",
                "--capture-crack-engine",
                "capture.plan",
                "--realtime-hrsegnet-crack-engine",
                "hrsegnet.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "7" * 64,
            ],
            [
                "--no-crack",
                "--realtime-hrsegnet-crack-engine",
                "hrsegnet.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "7" * 64,
            ],
            [
                "--realtime-test",
                "--realtime-hrsegnet-crack-engine",
                "hrsegnet.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "7" * 64,
                "--realtime-crack-engine",
                "bgcrack.plan",
            ],
            [
                "--realtime-test",
                "--realtime-hrsegnet-crack-engine",
                "hrsegnet.plan",
                "--realtime-hrsegnet-crack-engine-sha256",
                "7" * 64,
                "--realtime-multitask-engine",
                "multitask.plan",
                "--realtime-multitask-engine-sha256",
                "8" * 64,
            ],
            ["--realtime-test", "--realtime-hrsegnet-crack-engine", "hrsegnet.plan"],
        )
        for options in invalid_options:
            with self.subTest(options=options), patch.object(
                sys, "argv", ["run.py", *options]
            ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_accepts_capture_test_without_realtime_engines(self) -> None:
        with patch.dict(
            run_module.os.environ,
            {"DISPLAY": ":test"},
            clear=True,
        ), patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--capture-test",
                "--capture-hrsegnet-crack-engine",
                "capture-crack.plan",
                "--capture-hrsegnet-crack-engine-sha256",
                "c" * 64,
            ],
        ):
            args = parse_args()

        self.assertTrue(args.capture_test)
        self.assertIsNone(args.realtime_crack_engine)
        self.assertIsNone(args.realtime_multitask_engine)

    def test_capture_and_realtime_tests_are_mutually_exclusive(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--capture-test",
                "--realtime-test",
                "--capture-hrsegnet-crack-engine",
                "capture-crack.plan",
                "--capture-hrsegnet-crack-engine-sha256",
                "c" * 64,
            ],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()

    def test_accepts_stable_device_for_single_camera_capture_test(self) -> None:
        with patch.dict(
            run_module.os.environ,
            {"DISPLAY": ":test"},
            clear=True,
        ), patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--capture-test",
                "--capture-camera-device",
                "/dev/v4l/by-path/front-camera",
                "--capture-hrsegnet-crack-engine",
                "capture-crack.plan",
                "--capture-hrsegnet-crack-engine-sha256",
                "c" * 64,
            ],
        ):
            args = parse_args()

        self.assertEqual(
            args.capture_camera_device,
            "/dev/v4l/by-path/front-camera",
        )
        self.assertFalse(args.dual_camera)

    def test_rejects_capture_camera_device_outside_capture_test(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run.py", "--capture-camera-device", "/dev/video2"],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parse_args()

        self.assertEqual(raised.exception.code, 2)

    def test_accepts_front_capture_test_with_pinned_inputs(self) -> None:
        with patch.dict(
            run_module.os.environ,
            {"DISPLAY": ":test"},
            clear=True,
        ), patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--front-capture-test",
                "--capture-camera-device",
                "/dev/v4l/by-path/front-camera",
                "--capture-hrsegnet-crack-engine",
                "capture-crack.plan",
                "--capture-hrsegnet-crack-engine-sha256",
                "c" * 64,
                "--obstacle-engine",
                "obstacle.plan",
                "--obstacle-engine-sha256",
                "d" * 64,
            ],
        ):
            args = parse_args()

        self.assertTrue(args.front_capture_test)
        self.assertFalse(args.capture_test)
        self.assertIsNone(args.student_engine)
        self.assertEqual(args.obstacle_engine, Path("obstacle.plan"))

    def test_front_capture_test_requires_camera_crack_and_obstacle(self) -> None:
        complete = [
            "--front-capture-test",
            "--capture-camera-device",
            "/dev/video2",
            "--capture-hrsegnet-crack-engine",
            "crack.plan",
            "--capture-hrsegnet-crack-engine-sha256",
            "c" * 64,
            "--obstacle-engine",
            "obstacle.plan",
            "--obstacle-engine-sha256",
            "d" * 64,
        ]
        for option_and_value in (
            ("--capture-camera-device", "/dev/video2"),
            ("--capture-hrsegnet-crack-engine", "crack.plan"),
            ("--obstacle-engine", "obstacle.plan"),
        ):
            invalid = complete.copy()
            index = invalid.index(option_and_value[0])
            del invalid[index : index + 2]
            if option_and_value[0] == "--capture-hrsegnet-crack-engine":
                sha_index = invalid.index("--capture-hrsegnet-crack-engine-sha256")
                del invalid[sha_index : sha_index + 2]
            if option_and_value[0] == "--obstacle-engine":
                sha_index = invalid.index("--obstacle-engine-sha256")
                del invalid[sha_index : sha_index + 2]
            with self.subTest(missing=option_and_value[0]), patch.dict(
                run_module.os.environ,
                {"DISPLAY": ":test"},
                clear=True,
            ), patch.object(
                sys,
                "argv",
                ["run.py", *invalid],
            ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_rejects_legacy_pinned_optimized_multitask_realtime_test(self) -> None:
        error = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--realtime-test",
                "--realtime-multitask-engine",
                "optimized.plan",
                "--realtime-multitask-engine-sha256",
                "e" * 64,
            ],
        ), redirect_stderr(error), self.assertRaises(SystemExit):
            parse_args()

        self.assertIn("bottom-quarter 180-row crack ROI", error.getvalue())

    def test_rejects_legacy_hybrid_student_rust_and_multitask_crack(self) -> None:
        error = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--realtime-test",
                "--student-engine",
                "student.plan",
                "--student-engine-sha256",
                "b" * 64,
                "--realtime-multitask-engine",
                "optimized.plan",
                "--realtime-multitask-engine-sha256",
                "e" * 64,
            ],
        ), redirect_stderr(error), self.assertRaises(SystemExit):
            parse_args()

        self.assertIn("bottom-quarter 180-row crack ROI", error.getvalue())

    def test_rejects_legacy_optimized_rust_and_multitask_crack(self) -> None:
        error = io.StringIO()
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--realtime-test",
                "--optimized-student-engine",
                "optimized-rust.plan",
                "--optimized-student-engine-sha256",
                "b" * 64,
                "--realtime-multitask-engine",
                "optimized-crack.plan",
                "--realtime-multitask-engine-sha256",
                "e" * 64,
            ],
        ), redirect_stderr(error), self.assertRaises(SystemExit):
            parse_args()

        self.assertIn("bottom-quarter 180-row crack ROI", error.getvalue())

    def test_rejects_raw_and_optimized_student_options_together(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run.py",
                "--no-crack",
                "--student-engine",
                "raw.plan",
                "--optimized-student-engine",
                "optimized.plan",
            ],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()

    def test_rejects_unpinned_or_conflicting_multitask_engine(self) -> None:
        invalid_options = (
            ["--realtime-test", "--realtime-multitask-engine", "raw.plan"],
            [
                "--realtime-test",
                "--realtime-multitask-engine",
                "optimized.plan",
                "--realtime-multitask-engine-sha256",
                "e" * 64,
                "--realtime-crack-engine",
                "crack.plan",
            ],
        )
        for options in invalid_options:
            with self.subTest(options=options), patch.object(
                sys, "argv", ["run.py", *options]
            ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args()

    def test_accepts_no_crack_without_a_crack_engine(self) -> None:
        with patch.object(sys, "argv", ["run.py", "--no-crack", "--no-uart"]):
            args = parse_args()

        self.assertTrue(args.no_crack)
        self.assertIsNone(args.capture_crack_engine)
        self.assertIsNone(args.realtime_crack_engine)

    def test_rejects_no_crack_with_uart_enabled(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run.py", "--no-crack"],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()

    def test_rejects_no_crack_with_crack_configuration(self) -> None:
        conflicting_options = [
            ["--capture-crack-engine", "capture-crack.plan"],
            ["--realtime-crack-engine-sha256", "0" * 64],
        ]

        for options in conflicting_options:
            with self.subTest(options=options), patch.object(
                sys, "argv", ["run.py", "--no-crack", *options]
            ), self.assertRaises(SystemExit):
                parse_args()

    def test_realtime_test_requires_a_crack_engine(self) -> None:
        with patch.object(
            sys, "argv", ["run.py", "--realtime-test"]
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()

    def test_realtime_test_rejects_no_crack(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run.py", "--realtime-test", "--no-crack"],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()

    def test_rejects_legacy_separate_capture_and_realtime_crack_contracts(self) -> None:
        argv = [
            "run.py",
            "--capture-hrsegnet-crack-engine",
            "capture-crack.plan",
            "--capture-hrsegnet-crack-engine-sha256",
            "c" * 64,
            "--capture-hrsegnet-crack-probability-threshold",
            "0.4",
            "--realtime-crack-engine",
            "realtime-crack.plan",
            "--realtime-crack-engine-sha256",
            "d" * 64,
            "--realtime-crack-threshold",
            "0.7",
            "--capture-hrsegnet-crack-min-component-pixels",
            "12",
            "--realtime-crack-min-component-pixels",
            "34",
        ]

        error = io.StringIO()
        with patch.object(sys, "argv", argv), redirect_stderr(error), self.assertRaises(SystemExit):
            parse_args()

        self.assertIn("bottom-quarter 180-row crack ROI", error.getvalue())

    def test_normal_mode_rejects_only_one_crack_engine(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run.py", "--capture-crack-engine", "capture-crack.plan"],
        ), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args()

    def test_legacy_engine_option_only_aliases_the_student(self) -> None:
        argv = [
            "run.py",
            "--engine",
            "student.plan",
            "--teacher-engine",
            "teacher.plan",
            "--no-uart",
            "--no-crack",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.student_engine, Path("student.plan"))
        self.assertEqual(args.teacher_engine, Path("teacher.plan"))

    def test_accepts_matching_approved_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_path = Path(temporary_directory) / "model.plan"
            engine_bytes = b"test TensorRT engine bytes"
            engine_path.write_bytes(engine_bytes)
            expected_sha256 = hashlib.sha256(engine_bytes).hexdigest()

            validated_path, actual_sha256 = validate_engine(
                engine_path,
                expected_sha256.upper(),
            )

            self.assertEqual(validated_path, engine_path.resolve())
            self.assertEqual(actual_sha256, expected_sha256)

    def test_rejects_unapproved_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_path = Path(temporary_directory) / "model.plan"
            engine_path.write_bytes(b"unexpected engine")

            with self.assertRaisesRegex(ValueError, "approved digest"):
                validate_engine(engine_path, "0" * 64)

    def test_tf32_environment_must_be_exactly_zero(self) -> None:
        for value in (None, "1"):
            with self.subTest(value=value), patch.dict(
                run_module.os.environ,
                {},
                clear=True,
            ):
                if value is not None:
                    run_module.os.environ["NVIDIA_TF32_OVERRIDE"] = value
                with self.assertRaisesRegex(RuntimeError, "exactly 0"):
                    validate_tf32_runtime_environment()

        with patch.dict(
            run_module.os.environ,
            {"NVIDIA_TF32_OVERRIDE": "0"},
            clear=True,
        ):
            validate_tf32_runtime_environment()


class RealtimeTestModeTests(unittest.TestCase):
    def test_formats_running_average_pipeline_statistics(self) -> None:
        label = format_realtime_test_statistics(
            4,
            0.15,
            0.2,
            0.04,
            0.08,
            app_frame_count=4,
            total_app_seconds=0.24,
        )

        self.assertEqual(
            label,
            "CONTROL=20.0 FPS latency=50.0 ms | "
            "APP=16.7 FPS 60.0 ms | "
            "rust=10.0 ms crack=20.0 ms frames=4",
        )

    def test_formats_integrated_multitask_latency_without_fake_crack_zero(self) -> None:
        label = format_realtime_test_statistics(
            4,
            0.15,
            0.2,
            0.04,
            0.0,
            app_frame_count=4,
            total_app_seconds=0.24,
            multitask=True,
        )

        self.assertEqual(
            label,
            "CONTROL=20.0 FPS latency=50.0 ms | "
            "APP=16.7 FPS 60.0 ms | multitask=10.0 ms frames=4",
        )

    def test_first_frame_reports_rate_warmup_without_inventing_fps(self) -> None:
        label = format_realtime_test_statistics(1, 0.0, 0.04, 0.01, 0.02)

        self.assertEqual(
            label,
            "CONTROL=warming up latency=40.0 ms | APP=warming up | "
            "rust=10.0 ms crack=20.0 ms frames=1",
        )

    def test_headless_realtime_test_runs_inference_without_gui_rendering(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        student_path = Path("resolved-student.plan").resolve()
        crack_path = Path("resolved-crack.plan").resolve()
        args = SimpleNamespace(
            headless=True,
            headless_auto=True,
            no_crack=False,
            camera_index=0,
            student_engine=Path("student.plan"),
            student_engine_sha256=None,
            optimized_student_engine=None,
            optimized_student_engine_sha256=None,
            realtime_multitask_engine=None,
            realtime_crack_engine=Path("crack.plan"),
            realtime_crack_engine_sha256=None,
            realtime_crack_threshold=0.5,
            realtime_crack_min_component_pixels=20,
        )

        def rust_result_for(roi: np.ndarray) -> DetectionResult:
            height, width = roi.shape[:2]
            return DetectionResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                boxes=[],
                rust_ratio=0.0,
                method="deeplabv3plus-tensorrt/student/fake",
                class_map=np.zeros((height, width), dtype=np.uint8),
                class_ratios={},
            )

        def crack_result_for(roi: np.ndarray) -> CrackDetectionResult:
            height, width = roi.shape[:2]
            return CrackDetectionResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                boxes=[],
                crack_pixels=0,
                inspected_pixels=height * width,
                crack_ratio=0.0,
                detected=False,
                method="bgcrack-tensorrt/realtime/fake",
                probability_threshold=0.5,
            )

        student = Mock(
            method="deeplabv3plus-tensorrt/student/fake",
        )
        student.detect.side_effect = rust_result_for
        crack = Mock(method="bgcrack-tensorrt/realtime/fake")
        crack.detect.side_effect = crack_result_for
        camera = _FakeLatestCamera((frame,))
        display_factory = Mock()
        annotation = Mock()
        roi_guide = Mock()
        result_summary = Mock()
        statistics_draw = Mock()
        stdout = io.StringIO()

        with (
            patch.dict(run_module.os.environ, {"NVIDIA_TF32_OVERRIDE": "0"}),
            patch.object(
                run_module,
                "validate_engine",
                side_effect=[
                    (student_path, "b" * 64),
                    (crack_path, "c" * 64),
                ],
            ),
            patch.object(run_module, "RustDetector", return_value=student),
            patch.object(run_module, "CrackDetector", return_value=crack),
            patch.object(run_module, "open_latest_frame_camera", return_value=camera),
            patch.object(run_module, "open_latest_frame_display", display_factory),
            patch.object(
                run_module,
                "read_realtime_camera_frame",
                side_effect=[
                    (
                        CameraFrame(frame, 2, run_module.time.monotonic()),
                        False,
                    ),
                    RuntimeError("bounded test stop"),
                ],
            ),
            patch.object(
                run_module,
                "annotate_realtime_control_results",
                annotation,
            ),
            patch.object(run_module, "draw_realtime_roi_guide", roi_guide),
            patch.object(run_module, "draw_realtime_result_summary", result_summary),
            patch.object(run_module, "draw_realtime_test_statistics", statistics_draw),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.run_realtime_test(args)

        self.assertEqual(exit_code, 1)
        self.assertGreaterEqual(student.detect.call_count, 2)
        self.assertGreaterEqual(crack.detect.call_count, 1)
        display_factory.assert_not_called()
        annotation.assert_not_called()
        roi_guide.assert_not_called()
        result_summary.assert_not_called()
        statistics_draw.assert_not_called()
        self.assertEqual(camera.release_calls, 1)
        self.assertIn("Display mode: headless", stdout.getvalue())
        self.assertIn("[realtime-test][SIMULATED:NO-UART]", stdout.getvalue())

    def test_hybrid_initializes_optimized_rust_and_keeps_multitask_crack(self) -> None:
        optimized_path = Path("resolved-optimized-rust.plan").resolve()
        multitask_path = Path("resolved-multitask.plan").resolve()
        optimized_sha256 = "f" * 64
        multitask_sha256 = "e" * 64
        args = SimpleNamespace(
            no_crack=False,
            camera_index=0,
            student_engine=None,
            student_engine_sha256=None,
            optimized_student_engine=Path("optimized-rust.plan"),
            optimized_student_engine_sha256=optimized_sha256,
            realtime_multitask_engine=Path("multitask.plan"),
            realtime_multitask_engine_sha256=multitask_sha256,
            realtime_crack_engine=None,
            realtime_crack_engine_sha256=None,
            realtime_crack_threshold=0.5,
            realtime_crack_min_component_pixels=20,
        )
        optimized_factory = Mock(return_value=Mock())
        multitask_factory = Mock(return_value=Mock())

        with (
            patch.dict(
                run_module.os.environ,
                {"NVIDIA_TF32_OVERRIDE": "0"},
            ),
            patch.object(
                run_module,
                "validate_engine",
                side_effect=[
                    (multitask_path, multitask_sha256),
                    (optimized_path, optimized_sha256),
                ],
            ),
            patch.object(
                run_module,
                "OptimizedRustDetector",
                optimized_factory,
            ),
            patch.object(
                run_module,
                "OptimizedMultitaskDetector",
                multitask_factory,
            ),
            patch.object(
                run_module,
                "open_latest_frame_camera",
                side_effect=RuntimeError("stop after detector wiring"),
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.run_realtime_test(args)

        self.assertEqual(exit_code, 1)
        optimized_factory.assert_called_once_with(
            optimized_path,
            optimized_sha256,
        )
        multitask_factory.assert_called_once_with(
            multitask_path,
            multitask_sha256,
            0.5,
            20,
        )

    def test_hrsegnet_realtime_test_wires_only_test_detector_and_never_uart(self) -> None:
        student_path = Path("resolved-student.plan").resolve()
        hrsegnet_path = Path("resolved-hrsegnet.plan").resolve()
        student_sha256 = "b" * 64
        hrsegnet_sha256 = "7" * 64
        args = SimpleNamespace(
            no_crack=False,
            camera_index=0,
            student_engine=Path("student.plan"),
            student_engine_sha256=student_sha256,
            optimized_student_engine=None,
            optimized_student_engine_sha256=None,
            realtime_multitask_engine=None,
            realtime_crack_engine=None,
            realtime_crack_engine_sha256=None,
            realtime_hrsegnet_crack_engine=Path("hrsegnet.plan"),
            realtime_hrsegnet_crack_engine_sha256=hrsegnet_sha256,
            realtime_hrsegnet_crack_probability_threshold=0.73,
            realtime_hrsegnet_crack_min_component_pixels=31,
        )
        rust_factory = Mock(return_value=Mock())
        hrsegnet_factory = Mock(return_value=Mock())
        serial_factory = Mock()

        with (
            patch.dict(run_module.os.environ, {"NVIDIA_TF32_OVERRIDE": "0"}),
            patch.object(
                run_module,
                "validate_engine",
                side_effect=[
                    (student_path, student_sha256),
                    (hrsegnet_path, hrsegnet_sha256),
                ],
            ),
            patch.object(run_module, "RustDetector", rust_factory),
            patch.object(run_module, "HrSegNetCrackDetector", hrsegnet_factory),
            patch.object(run_module.serial, "Serial", serial_factory),
            patch.object(
                run_module,
                "open_latest_frame_camera",
                side_effect=RuntimeError("stop after detector wiring"),
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.run_realtime_test(args)

        self.assertEqual(exit_code, 1)
        rust_factory.assert_called_once_with(
            student_path,
            run_module.STUDENT_PROFILE,
            student_sha256,
            gpu_argmax=True,
        )
        hrsegnet_factory.assert_called_once_with(
            hrsegnet_path,
            hrsegnet_sha256,
            0.73,
            31,
            role="realtime",
        )
        serial_factory.assert_not_called()

    def test_normal_operation_wires_separate_capture_and_realtime_hrsegnet(self) -> None:
        teacher_path = Path("resolved-teacher.plan").resolve()
        student_path = Path("resolved-student.plan").resolve()
        capture_crack_path = Path("resolved-capture-crack.plan").resolve()
        hrsegnet_path = Path("resolved-hrsegnet.plan").resolve()
        teacher_sha256 = "a" * 64
        student_sha256 = "b" * 64
        capture_crack_sha256 = "c" * 64
        hrsegnet_sha256 = "7" * 64
        args = SimpleNamespace(
            capture_test=False,
            realtime_test=False,
            no_crack=False,
            no_uart=False,
            camera_index=0,
            teacher_engine=Path("teacher.plan"),
            teacher_engine_sha256=teacher_sha256,
            student_engine=Path("student.plan"),
            student_engine_sha256=student_sha256,
            optimized_student_engine=None,
            optimized_student_engine_sha256=None,
            capture_hrsegnet_crack_engine=Path("capture-crack.plan"),
            capture_hrsegnet_crack_engine_sha256=capture_crack_sha256,
            capture_hrsegnet_crack_probability_threshold=0.41,
            capture_hrsegnet_crack_min_component_pixels=12,
            realtime_crack_engine=None,
            realtime_crack_engine_sha256=None,
            realtime_crack_threshold=0.67,
            realtime_crack_min_component_pixels=34,
            realtime_multitask_engine=None,
            realtime_multitask_engine_sha256=None,
            realtime_hrsegnet_crack_engine=Path("hrsegnet.plan"),
            realtime_hrsegnet_crack_engine_sha256=hrsegnet_sha256,
            realtime_hrsegnet_crack_probability_threshold=0.5,
            realtime_hrsegnet_crack_min_component_pixels=20,
            report=Path("unused.xlsx"),
            serial_port="must-not-open",
            baud_rate=115200,
        )
        teacher = Mock(method="deeplabv3plus-tensorrt/teacher/fake")
        student = Mock(method="deeplabv3plus-tensorrt/student/fake")
        capture_crack = Mock(
            method="hrsegnet-b32-tensorrt/capture/crack/fake/logit-margin"
        )
        hrsegnet = Mock(
            method="hrsegnet-b32-tensorrt/realtime/crack/fake/logit-margin"
        )
        rust_factory = Mock(side_effect=[teacher, student])
        capture_crack_factory = Mock(return_value=capture_crack)
        hrsegnet_factory = Mock(side_effect=[capture_crack, hrsegnet])
        serial_factory = Mock()
        warmup_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        camera = Mock()
        camera.read_latest.return_value = CameraFrame(
            warmup_frame,
            sequence=1,
            read_completed_at=time.monotonic(),
        )
        workbook_factory = Mock(side_effect=RuntimeError("stop after report wiring"))

        with (
            patch.object(run_module, "parse_args", return_value=args),
            patch.dict(run_module.os.environ, {"NVIDIA_TF32_OVERRIDE": "0"}),
            patch.object(
                run_module,
                "validate_engine",
                side_effect=[
                    (teacher_path, teacher_sha256),
                    (student_path, student_sha256),
                    (hrsegnet_path, hrsegnet_sha256),
                    (capture_crack_path, capture_crack_sha256),
                ],
            ),
            patch.object(run_module, "RustDetector", rust_factory),
            patch.object(run_module, "HrSegNetCrackDetector", hrsegnet_factory),
            patch.object(run_module, "InspectionWorkbook", workbook_factory),
            patch.object(run_module.serial, "Serial", serial_factory),
            patch.object(
                run_module,
                "open_latest_frame_camera",
                return_value=camera,
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(
            rust_factory.call_args_list,
            [
                call(teacher_path, run_module.TEACHER_PROFILE, teacher_sha256),
                call(
                    student_path,
                    run_module.STUDENT_PROFILE,
                    student_sha256,
                    gpu_argmax=True,
                ),
            ],
        )
        self.assertEqual(hrsegnet_factory.call_count, 2)
        self.assertEqual(
            hrsegnet_factory.call_args_list[0],
            call(
            capture_crack_path,
            capture_crack_sha256,
            0.41,
            12,
            role="capture",
            ),
        )
        self.assertEqual(
            hrsegnet_factory.call_args_list[1],
            call(
            hrsegnet_path,
            hrsegnet_sha256,
            0.5,
            20,
            role="realtime",
            ),
        )
        self.assertEqual(teacher.detect.call_args.args[0].shape, (720, 1280, 3))
        self.assertEqual(capture_crack.detect.call_args.args[0].shape, (720, 1280, 3))
        self.assertEqual(student.detect.call_args.args[0].shape, (240, 1280, 3))
        self.assertEqual(hrsegnet.detect.call_args.args[0].shape, (180, 1280, 3))
        report_kwargs = workbook_factory.call_args.kwargs
        self.assertEqual(report_kwargs["crack_model_filename"], capture_crack_path.name)
        self.assertEqual(report_kwargs["crack_model_sha256"], capture_crack_sha256)
        self.assertEqual(report_kwargs["crack_detector"], capture_crack.method)
        self.assertEqual(report_kwargs["crack_probability_threshold"], 0.41)
        self.assertEqual(report_kwargs["capture_crack_min_component_pixels"], 12)
        self.assertEqual(
            report_kwargs["realtime_crack_model_filename"],
            hrsegnet_path.name,
        )
        self.assertEqual(
            report_kwargs["realtime_crack_model_sha256"],
            hrsegnet_sha256,
        )
        self.assertEqual(
            report_kwargs["realtime_crack_detector"],
            hrsegnet.method,
        )
        self.assertEqual(report_kwargs["realtime_crack_probability_threshold"], 0.5)
        self.assertEqual(report_kwargs["realtime_crack_min_component_pixels"], 20)
        serial_factory.assert_not_called()
        teacher.close.assert_called_once()
        student.close.assert_called_once()
        capture_crack.close.assert_called_once()
        hrsegnet.close.assert_called_once()
        camera.release.assert_called_once()

    def test_runs_only_student_and_crack_without_mission_resources(self) -> None:
        warmup_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rust_frame = np.ones((720, 1280, 3), dtype=np.uint8)
        crack_frame = np.full((720, 1280, 3), 2, dtype=np.uint8)

        student_path = Path("resolved-student.plan").resolve()
        crack_path = Path("resolved-crack.plan").resolve()
        student_sha256 = "b" * 64
        crack_sha256 = "c" * 64
        args = SimpleNamespace(
            realtime_test=True,
            no_crack=False,
            no_uart=False,
            camera_index=2,
            student_engine=Path("student.plan"),
            student_engine_sha256=None,
            teacher_engine=Path("missing-teacher.plan"),
            teacher_engine_sha256=None,
            realtime_crack_engine=Path("crack.plan"),
            realtime_crack_engine_sha256=None,
            realtime_crack_threshold=0.5,
            realtime_crack_min_component_pixels=20,
            report=Path("must-not-be-created.xlsx"),
            serial_port="must-not-open",
            baud_rate=115200,
        )
        def rust_result_for(frame: np.ndarray) -> DetectionResult:
            height, width = frame.shape[:2]
            return DetectionResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                boxes=[],
                rust_ratio=0.0,
                method="deeplabv3plus-tensorrt/student/fake",
                class_map=np.zeros((height, width), dtype=np.uint8),
                class_ratios={},
            )

        def crack_result_for(frame: np.ndarray) -> CrackDetectionResult:
            height, width = frame.shape[:2]
            return CrackDetectionResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                boxes=[],
                crack_pixels=0,
                inspected_pixels=height * width,
                crack_ratio=0.0,
                detected=False,
                method="bgcrack-tensorrt/realtime/fake",
                probability_threshold=0.5,
            )

        student = Mock()
        student.method = "deeplabv3plus-tensorrt/student/fake"
        student.detect.side_effect = rust_result_for
        crack = Mock()
        crack.method = "bgcrack-tensorrt/realtime/fake"
        crack.detect.side_effect = crack_result_for
        camera = _FakeLatestCamera((warmup_frame, rust_frame, crack_frame))
        rust_factory = Mock(return_value=student)
        crack_factory = Mock(return_value=crack)
        serial_factory = Mock()
        workbook_factory = Mock()
        worker_factory = Mock()
        draw_statistics = Mock(side_effect=lambda frame, _label: frame)
        display = _FakeDisplay((ord("x"), ord("q")))
        stdout = io.StringIO()

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    run_module.os.environ,
                    {"NVIDIA_TF32_OVERRIDE": "0"},
                )
            )
            stack.enter_context(
                patch.object(run_module, "parse_args", return_value=args)
            )
            validate = stack.enter_context(
                patch.object(
                    run_module,
                    "validate_engine",
                    side_effect=[
                        (student_path, student_sha256),
                        (crack_path, crack_sha256),
                    ],
                )
            )
            for target, replacement in (
                ("RustDetector", rust_factory),
                ("CrackDetector", crack_factory),
                ("InspectionWorkbook", workbook_factory),
                ("CaptureAnalysisWorker", worker_factory),
            ):
                stack.enter_context(patch.object(run_module, target, replacement))
            stack.enter_context(
                patch.object(run_module.serial, "Serial", serial_factory)
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "open_latest_frame_camera",
                    return_value=camera,
                )
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "open_latest_frame_display",
                    return_value=display,
                )
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "annotate",
                    side_effect=lambda frame, _result: frame,
                )
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "annotate_cracks",
                    side_effect=lambda frame, _result, _zone: frame,
                )
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "draw_realtime_roi_guide",
                    side_effect=lambda frame, _annotated_frame=None: frame,
                )
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "draw_realtime_test_statistics",
                    draw_statistics,
                )
            )
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(io.StringIO()))
            exit_code = run_module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(validate.call_count, 2)
        rust_factory.assert_called_once_with(
            student_path,
            run_module.STUDENT_PROFILE,
            student_sha256,
            gpu_argmax=True,
        )
        crack_factory.assert_called_once_with(
            crack_path,
            run_module.CRACK_REALTIME_PROFILE,
            crack_sha256,
            0.5,
            20,
        )
        self.assertEqual(student.detect.call_count, 2)
        self.assertEqual(crack.detect.call_count, 2)
        self.assertEqual(
            student.detect.call_args_list[0].args[0].shape, (240, 1280, 3)
        )
        self.assertEqual(
            student.detect.call_args_list[1].args[0].shape, (240, 1280, 3)
        )
        self.assertEqual(
            crack.detect.call_args_list[0].args[0].shape, (180, 1280, 3)
        )
        self.assertEqual(
            crack.detect.call_args_list[1].args[0].shape, (180, 1280, 3)
        )
        student.close.assert_called_once_with()
        crack.close.assert_called_once_with()
        self.assertEqual(camera.release_calls, 1)
        self.assertEqual(display.close_calls, 1)
        workbook_factory.assert_not_called()
        worker_factory.assert_not_called()
        serial_factory.assert_not_called()
        statistics_label = draw_statistics.call_args.args[1]
        self.assertIn("CONTROL=", statistics_label)
        self.assertIn("frames=2", statistics_label)
        output = stdout.getvalue()
        self.assertIn(str(student_path), output)
        self.assertIn(student_sha256, output)
        self.assertIn(str(crack_path), output)
        self.assertIn(crack_sha256, output)
        self.assertIn(
            "[realtime-test][SIMULATED:NO-UART] "
            "CLEANER_PWM=OFF WATER_PUMP=OFF",
            output,
        )
        self.assertNotIn(f"[realtime-test] {statistics_label}", output)


class CaptureTestModeTests(unittest.TestCase):
    def test_headless_capture_test_stops_before_models_camera_or_highgui(self) -> None:
        args = SimpleNamespace(headless=True)
        validate = Mock()
        open_camera = Mock()
        named_window = Mock()

        with (
            patch.object(run_module, "validate_engine", validate),
            patch.object(run_module, "open_latest_frame_camera", open_camera),
            patch.object(run_module.cv2, "namedWindow", named_window),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run_module.run_capture_test(args)

        self.assertEqual(exit_code, 2)
        validate.assert_not_called()
        open_camera.assert_not_called()
        named_window.assert_not_called()

    def test_s_runs_only_full_frame_capture_models_without_mission_resources(self) -> None:
        frames = tuple(
            np.full((720, 1280, 3), fill_value, dtype=np.uint8)
            for fill_value in (0, 1, 2, 3)
        )

        teacher_path = Path("resolved-teacher.plan").resolve()
        crack_path = Path("resolved-capture-crack.plan").resolve()
        teacher_sha256 = "a" * 64
        crack_sha256 = "c" * 64
        args = SimpleNamespace(
            capture_test=True,
            realtime_test=False,
            camera_index=2,
            teacher_engine=Path("teacher.plan"),
            teacher_engine_sha256=None,
            capture_hrsegnet_crack_engine=Path("capture-crack.plan"),
            capture_hrsegnet_crack_engine_sha256=None,
            capture_hrsegnet_crack_probability_threshold=0.5,
            capture_hrsegnet_crack_min_component_pixels=20,
        )

        def rust_result_for(frame: np.ndarray) -> DetectionResult:
            height, width = frame.shape[:2]
            class_map = np.zeros((height, width), dtype=np.uint8)
            class_map[:, : width // 4] = 1
            return DetectionResult(
                mask=(class_map != 0).astype(np.uint8),
                boxes=[],
                rust_ratio=0.25,
                method="deeplabv3plus-tensorrt/teacher/fake",
                class_map=class_map,
                class_ratios={"Good": 0.75, "Fair": 0.25},
            )

        def crack_result_for(frame: np.ndarray) -> CrackDetectionResult:
            height, width = frame.shape[:2]
            return CrackDetectionResult(
                mask=np.zeros((height, width), dtype=np.uint8),
                boxes=[],
                crack_pixels=0,
                inspected_pixels=height * width,
                crack_ratio=0.0,
                detected=False,
                method="hrsegnet-b32-tensorrt/capture/crack/fake/logit-margin",
                probability_threshold=0.5,
            )

        teacher = Mock()
        teacher.method = "deeplabv3plus-tensorrt/teacher/fake"
        teacher.detect.side_effect = rust_result_for
        crack = Mock()
        crack.method = "hrsegnet-b32-tensorrt/capture/crack/fake/logit-margin"
        crack.detect.side_effect = crack_result_for
        camera = _FakeLatestCamera(frames)
        teacher_factory = Mock(return_value=teacher)
        crack_factory = Mock(return_value=crack)
        serial_factory = Mock()
        workbook_factory = Mock()
        worker_factory = Mock()
        realtime_factory = Mock()
        stdout = io.StringIO()

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(run_module.os.environ, {"NVIDIA_TF32_OVERRIDE": "0"})
            )
            stack.enter_context(patch.object(run_module, "parse_args", return_value=args))
            validate = stack.enter_context(
                patch.object(
                    run_module,
                    "validate_engine",
                    side_effect=[
                        (teacher_path, teacher_sha256),
                        (crack_path, crack_sha256),
                    ],
                )
            )
            stack.enter_context(
                patch.object(run_module, "RustDetector", teacher_factory)
            )
            stack.enter_context(
                patch.object(run_module, "HrSegNetCrackDetector", crack_factory)
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "OptimizedMultitaskDetector",
                    realtime_factory,
                )
            )
            stack.enter_context(
                patch.object(run_module, "InspectionWorkbook", workbook_factory)
            )
            stack.enter_context(
                patch.object(run_module, "CaptureAnalysisWorker", worker_factory)
            )
            stack.enter_context(patch.object(run_module.serial, "Serial", serial_factory))
            stack.enter_context(
                patch.object(
                    run_module,
                    "open_latest_frame_camera",
                    return_value=camera,
                )
            )
            stack.enter_context(
                patch.object(run_module, "annotate", side_effect=lambda frame, _: frame)
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "annotate_cracks",
                    side_effect=lambda frame, _result, _zone: frame,
                )
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "draw_roi_guide",
                    side_effect=lambda frame, _display=None: frame,
                )
            )
            for cv2_name in ("namedWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(run_module.cv2, cv2_name))
            stack.enter_context(
                patch.object(
                    run_module.cv2,
                    "waitKey",
                    side_effect=[ord("x"), ord("s"), ord("q")],
                )
            )
            stack.enter_context(
                patch.object(
                    run_module.time,
                    "perf_counter",
                    side_effect=[1.0, 1.01, 1.03],
                )
            )
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(io.StringIO()))
            exit_code = run_module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(validate.call_count, 2)
        teacher_factory.assert_called_once_with(
            teacher_path,
            run_module.TEACHER_PROFILE,
            teacher_sha256,
        )
        crack_factory.assert_called_once_with(
            crack_path,
            crack_sha256,
            0.5,
            20,
            role="capture",
        )
        self.assertEqual(teacher.detect.call_count, 2)
        self.assertEqual(crack.detect.call_count, 2)
        for detector in (teacher, crack):
            self.assertEqual(detector.detect.call_args_list[0].args[0].shape, (720, 1280, 3))
            self.assertEqual(detector.detect.call_args_list[1].args[0].shape, (720, 1280, 3))
            detector.close.assert_called_once_with()
        self.assertEqual(int(teacher.detect.call_args_list[1].args[0][0, 0, 0]), 2)
        self.assertEqual(camera.release_calls, 1)
        serial_factory.assert_not_called()
        workbook_factory.assert_not_called()
        worker_factory.assert_not_called()
        realtime_factory.assert_not_called()
        output = stdout.getvalue()
        self.assertIn("[capture-test] rust=10.0 ms crack=20.0 ms total=30.0 ms", output)
        self.assertIn("rust_ratio=25.00% crack_detected=NO crack_ratio=0.00%", output)


class FrontObstacleCrackSuppressionTests(unittest.TestCase):
    @staticmethod
    def _result(*boxes) -> ObstacleDetectionResult:
        return ObstacleDetectionResult(
            detections=tuple(
                ObstacleDetection(tuple(box), 0.9, 0) for box in boxes
            ),
            method=f"{OBSTACLE_DETECTOR_PREFIX}test",
            confidence_threshold=0.424,
            control_roi_detected=bool(boxes),
        )

    def test_suppresses_only_box_crossed_by_crack(self) -> None:
        overlapping = (10.0, 10.0, 30.0, 30.0)
        separate = (50.0, 50.0, 70.0, 70.0)
        original = self._result(overlapping, separate)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 19:21] = 255

        filtered = suppress_front_obstacles_overlapping_cracks(original, mask)

        self.assertEqual(filtered.boxes, (separate,))
        self.assertTrue(filtered.detected)
        self.assertFalse(filtered.control_roi_detected)
        self.assertEqual(original.boxes, (overlapping, separate))

    def test_keeps_box_when_overlap_is_below_threshold(self) -> None:
        box = (10.0, 10.0, 20.0, 20.0)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10, 10] = 255

        filtered = suppress_front_obstacles_overlapping_cracks(
            self._result(box),
            mask,
            overlap_ratio_threshold=0.02,
        )

        self.assertEqual(filtered.boxes, (box,))

    def test_suppresses_box_at_exact_threshold(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[0, :10] = 255

        filtered = suppress_front_obstacles_overlapping_cracks(
            self._result((0.0, 0.0, 10.0, 10.0)),
            mask,
            overlap_ratio_threshold=0.10,
        )

        self.assertFalse(filtered.detected)
        self.assertFalse(filtered.control_roi_detected)

    def test_clamps_box_to_mask_and_accepts_empty_mask(self) -> None:
        box = (-5.0, -5.0, 5.0, 5.0)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:1, :5] = 255

        filtered = suppress_front_obstacles_overlapping_cracks(
            self._result(box),
            mask,
            overlap_ratio_threshold=0.20,
        )
        empty_filtered = suppress_front_obstacles_overlapping_cracks(
            self._result(box),
            np.zeros((0, 0), dtype=np.uint8),
        )

        self.assertFalse(filtered.detected)
        self.assertEqual(empty_filtered.boxes, (box,))

    def test_default_threshold_is_one_percent(self) -> None:
        self.assertEqual(FRONT_CAPTURE_OBSTACLE_CRACK_OVERLAP_RATIO, 0.01)


class FrontCaptureTestModeTests(unittest.TestCase):
    def test_s_runs_obstacle_roi_and_full_frame_crack_on_same_camera_frame(self) -> None:
        frames = tuple(
            np.full((720, 1280, 3), fill_value, dtype=np.uint8)
            for fill_value in (0, 1, 2, 3)
        )
        obstacle_path = Path("resolved-obstacle.plan").resolve()
        crack_path = Path("resolved-capture-crack.plan").resolve()
        obstacle_sha256 = "d" * 64
        crack_sha256 = "c" * 64
        args = SimpleNamespace(
            front_capture_test=True,
            capture_test=False,
            headless=False,
            capture_camera_device="/dev/v4l/by-path/front-camera",
            obstacle_engine=Path("obstacle.plan"),
            obstacle_engine_sha256=obstacle_sha256,
            obstacle_confidence_threshold=0.3,
            capture_hrsegnet_crack_engine=Path("capture-crack.plan"),
            capture_hrsegnet_crack_engine_sha256=crack_sha256,
            capture_hrsegnet_crack_probability_threshold=0.55,
            capture_hrsegnet_crack_min_component_pixels=20,
        )
        obstacle_result = ObstacleDetectionResult(
            detections=(
                ObstacleDetection((10.0, 10.0, 20.0, 20.0), 0.9, 0),
            ),
            method="yolo26n/fake",
            confidence_threshold=0.3,
            control_roi_detected=True,
        )

        def crack_result_for(frame: np.ndarray) -> CrackDetectionResult:
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[10:20, 10:11] = 255
            return CrackDetectionResult(
                mask=mask,
                boxes=[],
                crack_pixels=10,
                inspected_pixels=height * width,
                crack_ratio=10 / (height * width),
                detected=True,
                method="hrsegnet/capture/fake",
                probability_threshold=0.55,
            )

        obstacle = Mock(method="yolo26n/fake")
        obstacle.detect.return_value = obstacle_result
        crack = Mock(method="hrsegnet/capture/fake")
        crack.detect.side_effect = crack_result_for
        camera = _FakeLatestCamera(frames)

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(run_module.os.environ, {"NVIDIA_TF32_OVERRIDE": "0"})
            )
            stack.enter_context(patch.object(run_module, "parse_args", return_value=args))
            validate = stack.enter_context(
                patch.object(
                    run_module,
                    "validate_engine",
                    side_effect=[
                        (obstacle_path, obstacle_sha256),
                        (crack_path, crack_sha256),
                    ],
                )
            )
            obstacle_factory = stack.enter_context(
                patch.object(run_module, "ObstacleDetector", return_value=obstacle)
            )
            crack_factory = stack.enter_context(
                patch.object(run_module, "HrSegNetCrackDetector", return_value=crack)
            )
            stack.enter_context(
                patch.object(
                    run_module,
                    "open_latest_frame_camera",
                    return_value=camera,
                )
            )
            annotate = stack.enter_context(
                patch.object(
                    run_module,
                    "annotate_front_capture_results",
                    side_effect=lambda frame, _obstacle, _crack: frame,
                )
            )
            for cv2_name in ("namedWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(run_module.cv2, cv2_name))
            stack.enter_context(
                patch.object(
                    run_module.cv2,
                    "waitKey",
                    side_effect=[ord("x"), ord("s"), ord("q")],
                )
            )
            stack.enter_context(
                patch.object(
                    run_module.time,
                    "perf_counter",
                    side_effect=[1.0, 1.01, 1.03],
                )
            )
            stdout = stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = run_module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(validate.call_count, 2)
        obstacle_factory.assert_called_once_with(obstacle_path, obstacle_sha256, 0.3)
        crack_factory.assert_called_once_with(
            crack_path,
            crack_sha256,
            0.55,
            20,
            role="capture",
        )
        self.assertEqual(obstacle.detect.call_count, 2)
        self.assertEqual(crack.detect.call_count, 2)
        captured_obstacle_roi = obstacle.detect.call_args_list[1].args[0]
        captured_crack_frame = crack.detect.call_args_list[1].args[0]
        self.assertEqual(captured_obstacle_roi.shape, (720, 1280, 3))
        self.assertEqual(captured_crack_frame.shape, (720, 1280, 3))
        self.assertEqual(int(captured_obstacle_roi[0, 0, 0]), 2)
        self.assertEqual(int(captured_crack_frame[0, 0, 0]), 2)
        annotate.assert_called_once()
        displayed_obstacle_result = annotate.call_args.args[1]
        self.assertFalse(displayed_obstacle_result.detected)
        self.assertFalse(displayed_obstacle_result.control_roi_detected)
        obstacle.close.assert_called_once_with()
        crack.close.assert_called_once_with()
        self.assertEqual(camera.release_calls, 1)
        self.assertIn(
            "[front-capture-test] obstacle=10.0 ms crack=20.0 ms total=30.0 ms",
            stdout.getvalue(),
        )
        self.assertIn("obstacle_detected=NO crack_detected=YES", stdout.getvalue())
        self.assertIn(
            "suppressed_obstacles=1 reason=crack_overlap threshold=1.0%",
            stdout.getvalue(),
        )


class RoiGuideTests(unittest.TestCase):
    def test_hybrid_discards_multitask_rust_and_keeps_its_crack(self) -> None:
        water_roi = np.zeros((240, 1280, 3), dtype=np.uint8)
        crack_roi = np.zeros((180, 1280, 3), dtype=np.uint8)
        old_rust = object()
        unused_multitask_rust = object()
        multitask_crack = object()
        student = Mock()
        student.detect.return_value = old_rust
        multitask = Mock()
        multitask.detect.return_value = (unused_multitask_rust, multitask_crack)

        rust_result, crack_result = detect_realtime_control(
            water_roi,
            crack_roi,
            student,
            multitask,
            multitask_enabled=True,
            hybrid_enabled=True,
        )

        self.assertIs(rust_result, old_rust)
        self.assertIs(crack_result, multitask_crack)
        student.detect.assert_called_once_with(water_roi)
        multitask.detect.assert_called_once_with(water_roi)

    def test_extracts_only_realtime_control_regions(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(720, dtype=np.uint16)[:, None] % 256

        water_roi, crack_roi = extract_realtime_control_rois(frame)

        self.assertEqual(water_roi.shape, (240, 1280, 3))
        self.assertEqual(crack_roi.shape, (180, 1280, 3))
        self.assertEqual(int(water_roi[0, 0, 0]), 480 % 256)
        self.assertEqual(int(crack_roi[0, 0, 0]), 540 % 256)
        np.testing.assert_array_equal(water_roi[60:240], crack_roi)

    def test_realtime_annotation_leaves_unprocessed_area_unclassified(self) -> None:
        frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        class_map = np.zeros((240, 1280), dtype=np.uint8)
        class_map[24, 300] = 2
        rust_result = DetectionResult(
            mask=np.zeros((240, 1280), dtype=np.uint8),
            boxes=[],
            rust_ratio=0.0,
            method="deeplabv3plus-tensorrt/student/fake",
            class_map=class_map,
            class_ratios={},
        )

        annotated = annotate_realtime_control_results(frame, rust_result)

        self.assertFalse(np.array_equal(annotated[504, 300], frame[504, 300]))
        np.testing.assert_array_equal(annotated[300, 300], frame[300, 300])

    def test_realtime_annotation_offsets_crack_result_by_540_rows(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rust_result = DetectionResult(
            mask=np.zeros((240, 1280), dtype=np.uint8),
            boxes=[(5, 7, 8, 9)],
            rust_ratio=0.0,
            method="deeplabv3plus-tensorrt/student/fake",
            class_map=None,
            class_ratios={},
        )
        crack_mask = np.zeros((180, 1280), dtype=np.uint8)
        crack_mask[3, 20] = 255
        crack_result = CrackDetectionResult(
            mask=crack_mask,
            boxes=[(10, 3, 4, 5)],
            crack_pixels=1,
            inspected_pixels=crack_mask.size,
            crack_ratio=1.0 / crack_mask.size,
            detected=True,
            method="bgcrack-tensorrt/realtime/fake",
            probability_threshold=0.5,
        )

        with patch.object(run_module.cv2, "putText") as put_text:
            annotated = annotate_realtime_control_results(
                frame,
                rust_result,
                crack_result,
            )

        self.assertGreater(int(annotated[543, 20].max()), 0)
        self.assertEqual(int(annotated[3, 20].max()), 0)
        put_text.assert_not_called()

    def test_capture_preview_marks_full_native_frame(self) -> None:
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)

        display = draw_roi_guide(frame)

        self.assertEqual(display.shape, frame.shape)
        np.testing.assert_array_equal(display[1, 640], np.array([0, 255, 255]))
        np.testing.assert_array_equal(display[360, 1], np.array([0, 255, 255]))
        np.testing.assert_array_equal(display[480, 640], np.array([200, 200, 200]))

    def test_capture_preview_rejects_non_native_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "native 1280x720"):
            draw_roi_guide(np.zeros((720, 720, 3), dtype=np.uint8))

    def test_dual_camera_display_stacks_side_then_top(self) -> None:
        side = np.full((720, 1280, 3), 10, dtype=np.uint8)
        top = np.full((720, 1280, 3), 200, dtype=np.uint8)

        display = stack_dual_camera_displays(side, top)

        self.assertEqual(display.shape, (360, 1280, 3))
        np.testing.assert_array_equal(display[180, 320], np.array([10, 10, 10]))
        np.testing.assert_array_equal(display[180, 960], np.array([200, 200, 200]))

    def test_dual_camera_display_rejects_non_native_frame(self) -> None:
        native = np.zeros((720, 1280, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "native 1280x720"):
            stack_dual_camera_displays(native[:, :640], native)

    def test_realtime_preview_marks_full_width_fixed_roi_boundaries(self) -> None:
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)

        display = draw_realtime_roi_guide(frame)

        self.assertEqual(display.shape, frame.shape)
        np.testing.assert_array_equal(display[300, 100], display[300, 640])
        np.testing.assert_array_equal(display[540, 640], np.array([255, 0, 255]))
        np.testing.assert_array_equal(display[480, 640], np.array([255, 220, 0]))

    def test_dual_preview_labels_side_and_top_inference_coordinates(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        with patch.object(run_module.cv2, "putText") as put_text:
            side = draw_realtime_roi_guide(frame)
            side_labels = [call.args[1] for call in put_text.call_args_list]
            put_text.reset_mock()
            top = draw_realtime_roi_guide(
                frame,
                primary_roi_label="OBSTACLE CONTROL",
            )
            top_labels = [call.args[1] for call in put_text.call_args_list]

        self.assertEqual(
            side_labels,
            [
                "RUST ROI FULL WIDTH (Y=480:720)",
                "CRACK ROI FULL WIDTH (Y=540:720)",
            ],
        )
        self.assertEqual(
            top_labels,
            [
                "OBSTACLE CONTROL ROI FULL WIDTH (Y=480:720)",
                "CRACK ROI FULL WIDTH (Y=540:720)",
            ],
        )
        for display in (side, top):
            np.testing.assert_array_equal(
                display[540, 640],
                np.array([255, 0, 255]),
            )
            np.testing.assert_array_equal(
                display[480, 640],
                np.array([255, 220, 0]),
            )

    def test_realtime_summary_is_two_lines_inside_bottom_right(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rust_result = DetectionResult(
            mask=np.zeros((240, 1280), dtype=np.uint8),
            boxes=[],
            rust_ratio=0.25,
            method="rust/fake",
            class_map=np.zeros((240, 1280), dtype=np.uint8),
            class_ratios={"Fair": 0.2, "Poor": 0.04, "Severe": 0.01},
        )
        crack_result = CrackDetectionResult(
            mask=np.zeros((180, 1280), dtype=np.uint8),
            boxes=[],
            crack_pixels=0,
            inspected_pixels=180 * 1280,
            crack_ratio=0.0,
            detected=False,
            method="crack/fake",
            probability_threshold=0.5,
        )

        with patch.object(run_module.cv2, "putText") as put_text:
            display = draw_realtime_result_summary(
                frame,
                rust_result,
                crack_result,
            )

        self.assertEqual(display.shape, frame.shape)
        self.assertEqual(put_text.call_count, 2)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertTrue(labels[0].startswith("RUST 25.00%"))
        self.assertEqual(labels[1], "CRACK CLEAR  0.000%")
        for call in put_text.call_args_list:
            label = call.args[1]
            x, y = call.args[2]
            (width, _height), _baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                2,
            )
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + width, 1280)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, 720)

    def test_water_control_uses_exactly_y_480_through_719(self) -> None:
        class_map = np.zeros((720, 1280), dtype=np.uint8)
        class_map[479, 10] = 3
        class_map[480, 20] = 1

        roi_top, roi_bottom = water_control_roi_bounds(class_map)
        water_class_map = class_map[roi_top:roi_bottom]
        self.assertEqual((roi_top, roi_bottom), (480, 720))
        self.assertTrue(water_control_blocked_by_rust(water_class_map))

        class_map[719, 30] = 2
        self.assertTrue(water_control_blocked_by_rust(class_map[roi_top:roi_bottom]))

    def test_missing_class_map_blocks_water(self) -> None:
        self.assertTrue(water_control_blocked_by_rust(None))

    def test_wrong_shape_or_unknown_class_blocks_water(self) -> None:
        self.assertTrue(
            water_control_blocked_by_rust(np.zeros((239, 1280), dtype=np.uint8))
        )
        class_map = np.zeros((240, 1280), dtype=np.uint8)
        class_map[0, 0] = 255
        self.assertTrue(water_control_blocked_by_rust(class_map))

    def test_crack_stop_control_uses_exactly_y_540_through_719(self) -> None:
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[539, 10] = 255
        roi_top, roi_bottom = crack_stop_roi_bounds(mask)
        control_mask = mask[roi_top:roi_bottom]
        result = CrackDetectionResult(
            mask=control_mask,
            boxes=[],
            crack_pixels=0,
            inspected_pixels=control_mask.size,
            crack_ratio=0.0,
            detected=False,
            method="bgcrack-tensorrt/realtime/trt-10.3.0/cuda:0",
            probability_threshold=0.5,
        )

        self.assertEqual((roi_top, roi_bottom), (540, 720))
        self.assertFalse(cleaning_blocked_by_crack(result))

        mask[540, 20] = 255
        result.mask = mask[roi_top:roi_bottom]
        result.crack_pixels = 1
        result.crack_ratio = 1 / result.mask.size
        self.assertFalse(cleaning_blocked_by_crack(result))

        stop_pixels = int(result.mask.size * CRACK_CONTROL_STOP_RATIO) + 1
        result.mask.flat[:stop_pixels] = 255
        result.crack_pixels = stop_pixels
        result.crack_ratio = stop_pixels / result.mask.size
        self.assertTrue(cleaning_blocked_by_crack(result))

    def test_missing_or_unapproved_crack_result_blocks_cleaning(self) -> None:
        self.assertTrue(cleaning_blocked_by_crack(None))

        mask = np.zeros((720, 720), dtype=np.uint8)
        result = CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=0,
            inspected_pixels=mask.size,
            crack_ratio=0.0,
            detected=False,
            method="unapproved-crack-detector",
            probability_threshold=0.5,
        )
        self.assertTrue(cleaning_blocked_by_crack(result))

    def test_capture_crack_method_cannot_drive_realtime_cleaning(self) -> None:
        mask = np.zeros((180, 1280), dtype=np.uint8)
        result = CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=0,
            inspected_pixels=mask.size,
            crack_ratio=0.0,
            detected=False,
            method="bgcrack-tensorrt/capture/trt-10.3.0/cuda:0",
            probability_threshold=0.5,
        )

        self.assertTrue(cleaning_blocked_by_crack(result))

    def test_hrsegnet_realtime_method_can_clear_realtime_interlock(self) -> None:
        mask = np.zeros((180, 1280), dtype=np.uint8)
        result = CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=0,
            inspected_pixels=mask.size,
            crack_ratio=0.0,
            detected=False,
            method=(
                "hrsegnet-b32-tensorrt/realtime/crack/"
                "trt-10.3.0/cuda:0/logit-margin"
            ),
            probability_threshold=0.5,
        )

        self.assertFalse(cleaning_blocked_by_crack(result))
        result.mask = np.zeros((127, 1280), dtype=np.uint8)
        self.assertTrue(cleaning_blocked_by_crack(result))
        result.mask = mask
        result.method = "hrsegnet-b32-tensorrt/capture/crack/fake"
        self.assertTrue(cleaning_blocked_by_crack(result))

    def test_optimized_multitask_crack_method_can_clear_realtime_interlock(self) -> None:
        mask = np.zeros((180, 1280), dtype=np.uint8)
        result = CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=0,
            inspected_pixels=mask.size,
            crack_ratio=0.0,
            detected=False,
            method=(
                "multitask-segmentation-tensorrt/realtime/crack/"
                "optimized/trt-10.3.0/cuda:0"
            ),
            probability_threshold=0.5,
        )

        self.assertFalse(cleaning_blocked_by_crack(result))

    def test_wrong_shape_clear_crack_mask_blocks_cleaning(self) -> None:
        mask = np.zeros((127, 1280), dtype=np.uint8)
        result = CrackDetectionResult(
            mask=mask,
            boxes=[],
            crack_pixels=0,
            inspected_pixels=mask.size,
            crack_ratio=0.0,
            detected=False,
            method="bgcrack-tensorrt/realtime/trt-10.3.0/cuda:0",
            probability_threshold=0.5,
        )

        self.assertTrue(cleaning_blocked_by_crack(result))


class CaptureAnalysisTests(unittest.TestCase):
    def test_front_obstacle_capture_filters_cracks_and_exports_separate_metric(self):
        from obstacle_detector import ObstacleDetection, ObstacleDetectionResult, DETECTOR_PREFIX
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[10:20, 10:20] = 255
        crack = CrackDetectionResult(
            mask=mask, boxes=[], crack_pixels=100, inspected_pixels=mask.size,
            crack_ratio=100 / mask.size, detected=True,
            method="hrsegnet-b32-tensorrt/capture/crack/fake", probability_threshold=0.55)
        obstacle = ObstacleDetectionResult((
            ObstacleDetection((10, 10, 20, 20), 0.9, 0),
            ObstacleDetection((100, 300, 110, 310), 0.9, 0),
        ), DETECTOR_PREFIX + "fake", 0.424, True)
        engine = SimpleNamespace(detect=lambda image: obstacle,
                                 engine_path=Path("obstacle.plan"), engine_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = _FakeWorkbook(root / "report.xlsx")
            measurements = []
            workbook.append_obstacle_capture = lambda **kwargs: measurements.append(kwargs)
            task = CaptureAnalysisTask(root / "raw.jpg", INITIAL_PHASE, 1,
                                       CAPTURE_TRIGGER, datetime.now(), camera_role="top")
            with patch("node_runtime.inspection.export_top_crack_record") as export:
                result = capture_top_crack_only(frame, task, _FakeCrackDetector(crack),
                                               workbook, root / "captures",
                                               obstacle_detector=engine)
            self.assertIsNotNone(result)
            self.assertEqual(measurements[0]["measurement"]["positive_pixels"], 100)
            self.assertEqual(measurements[0]["measurement"]["inspected_pixels"], 720 * 1280)
            self.assertEqual(measurements[0]["measurement"]["detection_count"], 1)
            self.assertEqual(len(export.call_args.kwargs["obstacle_result"].detections), 1)
            self.assertEqual(len(workbook.captures), 0)

    def test_top_capture_records_xlsx_and_dashboard_camera_result(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        crack_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        crack_mask[20:30, 40:60] = 255
        crack_result = CrackDetectionResult(
            mask=crack_mask,
            boxes=[(40, 20, 20, 10)],
            crack_pixels=200,
            inspected_pixels=crack_mask.size,
            crack_ratio=200 / crack_mask.size,
            detected=True,
            method="hrsegnet-b32-tensorrt/capture/crack/fake",
            probability_threshold=0.55,
        )
        detector = _FakeCrackDetector(crack_result)
        captured_at = datetime(2026, 8, 25, 10, 0, 0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_path = root / "captures" / "raw" / "top" / "initial_01.jpg"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(b"raw-jpeg")
            task = CaptureAnalysisTask(
                raw_capture_path=raw_path,
                phase=INITIAL_PHASE,
                phase_sequence=1,
                trigger=CAPTURE_TRIGGER,
                captured_at=captured_at,
                camera_role=run_module.TOP_CAMERA_ROLE,
            )
            workbook = _FakeWorkbook(root / "report.xlsx")
            stderr = io.StringIO()
            with patch(
                "node_runtime.inspection.export_top_crack_record",
                side_effect=RuntimeError("dashboard unavailable"),
            ) as dashboard_export, redirect_stderr(stderr):
                display = capture_top_crack_only(
                    frame,
                    task,
                    detector,
                    workbook,
                    root / "captures",
                )

            self.assertIsNotNone(display)
            self.assertEqual(len(workbook.top_crack_captures), 1)
            recorded = workbook.top_crack_captures[0]
            self.assertEqual(recorded["raw_capture_path"], raw_path)
            self.assertEqual(recorded["phase"], INITIAL_PHASE)
            self.assertEqual(recorded["phase_sequence"], 1)
            self.assertEqual(recorded["crack_pixels"], 200)
            self.assertEqual(recorded["crack_inspected_pixels"], crack_mask.size)
            self.assertEqual(recorded["captured_at"], captured_at)
            dashboard_export.assert_called_once()
            self.assertIn("XLSX and captured images remain valid", stderr.getvalue())

    def test_crack_overlap_is_removed_from_copied_capture_rust_result(self) -> None:
        class_map = np.zeros((720, 1280), dtype=np.uint8)
        class_map[100:130, 100:130] = 1
        class_map[300:330, 300:330] = 3
        original_class_map = class_map.copy()
        original_mask = (class_map != 0).astype(np.uint8)
        rust_result = DetectionResult(
            mask=original_mask.copy(),
            boxes=[(100, 100, 30, 30), (300, 300, 30, 30)],
            rust_ratio=1800 / class_map.size,
            method="deeplabv3plus-tensorrt/teacher/fake",
            class_map=class_map,
            class_ratios={"Fair": 900 / class_map.size, "Severe": 900 / class_map.size},
        )
        crack_mask = np.zeros_like(class_map)
        crack_mask[100:130, 100:130] = 255
        crack_mask[300:310, 300:310] = 255
        crack_result = CrackDetectionResult(
            mask=crack_mask,
            boxes=[(100, 100, 30, 30), (300, 300, 10, 10)],
            crack_pixels=1000,
            inspected_pixels=crack_mask.size,
            crack_ratio=1000 / crack_mask.size,
            detected=True,
            method="hrsegnet-b32-tensorrt/capture/crack/fake",
            probability_threshold=0.5,
        )

        corrected = prioritize_capture_crack_over_rust(rust_result, crack_result)

        self.assertIsNot(corrected, rust_result)
        self.assertIsNot(corrected.class_map, rust_result.class_map)
        np.testing.assert_array_equal(rust_result.class_map, original_class_map)
        np.testing.assert_array_equal(rust_result.mask, original_mask)
        self.assertEqual(
            int(np.count_nonzero(corrected.mask)),
            800,
        )
        self.assertTrue(np.all(corrected.class_map[crack_mask != 0] == 0))
        self.assertEqual(corrected.boxes, [(300, 300, 30, 30)])
        self.assertEqual(corrected.class_ratios["Fair"], 0.0)
        self.assertAlmostEqual(corrected.class_ratios["Severe"], 800 / class_map.size)
        self.assertAlmostEqual(corrected.rust_ratio, 800 / class_map.size)
        self.assertEqual(corrected.method, rust_result.method)

    def test_capture_report_uses_crack_priority_rust_pixel_count(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        class_map = np.zeros(frame.shape[:2], dtype=np.uint8)
        class_map[100:130, 100:130] = 2
        detector_result = DetectionResult(
            mask=(class_map != 0).astype(np.uint8),
            boxes=[(100, 100, 30, 30)],
            rust_ratio=900 / class_map.size,
            method="deeplabv3plus-tensorrt/teacher/fake",
            class_map=class_map,
            class_ratios={"Poor": 900 / class_map.size},
        )
        crack_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        crack_mask[100:120, 100:130] = 255
        crack_result = CrackDetectionResult(
            mask=crack_mask,
            boxes=[(100, 100, 30, 20)],
            crack_pixels=600,
            inspected_pixels=crack_mask.size,
            crack_ratio=600 / crack_mask.size,
            detected=True,
            method="hrsegnet-b32-tensorrt/capture/crack/fake",
            probability_threshold=0.5,
        )
        detector = _FakeDetector(detector_result)
        crack_detector = _FakeCrackDetector(crack_result)

        with tempfile.TemporaryDirectory() as temporary_directory:
            workbook = _FakeWorkbook(Path(temporary_directory) / "report.xlsx")
            display = capture_and_analyze(
                frame,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                detector,
                workbook,
                Path(temporary_directory) / "captures",
                crack_detector=crack_detector,
            )

        self.assertIsNotNone(display)
        self.assertEqual(workbook.captures[0]["rust_pixels"], 300)
        self.assertEqual(workbook.captures[0]["inspected_pixels"], class_map.size)
        np.testing.assert_array_equal(detector_result.class_map, class_map)

    def test_saves_segmentation_overlay_and_records_its_path(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        class_map = np.zeros((720, 1280), dtype=np.uint8)
        class_map[:, 640:] = 3
        detector = _FakeDetector(
            DetectionResult(
                mask=(class_map != 0).astype(np.uint8),
                boxes=[(640, 0, 640, 720)],
                rust_ratio=0.5,
                method="deeplabv3plus-tensorrt/teacher/trt-10.3.0/cuda:0",
                class_map=class_map,
                class_ratios={"Fair": 0.0, "Poor": 0.0, "Severe": 0.5},
            )
        )
        crack_mask = np.zeros((720, 1280), dtype=np.uint8)
        crack_mask[130:135, 30:50] = 255
        crack_detector = _FakeCrackDetector(
            CrackDetectionResult(
                mask=crack_mask,
                boxes=[(30, 130, 20, 5)],
                crack_pixels=100,
                inspected_pixels=720 * 1280,
                crack_ratio=100 / (720 * 1280),
                detected=True,
                method=(
                    "hrsegnet-b32-tensorrt/capture/crack/"
                    "trt-10.3.0/cuda:0/logit-margin"
                ),
                probability_threshold=0.5,
            )
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "captures"
            workbook = _FakeWorkbook(Path(temporary_directory) / "report.xlsx")
            crack_zones: set[int] = set()

            display = capture_and_analyze(
                frame,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                detector,
                workbook,
                output_directory,
                crack_detector=crack_detector,
                crack_zones=crack_zones,
            )

            self.assertIsNotNone(display)
            self.assertEqual(len(detector.frames), 1)
            self.assertEqual(len(crack_detector.frames), 1)
            np.testing.assert_array_equal(detector.frames[0], frame)
            np.testing.assert_array_equal(crack_detector.frames[0], frame)
            self.assertEqual(display.shape, frame.shape)
            self.assertEqual(len(workbook.captures), 1)
            self.assertEqual(workbook.captures[0]["crack_status"], "ready")
            self.assertTrue(workbook.captures[0]["crack_detected"])
            self.assertEqual(workbook.captures[0]["crack_pixels"], 100)
            self.assertEqual(crack_zones, {1})
            self.assertEqual(
                workbook.captures[0]["crack_inspected_pixels"],
                720 * 1280,
            )
            capture_path = Path(workbook.captures[0]["capture_path"])
            self.assertTrue(capture_path.is_file())

            saved = cv2.imread(str(capture_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(saved)
            left_red = float(saved[400:500, 300:500, 2].mean())
            right_red = float(saved[400:500, 800:1000, 2].mean())
            self.assertGreater(right_red, left_red + 40.0)

    def test_rejects_student_result_for_capture_report(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector = _FakeDetector(
            DetectionResult(
                mask=np.zeros(frame.shape[:2], dtype=np.uint8),
                boxes=[],
                rust_ratio=0.0,
                method="deeplabv3plus-tensorrt/student/trt-10.3.0/cuda:0",
                class_map=np.zeros(frame.shape[:2], dtype=np.uint8),
                class_ratios={},
            )
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            workbook = _FakeWorkbook(Path(temporary_directory) / "report.xlsx")
            display = capture_and_analyze(
                frame,
                INITIAL_PHASE,
                1,
                CAPTURE_TRIGGER,
                detector,
                workbook,
                Path(temporary_directory) / "captures",
            )

            self.assertIsNone(display)
            self.assertEqual(workbook.captures, [])

    def test_rejects_wrong_role_crack_result_without_writing_a_row(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector = _FakeDetector(
            DetectionResult(
                mask=np.zeros(frame.shape[:2], dtype=np.uint8),
                boxes=[],
                rust_ratio=0.0,
                method="deeplabv3plus-tensorrt/teacher/trt-10.3.0/cuda:0",
                class_map=np.zeros(frame.shape[:2], dtype=np.uint8),
                class_ratios={},
            )
        )
        wrong_methods = (
            "bgcrack-tensorrt/realtime/trt-10.3.0/cuda:0",
            "hrsegnet-b32-tensorrt/realtime/crack/trt-10.3.0/cuda:0/logit-margin",
        )
        for method in wrong_methods:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temporary_directory:
                crack_detector = _FakeCrackDetector(
                    CrackDetectionResult(
                        mask=np.zeros(frame.shape[:2], dtype=np.uint8),
                        boxes=[],
                        crack_pixels=0,
                        inspected_pixels=720 * 1280,
                        crack_ratio=0.0,
                        detected=False,
                        method=method,
                        probability_threshold=0.5,
                    )
                )
                workbook = _FakeWorkbook(Path(temporary_directory) / "report.xlsx")
                display = capture_and_analyze(
                    frame,
                    INITIAL_PHASE,
                    1,
                    CAPTURE_TRIGGER,
                    detector,
                    workbook,
                    Path(temporary_directory) / "captures",
                    crack_detector=crack_detector,
                )

                self.assertIsNone(display)
                self.assertEqual(workbook.captures, [])
                self.assertFalse((Path(temporary_directory) / "captures").exists())


if __name__ == "__main__":
    unittest.main()
