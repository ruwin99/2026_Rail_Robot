from __future__ import annotations

import io
import sys
import types
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.error = RuntimeError
    for name, value in {
        "CAP_V4L2": 0,
        "CAP_PROP_FOURCC": 1,
        "CAP_PROP_FRAME_WIDTH": 2,
        "CAP_PROP_FRAME_HEIGHT": 3,
        "CAP_PROP_FPS": 4,
        "CAP_PROP_AUTOFOCUS": 5,
        "WINDOW_NORMAL": 0,
        "WND_PROP_VISIBLE": 1,
        "FONT_HERSHEY_SIMPLEX": 0,
        "LINE_AA": 0,
    }.items():
        setattr(cv2_stub, name, value)
    for name in (
        "VideoCapture",
        "VideoWriter_fourcc",
        "namedWindow",
        "getTextSize",
        "rectangle",
        "putText",
        "imshow",
        "waitKey",
        "getWindowProperty",
        "destroyAllWindows",
    ):
        setattr(cv2_stub, name, lambda *_args, **_kwargs: None)
    sys.modules["cv2"] = cv2_stub


serial_stub = types.ModuleType("serial")
serial_stub.SerialException = OSError
serial_stub.Serial = object
sys.modules.setdefault("serial", serial_stub)

JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

import run  # noqa: E402
from rust_detector import DetectionResult  # noqa: E402


def _camera_frame(value: int = 0) -> np.ndarray:
    return np.full((run.FRAME_HEIGHT, run.FRAME_WIDTH, 3), value, dtype=np.uint8)


class _FakeDetector:
    def __init__(
        self,
        role: str,
        digest: str,
        *,
        fail_on_detect: bool = False,
    ) -> None:
        self.engine_sha256 = digest
        self.method = f"deeplabv3plus-tensorrt/{role}/fake"
        self.fail_on_detect = fail_on_detect
        self.detected_frames: list[np.ndarray] = []
        self.close_calls = 0

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self.detected_frames.append(frame)
        if self.fail_on_detect:
            raise RuntimeError("warm-up failed")
        return DetectionResult(
            mask=np.zeros(frame.shape[:2], dtype=np.uint8),
            boxes=[],
            rust_ratio=0.0,
            method=self.method,
            class_map=np.zeros(frame.shape[:2], dtype=np.uint8),
            class_ratios={},
        )

    def close(self) -> None:
        self.close_calls += 1


class _FakeCamera:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)
        self.read_calls = 0
        self.release_calls = 0
        self.latest_sequence = 0

    def set(self, *_args: object) -> bool:
        return True

    def isOpened(self) -> bool:
        return True

    def read_latest(self, **_kwargs: object) -> run.CameraFrame:
        self.read_calls += 1
        try:
            frame = next(self.frames)
        except StopIteration:
            raise TimeoutError("fake camera exhausted")
        self.latest_sequence += 1
        return run.CameraFrame(frame, self.latest_sequence, run.time.monotonic())

    def release(self) -> None:
        self.release_calls += 1


class _FakeDisplay:
    def __init__(self, keys: list[int] | None = None) -> None:
        self.keys = list(keys or [])
        self.pending_keys: list[int] = []
        self.frames: list[np.ndarray] = []
        self.exit_requested = False
        self.close_calls = 0

    def check_status(self) -> bool:
        return self.exit_requested

    def submit(self, frame: np.ndarray) -> bool:
        self.frames.append(frame)
        if self.keys:
            key = self.keys.pop(0)
            if key in (ord("q"), 27):
                self.exit_requested = True
            elif key not in (0, 255):
                self.pending_keys.append(key)
        return not self.exit_requested

    def poll_key(self) -> int | None:
        return self.pending_keys.pop(0) if self.pending_keys else None

    def close(self) -> None:
        self.close_calls += 1


class _FakeSerial:
    def __init__(
        self,
        batches: list[bytes] | None = None,
        *,
        close_error: Exception | None = None,
        events: list[str] | None = None,
        write_hook=None,
    ) -> None:
        self.batches = list(batches or [])
        self.close_error = close_error
        self.events = events
        self.write_hook = write_hook
        self.writes: list[bytes] = []
        self.close_calls = 0

    @property
    def in_waiting(self) -> int:
        return len(self.batches[0]) if self.batches else 0

    def read(self, size: int) -> bytes:
        batch = self.batches.pop(0)
        if size != len(batch):
            raise AssertionError("main did not read the complete UART batch")
        return batch

    def write(self, data: bytes) -> int:
        if self.write_hook is not None:
            self.write_hook(data)
        if self.events is not None and data == run.CAPTURE_OK_COMMAND:
            self.events.append("ack")
        self.writes.append(data)
        return len(data)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _FakeAnalysisWorker:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.tasks: list[run.CaptureAnalysisTask] = []
        self.start_calls = 0
        self.wait_calls = 0
        self.shutdown_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def submit(self, task: run.CaptureAnalysisTask) -> None:
        self.tasks.append(task)

    def wait_until_idle(self, _timeout: float | None = None) -> bool:
        self.wait_calls += 1
        return True

    def failure_message(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        return None


def _args(
    *,
    no_uart: bool = False,
    crack: bool = False,
    no_crack: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        teacher_engine=Path("teacher.plan"),
        teacher_engine_sha256="a" * 64,
        student_engine=Path("student.plan"),
        student_engine_sha256="b" * 64,
        optimized_student_engine=None,
        optimized_student_engine_sha256=None,
        capture_crack_engine=None,
        capture_crack_engine_sha256=None,
        capture_hrsegnet_crack_engine=(
            Path("capture-hrsegnet-crack.plan") if crack else None
        ),
        capture_hrsegnet_crack_engine_sha256="c" * 64 if crack else None,
        capture_hrsegnet_crack_probability_threshold=0.5,
        capture_hrsegnet_crack_min_component_pixels=20,
        realtime_crack_engine=None,
        realtime_crack_engine_sha256=None,
        realtime_multitask_engine=None,
        realtime_multitask_engine_sha256=None,
        realtime_hrsegnet_crack_engine=(
            Path("realtime-hrsegnet-crack.plan") if crack else None
        ),
        realtime_hrsegnet_crack_engine_sha256="d" * 64 if crack else None,
        realtime_hrsegnet_crack_probability_threshold=0.5,
        realtime_hrsegnet_crack_min_component_pixels=20,
        no_crack=no_crack,
        capture_crack_threshold=0.5,
        realtime_crack_threshold=0.5,
        capture_crack_min_component_pixels=20,
        realtime_crack_min_component_pixels=20,
        report=Path("report.xlsx"),
        no_uart=no_uart,
        camera_index=0,
        serial_port="COM_TEST",
        baud_rate=115200,
        realtime_test=False,
        capture_test=False,
    )


def _validate_engine(
    engine_path: Path,
    expected_sha256: str | None,
    _sha_option: str,
) -> tuple[Path, str]:
    if expected_sha256 is None:
        raise AssertionError("tests require pinned fake engines")
    return engine_path, expected_sha256


class DualEngineMainTests(unittest.TestCase):
    def test_student_initialization_failure_closes_teacher_before_uart(self) -> None:
        teacher = _FakeDetector("teacher", "a" * 64)
        serial_factory = Mock()
        detector_factory = Mock(
            side_effect=[teacher, RuntimeError("student init failed")]
        )

        with (
            patch.object(run, "parse_args", return_value=_args()),
            patch.object(run, "validate_engine", side_effect=_validate_engine),
            patch.object(run, "RustDetector", detector_factory),
            patch.object(run.serial, "Serial", serial_factory),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(teacher.close_calls, 1)
        self.assertNotIn("gpu_argmax", detector_factory.call_args_list[0].kwargs)
        self.assertIs(
            detector_factory.call_args_list[1].kwargs["gpu_argmax"],
            True,
        )
        serial_factory.assert_not_called()

    def test_either_detector_warmup_failure_closes_both_and_skips_uart(self) -> None:
        for failing_role in ("teacher", "student"):
            with self.subTest(failing_role=failing_role):
                warmup_frame = _camera_frame()
                teacher = _FakeDetector(
                    "teacher",
                    "a" * 64,
                    fail_on_detect=failing_role == "teacher",
                )
                student = _FakeDetector(
                    "student",
                    "b" * 64,
                    fail_on_detect=failing_role == "student",
                )
                camera = _FakeCamera([warmup_frame])
                serial_factory = Mock()

                with (
                    patch.object(run, "parse_args", return_value=_args()),
                    patch.object(
                        run, "validate_engine", side_effect=_validate_engine
                    ),
                    patch.object(
                        run, "RustDetector", side_effect=[teacher, student]
                    ),
                    patch.object(
                        run,
                        "open_latest_frame_camera",
                        return_value=camera,
                    ),
                    patch.object(run.serial, "Serial", serial_factory),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(io.StringIO()),
                ):
                    exit_code = run.main()

                self.assertEqual(exit_code, 1)
                self.assertEqual(teacher.close_calls, 1)
                self.assertEqual(student.close_calls, 1)
                self.assertEqual(camera.release_calls, 1)
                serial_factory.assert_not_called()

    def test_crack_warmup_failure_closes_all_detectors_before_uart(self) -> None:
        warmup_frame = _camera_frame()
        teacher = _FakeDetector("teacher", "a" * 64)
        student = _FakeDetector("student", "b" * 64)
        capture_crack = _FakeDetector(
            "capture-crack",
            "c" * 64,
        )
        realtime_crack = _FakeDetector(
            "realtime-crack",
            "d" * 64,
            fail_on_detect=True,
        )
        camera = _FakeCamera([warmup_frame])
        serial_factory = Mock()

        with (
            patch.object(run, "parse_args", return_value=_args(crack=True)),
            patch.dict(run.os.environ, {"NVIDIA_TF32_OVERRIDE": "0"}),
            patch.object(run, "validate_engine", side_effect=_validate_engine),
            patch.object(run, "RustDetector", side_effect=[teacher, student]),
            patch.object(
                run,
                "HrSegNetCrackDetector",
                side_effect=[capture_crack, realtime_crack],
            ),
            patch.object(
                run,
                "open_latest_frame_camera",
                return_value=camera,
            ),
            patch.object(run.serial, "Serial", serial_factory),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(teacher.close_calls, 1)
        self.assertEqual(student.close_calls, 1)
        self.assertEqual(capture_crack.close_calls, 1)
        self.assertEqual(realtime_crack.close_calls, 1)
        np.testing.assert_array_equal(
            capture_crack.detected_frames[0],
            warmup_frame,
        )
        np.testing.assert_array_equal(
            realtime_crack.detected_frames[0],
            warmup_frame[run.REALTIME_CRACK_ROI_TOP:run.REALTIME_CRACK_ROI_BOTTOM, :],
        )
        self.assertEqual(camera.release_calls, 1)
        serial_factory.assert_not_called()

    def test_realtime_start_stays_off_until_model_results_are_ready(
        self,
    ) -> None:
        section_target = run.AUTOMATIC_RAIL_SECTION_TARGET
        self.assertEqual(section_target, 6)
        warmup_frame = _camera_frame(1)
        preview_frames = [
            _camera_frame(value)
            for value in range(10, 10 + section_target + 1)
        ]
        fresh_realtime_frame = _camera_frame(99)
        capture_frames = [
            _camera_frame(value)
            for value in range(30, 30 + section_target)
        ]
        camera = _FakeCamera(
            [warmup_frame, *preview_frames, fresh_realtime_frame]
        )
        teacher = _FakeDetector("teacher", "a" * 64)
        student = _FakeDetector("student", "b" * 64)
        events: list[str] = []
        realtime_order: list[str] = []
        uart = _FakeSerial(
            [
                b"STARTED\nCAMERA_CAPTURE\n",
                *([b"CAMERA_CAPTURE\n"] * (section_target - 1)),
                b"RETURN_START\nREALTIME_START\n",
            ],
            events=events,
            write_hook=lambda command: (
                realtime_order.append("uart-control")
                if command == run.CLEANER_ON_COMMAND
                else None
            ),
        )
        workbook = SimpleNamespace(path=Path("report.xlsx"))
        analysis_worker = _FakeAnalysisWorker()
        capture_calls: list[tuple[np.ndarray, int, object]] = []

        def fake_queue_capture_for_analysis(
            frame: np.ndarray,
            phase: str,
            phase_sequence: int,
            trigger: str,
            worker: object,
            _output_directory: Path = run.CAPTURE_DIRECTORY,
        ) -> run.CaptureAnalysisTask:
            self.assertEqual(phase, run.INITIAL_PHASE)
            self.assertEqual(trigger, run.CAPTURE_TRIGGER)
            self.assertIs(worker, analysis_worker)
            events.extend(("save", "queue"))
            task = run.CaptureAnalysisTask(
                Path(f"raw_{phase_sequence}.jpg"),
                phase,
                phase_sequence,
                trigger,
                run.datetime.now(),
            )
            worker.submit(task)
            capture_calls.append((frame, phase_sequence, worker))
            return task

        serial_factory = Mock(return_value=uart)
        display = _FakeDisplay([0] * section_target + [ord("q")])
        put_text = Mock()
        stdout = io.StringIO()
        analysis_worker_patcher = patch.object(
            run,
            "CaptureAnalysisWorker",
            return_value=analysis_worker,
        )
        analysis_worker_patcher.start()
        self.addCleanup(analysis_worker_patcher.stop)

        with (
            patch.object(run, "parse_args", return_value=_args(no_crack=True)),
            patch.object(run, "validate_engine", side_effect=_validate_engine),
            patch.object(run, "RustDetector", side_effect=[teacher, student]),
            patch.object(
                run,
                "open_latest_frame_camera",
                return_value=camera,
            ),
            patch.object(
                run,
                "open_latest_frame_display",
                return_value=display,
            ),
            patch.object(run, "InspectionWorkbook", return_value=workbook),
            patch.object(run.serial, "Serial", serial_factory),
            patch.object(
                run,
                "read_terminal_command",
                side_effect=[("start", True), (None, False)],
            ),
            patch.object(
                run,
                "read_stopped_capture_frame",
                side_effect=[
                    run.CameraFrame(frame, 100 + index, run.time.monotonic())
                    for index, frame in enumerate(capture_frames)
                ],
            ),
            patch.object(
                run,
                "queue_capture_for_analysis",
                side_effect=fake_queue_capture_for_analysis,
            ),
            patch.object(run, "annotate", side_effect=lambda frame, _result: frame),
            patch.object(
                run,
                "annotate_realtime_control_results",
                side_effect=lambda frame, *_results: (
                    realtime_order.append("ui-render") or frame
                ),
            ),
            patch.multiple(
                run,
                draw_roi_guide=Mock(
                    side_effect=lambda frame, _roi_display=None: frame
                ),
                draw_realtime_roi_guide=Mock(
                    side_effect=lambda frame, _annotated_frame=None: frame
                ),
            ),
            patch.object(run.cv2, "getTextSize", return_value=((20, 8), 2)),
            patch.object(run.cv2, "rectangle"),
            patch.object(run.cv2, "putText", put_text),
            redirect_stdout(stdout),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 0)
        # START 뒤에는 STM legacy 통신 감시용 LINK_OK가 주기적으로 섞인다.
        self.assertIn(run.LINK_OK_COMMAND, uart.writes[1:])
        self.assertEqual(
            [write for write in uart.writes if write != run.LINK_OK_COMMAND],
            [
                run.START_COMMAND,
                *([run.CAPTURE_OK_COMMAND] * section_target),
                run.CLEANER_OFF_COMMAND,
                run.PUMP_OFF_COMMAND,
                run.CLEANER_OFF_COMMAND,
                run.PUMP_OFF_COMMAND,
            ],
        )
        self.assertIn("--no-crack TEST MODE", stdout.getvalue())
        self.assertIn("REALTIME TEST MODE", stdout.getvalue())
        self.assertNotIn("uart-control", realtime_order)
        self.assertIn("ui-render", realtime_order)
        self.assertTrue(
            any("BYPASS(TEST)" in call.args[1] for call in put_text.call_args_list)
        )
        self.assertEqual(events, ["save", "queue", "ack"] * section_target)
        self.assertEqual(
            [call[1] for call in capture_calls],
            list(range(1, section_target + 1)),
        )
        self.assertTrue(all(call[2] is analysis_worker for call in capture_calls))
        self.assertEqual(len(analysis_worker.tasks), section_target)
        self.assertEqual(analysis_worker.start_calls, 1)
        self.assertEqual(analysis_worker.shutdown_calls, 1)
        self.assertEqual(len(teacher.detected_frames), 1)
        np.testing.assert_array_equal(
            teacher.detected_frames[0], warmup_frame
        )
        self.assertEqual(len(student.detected_frames), 2)
        np.testing.assert_array_equal(
            student.detected_frames[0], warmup_frame[0:240, :]
        )
        np.testing.assert_array_equal(
            student.detected_frames[1], fresh_realtime_frame[0:240, :]
        )
        self.assertFalse(
            np.array_equal(student.detected_frames[1], preview_frames[-1][0:240, :])
        )
        self.assertEqual(camera.read_calls, section_target + 3)
        self.assertEqual(uart.close_calls, 1)
        self.assertEqual(student.close_calls, 1)
        self.assertEqual(teacher.close_calls, 1)
        self.assertEqual(camera.release_calls, 1)
        self.assertEqual(display.close_calls, 1)

    def test_capture_queue_failure_sends_no_capture_ack(self) -> None:
        warmup_frame = _camera_frame()
        preview_frame = _camera_frame(1)
        capture_frame = _camera_frame(2)
        camera = _FakeCamera([warmup_frame, preview_frame])
        teacher = _FakeDetector("teacher", "a" * 64)
        student = _FakeDetector("student", "b" * 64)
        uart = _FakeSerial([b"STARTED\nCAMERA_CAPTURE\n"])
        workbook = SimpleNamespace(path=Path("report.xlsx"))
        analysis_worker = _FakeAnalysisWorker()
        display = _FakeDisplay()
        stderr = io.StringIO()

        with (
            patch.object(run, "parse_args", return_value=_args(no_crack=True)),
            patch.object(run, "validate_engine", side_effect=_validate_engine),
            patch.object(run, "RustDetector", side_effect=[teacher, student]),
            patch.object(
                run,
                "open_latest_frame_camera",
                return_value=camera,
            ),
            patch.object(
                run,
                "open_latest_frame_display",
                return_value=display,
            ),
            patch.object(run, "InspectionWorkbook", return_value=workbook),
            patch.object(
                run,
                "CaptureAnalysisWorker",
                return_value=analysis_worker,
            ),
            patch.object(run.serial, "Serial", return_value=uart),
            patch.object(run, "read_terminal_command", return_value=("start", False)),
            patch.object(
                run,
                "read_stopped_capture_frame",
                return_value=run.CameraFrame(
                    capture_frame,
                    100,
                    run.time.monotonic(),
                ),
            ),
            patch.object(
                run,
                "queue_capture_for_analysis",
                side_effect=RuntimeError("capture analysis queue is full"),
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 1)
        self.assertNotIn(run.CAPTURE_OK_COMMAND, uart.writes)
        self.assertEqual(
            [write for write in uart.writes if write != run.LINK_OK_COMMAND],
            [
                run.START_COMMAND,
                run.CLEANER_OFF_COMMAND,
                run.PUMP_OFF_COMMAND,
            ],
        )
        self.assertIn("queue is full", stderr.getvalue())
        self.assertEqual(analysis_worker.shutdown_calls, 1)
        self.assertEqual(display.close_calls, 1)

    def test_dual_first_capture_queues_both_roles_before_single_ack(self) -> None:
        side_warmup = _camera_frame(1)
        side_preview = _camera_frame(2)
        top_warmup = _camera_frame(3)
        top_preview = _camera_frame(6)
        side_capture = run.CameraFrame(
            _camera_frame(4), 101, run.time.monotonic()
        )
        top_capture = run.CameraFrame(
            _camera_frame(5), 201, run.time.monotonic()
        )
        side_camera = _FakeCamera([side_warmup, side_preview])
        top_camera = _FakeCamera([top_warmup, top_preview])
        teacher = _FakeDetector("teacher", "a" * 64)
        student = _FakeDetector("student", "b" * 64)
        capture_crack = _FakeDetector("capture-crack", "c" * 64)
        realtime_crack = _FakeDetector("realtime-crack", "d" * 64)
        obstacle = _FakeDetector("obstacle", "e" * 64)
        uart = _FakeSerial([b"STARTED\nCAMERA_CAPTURE\n"])
        workbook = SimpleNamespace(path=Path("report.xlsx"))
        analysis_worker = _FakeAnalysisWorker()
        display = _FakeDisplay()
        queued_roles: list[tuple[str, np.ndarray]] = []

        def fake_queue_capture_for_analysis(
            frame: np.ndarray,
            phase: str,
            phase_sequence: int,
            trigger: str,
            worker: object,
            _output_directory: Path = run.CAPTURE_DIRECTORY,
            *,
            camera_role: str | None = None,
            captured_at=None,
            frame_read_completed_at=None,
        ) -> run.CaptureAnalysisTask:
            self.assertEqual(phase, run.INITIAL_PHASE)
            self.assertEqual(phase_sequence, 1)
            self.assertEqual(trigger, run.CAPTURE_TRIGGER)
            self.assertIs(worker, analysis_worker)
            self.assertIsNotNone(captured_at)
            self.assertIsNotNone(frame_read_completed_at)
            queued_roles.append((camera_role, frame))
            task = run.CaptureAnalysisTask(
                Path(f"{camera_role}.jpg"),
                phase,
                phase_sequence,
                trigger,
                captured_at,
                camera_role=camera_role,
                frame_read_completed_at=frame_read_completed_at,
            )
            worker.submit(task)
            return task

        args = _args(crack=True)
        args.side_camera_device = Path("side-camera")
        args.top_camera_device = Path("top-camera")
        args.obstacle_engine = Path("obstacle.plan")
        args.obstacle_engine_sha256 = "e" * 64
        args.obstacle_confidence_threshold = 0.30
        args.dual_camera = True
        args.headless = False
        args.headless_auto = False

        with (
            patch.object(run, "parse_args", return_value=args),
            patch.dict(run.os.environ, {"NVIDIA_TF32_OVERRIDE": "0"}),
            patch.object(run, "validate_engine", side_effect=_validate_engine),
            patch.object(run, "validate_distinct_camera_devices"),
            patch.object(run, "RustDetector", side_effect=[teacher, student]),
            patch.object(
                run,
                "HrSegNetCrackDetector",
                side_effect=[capture_crack, realtime_crack],
            ),
            patch.object(run, "ObstacleDetector", return_value=obstacle),
            patch.object(
                run,
                "open_latest_frame_camera",
                side_effect=[side_camera, top_camera],
            ),
            patch.object(
                run,
                "open_latest_frame_display",
                return_value=display,
            ),
            patch.object(run, "InspectionWorkbook", return_value=workbook),
            patch.object(
                run,
                "CaptureAnalysisWorker",
                return_value=analysis_worker,
            ),
            patch.object(run.serial, "Serial", return_value=uart),
            patch.object(
                run,
                "read_terminal_command",
                side_effect=[("start", True), (None, False)],
            ),
            patch.object(
                run,
                "read_stopped_capture_pair",
                return_value=(side_capture, top_capture),
            ) as pair_reader,
            patch.object(
                run,
                "queue_capture_for_analysis",
                side_effect=fake_queue_capture_for_analysis,
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            exit_code = run.main()

        # The fake side camera is exhausted on the next preview loop, after the
        # paired capture has been accepted. This keeps the test finite.
        self.assertEqual(exit_code, 1)
        pair_reader.assert_called_once_with(side_camera, top_camera)
        self.assertEqual([role for role, _frame in queued_roles], ["side", "top"])
        np.testing.assert_array_equal(queued_roles[0][1], side_capture.frame)
        np.testing.assert_array_equal(queued_roles[1][1], top_capture.frame)
        self.assertEqual(uart.writes.count(run.CAPTURE_OK_COMMAND), 1)
        self.assertEqual(
            uart.writes[-4:],
            [
                run.FRONT_CLEANER_OFF_COMMAND,
                run.FRONT_PUMP_OFF_COMMAND,
                run.SIDE_CLEANER_OFF_COMMAND,
                run.SIDE_PUMP_OFF_COMMAND,
            ],
        )
        self.assertNotIn(run.CLEANER_OFF_COMMAND, uart.writes)
        self.assertNotIn(run.PUMP_OFF_COMMAND, uart.writes)
        self.assertEqual(len(analysis_worker.tasks), 2)
        self.assertEqual(len(display.frames), 1)
        self.assertEqual(display.frames[0].shape, (360, 1280, 3))
        np.testing.assert_array_equal(display.frames[0][100, 100], [4, 4, 4])
        np.testing.assert_array_equal(display.frames[0][100, 900], [5, 5, 5])
        self.assertEqual(side_camera.release_calls, 1)
        self.assertEqual(top_camera.release_calls, 1)
        self.assertEqual(display.close_calls, 1)

    def test_initial_and_rescan_each_ack_every_rail_section_then_done_drains(
        self,
    ) -> None:
        section_target = run.AUTOMATIC_RAIL_SECTION_TARGET
        self.assertEqual(section_target, 6)
        warmup_frame = _camera_frame()
        preview_frames = [
            _camera_frame(value)
            for value in range(section_target * 2 + 3)
        ]
        stopped_frames = [
            _camera_frame(value)
            for value in range(40, 40 + section_target * 2)
        ]
        camera = _FakeCamera([warmup_frame, *preview_frames])
        teacher = _FakeDetector("teacher", "a" * 64)
        student = _FakeDetector("student", "b" * 64)
        events: list[str] = []
        uart = _FakeSerial(
            [
                b"STARTED\nCAMERA_CAPTURE\n",
                *([b"CAMERA_CAPTURE\n"] * (section_target - 1)),
                b"RETURN_START\nREALTIME_START\n",
                b"RESCAN_RETURN_START\nRESCAN_START\nCAMERA_CAPTURE\n",
                *([b"CAMERA_CAPTURE\n"] * (section_target - 1)),
                b"RESCAN_DONE\nDONE\n",
            ],
            events=events,
        )
        workbook = SimpleNamespace(path=Path("report.xlsx"))
        analysis_worker = _FakeAnalysisWorker()
        display_factory = Mock()
        draw_roi_guide = Mock(side_effect=lambda frame, _roi_display=None: frame)
        draw_realtime_roi_guide = Mock(
            side_effect=lambda frame, _annotated_frame=None: frame
        )
        get_text_size = Mock(return_value=((20, 8), 2))
        rectangle = Mock()
        put_text = Mock()
        capture_calls: list[tuple[str, int, str]] = []

        def fake_queue_capture_for_analysis(
            _frame: np.ndarray,
            phase: str,
            phase_sequence: int,
            trigger: str,
            worker: object,
            _output_directory: Path = run.CAPTURE_DIRECTORY,
        ) -> run.CaptureAnalysisTask:
            self.assertIs(worker, analysis_worker)
            events.append(f"queue:{phase}:{phase_sequence}")
            task = run.CaptureAnalysisTask(
                Path(f"{phase}_{phase_sequence}.jpg"),
                phase,
                phase_sequence,
                trigger,
                run.datetime.now(),
            )
            worker.submit(task)
            capture_calls.append((phase, phase_sequence, trigger))
            return task

        stdout = io.StringIO()
        args = _args(no_crack=True)
        args.headless = True
        args.headless_auto = False
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(run, "parse_args", return_value=args)
            )
            stack.enter_context(
                patch.object(run, "validate_engine", side_effect=_validate_engine)
            )
            stack.enter_context(
                patch.object(run, "RustDetector", side_effect=[teacher, student])
            )
            stack.enter_context(
                patch.object(
                    run,
                    "open_latest_frame_camera",
                    return_value=camera,
                )
            )
            stack.enter_context(
                patch.object(
                    run,
                    "open_latest_frame_display",
                    display_factory,
                )
            )
            stack.enter_context(
                patch.object(run, "InspectionWorkbook", return_value=workbook)
            )
            stack.enter_context(
                patch.object(
                    run,
                    "CaptureAnalysisWorker",
                    return_value=analysis_worker,
                )
            )
            stack.enter_context(patch.object(run.serial, "Serial", return_value=uart))
            stack.enter_context(
                patch.object(
                    run,
                    "read_terminal_command",
                    return_value=("start", False),
                )
            )
            stack.enter_context(
                patch.object(
                    run,
                    "read_stopped_capture_frame",
                    side_effect=[
                        run.CameraFrame(frame, 100 + index, run.time.monotonic())
                        for index, frame in enumerate(stopped_frames)
                    ],
                )
            )
            stack.enter_context(
                patch.object(
                    run,
                    "queue_capture_for_analysis",
                    side_effect=fake_queue_capture_for_analysis,
                )
            )
            stack.enter_context(
                patch.object(run, "annotate", side_effect=lambda frame, _result: frame)
            )
            stack.enter_context(
                patch.object(
                    run,
                    "draw_roi_guide",
                    draw_roi_guide,
                )
            )
            stack.enter_context(
                patch.object(
                    run,
                    "draw_realtime_roi_guide",
                    draw_realtime_roi_guide,
                )
            )
            stack.enter_context(
                patch.object(run.cv2, "getTextSize", get_text_size)
            )
            stack.enter_context(patch.object(run.cv2, "rectangle", rectangle))
            stack.enter_context(patch.object(run.cv2, "putText", put_text))
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(io.StringIO()))
            exit_code = run.main()

        expected_captures = [
            (run.INITIAL_PHASE, sequence, run.CAPTURE_TRIGGER)
            for sequence in range(1, section_target + 1)
        ] + [
            (run.RESCAN_PHASE, sequence, run.CAPTURE_TRIGGER)
            for sequence in range(1, section_target + 1)
        ]
        expected_events: list[str] = []
        for phase, sequence, _trigger in expected_captures:
            expected_events.extend((f"queue:{phase}:{sequence}", "ack"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(capture_calls, expected_captures)
        self.assertEqual(events, expected_events)
        self.assertEqual(
            uart.writes.count(run.CAPTURE_OK_COMMAND),
            section_target * 2,
        )
        self.assertEqual(len(analysis_worker.tasks), section_target * 2)
        # Capture analysis is drained before REALTIME_START and again before
        # accepting DONE, both with the bounded worker API.
        self.assertEqual(analysis_worker.wait_calls, 2)
        self.assertEqual(analysis_worker.shutdown_calls, 1)
        display_factory.assert_not_called()
        draw_roi_guide.assert_not_called()
        draw_realtime_roi_guide.assert_not_called()
        get_text_size.assert_not_called()
        rectangle.assert_not_called()
        put_text.assert_not_called()
        self.assertIn("Mode: COMPLETE", stdout.getvalue())
        self.assertIn("Headless mission complete", stdout.getvalue())

    def test_finally_continues_after_one_resource_close_raises(self) -> None:
        warmup_frame = _camera_frame()
        preview_frame = _camera_frame(1)
        camera = _FakeCamera([warmup_frame, preview_frame])
        teacher = _FakeDetector("teacher", "a" * 64)
        student = _FakeDetector("student", "b" * 64)
        uart = _FakeSerial(close_error=RuntimeError("close failed"))
        workbook = SimpleNamespace(path=Path("report.xlsx"))
        display = _FakeDisplay([ord("q")])
        stderr = io.StringIO()

        with (
            patch.object(run, "parse_args", return_value=_args()),
            patch.object(run, "validate_engine", side_effect=_validate_engine),
            patch.object(run, "RustDetector", side_effect=[teacher, student]),
            patch.object(
                run,
                "open_latest_frame_camera",
                return_value=camera,
            ),
            patch.object(
                run,
                "open_latest_frame_display",
                return_value=display,
            ),
            patch.object(run, "InspectionWorkbook", return_value=workbook),
            patch.object(run.serial, "Serial", return_value=uart),
            patch.object(run, "read_terminal_command", return_value=(None, False)),
            patch.object(
                run,
                "draw_roi_guide",
                side_effect=lambda frame, _roi_display=None: frame,
            ),
            patch.object(run.cv2, "getTextSize", return_value=((20, 8), 2)),
            patch.object(run.cv2, "rectangle"),
            patch.object(run.cv2, "putText"),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            uart.writes,
            [run.CLEANER_OFF_COMMAND, run.PUMP_OFF_COMMAND],
        )
        self.assertEqual(uart.close_calls, 1)
        self.assertEqual(student.close_calls, 1)
        self.assertEqual(teacher.close_calls, 1)
        self.assertEqual(camera.release_calls, 1)
        self.assertEqual(display.close_calls, 1)
        self.assertIn("Could not close UART: close failed", stderr.getvalue())

    def test_ui_failure_returns_error_and_fails_actuators_closed(self) -> None:
        warmup_frame = _camera_frame()
        camera = _FakeCamera([warmup_frame])
        teacher = _FakeDetector("teacher", "a" * 64)
        student = _FakeDetector("student", "b" * 64)
        uart = _FakeSerial()
        workbook = SimpleNamespace(path=Path("report.xlsx"))
        analysis_worker = _FakeAnalysisWorker()
        display = Mock()
        display.check_status.side_effect = RuntimeError("HighGUI display failed")
        stderr = io.StringIO()

        with (
            patch.object(run, "parse_args", return_value=_args(no_crack=True)),
            patch.object(run, "validate_engine", side_effect=_validate_engine),
            patch.object(run, "RustDetector", side_effect=[teacher, student]),
            patch.object(run, "open_latest_frame_camera", return_value=camera),
            patch.object(run, "open_latest_frame_display", return_value=display),
            patch.object(run, "InspectionWorkbook", return_value=workbook),
            patch.object(
                run,
                "CaptureAnalysisWorker",
                return_value=analysis_worker,
            ),
            patch.object(run.serial, "Serial", return_value=uart),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            exit_code = run.main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(
            uart.writes,
            [run.CLEANER_OFF_COMMAND, run.PUMP_OFF_COMMAND],
        )
        display.close.assert_called_once_with()
        self.assertIn("HighGUI display failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
