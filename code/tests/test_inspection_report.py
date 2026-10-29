from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from inspection_report import (  # noqa: E402
    INITIAL_PHASE,
    InspectionWorkbook,
    MANUAL_PHASE,
    PIPELINE_VERSION,
    RAIL_SECTION_TARGET,
    RESCAN_PHASE,
    TOP_CRACK_HEADERS,
    TOP_CRACK_SHEET_NAME,
)


class InspectionWorkbookTests(unittest.TestCase):
    def test_obstacle_report_is_separate_and_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.xlsx"
            workbook = InspectionWorkbook(path, **self._workbook_kwargs())
            captured_at = datetime(2026, 9, 5, 10)
            workbook.append_capture(
                capture_path="side.jpg", phase=INITIAL_PHASE, trigger="CAMERA_CAPTURE",
                detector=self._workbook_kwargs()["capture_detector"],
                rust_pixels=20, inspected_pixels=100, captured_at=captured_at)
            kwargs = dict(phase=INITIAL_PHASE, phase_sequence=1, captured_at=captured_at,
                          capture_path="top.jpg", measurement={
                              "positive_pixels": 0, "inspected_pixels": 100,
                              "detection_count": 0, "detector_method": "approved-obstacle"})
            workbook.append_obstacle_capture(**kwargs)
            from openpyxl import load_workbook
            sheet = load_workbook(path)["Front Obstacle"]
            self.assertEqual(sheet.cell(2, 5).value, "bbox_area_ratio")
            self.assertEqual(sheet.cell(2, 8).value, 0)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                workbook.append_obstacle_capture(**kwargs)

    @staticmethod
    def _workbook_kwargs() -> dict[str, str]:
        return {
            "capture_model_filename": (
                "corrosion-capture-r101-os8-w1280-h720-fp32.plan"
            ),
            "capture_model_sha256": "a" * 64,
            "realtime_model_filename": (
                "realtime-rust-mnv2-os8-w1280-h240-fp16.plan"
            ),
            "realtime_model_sha256": "b" * 64,
            "capture_detector": (
                "deeplabv3plus-tensorrt/teacher/trt-10.3.0/cuda:0"
            ),
            "realtime_detector": (
                "deeplabv3plus-tensorrt/student/trt-10.3.0/cuda:0"
            ),
        }

    def test_records_tensorrt_engine_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.xlsx"
            workbook = InspectionWorkbook(
                report_path,
                **self._workbook_kwargs(),
            )

            ratio = workbook.append_capture(
                capture_path="capture.jpg",
                phase=MANUAL_PHASE,
                trigger="manual",
                detector="deeplabv3plus-tensorrt/teacher/trt-10.3.0/cuda:0",
                rust_pixels=25,
                inspected_pixels=100,
                captured_at=datetime(2026, 8, 4, 12, 0, 0),
            )

            self.assertEqual(ratio, 0.25)
            self.assertTrue(report_path.is_file())
            settings = {
                workbook.settings_sheet.cell(row, 1).value:
                workbook.settings_sheet.cell(row, 2).value
                for row in range(2, workbook.settings_sheet.max_row + 1)
            }
            self.assertEqual(settings["파이프라인 버전"], PIPELINE_VERSION)
            self.assertEqual(settings["Rail Section 목표"], RAIL_SECTION_TARGET)
            self.assertEqual(
                settings["캡처 모델 파일"],
                "corrosion-capture-r101-os8-w1280-h720-fp32.plan",
            )
            self.assertEqual(
                settings["실시간 모델 파일"],
                "realtime-rust-mnv2-os8-w1280-h240-fp16.plan",
            )
            self.assertIn("[1,3,720,1280]", settings["캡처 모델 I/O 계약"])
            self.assertIn("y=480:720", settings["실시간 입력 전처리"])
            self.assertEqual(settings["클래스 순서"], "Good,Fair,Poor,Severe")
            self.assertEqual(settings["균열 분석 상태"], "disabled")
            self.assertEqual(workbook.sheet.cell(7, 11).value, "수동")
            self.assertEqual(workbook.sheet.cell(7, 12).value, "disabled")
            self.assertIsNone(workbook.sheet.cell(7, 17).value)

    def test_records_crack_candidate_with_one_based_zone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "crack_model_filename": (
                        "hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan"
                    ),
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": (
                        "hrsegnet-b32-tensorrt/capture/crack/trt-10.3.0/cuda:0/logit-margin"
                    ),
                    "realtime_crack_model_filename": (
                        "realtime-crack-bgcrack-w1280-h128-fp32.plan"
                    ),
                    "realtime_crack_model_sha256": "d" * 64,
                    "realtime_crack_detector": (
                        "bgcrack-tensorrt/realtime/trt-10.3.0/cuda:0"
                    ),
                    "realtime_crack_probability_threshold": 0.55,
                    "crack_probability_threshold": 0.45,
                    "capture_crack_min_component_pixels": 12,
                    "realtime_crack_min_component_pixels": 34,
                }
            )
            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx",
                **kwargs,
            )

            workbook.append_capture(
                capture_path="initial_01.jpg",
                phase=INITIAL_PHASE,
                trigger="CAMERA_CAPTURE",
                detector="deeplabv3plus-tensorrt/teacher/trt-10.3.0/cuda:0",
                rust_pixels=10,
                inspected_pixels=100,
                crack_status="ready",
                crack_detector=(
                    "hrsegnet-b32-tensorrt/capture/crack/trt-10.3.0/cuda:0/logit-margin"
                ),
                crack_detected=True,
                crack_pixels=4,
                crack_inspected_pixels=100,
            )

            self.assertEqual(workbook.sheet.cell(7, 3).value, 1)
            self.assertEqual(workbook.sheet.cell(7, 11).value, "1번 레일 구간")
            self.assertEqual(workbook.sheet.cell(7, 12).value, "ready")
            self.assertTrue(workbook.sheet.cell(7, 14).value)
            self.assertEqual(workbook.sheet.cell(7, 15).value, 4)
            settings = {
                workbook.settings_sheet.cell(row, 1).value:
                workbook.settings_sheet.cell(row, 2).value
                for row in range(2, workbook.settings_sheet.max_row + 1)
            }
            self.assertEqual(settings["캡처 균열 최소 연결성분 픽셀"], 12)
            self.assertEqual(settings["실시간 균열 최소 연결성분 픽셀"], 34)
            self.assertEqual(
                workbook.sheet.cell(7, 17).value,
                "=IF(P7=0,0,O7/P7)",
            )

    def test_records_top_crack_without_changing_side_rust_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.xlsx"
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "crack_model_filename": "capture-hrsegnet.plan",
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": (
                        "hrsegnet-b32-tensorrt/capture/crack/fake"
                    ),
                    "realtime_crack_model_filename": "realtime-hrsegnet.plan",
                    "realtime_crack_model_sha256": "d" * 64,
                    "realtime_crack_detector": (
                        "hrsegnet-b32-tensorrt/realtime/crack/fake"
                    ),
                }
            )
            captured_at = datetime(2026, 8, 25, 9, 30, 0)
            workbook = InspectionWorkbook(report_path, **kwargs)
            workbook.append_capture(
                capture_path="side_initial_01.jpg",
                phase=INITIAL_PHASE,
                trigger="CAMERA_CAPTURE",
                detector=kwargs["capture_detector"],
                rust_pixels=25,
                inspected_pixels=100,
                captured_at=captured_at,
                crack_status="ready",
                crack_detector=kwargs["crack_detector"],
                crack_detected=False,
                crack_pixels=0,
                crack_inspected_pixels=100,
            )

            top_ratio = workbook.append_top_crack_capture(
                raw_capture_path="raw/top/initial_01.jpg",
                capture_path="top_crack/initial_01.jpg",
                phase=INITIAL_PHASE,
                phase_sequence=1,
                trigger="CAMERA_CAPTURE",
                crack_detector=kwargs["crack_detector"],
                crack_detected=True,
                crack_pixels=5,
                crack_inspected_pixels=100,
                captured_at=captured_at,
            )

            self.assertEqual(top_ratio, 0.05)
            self.assertEqual(workbook.sheet.max_row, 7)
            self.assertEqual(workbook._phase_counts()[INITIAL_PHASE], 1)
            self.assertEqual(workbook.top_crack_sheet.title, TOP_CRACK_SHEET_NAME)
            self.assertNotIn("부식", " ".join(TOP_CRACK_HEADERS))
            self.assertEqual(workbook.top_crack_sheet.cell(2, 3).value, 1)
            self.assertEqual(workbook.top_crack_sheet.cell(2, 8).value, "top")
            self.assertTrue(workbook.top_crack_sheet.cell(2, 11).value)
            self.assertEqual(workbook.top_crack_sheet.cell(2, 12).value, 5)
            self.assertEqual(
                workbook.top_crack_sheet.cell(2, 14).value,
                "=IF(M2=0,0,L2/M2)",
            )
            with self.assertRaisesRegex(
                ValueError,
                "Expected top initial rail section 2",
            ):
                workbook.append_top_crack_capture(
                    raw_capture_path="duplicate.jpg",
                    capture_path="duplicate-analyzed.jpg",
                    phase=INITIAL_PHASE,
                    phase_sequence=1,
                    trigger="CAMERA_CAPTURE",
                    crack_detector=kwargs["crack_detector"],
                    crack_detected=False,
                    crack_pixels=0,
                    crack_inspected_pixels=100,
                    captured_at=captured_at,
                )

            workbook._validate_top_crack_sheet()
            with self.assertRaisesRegex(ValueError, "automatic inspection rows"):
                InspectionWorkbook(report_path, **kwargs)

    def test_top_crack_requires_matching_side_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "crack_model_filename": "capture-hrsegnet.plan",
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": (
                        "hrsegnet-b32-tensorrt/capture/crack/fake"
                    ),
                    "realtime_crack_model_filename": "realtime-hrsegnet.plan",
                    "realtime_crack_model_sha256": "d" * 64,
                    "realtime_crack_detector": (
                        "hrsegnet-b32-tensorrt/realtime/crack/fake"
                    ),
                }
            )
            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx",
                **kwargs,
            )
            with self.assertRaisesRegex(ValueError, "matching SIDE"):
                workbook.append_top_crack_capture(
                    raw_capture_path="raw/top/initial_01.jpg",
                    capture_path="top_crack/initial_01.jpg",
                    phase=INITIAL_PHASE,
                    phase_sequence=1,
                    trigger="CAMERA_CAPTURE",
                    crack_detector=kwargs["crack_detector"],
                    crack_detected=False,
                    crack_pixels=0,
                    crack_inspected_pixels=100,
                )

    def test_records_hrsegnet_realtime_crack_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "crack_model_filename": "capture-hrsegnet.plan",
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": "hrsegnet-b32-tensorrt/capture/crack/fake",
                    "realtime_crack_model_filename": "hrsegnet-b32.plan",
                    "realtime_crack_model_sha256": "7" * 64,
                    "realtime_crack_detector": (
                        "hrsegnet-b32-tensorrt/realtime/crack/"
                        "trt-10.3.0/cuda:0/logit-margin"
                    ),
                    "realtime_crack_probability_threshold": 0.5,
                    "capture_crack_min_component_pixels": 12,
                    "realtime_crack_min_component_pixels": 20,
                }
            )
            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx",
                **kwargs,
            )

            self.assertTrue(workbook.realtime_crack_hrsegnet_enabled)
            self.assertEqual(
                workbook.realtime_crack_model_filename,
                "hrsegnet-b32.plan",
            )
            self.assertEqual(workbook.realtime_crack_model_sha256, "7" * 64)
            self.assertIn("y=540:720", workbook.realtime_crack_input_preprocessing)
            self.assertIn("INTER_AREA", workbook.realtime_crack_input_preprocessing)
            self.assertIn("INTER_NEAREST", workbook.realtime_crack_model_io_contract)
            self.assertIn("/127.5-1.0", workbook.realtime_crack_input_preprocessing)
            self.assertIn("[1,2,128,1280]", workbook.realtime_crack_model_io_contract)
            self.assertIn("logit margin", workbook.realtime_crack_model_io_contract)
            self.assertEqual(workbook.realtime_crack_probability_threshold, 0.5)
            self.assertEqual(workbook.realtime_crack_min_component_pixels, 20)

    def test_records_and_reopens_capture_hrsegnet_full_frame_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.xlsx"
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "crack_model_filename": "capture-hrsegnet.plan",
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": (
                        "hrsegnet-b32-tensorrt/capture/crack/"
                        "trt-10.3.0/cuda:0/logit-margin"
                    ),
                    "realtime_crack_model_filename": "realtime-hrsegnet.plan",
                    "realtime_crack_model_sha256": "d" * 64,
                    "realtime_crack_detector": (
                        "hrsegnet-b32-tensorrt/realtime/crack/"
                        "trt-10.3.0/cuda:0/logit-margin"
                    ),
                }
            )

            workbook = InspectionWorkbook(report_path, **kwargs)
            self.assertTrue(workbook.capture_crack_hrsegnet_enabled)
            self.assertEqual(workbook.crack_probability_threshold, 0.55)
            self.assertEqual(workbook.realtime_crack_probability_threshold, 0.55)
            self.assertIn("original HrSegNet-B32", workbook.capture_crack_model_architecture)
            self.assertIn("1280x720", workbook.capture_crack_input_preprocessing)
            self.assertIn("no resize/padding", workbook.capture_crack_input_preprocessing)
            self.assertIn("[1,2,720,1280]", workbook.capture_crack_model_io_contract)

            reopened = InspectionWorkbook(report_path, **kwargs)
            self.assertTrue(reopened.capture_crack_hrsegnet_enabled)

    def test_records_single_plan_for_both_multitask_realtime_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "realtime_model_filename": "optimized-multitask.plan",
                    "realtime_model_sha256": "e" * 64,
                    "realtime_detector": (
                        "multitask-segmentation-tensorrt/realtime/rust/"
                        "optimized/trt-10.3.0/cuda:0"
                    ),
                    "crack_model_filename": "capture-crack.plan",
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": "hrsegnet-b32-tensorrt/capture/crack/fake",
                    "realtime_crack_model_filename": "optimized-multitask.plan",
                    "realtime_crack_model_sha256": "e" * 64,
                    "realtime_crack_detector": (
                        "multitask-segmentation-tensorrt/realtime/crack/"
                        "optimized/trt-10.3.0/cuda:0"
                    ),
                }
            )
            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx", **kwargs
            )
            settings = {
                workbook.settings_sheet.cell(row, 1).value:
                workbook.settings_sheet.cell(row, 2).value
                for row in range(2, workbook.settings_sheet.max_row + 1)
            }

            self.assertEqual(settings["실시간 모델 파일"], "optimized-multitask.plan")
            self.assertEqual(
                settings["실시간 균열 모델 파일"], "optimized-multitask.plan"
            )
            self.assertIn("rust_class_map", settings["실시간 모델 I/O 계약"])
            self.assertIn(
                "multitask_outputs_finite", settings["실시간 모델 I/O 계약"]
            )
            self.assertIn(
                "crack_candidate_map", settings["실시간 균열 모델 I/O 계약"]
            )

    def test_records_separate_student_rust_and_multitask_crack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "crack_model_filename": "capture-crack.plan",
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": "hrsegnet-b32-tensorrt/capture/crack/fake",
                    "realtime_crack_model_filename": "optimized-multitask.plan",
                    "realtime_crack_model_sha256": "e" * 64,
                    "realtime_crack_detector": (
                        "multitask-segmentation-tensorrt/realtime/crack/fake"
                    ),
                }
            )

            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx", **kwargs
            )

            self.assertFalse(workbook.realtime_multitask_enabled)
            self.assertTrue(workbook.realtime_crack_multitask_enabled)
            self.assertEqual(
                workbook.realtime_model_filename,
                "realtime-rust-mnv2-os8-w1280-h240-fp16.plan",
            )
            self.assertEqual(
                workbook.realtime_crack_model_filename,
                "optimized-multitask.plan",
            )
            self.assertIn(
                "logits float32 [1,4,240,1280]",
                workbook.realtime_model_io_contract,
            )
            self.assertIn(
                "crack_candidate_map",
                workbook.realtime_crack_model_io_contract,
            )

    def test_records_optimized_rust_only_contract_in_hybrid_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kwargs = self._workbook_kwargs()
            kwargs.update(
                {
                    "realtime_model_filename": "optimized-rust.plan",
                    "realtime_model_sha256": "f" * 64,
                    "realtime_detector": (
                        "deeplabv3plus-tensorrt/student/optimized-compact-v2/"
                        "trt-10.3.0/cuda:0"
                    ),
                    "crack_model_filename": "capture-crack.plan",
                    "crack_model_sha256": "c" * 64,
                    "crack_detector": "hrsegnet-b32-tensorrt/capture/crack/fake",
                    "realtime_crack_model_filename": "optimized-multitask.plan",
                    "realtime_crack_model_sha256": "e" * 64,
                    "realtime_crack_detector": (
                        "multitask-segmentation-tensorrt/realtime/crack/fake"
                    ),
                }
            )

            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx", **kwargs
            )

            self.assertTrue(workbook.realtime_optimized_enabled)
            self.assertFalse(workbook.realtime_multitask_enabled)
            self.assertIn("rust_class_map", workbook.realtime_model_io_contract)
            self.assertIn("rust_logits_not_nan", workbook.realtime_model_io_contract)
            self.assertIn("rust_logits_finite_abs", workbook.realtime_model_io_contract)
            self.assertIn("raw 4-channel D2H forbidden", workbook.realtime_model_io_contract)
            self.assertEqual(workbook.realtime_model_sha256, "f" * 64)

    def test_records_external_cuda_argmax_contract_for_raw_student_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            kwargs = self._workbook_kwargs()
            kwargs["realtime_detector"] = (
                "deeplabv3plus-tensorrt/student/trt-10.3.0/cuda:0/gpu-argmax"
            )

            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx", **kwargs
            )

            self.assertTrue(workbook.realtime_cuda_argmax_enabled)
            self.assertFalse(workbook.realtime_optimized_enabled)
            self.assertIn("same-stream CUDA", workbook.realtime_model_io_contract)
            self.assertIn("first-tie argmax", workbook.realtime_model_io_contract)
            self.assertIn("invalid uint32", workbook.realtime_model_io_contract)
            self.assertIn("raw logits D2H forbidden", workbook.realtime_model_io_contract)

    def test_rejects_hybrid_roles_with_shared_plan_or_digest(self) -> None:
        base = self._workbook_kwargs()
        base.update(
            {
                "crack_model_filename": "capture-crack.plan",
                "crack_model_sha256": "c" * 64,
                "crack_detector": "hrsegnet-b32-tensorrt/capture/crack/fake",
                "realtime_crack_model_filename": "optimized-multitask.plan",
                "realtime_crack_model_sha256": "e" * 64,
                "realtime_crack_detector": (
                    "multitask-segmentation-tensorrt/realtime/crack/fake"
                ),
            }
        )
        invalid = (
            ("realtime_crack_model_filename", base["realtime_model_filename"]),
            ("realtime_crack_model_sha256", base["realtime_model_sha256"]),
        )
        for field, value in invalid:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                kwargs = dict(base)
                kwargs[field] = value
                with self.assertRaisesRegex(ValueError, "Hybrid realtime"):
                    InspectionWorkbook(Path(directory) / "report.xlsx", **kwargs)

    def test_rejects_mixed_or_mismatched_multitask_provenance(self) -> None:
        base = self._workbook_kwargs()
        base.update(
            {
                "realtime_model_filename": "optimized-multitask.plan",
                "realtime_model_sha256": "e" * 64,
                "realtime_detector": (
                    "multitask-segmentation-tensorrt/realtime/rust/fake"
                ),
                "crack_model_filename": "capture-crack.plan",
                "crack_model_sha256": "c" * 64,
                "crack_detector": "hrsegnet-b32-tensorrt/capture/crack/fake",
                "realtime_crack_model_filename": "optimized-multitask.plan",
                "realtime_crack_model_sha256": "e" * 64,
                "realtime_crack_detector": (
                    "multitask-segmentation-tensorrt/realtime/crack/fake"
                ),
            }
        )
        invalid = (
            ("realtime_crack_detector", "bgcrack-tensorrt/realtime/fake"),
            ("realtime_crack_model_sha256", "f" * 64),
        )
        for field, value in invalid:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                kwargs = dict(base)
                kwargs[field] = value
                with self.assertRaisesRegex(ValueError, "multitask|both use"):
                    InspectionWorkbook(Path(directory) / "report.xlsx", **kwargs)

    def test_rejects_existing_report_with_different_student_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = Path(temporary_directory) / "report.xlsx"
            InspectionWorkbook(report_path, **self._workbook_kwargs())
            changed = self._workbook_kwargs()
            changed["realtime_model_sha256"] = "c" * 64

            with self.assertRaisesRegex(ValueError, "settings differ"):
                InspectionWorkbook(report_path, **changed)

    def test_rejects_swapped_capture_and_realtime_crack_methods(self) -> None:
        crack_kwargs = {
            "crack_model_filename": "capture-crack.plan",
            "crack_model_sha256": "c" * 64,
            "crack_detector": "hrsegnet-b32-tensorrt/capture/crack/fake",
            "realtime_crack_model_filename": "realtime-crack.plan",
            "realtime_crack_model_sha256": "d" * 64,
            "realtime_crack_detector": "bgcrack-tensorrt/realtime/fake",
        }
        invalid_methods = (
            (
                "crack_detector",
                "bgcrack-tensorrt/realtime/fake",
                "capture HrSegNet",
            ),
            (
                "realtime_crack_detector",
                "hrsegnet-b32-tensorrt/capture/crack/fake",
                "realtime BGCrack",
            ),
        )

        for field, method, message in invalid_methods:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary_directory:
                kwargs = self._workbook_kwargs()
                kwargs.update(crack_kwargs)
                kwargs[field] = method
                with self.assertRaisesRegex(ValueError, message):
                    InspectionWorkbook(
                        Path(temporary_directory) / "report.xlsx",
                        **kwargs,
                    )

    def test_requires_every_initial_and_rescan_rail_section(self) -> None:
        self.assertEqual(RAIL_SECTION_TARGET, 6)
        with tempfile.TemporaryDirectory() as temporary_directory:
            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx",
                **self._workbook_kwargs(),
            )
            capture = {
                "detector": "deeplabv3plus-tensorrt/teacher/trt-10.3.0/cuda:0",
                "rust_pixels": 1,
                "inspected_pixels": 10,
            }

            for sequence in range(1, RAIL_SECTION_TARGET):
                workbook.append_capture(
                    capture_path=f"initial_{sequence:02d}.jpg",
                    phase=INITIAL_PHASE,
                    trigger="CAMERA_CAPTURE",
                    **capture,
                )
            with self.assertRaisesRegex(
                ValueError,
                "exactly 6 initial rail sections",
            ):
                workbook.append_capture(
                    capture_path="rescan_01.jpg",
                    phase=RESCAN_PHASE,
                    trigger="CAMERA_CAPTURE",
                    **capture,
                )
            workbook.append_capture(
                capture_path="initial_06.jpg",
                phase=INITIAL_PHASE,
                trigger="CAMERA_CAPTURE",
                **capture,
            )
            with self.assertRaisesRegex(ValueError, "6 initial rail sections"):
                workbook.append_capture(
                    capture_path="initial_07.jpg",
                    phase=INITIAL_PHASE,
                    trigger="CAMERA_CAPTURE",
                    **capture,
                )

            for sequence in range(1, RAIL_SECTION_TARGET + 1):
                workbook.append_capture(
                    capture_path=f"rescan_{sequence:02d}.jpg",
                    phase=RESCAN_PHASE,
                    trigger="CAMERA_CAPTURE",
                    **capture,
                )
            with self.assertRaisesRegex(ValueError, "6 rescan rail sections"):
                workbook.append_capture(
                    capture_path="rescan_07.jpg",
                    phase=RESCAN_PHASE,
                    trigger="CAMERA_CAPTURE",
                    **capture,
                )

            self.assertEqual(
                workbook.sheet["H2"].value,
                '=IF(OR(B3<>6,B4<>6),"",E3-E4)',
            )
            self.assertEqual(workbook.sheet.cell(2, 2).value, "Rail Section 수")
            self.assertEqual(workbook.sheet.cell(6, 3).value, "Rail Section 번호")
            self.assertEqual(workbook.sheet.cell(12, 11).value, "6번 레일 구간")
            self.assertEqual(workbook.sheet.cell(18, 11).value, "6번 레일 구간")

    def test_rejects_non_tensorrt_detector_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workbook = InspectionWorkbook(
                Path(temporary_directory) / "report.xlsx",
                capture_model_filename="teacher.plan",
                capture_model_sha256="a" * 64,
                realtime_model_filename="student.plan",
                realtime_model_sha256="b" * 64,
                capture_detector="deeplabv3plus-tensorrt/teacher/cuda:0",
                realtime_detector="deeplabv3plus-tensorrt/student/cuda:0",
            )

            with self.assertRaisesRegex(ValueError, "exact capture teacher"):
                workbook.append_capture(
                    capture_path="capture.jpg",
                    phase=MANUAL_PHASE,
                    trigger="manual",
                    detector="deeplabv3plus-tensorrt/student/cuda:0",
                    rust_pixels=1,
                    inspected_pixels=100,
                )


if __name__ == "__main__":
    unittest.main()
