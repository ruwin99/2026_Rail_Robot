from __future__ import annotations

import json
import io
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def _pillow_imencode(extension, image, _parameters=()):
    output = io.BytesIO()
    image_format = "PNG" if extension.lower() == ".png" else "JPEG"
    Image.fromarray(image[:, :, ::-1]).save(output, format=image_format)
    return True, np.frombuffer(output.getvalue(), dtype=np.uint8)


try:
    import cv2  # type: ignore
except ImportError:
    cv2 = types.ModuleType("cv2")
if not hasattr(cv2, "imencode"):
    cv2.error = RuntimeError
    cv2.IMWRITE_PNG_COMPRESSION = 16
    cv2.imencode = _pillow_imencode
    sys.modules["cv2"] = cv2


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
if str(JETSON_CODE) not in sys.path:
    sys.path.insert(0, str(JETSON_CODE))

from dashboard_export import (  # noqa: E402
    export_capture_record,
    export_top_crack_record,
    finalize_dashboard_run,
    render_crack_mask_preview,
    render_rust_mask_preview,
    obstacle_bbox_measurement,
)
from inspection_report import RAIL_SECTION_TARGET  # noqa: E402


@dataclass
class _RustResult:
    class_map: np.ndarray
    method: str = "deeplabv3plus-tensorrt/teacher/fake"


@dataclass
class _CrackResult:
    mask: np.ndarray
    method: str = "hrsegnet-b32-tensorrt/capture/crack/fake"


class _Workbook:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture_model_filename = "capture-rust.plan"
        self.capture_model_sha256 = "1" * 64
        self.capture_detector = "deeplabv3plus-tensorrt/teacher/fake"
        self.crack_model_filename = "capture-crack.plan"
        self.crack_model_sha256 = "2" * 64
        self.crack_detector = "hrsegnet-b32-tensorrt/capture/crack/fake"
        self.crack_probability_threshold = 0.55
        self.capture_crack_min_component_pixels = 20


def _write_jpeg(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame[:, :, ::-1]).save(path, format="JPEG")


class DashboardPreviewTests(unittest.TestCase):
    def test_obstacle_union_clips_and_does_not_double_count(self):
        from obstacle_detector import ObstacleDetection, ObstacleDetectionResult, DETECTOR_PREFIX
        result = ObstacleDetectionResult((
            ObstacleDetection((-2, -2, 3, 3), 0.9, 0),
            ObstacleDetection((1, 1, 8, 8), 0.8, 0),
        ), DETECTOR_PREFIX + "test", 0.424, True)
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        measurement, preview = obstacle_bbox_measurement(frame, result)
        self.assertEqual(measurement["positive_pixels"], 17)
        self.assertEqual(measurement["inspected_pixels"], 20)
        self.assertEqual(measurement["detection_count"], 2)
        self.assertEqual(measurement["ratio_fraction"], 0.85)
        self.assertEqual(preview.shape, frame.shape)

    def test_obstacle_export_preserves_clean_after_image_and_missing_history(self):
        from obstacle_detector import ObstacleDetectionResult, DETECTOR_PREFIX
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workbook = _Workbook(root / "report.xlsx")
            frame = np.zeros((4, 5, 3), dtype=np.uint8)
            result = ObstacleDetectionResult((), DETECTOR_PREFIX + "test", 0.424, False)
            for sequence, obstacle in ((1, None), (2, result)):
                raw = root / "captures" / "raw" / f"{sequence}.jpg"
                analyzed = root / "captures" / f"{sequence}.jpg"
                _write_jpeg(raw, frame)
                _write_jpeg(analyzed, frame)
                path = export_top_crack_record(
                    frame=frame, crack_result=_CrackResult(np.zeros((4, 5), dtype=np.uint8)),
                    workbook=workbook, capture_path=analyzed, raw_capture_path=raw,
                    phase="initial", phase_sequence=sequence, trigger="CAMERA_CAPTURE",
                    captured_at=datetime(2026, 9, 5), output_directory=root / "captures",
                    obstacle_result=obstacle,
                )
            document = json.loads(path.read_text(encoding="utf-8"))
            analyses = [a for a in document["analyses"] if a["defect_type"] == "obstacle"]
            self.assertEqual(len(analyses), 1)
            self.assertEqual(analyses[0]["ratio_fraction"], 0.0)
            self.assertEqual(analyses[0]["metric"], "bbox_area_ratio")
            artifacts = [a for a in document["artifacts"] if a["artifact_type"] == "obstacle_preview"]
            self.assertEqual(len(artifacts), 1)
            self.assertTrue((root / artifacts[0]["object_key"]).is_file())

    def test_previews_only_change_selected_mask_pixels(self) -> None:
        frame = np.full((4, 5, 3), 80, dtype=np.uint8)
        class_map = np.zeros((4, 5), dtype=np.uint8)
        class_map[1, 2] = 1
        crack_mask = np.zeros((4, 5), dtype=np.uint8)
        crack_mask[2, 3] = 255

        rust_preview = render_rust_mask_preview(frame, class_map)
        crack_preview = render_crack_mask_preview(frame, crack_mask)

        np.testing.assert_array_equal(rust_preview[class_map == 0], frame[class_map == 0])
        np.testing.assert_array_equal(crack_preview[crack_mask == 0], frame[crack_mask == 0])
        self.assertFalse(np.array_equal(rust_preview[1, 2], frame[1, 2]))
        self.assertFalse(np.array_equal(crack_preview[2, 3], frame[2, 3]))

    def test_rejects_non_binary_crack_mask(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "0 and 255"):
            render_crack_mask_preview(frame, np.ones((2, 2), dtype=np.uint8))


class DashboardExportTests(unittest.TestCase):
    def test_exports_separate_pngs_and_weighted_complete_summary(self) -> None:
        frame = np.full((4, 5, 3), 90, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "captures"
            workbook = _Workbook(root / "report.xlsx")
            workbook.path.write_bytes(b"report")
            manifest_path = None

            sections = range(1, RAIL_SECTION_TARGET + 1)
            initial_counts = tuple(2 * section for section in sections)
            rescan_counts = tuple(sections)
            for phase, rust_counts in (
                ("initial", initial_counts),
                ("rescan", rescan_counts),
            ):
                for phase_sequence, rust_pixels in enumerate(rust_counts, start=1):
                    class_map = np.zeros((4, 5), dtype=np.uint8)
                    class_map.ravel()[:rust_pixels] = 1
                    crack_mask = np.zeros((4, 5), dtype=np.uint8)
                    if phase_sequence == 2:
                        crack_mask[3, 4] = 255
                    stem = f"{phase}_{phase_sequence:02d}"
                    raw_path = output_directory / "raw" / f"{stem}.jpg"
                    analyzed_path = output_directory / f"{stem}.jpg"
                    _write_jpeg(raw_path, frame)
                    _write_jpeg(analyzed_path, frame)
                    manifest_path = export_capture_record(
                        frame=frame,
                        rust_result=_RustResult(class_map),
                        crack_result=_CrackResult(crack_mask),
                        workbook=workbook,
                        capture_path=analyzed_path,
                        raw_capture_path=raw_path,
                        phase=phase,
                        phase_sequence=phase_sequence,
                        trigger="CAMERA_CAPTURE",
                        captured_at=datetime(2026, 8, 20, 9, phase_sequence),
                        output_directory=output_directory,
                    )

            assert manifest_path is not None
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(document["runs"][0]["capture_target"], RAIL_SECTION_TARGET)
            self.assertEqual(document["runs"][0]["crane_id"], "CRANE-01")
            self.assertEqual(document["runs"][0]["crane_label"], "1번 크레인")
            self.assertEqual(len(document["captures"]), 2 * RAIL_SECTION_TARGET)
            self.assertEqual(len(document["analyses"]), 4 * RAIL_SECTION_TARGET)
            self.assertEqual(len(document["artifacts"]), 8 * RAIL_SECTION_TARGET)
            summary = document["run_summaries"][0]
            self.assertTrue(summary["summary_complete"])
            total_pixels = 20 * RAIL_SECTION_TARGET
            self.assertAlmostEqual(summary["before_ratio_fraction"], sum(initial_counts) / total_pixels)
            self.assertAlmostEqual(summary["after_ratio_fraction"], sum(rescan_counts) / total_pixels)
            self.assertAlmostEqual(summary["absolute_reduction_fraction"], sum(rescan_counts) / total_pixels)
            self.assertAlmostEqual(summary["relative_improvement_fraction"], 0.5)
            self.assertEqual(document["runs"][0]["status"], "in_progress")
            self.assertIsNone(document["runs"][0]["finished_at_utc"])

            finalized_path = finalize_dashboard_run(
                workbook=workbook,
                output_directory=output_directory,
                status="complete",
                finished_at=datetime(2026, 8, 20, 10, 0),
            )
            self.assertEqual(finalized_path, manifest_path)
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(document["runs"][0]["status"], "complete")
            self.assertEqual(document["runs"][0]["finished_at_utc"], "2026-08-20T01:00:00Z")
            self.assertIsNone(document["runs"][0]["failure_reason"])

            previews = [
                artifact
                for artifact in document["artifacts"]
                if artifact["artifact_type"] in ("rust_preview", "crack_preview")
            ]
            self.assertEqual(len(previews), 4 * RAIL_SECTION_TARGET)
            for artifact in previews:
                preview_path = root / artifact["object_key"]
                self.assertTrue(preview_path.is_file())
                with Image.open(preview_path) as loaded:
                    self.assertEqual((loaded.height, loaded.width, 3), frame.shape)

    def test_incomplete_run_is_not_reported_as_complete_or_reduced(self) -> None:
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        class_map = np.zeros((2, 3), dtype=np.uint8)
        crack_mask = np.zeros((2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "captures"
            workbook = _Workbook(root / "report.xlsx")
            workbook.path.write_bytes(b"report")
            raw_path = output_directory / "raw" / "initial_01.jpg"
            analyzed_path = output_directory / "initial_01.jpg"
            _write_jpeg(raw_path, frame)
            _write_jpeg(analyzed_path, frame)

            manifest_path = export_capture_record(
                frame=frame,
                rust_result=_RustResult(class_map),
                crack_result=_CrackResult(crack_mask),
                workbook=workbook,
                capture_path=analyzed_path,
                raw_capture_path=raw_path,
                phase="initial",
                phase_sequence=1,
                trigger="CAMERA_CAPTURE",
                captured_at=datetime(2026, 8, 20, 23, 59, 59),
                output_directory=output_directory,
            )

            summary = json.loads(manifest_path.read_text(encoding="utf-8"))[
                "run_summaries"
            ][0]
            self.assertFalse(summary["summary_complete"])
            self.assertIsNone(summary["absolute_reduction_fraction"])
            self.assertIsNone(summary["relative_improvement_fraction"])

            finalized_path = finalize_dashboard_run(
                workbook=workbook,
                output_directory=output_directory,
                status="failed",
                failure_reason="camera timeout",
                finished_at=datetime(2026, 8, 21, 0, 1),
            )
            self.assertEqual(finalized_path, manifest_path)
            run = json.loads(manifest_path.read_text(encoding="utf-8"))["runs"][0]
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["failure_reason"], "camera timeout")
            self.assertEqual(run["finished_at_utc"], "2026-08-20T15:01:00Z")

    def test_exports_top_crack_without_changing_side_summary(
        self,
    ) -> None:
        frame = np.full((4, 5, 3), 90, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "captures"
            workbook = _Workbook(root / "report.xlsx")
            workbook.path.write_bytes(b"report")
            manifest_path = None

            for phase in ("initial", "rescan"):
                for phase_sequence in range(1, RAIL_SECTION_TARGET + 1):
                    class_map = np.zeros((4, 5), dtype=np.uint8)
                    class_map.ravel()[:phase_sequence] = 1
                    side_crack_mask = np.zeros((4, 5), dtype=np.uint8)
                    top_crack_mask = np.zeros((4, 5), dtype=np.uint8)
                    top_crack_mask.ravel()[:phase_sequence] = 255
                    stem = f"{phase}_{phase_sequence:02d}"
                    side_raw = output_directory / "raw" / "side" / f"{stem}.jpg"
                    side_analyzed = output_directory / f"{stem}.jpg"
                    top_raw = output_directory / "raw" / "top" / f"{stem}.jpg"
                    top_analyzed = output_directory / "top_crack" / f"{stem}.jpg"
                    for path in (side_raw, side_analyzed, top_raw, top_analyzed):
                        _write_jpeg(path, frame)
                    captured_at = datetime(2026, 8, 25, 9, phase_sequence)
                    manifest_path = export_capture_record(
                        frame=frame,
                        rust_result=_RustResult(class_map),
                        crack_result=_CrackResult(side_crack_mask),
                        workbook=workbook,
                        capture_path=side_analyzed,
                        raw_capture_path=side_raw,
                        phase=phase,
                        phase_sequence=phase_sequence,
                        trigger="CAMERA_CAPTURE",
                        captured_at=captured_at,
                        output_directory=output_directory,
                    )
                    if phase == "initial" and phase_sequence == 1:
                        legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
                        legacy["captures"][0].pop("camera_role")
                        manifest_path.write_text(
                            json.dumps(legacy),
                            encoding="utf-8",
                        )
                    manifest_path = export_top_crack_record(
                        frame=frame,
                        crack_result=_CrackResult(top_crack_mask),
                        workbook=workbook,
                        capture_path=top_analyzed,
                        raw_capture_path=top_raw,
                        phase=phase,
                        phase_sequence=phase_sequence,
                        trigger="CAMERA_CAPTURE",
                        captured_at=captured_at,
                        output_directory=output_directory,
                    )

            assert manifest_path is not None
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(document["captures"]), 4 * RAIL_SECTION_TARGET)
            self.assertEqual(
                [capture["camera_role"] for capture in document["captures"]].count(
                    "side"
                ),
                2 * RAIL_SECTION_TARGET,
            )
            self.assertEqual(
                [capture["camera_role"] for capture in document["captures"]].count(
                    "top"
                ),
                2 * RAIL_SECTION_TARGET,
            )
            summary = document["run_summaries"][0]
            self.assertEqual(summary["initial_capture_count"], RAIL_SECTION_TARGET)
            self.assertEqual(summary["rescan_capture_count"], RAIL_SECTION_TARGET)
            self.assertTrue(summary["summary_complete"])
            top_ids = {
                capture["capture_id"]
                for capture in document["captures"]
                if capture["camera_role"] == "top"
            }
            top_analyses = [
                analysis
                for analysis in document["analyses"]
                if analysis["capture_id"] in top_ids
            ]
            self.assertEqual(len(top_analyses), 2 * RAIL_SECTION_TARGET)
            self.assertTrue(
                all(analysis["defect_type"] == "crack" for analysis in top_analyses)
            )
            top_artifacts = [
                artifact
                for artifact in document["artifacts"]
                if artifact["capture_id"] in top_ids
            ]
            self.assertEqual(len(top_artifacts), 6 * RAIL_SECTION_TARGET)
            self.assertNotIn(
                "rust_preview",
                {artifact["artifact_type"] for artifact in top_artifacts},
            )
            finalize_dashboard_run(
                workbook=workbook,
                output_directory=output_directory,
                status="complete",
            )
            with self.assertRaisesRegex(ValueError, "already contains TOP"):
                export_top_crack_record(
                    frame=frame,
                    crack_result=_CrackResult(np.zeros((4, 5), dtype=np.uint8)),
                    workbook=workbook,
                    capture_path=output_directory / "top_crack" / "initial_01.jpg",
                    raw_capture_path=output_directory / "raw" / "top" / "initial_01.jpg",
                    phase="initial",
                    phase_sequence=1,
                    trigger="CAMERA_CAPTURE",
                    captured_at=datetime(2026, 8, 25, 9, 1),
                    output_directory=output_directory,
                )

    def test_top_only_captures_cannot_complete_side_summary(self) -> None:
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        crack_mask = np.zeros((2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "captures"
            workbook = _Workbook(root / "report.xlsx")
            workbook.path.write_bytes(b"report")
            manifest_path = None
            for phase in ("initial", "rescan"):
                for phase_sequence in range(1, RAIL_SECTION_TARGET + 1):
                    stem = f"{phase}_{phase_sequence:02d}"
                    raw_path = output_directory / "raw" / "top" / f"{stem}.jpg"
                    analyzed_path = output_directory / "top_crack" / f"{stem}.jpg"
                    _write_jpeg(raw_path, frame)
                    _write_jpeg(analyzed_path, frame)
                    manifest_path = export_top_crack_record(
                        frame=frame,
                        crack_result=_CrackResult(crack_mask),
                        workbook=workbook,
                        capture_path=analyzed_path,
                        raw_capture_path=raw_path,
                        phase=phase,
                        phase_sequence=phase_sequence,
                        trigger="CAMERA_CAPTURE",
                        captured_at=datetime(2026, 8, 25, 12, phase_sequence),
                        output_directory=output_directory,
                    )

            assert manifest_path is not None
            summary = json.loads(manifest_path.read_text(encoding="utf-8"))[
                "run_summaries"
            ][0]
            self.assertEqual(summary["initial_capture_count"], 0)
            self.assertEqual(summary["rescan_capture_count"], 0)
            self.assertFalse(summary["summary_complete"])
            with self.assertRaisesRegex(
                ValueError, rf"{RAIL_SECTION_TARGET}\+{RAIL_SECTION_TARGET} rail-section"
            ):
                finalize_dashboard_run(
                    workbook=workbook,
                    output_directory=output_directory,
                    status="complete",
                )


if __name__ == "__main__":
    unittest.main()
