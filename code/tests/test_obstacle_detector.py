from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from obstacle_detector import (  # noqa: E402
    CAMERA_FRAME_SHAPE,
    CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    PAD_BOTTOM,
    PAD_TOP,
    PAD_VALUE,
    SOURCE_ONNX_SHA256,
    ObstacleDetector,
    ObstacleStageTiming,
    preprocess_frame,
    result_from_compact_output,
    result_from_output,
)


class ObstaclePreprocessTests(unittest.TestCase):
    def test_converts_full_frame_bgr_to_padded_normalized_rgb(self) -> None:
        frame = np.zeros(CAMERA_FRAME_SHAPE, dtype=np.uint8)
        frame[0, 0] = (0, 127, 255)

        tensor = preprocess_frame(frame)

        self.assertEqual(tensor.shape, INPUT_SHAPE)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(
            tensor[0, :, PAD_TOP, 0],
            np.array([1.0, 127.0 / 255.0, 0.0], dtype=np.float32),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            tensor[:, :, :PAD_TOP, :],
            np.float32(PAD_VALUE / 255.0),
            atol=0.0,
        )
        np.testing.assert_allclose(
            tensor[:, :, -PAD_BOTTOM:, :],
            np.float32(PAD_VALUE / 255.0),
            atol=0.0,
        )

    def test_rejects_resize_or_non_uint8_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "1280x720"):
            preprocess_frame(np.zeros((240, 1280, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8"):
            preprocess_frame(np.zeros(CAMERA_FRAME_SHAPE, dtype=np.float32))


class ObstaclePostprocessTests(unittest.TestCase):
    @staticmethod
    def _output() -> np.ndarray:
        output = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
        output[0, :, 2] = 1.0
        output[0, :, 3] = 1.0
        return output

    def test_filters_at_metadata_threshold_and_restores_camera_coordinates(self) -> None:
        output = self._output()
        output[0, 0] = (-5.0, 4.0, 1300.0, 260.0, 0.80, 0.0)
        output[0, 1] = (10.0, 20.0, 30.0, 40.0, 0.424, 0.0)
        output[0, 2] = (50.0, 50.0, 60.0, 60.0, 0.423, 0.0)
        output[0, 3] = (10.0, 0.0, 20.0, 7.0, 0.90, 0.0)

        result = result_from_output(output, "test")

        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.424)
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.detected)
        self.assertEqual(
            result.boxes,
            (
                (0.0, 0.0, 1280.0, 252.0),
                (10.0, 12.0, 30.0, 32.0),
            ),
        )
        self.assertAlmostEqual(result.scores[0], 0.80, places=6)
        self.assertAlmostEqual(result.scores[1], 0.424, places=6)
        self.assertEqual(result.class_ids, (0, 0))
        self.assertFalse(result.control_roi_detected)
        self.assertEqual(result.detections[0].class_name, "obstacle")
        self.assertEqual(CLASS_NAMES, ("obstacle",))

    def test_gpu_compact_result_matches_cpu_reference(self) -> None:
        output = self._output()
        output[0, 0] = (-5.0, 4.0, 1300.0, 260.0, 0.80, 0.0)
        output[0, 1] = (10.0, 487.0, 30.0, 500.0, 0.424, 0.0)
        output[0, 2] = (10.0, 0.0, 20.0, 7.0, 0.90, 0.0)
        output[0, 3] = (50.0, 50.0, 60.0, 60.0, 0.423, 0.0)
        reference = result_from_output(output, "cpu")
        compact = np.array(
            [
                (*detection.box_xyxy, detection.confidence, detection.class_id)
                for detection in reference.detections
            ],
            dtype=np.float32,
        )

        gpu = result_from_compact_output(
            compact,
            reference.control_roi_detected,
            "gpu",
        )

        self.assertEqual(gpu.detections, reference.detections)
        self.assertEqual(
            gpu.control_roi_detected,
            reference.control_roi_detected,
        )

    def test_gpu_compact_result_fails_closed_on_contract_mismatch(self) -> None:
        valid = np.array(
            [[10.0, 500.0, 30.0, 520.0, 0.424, 0.0]],
            dtype=np.float32,
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            result_from_compact_output(valid, False, "gpu")
        with self.assertRaisesRegex(ValueError, "class other than 0"):
            invalid_class = valid.copy()
            invalid_class[0, 5] = 1.0
            result_from_compact_output(invalid_class, True, "gpu")
        with self.assertRaisesRegex(ValueError, "locked threshold"):
            below_threshold = valid.copy()
            below_threshold[0, 4] = 0.423
            result_from_compact_output(below_threshold, True, "gpu")

    def test_control_roi_is_the_bottom_240_camera_rows(self) -> None:
        output = self._output()
        output[0, 0] = (10.0, 400.0, 30.0, 488.0, 0.80, 0.0)
        self.assertFalse(result_from_output(output, "cpu").control_roi_detected)

        output[0, 0] = (10.0, 487.0, 30.0, 500.0, 0.80, 0.0)
        self.assertTrue(result_from_output(output, "cpu").control_roi_detected)

    def test_bottom_padding_only_box_is_dropped(self) -> None:
        output = self._output()
        output[0, 0] = (10.0, 728.0, 30.0, 736.0, 0.80, 0.0)
        self.assertFalse(result_from_output(output, "cpu").detected)

    def test_keeps_overlapping_one_to_one_topk_rows_without_extra_nms(self) -> None:
        output = self._output()
        output[0, 0] = (10.0, 18.0, 100.0, 108.0, 0.90, 0.0)
        output[0, 1] = (11.0, 19.0, 99.0, 107.0, 0.80, 0.0)

        result = result_from_output(output, "test")

        self.assertEqual(len(result.detections), 2)

    def test_rejects_invalid_output_before_control_can_use_it(self) -> None:
        invalid_outputs = []

        wrong_dtype = self._output().astype(np.float64)
        invalid_outputs.append(wrong_dtype)
        invalid_outputs.append(np.zeros((1, 299, 6), dtype=np.float32))

        nan_output = self._output()
        nan_output[0, 0, 4] = np.nan
        invalid_outputs.append(nan_output)

        invalid_score = self._output()
        invalid_score[0, 0, 4] = 1.01
        invalid_outputs.append(invalid_score)

        fractional_class = self._output()
        fractional_class[0, 0, 5] = 0.5
        invalid_outputs.append(fractional_class)

        unknown_class = self._output()
        unknown_class[0, 0, 5] = 1.0
        invalid_outputs.append(unknown_class)

        reversed_box = self._output()
        reversed_box[0, 0, :4] = (10.0, 10.0, 9.0, 20.0)
        invalid_outputs.append(reversed_box)

        for output in invalid_outputs:
            with self.subTest(shape=output.shape, dtype=output.dtype):
                with self.assertRaises(ValueError):
                    result_from_output(output, "test")

    def test_rejects_invalid_confidence_threshold(self) -> None:
        output = self._output()
        for threshold in (0.0, 1.0, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                result_from_output(output, "test", threshold)


class _FakeTensorIOMode:
    INPUT = "input"
    OUTPUT = "output"


class _FakeTensorFormat:
    LINEAR = "linear"


class _FakeTensorLocation:
    DEVICE = "device"


class _FakeTensorRT:
    TensorIOMode = _FakeTensorIOMode
    TensorFormat = _FakeTensorFormat
    TensorLocation = _FakeTensorLocation

    @staticmethod
    def nptype(dtype):
        return dtype


class _FakeEngine:
    def __init__(self) -> None:
        self.names = [OUTPUT_NAME, INPUT_NAME]
        self.num_io_tensors = len(self.names)
        self.modes = {
            INPUT_NAME: _FakeTensorIOMode.INPUT,
            OUTPUT_NAME: _FakeTensorIOMode.OUTPUT,
        }
        self.shapes = {INPUT_NAME: INPUT_SHAPE, OUTPUT_NAME: OUTPUT_SHAPE}
        self.dtypes = {INPUT_NAME: np.float32, OUTPUT_NAME: np.float32}
        self.formats = {
            INPUT_NAME: _FakeTensorFormat.LINEAR,
            OUTPUT_NAME: _FakeTensorFormat.LINEAR,
        }
        self.locations = {
            INPUT_NAME: _FakeTensorLocation.DEVICE,
            OUTPUT_NAME: _FakeTensorLocation.DEVICE,
        }

    def get_tensor_name(self, index: int) -> str:
        return self.names[index]

    def get_tensor_mode(self, name: str) -> str:
        return self.modes[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str):
        return self.dtypes[name]

    def get_tensor_format(self, name: str) -> str:
        return self.formats[name]

    def get_tensor_location(self, name: str) -> str:
        return self.locations[name]


class ObstacleEngineContractTests(unittest.TestCase):
    @staticmethod
    def _detector() -> ObstacleDetector:
        detector = ObstacleDetector.__new__(ObstacleDetector)
        detector._trt = _FakeTensorRT()
        detector._engine = _FakeEngine()
        return detector

    def test_accepts_exact_static_fp32_linear_device_contract(self) -> None:
        self._detector()._validate_engine_contract()

    def test_rejects_any_engine_contract_change(self) -> None:
        mutations = (
            lambda engine: engine.names.append("extra"),
            lambda engine: engine.modes.__setitem__(INPUT_NAME, "output"),
            lambda engine: engine.modes.__setitem__(OUTPUT_NAME, "input"),
            lambda engine: engine.shapes.__setitem__(INPUT_NAME, (1, 3, 720, 1280)),
            lambda engine: engine.shapes.__setitem__(OUTPUT_NAME, (1, 300, 7)),
            lambda engine: engine.dtypes.__setitem__(INPUT_NAME, np.float16),
            lambda engine: engine.dtypes.__setitem__(OUTPUT_NAME, np.float16),
            lambda engine: engine.formats.__setitem__(INPUT_NAME, "chw32"),
            lambda engine: engine.locations.__setitem__(OUTPUT_NAME, "host"),
        )
        for mutate in mutations:
            detector = self._detector()
            mutate(detector._engine)
            detector._engine.num_io_tensors = len(detector._engine.names)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                detector._validate_engine_contract()

    def test_rejects_invalid_expected_digest_before_engine_or_cuda_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_sha256"):
            ObstacleDetector(Path("missing.plan"), "invalid")

    def test_records_the_inspected_source_onnx_digest(self) -> None:
        self.assertEqual(
            SOURCE_ONNX_SHA256,
            "c55fb45efd33345d31ec99c4b2ff81c5ccc7973d5810c88ac51210310a681341",
        )


class _FakeTimingCudaRuntime:
    class cudaError_t:
        cudaSuccess = 0

    def __init__(self) -> None:
        self.records = []

    def cudaEventRecord(self, event, stream):
        self.records.append((event, stream))
        return (0,)

    @staticmethod
    def cudaEventElapsedTime(start, end):
        return (0, float(end - start))


class ObstacleStageTimingTests(unittest.TestCase):
    def test_opt_in_timing_reports_all_detector_stages(self) -> None:
        detector = ObstacleDetector.__new__(ObstacleDetector)
        detector._context = Mock()
        detector._context.execute_async_v3.return_value = True
        detector._stream = 50
        detector._device_input = 60
        detector._device_output = 70
        detector._gpu_pipeline = Mock()
        detector._gpu_pipeline.upload_and_preprocess.return_value = 1.0
        detector._gpu_pipeline.postprocess.return_value = SimpleNamespace(
            records=np.empty((0, 6), dtype=np.float32),
            control_roi_detected=False,
        )
        detector._stage_timing = True
        detector._timing_events = [1, 3, 6, 10, 15, 21]
        detector._cudart = _FakeTimingCudaRuntime()
        detector.last_stage_timing = None
        detector.method = "test"
        detector.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

        with patch(
            "obstacle_detector.time.perf_counter",
            side_effect=[10.0, 10.005, 10.007, 10.010],
        ):
            result = detector.detect(np.zeros(CAMERA_FRAME_SHAPE, dtype=np.uint8))

        self.assertFalse(result.detected)
        expected = ObstacleStageTiming(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 10.0
        )
        for field_name in expected.__dataclass_fields__:
            self.assertAlmostEqual(
                getattr(detector.last_stage_timing, field_name),
                getattr(expected, field_name),
                places=6,
            )
        detector._gpu_pipeline.upload_and_preprocess.assert_called_once_with(
            ANY,
            60,
            50,
            (1, 3, 6),
        )
        detector._gpu_pipeline.postprocess.assert_called_once_with(
            70,
            50,
            (15, 21),
        )

    def test_normal_detection_does_not_record_cuda_events(self) -> None:
        detector = ObstacleDetector.__new__(ObstacleDetector)
        detector._context = Mock()
        detector._context.execute_async_v3.return_value = True
        detector._stream = 50
        detector._device_input = 60
        detector._device_output = 70
        detector._gpu_pipeline = Mock()
        detector._gpu_pipeline.postprocess.return_value = SimpleNamespace(
            records=np.empty((0, 6), dtype=np.float32),
            control_roi_detected=False,
        )
        detector._stage_timing = False
        detector._timing_events = []
        detector._cudart = _FakeTimingCudaRuntime()
        detector.last_stage_timing = None
        detector.method = "test"
        detector.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

        detector.detect(np.zeros(CAMERA_FRAME_SHAPE, dtype=np.uint8))

        self.assertEqual(detector._cudart.records, [])
        self.assertIsNone(detector.last_stage_timing)
        detector._gpu_pipeline.upload_and_preprocess.assert_called_once_with(
            ANY,
            60,
            50,
            None,
        )


if __name__ == "__main__":
    unittest.main()
