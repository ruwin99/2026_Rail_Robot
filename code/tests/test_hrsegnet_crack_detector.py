from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from hrsegnet_crack_detector import (  # noqa: E402
    CAPTURE_DETECTOR_PREFIX,
    CAPTURE_INPUT_SHAPE,
    CAPTURE_OUTPUT_SHAPE,
    DEFAULT_PROBABILITY_THRESHOLD,
    DETECTOR_PREFIX,
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    HrSegNetCrackDetector,
    preprocess_frame,
    preprocess_into,
    probability_threshold_to_logit_margin,
    resize_realtime_roi,
    result_from_candidate_map,
    result_from_logits,
)
from cuda_logit_margin import _kernel_source  # noqa: E402


class HrSegNetRealtimeRoiTests(unittest.TestCase):
    def _detector(self, candidate, *, role="realtime"):
        detector = HrSegNetCrackDetector.__new__(HrSegNetCrackDetector)
        detector.role = role
        detector.input_shape = INPUT_SHAPE if role == "realtime" else CAPTURE_INPUT_SHAPE
        detector.method = "test-roi"
        detector.probability_threshold = DEFAULT_PROBABILITY_THRESHOLD
        detector.min_component_pixels = 20
        detector.stage_timing = False
        detector._context = Mock()
        detector._context.execute_async_v3.return_value = True
        detector._stream = 17
        detector._device_input = 100
        detector._device_output = 200
        detector._host_input = np.empty(detector.input_shape, dtype=np.float32)
        detector._trt_timing_events = ()
        detector._cudart = Mock()
        detector._cuda_call = Mock()
        detector._gpu_postprocessor = Mock()
        detector._gpu_preprocessor = None
        detector._gpu_postprocessor.process.return_value = SimpleNamespace(
            candidate_map=candidate, max_margin=2.0
        )
        return detector

    def test_180_row_roi_uses_static_engine_and_returns_original_coordinates(self):
        frame = np.zeros((180, 1280, 3), dtype=np.uint8)
        frame[:, :, 0] = np.arange(180, dtype=np.uint8)[:, None]
        original = frame.copy()
        candidate = np.zeros((128, 1280), dtype=np.uint8)
        candidate[:64] = 1
        detector = self._detector(candidate)

        result = detector.detect(frame)

        self.assertEqual(result.mask.shape, (180, 1280))
        self.assertEqual(result.inspected_pixels, 180 * 1280)
        self.assertEqual(result.crack_pixels, 90 * 1280)
        self.assertEqual(result.crack_ratio, 0.5)
        self.assertEqual(result.boxes, [(0, 0, 1280, 90)])
        self.assertEqual(result.candidate_pixels, 90 * 1280)
        expected_mask = np.zeros((180, 1280), dtype=np.uint8)
        expected_mask[:90] = 255
        np.testing.assert_array_equal(result.mask, expected_mask)
        expected_tensor = preprocess_frame(
            cv2.resize(frame, (1280, 128), interpolation=cv2.INTER_AREA)
        )
        np.testing.assert_array_equal(detector._host_input, expected_tensor)
        np.testing.assert_array_equal(frame, original)
        detector._gpu_postprocessor.process.assert_called_once_with(200, 17)

    def test_component_threshold_is_applied_after_mask_restoration(self):
        candidate = np.zeros((128, 1280), dtype=np.uint8)
        candidate[0, :10] = 1  # Becomes 20 source-ROI pixels, so it must survive.
        candidate[-1, 100:119] = 1  # Remains 19 source-ROI pixels, so it is dropped.
        result = self._detector(candidate).detect(
            np.zeros((180, 1280, 3), dtype=np.uint8)
        )
        self.assertEqual(result.candidate_pixels, 39)
        self.assertEqual(result.filtered_pixels, 20)
        self.assertEqual(result.boxes, [(0, 0, 10, 2)])
        self.assertAlmostEqual(result.crack_ratio, 20 / (180 * 1280))
        self.assertFalse(np.any(result.mask[-1]))

    def test_native_realtime_and_full_capture_results_keep_their_dimensions(self):
        for role, height in (("realtime", 128), ("capture", 720)):
            with self.subTest(role=role):
                candidate = np.ones((height, 1280), dtype=np.uint8)
                detector = self._detector(candidate, role=role)
                result = detector.detect(np.zeros((height, 1280, 3), dtype=np.uint8))
                self.assertEqual(result.mask.shape, (height, 1280))
                self.assertEqual(result.inspected_pixels, height * 1280)
                self.assertEqual(result.crack_ratio, 1.0)
                self.assertEqual(result.boxes, [(0, 0, 1280, height)])

    def test_adapter_rejects_invalid_roi_and_capture_never_uses_adapter(self):
        detector = self._detector(np.zeros((128, 1280), dtype=np.uint8))
        for frame in (
            np.zeros((179, 1280, 3), dtype=np.uint8),
            np.zeros((180, 1279, 3), dtype=np.uint8),
            np.zeros((180, 1280, 3), dtype=np.float32),
        ):
            with self.subTest(shape=frame.shape, dtype=frame.dtype):
                with self.assertRaises(ValueError):
                    detector.detect(frame)
        detector.role = "capture"
        detector.input_shape = CAPTURE_INPUT_SHAPE
        with self.assertRaisesRegex(ValueError, "1280x720"):
            detector.detect(np.zeros((180, 1280, 3), dtype=np.uint8))
        detector._context.execute_async_v3.assert_not_called()


def _reference_preprocess(frame: np.ndarray) -> np.ndarray:
    """최적화 전 구현. 새 경로가 이것과 비트 단위로 같아야 엔진 입력이 바뀌지 않는다."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    return np.ascontiguousarray((rgb / 127.5 - 1.0).transpose(2, 0, 1)[None])


def _reference_filter(candidate: np.ndarray, min_pixels: int):
    """최적화 전 연결 성분 필터(성분마다 Python 반복)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    filtered = np.zeros(candidate.shape, dtype=np.uint8)
    boxes = []
    for component_id in range(1, count):
        if stats[component_id, cv2.CC_STAT_AREA] < min_pixels:
            continue
        filtered[labels == component_id] = 255
        boxes.append(tuple(int(value) for value in stats[component_id, :4]))
    return filtered, boxes


class HrSegNetOptimizedPathEquivalenceTests(unittest.TestCase):
    def test_preprocess_into_pinned_buffer_is_bit_identical_for_both_roles(self):
        rng = np.random.default_rng(7)
        for shape in (INPUT_SHAPE, CAPTURE_INPUT_SHAPE):
            with self.subTest(shape=shape):
                frame = rng.integers(0, 256, (shape[2], shape[3], 3), dtype=np.uint8)
                buffer = np.full(shape, np.nan, dtype=np.float32)
                preprocess_into(frame, buffer)
                np.testing.assert_array_equal(buffer, _reference_preprocess(frame))
                np.testing.assert_array_equal(
                    preprocess_frame(frame, shape), _reference_preprocess(frame)
                )

    def test_preprocess_into_rejects_wrong_buffer(self):
        frame = np.zeros((128, 1280, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_into(frame, np.empty(INPUT_SHAPE, dtype=np.float64))
        with self.assertRaises(ValueError):
            preprocess_into(frame, np.empty((1, 3, 1280, 128), dtype=np.float32).transpose(0, 1, 3, 2))
        with self.assertRaisesRegex(ValueError, "does not resize"):
            preprocess_into(np.zeros((127, 1280, 3), dtype=np.uint8), np.empty(INPUT_SHAPE, np.float32))

    def test_single_channel_roi_resize_matches_three_channel_inter_area(self):
        rng = np.random.default_rng(11)
        for _ in range(5):
            frame = rng.integers(0, 256, (180, 1280, 3), dtype=np.uint8)
            expected = cv2.resize(frame, (1280, 128), interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(resize_realtime_roi(frame), expected)
        # 연속 메모리가 아닌 ROI view(원본 프레임의 일부 열)도 같은 결과다.
        wide = rng.integers(0, 256, (180, 1300, 3), dtype=np.uint8)
        view = wide[:, 10:1290]
        np.testing.assert_array_equal(
            resize_realtime_roi(view),
            cv2.resize(view, (1280, 128), interpolation=cv2.INTER_AREA),
        )

    def test_component_filter_matches_reference_on_every_branch(self):
        rng = np.random.default_rng(3)
        cases = [np.zeros((180, 1280), dtype=np.uint8)]
        # 남는 성분 0개, 소수(성분별 칠하기), 다수(표 변환) 분기를 모두 지난다.
        for density in (0.0005, 0.02, 0.08, 0.3):
            cases.append((rng.random((180, 1280)) < density).astype(np.uint8))
        few = np.zeros((180, 1280), dtype=np.uint8)
        few[10:15, 10:20] = 1
        few[100:102, 500:530] = 1
        cases.append(few)
        many = np.zeros((180, 1280), dtype=np.uint8)
        for index in range(12):
            many[5:10, index * 100:index * 100 + 6] = 1
        cases.append(many)
        for min_pixels in (1, 2, 20):
            for candidate in cases:
                with self.subTest(min_pixels=min_pixels, pixels=int(candidate.sum())):
                    result = result_from_candidate_map(candidate, 0.5, "eq", 0.55, min_pixels)
                    mask, boxes = _reference_filter(candidate, min_pixels)
                    np.testing.assert_array_equal(result.mask, mask)
                    self.assertEqual(result.boxes, boxes)
                    self.assertTrue(all(type(value) is int for box in result.boxes for value in box))
                    self.assertEqual(result.crack_pixels, int(np.count_nonzero(mask)))


class HrSegNetPostprocessTests(unittest.TestCase):
    def test_preprocess_uses_rgb_and_half_range_normalization_without_resize(self) -> None:
        frame = np.zeros((128, 1280, 3), dtype=np.uint8)
        frame[0, 0] = (0, 127, 255)

        tensor = preprocess_frame(frame)

        self.assertEqual(tensor.shape, INPUT_SHAPE)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(
            tensor[0, :, 0, 0],
            np.array([1.0, 127.0 / 127.5 - 1.0, -1.0], dtype=np.float32),
            atol=1e-7,
        )
        with self.assertRaisesRegex(ValueError, "does not resize"):
            preprocess_frame(np.zeros((127, 1280, 3), dtype=np.uint8))

    def test_capture_preprocess_uses_native_full_frame_without_resize_or_padding(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[0, 0] = (0, 127, 255)

        tensor = preprocess_frame(frame, CAPTURE_INPUT_SHAPE)

        self.assertEqual(tensor.shape, CAPTURE_INPUT_SHAPE)
        np.testing.assert_allclose(
            tensor[0, :, 0, 0],
            np.array([1.0, 127.0 / 127.5 - 1.0, -1.0], dtype=np.float32),
            atol=1e-7,
        )
        with self.assertRaisesRegex(ValueError, "does not resize or pad"):
            preprocess_frame(np.zeros((719, 1280, 3), dtype=np.uint8), CAPTURE_INPUT_SHAPE)

    def test_logit_margin_is_equivalent_to_two_class_softmax_probability(self) -> None:
        self.assertEqual(DEFAULT_PROBABILITY_THRESHOLD, 0.55)
        for probability_threshold in (0.1, 0.5, 0.55, 0.73, 0.85, 0.99):
            with self.subTest(probability_threshold=probability_threshold):
                margin = probability_threshold_to_logit_margin(probability_threshold)
                probability = 1.0 / (1.0 + math.exp(-margin))
                self.assertAlmostEqual(probability, probability_threshold, places=12)

    def test_tie_behavior_matches_probability_threshold(self) -> None:
        logits = np.zeros(OUTPUT_SHAPE, dtype=np.float32)

        at_half = result_from_logits(logits, "test", 0.5, 1)
        above_half = result_from_logits(logits, "test", 0.5001, 1)

        self.assertEqual(at_half.crack_pixels, 128 * 1280)
        self.assertEqual(above_half.crack_pixels, 0)

    def test_rejects_nan_infinity_and_non_float32_logits(self) -> None:
        invalid_logits = (
            np.zeros(OUTPUT_SHAPE, dtype=np.float16),
            np.full(OUTPUT_SHAPE, np.nan, dtype=np.float32),
            np.full(OUTPUT_SHAPE, np.inf, dtype=np.float32),
        )
        for logits in invalid_logits:
            with self.subTest(dtype=logits.dtype, value=logits.flat[0]):
                with self.assertRaises(ValueError):
                    result_from_logits(logits, "test")

    def test_filters_8_connected_components_and_returns_control_result(self) -> None:
        logits = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
        logits[:, 0] = 1.0
        logits[0, 1, 10, 10] = 2.0
        logits[0, 1, 11, 11] = 2.0
        logits[0, 1, 40, 40:43] = 2.0
        logits[0, 1, 80, 80] = 2.0

        result = result_from_logits(
            logits,
            f"{DETECTOR_PREFIX}test",
            probability_threshold=0.5,
            min_component_pixels=2,
        )

        self.assertEqual(result.crack_pixels, 5)
        self.assertEqual(result.boxes, [(10, 10, 2, 2), (40, 40, 3, 1)])
        self.assertEqual(result.mask.dtype, np.uint8)
        self.assertEqual(result.mask.shape, (128, 1280))
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.detected)
        self.assertAlmostEqual(result.max_crack_probability, 0.7310585786)
        self.assertEqual(result.candidate_pixels, 6)
        self.assertEqual(result.filtered_pixels, 5)
        self.assertEqual(result.min_component_pixels, 2)

    def test_gpu_candidate_contract_preserves_8_connectivity_and_area_boundary(self) -> None:
        candidate = np.zeros((8, 12), dtype=np.uint8)
        candidate[0, 0] = 1
        candidate[1, 1] = 1
        candidate[4, 5:8] = 1

        result = result_from_candidate_map(
            candidate,
            0.75,
            "gpu-test",
            probability_threshold=0.55,
            min_component_pixels=2,
        )

        self.assertEqual(result.candidate_pixels, 5)
        self.assertEqual(result.filtered_pixels, 5)
        self.assertEqual(result.boxes, [(0, 0, 2, 2), (5, 4, 3, 1)])
        self.assertEqual(set(np.unique(result.mask)), {0, 255})

    def test_default_filter_drops_19_pixels_and_keeps_20_and_21(self) -> None:
        candidate = np.zeros((5, 64), dtype=np.uint8)
        candidate[0, 0:19] = 1
        candidate[2, 20:40] = 1
        candidate[4, 42:63] = 1

        result = result_from_candidate_map(candidate, 0.75, "gpu-test")

        self.assertEqual(result.candidate_pixels, 60)
        self.assertEqual(result.filtered_pixels, 41)
        self.assertEqual(result.boxes, [(20, 2, 20, 1), (42, 4, 21, 1)])

    def test_gpu_candidate_contract_rejects_invalid_maps_and_probability(self) -> None:
        for candidate in (
            np.zeros((2, 2), dtype=np.float32),
            np.full((2, 2), 2, dtype=np.uint8),
            np.zeros((1, 2, 2), dtype=np.uint8),
        ):
            with self.subTest(shape=candidate.shape, dtype=candidate.dtype):
                with self.assertRaises(ValueError):
                    result_from_candidate_map(candidate, 0.5, "test")
        with self.assertRaises(ValueError):
            result_from_candidate_map(
                np.zeros((2, 2), dtype=np.uint8), np.nan, "test"
            )

    def test_cuda_kernel_uses_margin_greater_equal_and_full_finite_gate(self) -> None:
        source = _kernel_source(128 * 1280, 640, 0.200670695462151)

        self.assertIn("margin = crack - background", source)
        self.assertIn("margin >=", source)
        self.assertIn("finite_float(background)", source)
        self.assertIn("finite_float(crack)", source)
        self.assertIn("finite_float(margin)", source)
        self.assertIn("atomicExch(invalid_flag, 1u)", source)

    def test_cuda_threshold_literals_are_valid_for_zero_positive_and_negative(self) -> None:
        zero = _kernel_source(1, 1, 0.0)
        positive = _kernel_source(1, 1, 1.0)
        negative = _kernel_source(1, 1, -1.0)

        self.assertIn("margin >= 0.0f", zero)
        self.assertNotIn("margin >= 0f", zero)
        self.assertIn("margin >= 1.0f", positive)
        self.assertIn("margin >= -1.0f", negative)

    def test_diagnostic_max_probability_uses_overflow_safe_sigmoid(self) -> None:
        logits = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
        logits[:, 0] = 100.0
        logits[:, 1] = -100.0
        result = result_from_logits(logits, "test", 0.5, 1)
        self.assertAlmostEqual(
            result.max_crack_probability,
            math.exp(-200.0) / (1.0 + math.exp(-200.0)),
        )
        self.assertEqual(result.candidate_pixels, 0)

        logits[:, 0] = -100.0
        logits[:, 1] = 100.0
        result = result_from_logits(logits, "test", 0.5, 1)
        self.assertEqual(result.max_crack_probability, 1.0)
        self.assertEqual(result.candidate_pixels, 128 * 1280)


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
        return _FakeTensorIOMode.INPUT if name == INPUT_NAME else _FakeTensorIOMode.OUTPUT

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str):
        return self.dtypes[name]

    def get_tensor_format(self, name: str) -> str:
        return self.formats[name]

    def get_tensor_location(self, name: str) -> str:
        return self.locations[name]


class HrSegNetEngineContractTests(unittest.TestCase):
    def _detector(self) -> HrSegNetCrackDetector:
        detector = HrSegNetCrackDetector.__new__(HrSegNetCrackDetector)
        detector._trt = _FakeTensorRT()
        detector._engine = _FakeEngine()
        detector.input_shape = INPUT_SHAPE
        detector.output_shape = OUTPUT_SHAPE
        return detector

    def test_accepts_exact_static_fp32_linear_device_contract(self) -> None:
        self._detector()._validate_engine_contract()

    def test_accepts_exact_capture_contract_and_uses_capture_method_prefix(self) -> None:
        detector = self._detector()
        detector.input_shape = CAPTURE_INPUT_SHAPE
        detector.output_shape = CAPTURE_OUTPUT_SHAPE
        detector._engine.shapes = {
            INPUT_NAME: CAPTURE_INPUT_SHAPE,
            OUTPUT_NAME: CAPTURE_OUTPUT_SHAPE,
        }

        detector._validate_engine_contract()
        capture_logits = np.zeros(CAPTURE_OUTPUT_SHAPE, dtype=np.float32)
        result = result_from_logits(
            capture_logits,
            f"{CAPTURE_DETECTOR_PREFIX}test",
            output_shape=CAPTURE_OUTPUT_SHAPE,
        )
        self.assertEqual(result.mask.shape, (720, 1280))
        self.assertTrue(result.method.startswith(CAPTURE_DETECTOR_PREFIX))

    def test_rejects_contract_changes(self) -> None:
        mutations = (
            lambda engine: engine.names.append("extra"),
            lambda engine: engine.shapes.__setitem__(OUTPUT_NAME, (1, 1, 128, 1280)),
            lambda engine: engine.dtypes.__setitem__(INPUT_NAME, np.float16),
            lambda engine: engine.formats.__setitem__(OUTPUT_NAME, "chw32"),
            lambda engine: engine.locations.__setitem__(INPUT_NAME, "host"),
        )
        for mutate in mutations:
            detector = self._detector()
            mutate(detector._engine)
            detector._engine.num_io_tensors = len(detector._engine.names)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                detector._validate_engine_contract()

    def test_rejects_wrong_input_or_output_tensor_mode(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            detector = self._detector()
            original = detector._engine.get_tensor_mode
            detector._engine.get_tensor_mode = (
                lambda name, target=tensor_name: (
                    _FakeTensorIOMode.OUTPUT
                    if name == target and target == INPUT_NAME
                    else (
                        _FakeTensorIOMode.INPUT
                        if name == target
                        else original(name)
                    )
                )
            )
            with self.subTest(tensor_name=tensor_name), self.assertRaises(RuntimeError):
                detector._validate_engine_contract()


if __name__ == "__main__":
    unittest.main()
