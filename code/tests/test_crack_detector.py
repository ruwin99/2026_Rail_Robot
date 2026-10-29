from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from crack_detector import (  # noqa: E402
    CAPTURE_INPUT_SHAPE,
    CAPTURE_OUTPUT_SHAPE,
    CAPTURE_PROFILE,
    INPUT_NAME,
    OUTPUT_NAME,
    PROBABILITY_OUTPUT_NAME,
    REALTIME_INPUT_SHAPE,
    REALTIME_OUTPUT_SHAPE,
    REALTIME_PROFILE,
    CrackDetector,
    CrackDetectionResult,
    annotate_cracks,
    preprocess_frame,
    result_from_logits,
    result_from_output,
    restore_probability_map,
)


class CrackPreprocessingTests(unittest.TestCase):
    def test_converts_bgr_to_normalized_rgb_nchw(self) -> None:
        frame = np.zeros((128, 1280, 3), dtype=np.uint8)
        frame[0, 0] = (0, 128, 255)

        tensor, original_shape = preprocess_frame(frame)

        self.assertEqual(tensor.shape, REALTIME_INPUT_SHAPE)
        self.assertEqual(original_shape, (128, 1280))
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(
            float(tensor[0, 1, 0, 0]),
            128.0 / 127.5 - 1.0,
            places=6,
        )
        self.assertAlmostEqual(float(tensor[0, 2, 0, 0]), -1.0, places=6)

    def test_capture_profile_uses_full_frame_without_resize(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tensor, original_shape = preprocess_frame(frame, CAPTURE_PROFILE)

        self.assertEqual(tensor.shape, CAPTURE_INPUT_SHAPE)
        self.assertEqual(original_shape, (720, 1280))

    def test_profiles_reject_mismatched_shapes_instead_of_resizing(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not resize"):
            preprocess_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "do not resize"):
            preprocess_frame(
                np.zeros((128, 1280, 3), dtype=np.uint8),
                CAPTURE_PROFILE,
            )

    def test_rejects_non_uint8_camera_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite uint8"):
            preprocess_frame(
                np.zeros((128, 1280, 3), dtype=np.float32),
            )


class CrackPostprocessingTests(unittest.TestCase):
    def test_filters_tiny_components_and_reports_pixel_ratio(self) -> None:
        probability = np.zeros(REALTIME_OUTPUT_SHAPE, dtype=np.float32)
        probability[0, 0, 20:25, 30:40] = 0.8
        probability[0, 0, 100, 100] = 0.8

        result = result_from_logits(
            probability,
            (128, 1280),
            "bgcrack-tensorrt/test",
            probability_threshold=0.5,
            min_component_pixels=20,
        )

        self.assertTrue(result.detected)
        self.assertEqual(result.crack_pixels, 50)
        self.assertEqual(result.inspected_pixels, 128 * 1280)
        self.assertEqual(result.boxes, [(30, 20, 10, 5)])
        self.assertAlmostEqual(result.crack_ratio, 50 / (128 * 1280))
        self.assertEqual(int(result.mask[100, 100]), 0)

    def test_rejects_an_unexpected_output_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have shape"):
            result_from_logits(
                np.zeros((1, 2, 128, 1280), dtype=np.float32),
                (128, 1280),
                "bgcrack-tensorrt/test",
            )

    def test_probability_output_is_not_passed_through_sigmoid_again(self) -> None:
        probability = np.full(REALTIME_OUTPUT_SHAPE, 0.2, dtype=np.float32)
        probability[0, 0, 20:25, 30:40] = 0.8

        restored = restore_probability_map(
            probability,
            (128, 1280),
            PROBABILITY_OUTPUT_NAME,
        )
        result = result_from_output(
            probability,
            (128, 1280),
            "bgcrack-tensorrt/test",
            PROBABILITY_OUTPUT_NAME,
            probability_threshold=0.7,
            min_component_pixels=20,
        )

        self.assertAlmostEqual(float(restored[0, 0]), 0.2, places=6)
        self.assertAlmostEqual(float(restored[22, 35]), 0.8, places=6)
        self.assertEqual(result.crack_pixels, 50)
        self.assertEqual(result.boxes, [(30, 20, 10, 5)])

    def test_probability_output_rejects_values_outside_zero_to_one(self) -> None:
        for invalid_value in (-0.01, 1.01):
            with self.subTest(invalid_value=invalid_value):
                probability = np.zeros(REALTIME_OUTPUT_SHAPE, dtype=np.float32)
                probability[0, 0, 0, 0] = invalid_value

                with self.assertRaisesRegex(ValueError, "between zero and one"):
                    restore_probability_map(
                        probability,
                        (128, 1280),
                        PROBABILITY_OUTPUT_NAME,
                    )

    def test_probability_output_rejects_non_finite_values(self) -> None:
        probability = np.zeros(REALTIME_OUTPUT_SHAPE, dtype=np.float32)
        probability[0, 0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            restore_probability_map(
                probability,
                (128, 1280),
                PROBABILITY_OUTPUT_NAME,
            )

    def test_probability_output_rejects_non_float32_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "dtype float32"):
            restore_probability_map(
                np.zeros(REALTIME_OUTPUT_SHAPE, dtype=np.float16),
                (128, 1280),
                PROBABILITY_OUTPUT_NAME,
            )

    def test_annotation_adds_rail_section_and_mask_overlay(self) -> None:
        frame = np.zeros((160, 200, 3), dtype=np.uint8)
        mask = np.zeros((160, 200), dtype=np.uint8)
        mask[130:140, 30:60] = 255
        result = CrackDetectionResult(
            mask=mask,
            boxes=[(30, 130, 30, 10)],
            crack_pixels=300,
            inspected_pixels=160 * 200,
            crack_ratio=300 / (160 * 200),
            detected=True,
            method="bgcrack-tensorrt/test",
            probability_threshold=0.5,
        )

        with patch("crack_detector.cv2.putText") as put_text:
            annotated = annotate_cracks(frame, result, zone_number=1)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertGreater(int(annotated[135, 40].sum()), 0)
        self.assertGreater(int(annotated[100, 40].sum()), 0)
        self.assertIn("RAIL SECTION 1", put_text.call_args.args[1])


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
    def __init__(self, output_name: str, extra_name: str | None = None) -> None:
        self.names = [output_name, INPUT_NAME]
        if extra_name is not None:
            self.names.append(extra_name)
        self.num_io_tensors = len(self.names)
        self.output_name = output_name
        self.shapes = {
            INPUT_NAME: REALTIME_INPUT_SHAPE,
            output_name: REALTIME_OUTPUT_SHAPE,
        }
        self.dtypes = {
            INPUT_NAME: np.float32,
            output_name: np.float32,
        }
        self.formats = {
            INPUT_NAME: _FakeTensorFormat.LINEAR,
            output_name: _FakeTensorFormat.LINEAR,
        }
        self.locations = {
            INPUT_NAME: _FakeTensorLocation.DEVICE,
            output_name: _FakeTensorLocation.DEVICE,
        }

    def get_tensor_name(self, index: int) -> str:
        return self.names[index]

    def get_tensor_mode(self, name: str) -> str:
        return (
            _FakeTensorIOMode.INPUT
            if name == INPUT_NAME
            else _FakeTensorIOMode.OUTPUT
        )

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str):
        return self.dtypes[name]

    def get_tensor_format(self, name: str) -> str:
        return self.formats[name]

    def get_tensor_location(self, name: str) -> str:
        return self.locations[name]


class CrackEngineContractTests(unittest.TestCase):
    def _detector_with_output(self, output_name: str, extra_name: str | None = None):
        detector = CrackDetector.__new__(CrackDetector)
        detector._trt = _FakeTensorRT()
        detector._engine = _FakeEngine(output_name, extra_name)
        detector.profile = REALTIME_PROFILE
        detector.input_shape = REALTIME_INPUT_SHAPE
        detector.output_shape = REALTIME_OUTPUT_SHAPE
        return detector

    def test_accepts_probability_output_contract(self) -> None:
        detector = self._detector_with_output(OUTPUT_NAME)

        self.assertEqual(detector._validate_engine_contract(), OUTPUT_NAME)

    def test_rejects_extra_engine_io(self) -> None:
        detector = self._detector_with_output(
            PROBABILITY_OUTPUT_NAME,
            "unexpected_output",
        )

        with self.assertRaisesRegex(RuntimeError, "I/O names must be exactly"):
            detector._validate_engine_contract()

    def test_rejects_an_unsupported_output_name(self) -> None:
        detector = self._detector_with_output("crack_logits")

        with self.assertRaisesRegex(RuntimeError, "I/O names must be exactly"):
            detector._validate_engine_contract()

    def test_rejects_an_output_marked_as_input(self) -> None:
        detector = self._detector_with_output(PROBABILITY_OUTPUT_NAME)
        detector._engine.get_tensor_mode = lambda name: _FakeTensorIOMode.INPUT

        with self.assertRaisesRegex(RuntimeError, "is not an engine output"):
            detector._validate_engine_contract()

    def test_rejects_wrong_static_input_and_output_shapes(self) -> None:
        invalid_shapes = (
            (INPUT_NAME, (1, 3, 127, 1280), "input shape"),
            (OUTPUT_NAME, (1, 1, 127, 1280), "output shape"),
        )
        for tensor_name, shape, message in invalid_shapes:
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector_with_output(OUTPUT_NAME)
                detector._engine.shapes[tensor_name] = shape
                with self.assertRaisesRegex(RuntimeError, message):
                    detector._validate_engine_contract()

    def test_rejects_non_float32_input_and_output_dtypes(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector_with_output(OUTPUT_NAME)
                detector._engine.dtypes[tensor_name] = np.float16
                with self.assertRaisesRegex(RuntimeError, "dtype must be float32"):
                    detector._validate_engine_contract()

    def test_rejects_non_linear_input_and_output_formats(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector_with_output(OUTPUT_NAME)
                detector._engine.formats[tensor_name] = "chw32"
                with self.assertRaisesRegex(RuntimeError, "LINEAR/CHW"):
                    detector._validate_engine_contract()

    def test_rejects_non_device_input_and_output_locations(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector_with_output(OUTPUT_NAME)
                detector._engine.locations[tensor_name] = "host"
                with self.assertRaisesRegex(RuntimeError, "GPU device"):
                    detector._validate_engine_contract()


if __name__ == "__main__":
    unittest.main()
