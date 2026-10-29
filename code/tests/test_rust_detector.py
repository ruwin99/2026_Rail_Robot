from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

import rust_detector as rust_detector_module  # noqa: E402
from rust_detector import (  # noqa: E402
    CLASS_NAMES,
    INPUT_NAME,
    OUTPUT_NAME,
    STUDENT_INPUT_SHAPE,
    STUDENT_OUTPUT_SHAPE,
    STUDENT_PROFILE,
    TEACHER_INPUT_SHAPE,
    TEACHER_OUTPUT_SHAPE,
    TEACHER_PROFILE,
    RustDetector,
    preprocess_frame,
    result_from_class_map,
    result_from_logits,
    restore_class_map,
    validate_native_frame,
)
from cuda_argmax import CudaArgmaxPostprocessor, _kernel_source  # noqa: E402


class _FakeCudaError:
    cudaSuccess = 0


class _FakeCudaRuntime:
    cudaError_t = _FakeCudaError


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
    num_io_tensors = 2

    def __init__(self) -> None:
        self.names = (INPUT_NAME, OUTPUT_NAME)
        self.modes = {
            INPUT_NAME: _FakeTensorIOMode.INPUT,
            OUTPUT_NAME: _FakeTensorIOMode.OUTPUT,
        }
        self.shapes = {
            INPUT_NAME: STUDENT_INPUT_SHAPE,
            OUTPUT_NAME: STUDENT_OUTPUT_SHAPE,
        }
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


class CudaResultTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = RustDetector.__new__(RustDetector)
        self.detector._cudart = _FakeCudaRuntime()

    def test_cuda_call_returns_allocated_pointer(self) -> None:
        self.assertEqual(
            self.detector._cuda_call((0, 1234), "cudaMalloc"),
            1234,
        )

    def test_cuda_call_rejects_runtime_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "CUDA error 2"):
            self.detector._cuda_call((2,), "cudaMalloc")

    def test_host_array_wraps_fp16_memory_without_copy(self) -> None:
        host_buffer = (ctypes.c_ubyte * 8)()

        array = self.detector._host_array(
            ctypes.addressof(host_buffer),
            (2, 2),
            np.dtype(np.float16),
        )
        array[1, 1] = 7.5

        wrapped = np.ctypeslib.as_array(host_buffer).view(np.float16).reshape(2, 2)
        self.assertEqual(wrapped[1, 1], 7.5)

    def test_detector_rejects_invalid_expected_engine_digest_before_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_sha256"):
            RustDetector(Path("missing.plan"), expected_sha256="invalid")

    def test_cuda_argmax_rejects_invalid_output_shape_before_importing_cuda(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch size one"):
            CudaArgmaxPostprocessor(_FakeCudaRuntime(), (2, 4, 240, 1280))

    def test_cuda_kernel_preserves_first_tie_and_checks_finite_values(self) -> None:
        source = _kernel_source(4, 240 * 1280)

        self.assertIn("candidate > best_value", source)
        self.assertNotIn("candidate >= best_value", source)
        self.assertIn("finite_float(best_value)", source)
        self.assertIn("finite_float(candidate)", source)
        self.assertIn("0x7f800000u", source)
        self.assertIn("atomicExch(invalid_flag, 1u)", source)


class RustEngineContractTests(unittest.TestCase):
    @staticmethod
    def _detector() -> RustDetector:
        detector = RustDetector.__new__(RustDetector)
        detector._trt = _FakeTensorRT()
        detector._engine = _FakeEngine()
        detector.profile = STUDENT_PROFILE
        detector.input_shape = STUDENT_INPUT_SHAPE
        detector.output_shape = STUDENT_OUTPUT_SHAPE
        return detector

    def test_accepts_native_student_contract(self) -> None:
        self._detector()._validate_engine_contract()

    def test_rejects_wrong_static_input_and_output_shapes(self) -> None:
        invalid_shapes = (
            (INPUT_NAME, (1, 3, 239, 1280), "input shape"),
            (OUTPUT_NAME, (1, 4, 239, 1280), "output shape"),
        )
        for tensor_name, shape, message in invalid_shapes:
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector()
                detector._engine.shapes[tensor_name] = shape
                with self.assertRaisesRegex(RuntimeError, message):
                    detector._validate_engine_contract()

    def test_rejects_non_float32_input_and_output_dtypes(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector()
                detector._engine.dtypes[tensor_name] = np.float16
                with self.assertRaisesRegex(RuntimeError, "dtype must be float32"):
                    detector._validate_engine_contract()

    def test_rejects_non_linear_input_and_output_formats(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector()
                detector._engine.formats[tensor_name] = "chw32"
                with self.assertRaisesRegex(RuntimeError, "LINEAR/CHW"):
                    detector._validate_engine_contract()

    def test_rejects_non_device_input_and_output_locations(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            with self.subTest(tensor_name=tensor_name):
                detector = self._detector()
                detector._engine.locations[tensor_name] = "host"
                with self.assertRaisesRegex(RuntimeError, "GPU device"):
                    detector._validate_engine_contract()


class TensorRTPreprocessingTests(unittest.TestCase):
    def test_preprocessing_converts_bgr_to_normalized_rgb(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        frame[0, 0] = (0, 127, 255)

        tensor, _ = preprocess_frame(frame)

        expected_rgb = np.array(
            [
                (1.0 - 0.485) / 0.229,
                ((127.0 / 255.0) - 0.456) / 0.224,
                (0.0 - 0.406) / 0.225,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected_rgb, rtol=1e-6)

    def test_teacher_preprocessing_preserves_bgr_zero_to_255_values(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[0, 0] = (5, 127, 250)

        tensor, _ = preprocess_frame(frame, TEACHER_PROFILE)

        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(
            tensor[0, :, 0, 0],
            np.array([5.0, 127.0, 250.0], dtype=np.float32),
        )

    def test_unknown_preprocessing_profile_is_rejected(self) -> None:
        frame = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Unknown TensorRT"):
            preprocess_frame(frame, "unknown")

        self.assertEqual(STUDENT_PROFILE, "student")

    def test_student_native_shape_has_no_resize_or_padding(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)

        tensor, transform = preprocess_frame(frame)

        self.assertEqual(tensor.shape, STUDENT_INPUT_SHAPE)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(transform.resized_width, 1280)
        self.assertEqual(transform.resized_height, 240)
        self.assertEqual(transform.left, 0)
        self.assertEqual(transform.top, 0)

    def test_teacher_native_shape_has_no_resize_or_padding(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tensor, transform = preprocess_frame(frame, TEACHER_PROFILE)

        self.assertEqual(tensor.shape, TEACHER_INPUT_SHAPE)
        self.assertEqual(transform.resized_width, 1280)
        self.assertEqual(transform.resized_height, 720)
        self.assertEqual((transform.top, transform.left), (0, 0))

    def test_native_profiles_reject_mismatched_shapes_instead_of_resizing(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not resize"):
            preprocess_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "do not resize"):
            preprocess_frame(
                np.zeros((240, 1280, 3), dtype=np.uint8),
                TEACHER_PROFILE,
            )

    def test_rejects_non_uint8_camera_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite uint8"):
            preprocess_frame(np.zeros((240, 1280, 3), dtype=np.float32))

    def test_gpu_validation_preserves_native_student_contract(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)

        transform = validate_native_frame(frame, STUDENT_PROFILE)

        self.assertEqual(transform.original_height, 240)
        self.assertEqual(transform.original_width, 1280)
        with self.assertRaisesRegex(ValueError, "finite uint8"):
            validate_native_frame(frame.astype(np.float32), STUDENT_PROFILE)
        nan_frame = frame.astype(np.float32)
        nan_frame[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite uint8"):
            validate_native_frame(nan_frame, STUDENT_PROFILE)
        with self.assertRaisesRegex(ValueError, "do not resize"):
            validate_native_frame(frame[:-1], STUDENT_PROFILE)

    def test_gpu_argmax_path_uses_cuda_preprocess_without_cpu_tensor(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        class_map = np.zeros((240, 1280), dtype=np.uint8)
        expected_result = object()
        detector = RustDetector.__new__(RustDetector)
        detector.profile = STUDENT_PROFILE
        detector.stage_timing = False
        detector.last_stage_timing = None
        detector.method = "test/gpu-preprocess/gpu-argmax"
        detector._context = Mock()
        detector._context.execute_async_v3.return_value = True
        detector._stream = 17
        detector._device_input = 100
        detector._device_output = 200
        detector._gpu_preprocessor = Mock()
        detector._gpu_postprocessor = Mock()
        detector._gpu_postprocessor.process.return_value = class_map
        detector._timing_events = ()

        with (
            patch.object(
                rust_detector_module,
                "preprocess_frame",
                side_effect=AssertionError("CPU preprocessing must not run"),
            ),
            patch.object(
                rust_detector_module,
                "result_from_class_map",
                return_value=expected_result,
            ),
        ):
            result = detector.detect(frame)

        self.assertIs(result, expected_result)
        detector._gpu_preprocessor.stage.assert_called_once_with(frame)
        detector._gpu_preprocessor.enqueue.assert_called_once_with(100, 17)
        detector._gpu_postprocessor.process.assert_called_once_with(200, 17)

    def test_non_good_classes_form_corrosion_mask(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        _, transform = preprocess_frame(frame)
        logits = np.zeros(STUDENT_OUTPUT_SHAPE, dtype=np.float32)
        logits[:, 2, :, :] = 1.0

        result = result_from_logits(
            logits,
            transform,
            "deeplabv3plus-tensorrt/test",
        )

        self.assertEqual(result.mask.shape, frame.shape[:2])
        self.assertTrue(np.all(result.mask == 255))
        self.assertEqual(result.rust_ratio, 1.0)
        self.assertEqual(result.class_ratios[CLASS_NAMES[2]], 1.0)
        self.assertEqual(result.boxes, [(0, 0, 1280, 240)])

    def test_class_map_result_matches_finite_logits_with_first_tie(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        _, transform = preprocess_frame(frame)
        logits = np.zeros(STUDENT_OUTPUT_SHAPE, dtype=np.float32)
        logits[:, 1, :, :] = 2.0
        logits[:, 2, 0, 0] = 2.0
        logits[:, 3, 0, 1] = 3.0

        raw_result = result_from_logits(logits, transform, "raw")
        map_result = result_from_class_map(
            np.argmax(logits[0], axis=0).astype(np.uint8),
            transform,
            "gpu",
        )

        np.testing.assert_array_equal(map_result.class_map, raw_result.class_map)
        np.testing.assert_array_equal(map_result.mask, raw_result.mask)
        self.assertEqual(map_result.boxes, raw_result.boxes)
        self.assertEqual(map_result.class_ratios, raw_result.class_ratios)
        self.assertEqual(map_result.rust_ratio, raw_result.rust_ratio)

    def test_class_map_result_rejects_wrong_dtype_shape_and_class_id(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        _, transform = preprocess_frame(frame)
        invalid_maps = (
            (np.zeros((240, 1280), dtype=np.int32), "dtype uint8"),
            (np.zeros((239, 1280), dtype=np.uint8), "must have shape"),
            (np.full((240, 1280), 4, dtype=np.uint8), "invalid class ID"),
        )
        for class_map, message in invalid_maps:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    result_from_class_map(class_map, transform, "gpu")

    def test_non_finite_logits_are_rejected(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        _, transform = preprocess_frame(frame)
        logits = np.zeros(STUDENT_OUTPUT_SHAPE, dtype=np.float32)
        logits[0, 0, 0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            result_from_logits(logits, transform, "test")

    def test_rejects_non_float32_logits(self) -> None:
        frame = np.zeros((240, 1280, 3), dtype=np.uint8)
        _, transform = preprocess_frame(frame)

        with self.assertRaisesRegex(ValueError, "dtype float32"):
            result_from_logits(
                np.zeros(STUDENT_OUTPUT_SHAPE, dtype=np.float16),
                transform,
                "test",
            )


if __name__ == "__main__":
    unittest.main()
