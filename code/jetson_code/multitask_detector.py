from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crack_detector import CrackDetectionResult
from rust_detector import (
    CLASS_NAMES,
    MIN_BOX_AREA,
    DetectionResult,
    STUDENT_PROFILE,
    preprocess_frame,
)


INPUT_NAME = "images"
RUST_MAP_NAME = "rust_class_map"
RUST_COUNTS_NAME = "rust_class_counts"
RUST_BLOCKED_NAME = "rust_poor_severe"
CRACK_MAP_NAME = "crack_candidate_map"
CRACK_PIXELS_NAME = "crack_candidate_pixels"
CRACK_THRESHOLD_NAME = "crack_probability_threshold"
OUTPUTS_FINITE_NAME = "multitask_outputs_finite"
RAW_LOGITS_NAME = "multitask_logits"

INPUT_SHAPE = (1, 3, 240, 1280)
OUTPUT_CONTRACTS = {
    RUST_MAP_NAME: ((1, 240, 1280), np.dtype(np.uint8)),
    RUST_COUNTS_NAME: ((4,), np.dtype(np.int32)),
    RUST_BLOCKED_NAME: ((1,), np.dtype(np.uint8)),
    CRACK_MAP_NAME: ((1, 128, 1280), np.dtype(np.uint8)),
    CRACK_PIXELS_NAME: ((1,), np.dtype(np.int32)),
    CRACK_THRESHOLD_NAME: ((1,), np.dtype(np.float32)),
    OUTPUTS_FINITE_NAME: ((1,), np.dtype(np.uint8)),
}
TENSORRT_VERSION_PREFIX = "10.3."
RUST_METHOD_PREFIX = "multitask-segmentation-tensorrt/realtime/rust/optimized/"
CRACK_METHOD_PREFIX = "multitask-segmentation-tensorrt/realtime/crack/optimized/"


def results_from_postprocessed_outputs(
    outputs: dict[str, np.ndarray],
    rust_method: str,
    crack_method: str,
    probability_threshold: float,
    min_component_pixels: int,
) -> tuple[DetectionResult, CrackDetectionResult]:
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be between zero and one.")
    if min_component_pixels <= 0:
        raise ValueError("min_component_pixels must be positive.")
    if set(outputs) != set(OUTPUT_CONTRACTS):
        raise ValueError("Optimized multitask outputs do not match the strict contract.")
    for name, (shape, dtype) in OUTPUT_CONTRACTS.items():
        value = outputs[name]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Optimized output {name!r} must be a NumPy array.")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"Optimized output {name!r} must be {dtype} {shape}, got "
                f"{value.dtype} {value.shape}."
            )

    outputs_finite = int(outputs[OUTPUTS_FINITE_NAME][0])
    if outputs_finite != 1:
        raise ValueError(
            "Optimized multitask input or raw logits contain NaN or infinity."
        )

    class_map = outputs[RUST_MAP_NAME][0]
    if not np.isin(class_map, (0, 1, 2, 3)).all():
        raise ValueError("Rust class map contains an invalid class ID.")
    class_counts = outputs[RUST_COUNTS_NAME].astype(np.int64, copy=False)
    measured_counts = np.bincount(class_map.ravel(), minlength=4)
    if np.any(class_counts < 0) or not np.array_equal(class_counts, measured_counts):
        raise ValueError("Rust class counts do not match the transferred class map.")
    blocked = int(outputs[RUST_BLOCKED_NAME][0])
    expected_blocked = int(class_counts[2] + class_counts[3] > 0)
    if blocked not in (0, 1) or blocked != expected_blocked:
        raise ValueError("Rust Poor/Severe control flag does not match class counts.")

    rust_mask = np.where(class_map > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rust_boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) >= MIN_BOX_AREA
    ]
    inspected_pixels = int(class_map.size)
    class_ratios = {
        name: float(class_counts[index] / inspected_pixels)
        for index, name in enumerate(CLASS_NAMES)
    }
    rust_result = DetectionResult(
        mask=rust_mask,
        boxes=rust_boxes,
        rust_ratio=float((inspected_pixels - class_counts[0]) / inspected_pixels),
        method=rust_method,
        class_map=class_map.copy(),
        class_ratios=class_ratios,
    )

    crack_candidate = outputs[CRACK_MAP_NAME][0]
    if not np.isin(crack_candidate, (0, 1)).all():
        raise ValueError("Crack candidate map must contain only zero and one.")
    candidate_pixels = int(outputs[CRACK_PIXELS_NAME][0])
    measured_candidate_pixels = int(np.count_nonzero(crack_candidate))
    if candidate_pixels < 0 or candidate_pixels != measured_candidate_pixels:
        raise ValueError("Crack candidate count does not match the transferred map.")
    embedded_threshold = float(outputs[CRACK_THRESHOLD_NAME][0])
    if not np.isfinite(embedded_threshold):
        raise ValueError("Embedded crack probability threshold must be finite.")
    if not np.isclose(embedded_threshold, probability_threshold, rtol=0.0, atol=1e-6):
        raise ValueError(
            "Embedded crack probability threshold does not match the runtime "
            f"setting: engine={embedded_threshold}, runtime={probability_threshold}."
        )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        crack_candidate, connectivity=8
    )
    filtered = np.zeros(crack_candidate.shape, dtype=np.uint8)
    crack_boxes: list[tuple[int, int, int, int]] = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        filtered[labels == component_id] = 255
        crack_boxes.append(
            (
                int(stats[component_id, cv2.CC_STAT_LEFT]),
                int(stats[component_id, cv2.CC_STAT_TOP]),
                int(stats[component_id, cv2.CC_STAT_WIDTH]),
                int(stats[component_id, cv2.CC_STAT_HEIGHT]),
            )
        )
    crack_pixels = int(np.count_nonzero(filtered))
    crack_inspected_pixels = int(filtered.size)
    crack_result = CrackDetectionResult(
        mask=filtered,
        boxes=crack_boxes,
        crack_pixels=crack_pixels,
        inspected_pixels=crack_inspected_pixels,
        crack_ratio=crack_pixels / crack_inspected_pixels,
        detected=crack_pixels > 0,
        method=crack_method,
        probability_threshold=float(probability_threshold),
    )
    return rust_result, crack_result


class OptimizedMultitaskDetector:
    """TensorRT runtime accepting only a GPU-postprocessed multitask plan."""

    def __init__(
        self,
        engine_path: Path,
        expected_sha256: str,
        probability_threshold: float,
        min_component_pixels: int,
    ) -> None:
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        if not 0.0 < probability_threshold < 1.0:
            raise ValueError("probability_threshold must be between zero and one.")
        if min_component_pixels <= 0:
            raise ValueError("min_component_pixels must be positive.")

        self.engine_path = Path(engine_path).expanduser().resolve()
        self.probability_threshold = float(probability_threshold)
        self.min_component_pixels = int(min_component_pixels)
        self.method = RUST_METHOD_PREFIX
        self.crack_method = CRACK_METHOD_PREFIX
        self.engine_sha256 = ""
        self._trt: Any = None
        self._cudart: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._host_input_pointer: Any = None
        self._host_input: np.ndarray | None = None
        self._device_input: Any = None
        self._host_output_pointers: dict[str, Any] = {}
        self._host_outputs: dict[str, np.ndarray] = {}
        self._device_outputs: dict[str, Any] = {}

        try:
            import tensorrt as trt
            from cuda import cudart

            if not str(trt.__version__).startswith(TENSORRT_VERSION_PREFIX):
                raise RuntimeError(
                    f"TensorRT {TENSORRT_VERSION_PREFIX}x is required; got {trt.__version__}."
                )
            self._trt = trt
            self._cudart = cudart
            logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(logger)
            serialized_engine = self.engine_path.read_bytes()
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if self.engine_sha256 != expected_sha256:
                raise RuntimeError(
                    "Optimized multitask engine SHA-256 changed: "
                    f"expected {expected_sha256}, got {self.engine_sha256}."
                )
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the multitask engine.")
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create a multitask context.")

            input_nbytes = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
            self._host_input_pointer = self._cuda_call(
                cudart.cudaHostAlloc(input_nbytes, 0), "cudaHostAlloc(multitask input)"
            )
            self._host_input = self._host_array(
                self._host_input_pointer, INPUT_SHAPE, np.dtype(np.float32)
            )
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes), "cudaMalloc(multitask input)"
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(), "cudaStreamCreate(multitask)"
            )
            if not self._context.set_tensor_address(INPUT_NAME, int(self._device_input)):
                raise RuntimeError("TensorRT rejected the multitask input address.")
            for name, (shape, dtype) in OUTPUT_CONTRACTS.items():
                nbytes = int(np.prod(shape)) * dtype.itemsize
                host_pointer = self._cuda_call(
                    cudart.cudaHostAlloc(nbytes, 0), f"cudaHostAlloc({name})"
                )
                device_pointer = self._cuda_call(
                    cudart.cudaMalloc(nbytes), f"cudaMalloc({name})"
                )
                self._host_output_pointers[name] = host_pointer
                self._host_outputs[name] = self._host_array(host_pointer, shape, dtype)
                self._device_outputs[name] = device_pointer
                if not self._context.set_tensor_address(name, int(device_pointer)):
                    raise RuntimeError(f"TensorRT rejected output address {name!r}.")
            version = str(trt.__version__)
            self.method = f"{RUST_METHOD_PREFIX}trt-{version}/cuda:0"
            self.crack_method = f"{CRACK_METHOD_PREFIX}trt-{version}/cuda:0"
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> None:
        trt = self._trt
        engine = self._engine
        expected_names = {INPUT_NAME, *OUTPUT_CONTRACTS}
        tensor_names = {
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        }
        if RAW_LOGITS_NAME in tensor_names:
            raise RuntimeError(
                "Raw 5-channel logits are forbidden in the optimized multitask plan."
            )
        if tensor_names != expected_names:
            raise RuntimeError(
                "Optimized multitask TensorRT I/O names must be exactly "
                f"{sorted(expected_names)}, got {sorted(tensor_names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError("Multitask images tensor is not an input.")
        if tuple(engine.get_tensor_shape(INPUT_NAME)) != INPUT_SHAPE:
            raise RuntimeError(f"Multitask input shape must be {INPUT_SHAPE}.")
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(INPUT_NAME)))
        if input_dtype != np.dtype(np.float32):
            raise RuntimeError("Multitask input dtype must be float32.")
        for name, (shape, dtype) in OUTPUT_CONTRACTS.items():
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                raise RuntimeError(f"Optimized tensor {name!r} is not an output.")
            actual_shape = tuple(engine.get_tensor_shape(name))
            actual_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if actual_shape != shape or actual_dtype != dtype:
                raise RuntimeError(
                    f"Optimized tensor {name!r} must be {dtype} {shape}, got "
                    f"{actual_dtype} {actual_shape}."
                )
        for name in expected_names:
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f"Tensor {name!r} must use LINEAR format.")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(f"Tensor {name!r} must be on the GPU device.")

    def detect(
        self, frame: np.ndarray
    ) -> tuple[DetectionResult, CrackDetectionResult]:
        if self._context is None or self._stream is None:
            raise RuntimeError("Optimized multitask detector is closed.")
        tensor, _ = preprocess_frame(frame, STUDENT_PROFILE)
        try:
            np.copyto(self._host_input, tensor, casting="no")
            self._cuda_call(
                self._cudart.cudaMemcpyAsync(
                    self._device_input,
                    self._host_input.ctypes.data,
                    self._host_input.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self._stream,
                ),
                "cudaMemcpyAsync(multitask input)",
            )
            if not self._context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError("TensorRT multitask execute_async_v3 returned false.")
            for name in OUTPUT_CONTRACTS:
                host = self._host_outputs[name]
                self._cuda_call(
                    self._cudart.cudaMemcpyAsync(
                        host.ctypes.data,
                        self._device_outputs[name],
                        host.nbytes,
                        self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self._stream,
                    ),
                    f"cudaMemcpyAsync({name})",
                )
            self._cuda_call(
                self._cudart.cudaStreamSynchronize(self._stream),
                "cudaStreamSynchronize(multitask)",
            )
        except Exception as exc:
            raise RuntimeError(f"Optimized multitask inference failed: {exc}") from exc
        try:
            return results_from_postprocessed_outputs(
                self._host_outputs,
                self.method,
                self.crack_method,
                self.probability_threshold,
                self.min_component_pixels,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Optimized multitask output validation failed: {exc}") from exc

    def _cuda_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA status.")
        error = result[0]
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed with CUDA error {error}.")
        return None if len(result) == 1 else result[1]

    @staticmethod
    def _host_array(pointer: Any, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)), ctypes.POINTER(ctypes.c_ubyte)
        )
        return np.ctypeslib.as_array(byte_pointer, shape=(byte_count,)).view(dtype).reshape(shape)

    def close(self) -> None:
        cudart = self._cudart
        if cudart is not None and self._stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(self._stream),
                    "cudaStreamSynchronize(multitask close)",
                )
            except Exception:
                pass
        for pointer in [self._device_input, *self._device_outputs.values()]:
            if pointer is not None and cudart is not None:
                try:
                    self._cuda_call(cudart.cudaFree(pointer), "cudaFree(multitask)")
                except Exception:
                    pass
        if self._stream is not None and cudart is not None:
            try:
                self._cuda_call(cudart.cudaStreamDestroy(self._stream), "cudaStreamDestroy")
            except Exception:
                pass
        for pointer in [self._host_input_pointer, *self._host_output_pointers.values()]:
            if pointer is not None and cudart is not None:
                try:
                    self._cuda_call(cudart.cudaFreeHost(pointer), "cudaFreeHost(multitask)")
                except Exception:
                    pass
        self._context = None
        self._engine = None
        self._runtime = None
        self._stream = None
        self._host_input_pointer = None
        self._host_input = None
        self._device_input = None
        self._host_output_pointers = {}
        self._host_outputs = {}
        self._device_outputs = {}
        self._cudart = None
