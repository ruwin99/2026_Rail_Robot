from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rust_detector import (
    CLASS_NAMES,
    MIN_BOX_AREA,
    STUDENT_PROFILE,
    DetectionResult,
    preprocess_frame,
)


INPUT_NAME = "images"
RUST_MAP_NAME = "rust_class_map"
LOGITS_NOT_NAN_NAME = "rust_logits_not_nan"
LOGITS_FINITE_ABS_NAME = "rust_logits_finite_abs"
RAW_LOGITS_NAME = "logits"
INPUT_SHAPE = (1, 3, 240, 1280)
OUTPUT_CONTRACTS = {
    RUST_MAP_NAME: ((1, 240, 1280), np.dtype(np.uint8)),
    LOGITS_NOT_NAN_NAME: ((1,), np.dtype(np.uint8)),
    LOGITS_FINITE_ABS_NAME: ((1,), np.dtype(np.uint8)),
}
TENSORRT_VERSION_PREFIX = "10.3."
METHOD_PREFIX = "deeplabv3plus-tensorrt/student/optimized-compact-v2/"


def result_from_postprocessed_outputs(
    outputs: dict[str, np.ndarray],
    method: str,
) -> DetectionResult:
    if set(outputs) != set(OUTPUT_CONTRACTS):
        raise ValueError("Optimized rust outputs do not match the strict contract.")
    for name, (shape, dtype) in OUTPUT_CONTRACTS.items():
        value = outputs[name]
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Optimized output {name!r} must be a NumPy array.")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"Optimized output {name!r} must be {dtype} {shape}, got "
                f"{value.dtype} {value.shape}."
            )

    not_nan = int(outputs[LOGITS_NOT_NAN_NAME][0])
    finite_abs = int(outputs[LOGITS_FINITE_ABS_NAME][0])
    if not_nan != 1 or finite_abs != 1:
        raise ValueError("Optimized raw rust logits contain NaN or infinity.")

    class_map = outputs[RUST_MAP_NAME][0]
    if np.any(class_map >= len(CLASS_NAMES)):
        raise ValueError("Rust class map contains an invalid class ID.")
    class_counts = np.bincount(class_map.ravel(), minlength=len(CLASS_NAMES))

    mask = np.where(class_map > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) >= MIN_BOX_AREA
    ]
    inspected_pixels = int(class_map.size)
    class_ratios = {
        class_name: float(class_counts[class_id] / inspected_pixels)
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    return DetectionResult(
        mask=mask,
        boxes=boxes,
        rust_ratio=float((inspected_pixels - class_counts[0]) / inspected_pixels),
        method=method,
        class_map=class_map.copy(),
        class_ratios=class_ratios,
    )


class OptimizedRustDetector:
    """TensorRT 10.3 runtime accepting only GPU-postprocessed rust outputs."""

    def __init__(self, engine_path: Path, expected_sha256: str) -> None:
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(f"TensorRT engine was not found: {self.engine_path}")
        self.method = METHOD_PREFIX
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
                    f"TensorRT {TENSORRT_VERSION_PREFIX}x is required; "
                    f"got {trt.__version__}."
                )
            self._trt = trt
            self._cudart = cudart
            logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(logger, "")
            self._runtime = trt.Runtime(logger)
            serialized_engine = self.engine_path.read_bytes()
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if self.engine_sha256 != expected_sha256:
                raise RuntimeError(
                    "Optimized rust engine SHA-256 changed: "
                    f"expected {expected_sha256}, got {self.engine_sha256}."
                )
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the optimized rust engine.")
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an optimized rust context.")

            input_nbytes = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
            self._host_input_pointer = self._cuda_call(
                cudart.cudaHostAlloc(input_nbytes, 0),
                "cudaHostAlloc(optimized rust input)",
            )
            self._host_input = self._host_array(
                self._host_input_pointer, INPUT_SHAPE, np.dtype(np.float32)
            )
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes), "cudaMalloc(optimized rust input)"
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(), "cudaStreamCreate(optimized rust)"
            )
            if not self._context.set_tensor_address(
                INPUT_NAME, int(self._device_input)
            ):
                raise RuntimeError("TensorRT rejected the optimized rust input address.")
            for name, (shape, dtype) in OUTPUT_CONTRACTS.items():
                nbytes = int(np.prod(shape)) * dtype.itemsize
                host_pointer = self._cuda_call(
                    cudart.cudaHostAlloc(nbytes, 0), f"cudaHostAlloc({name})"
                )
                device_pointer = self._cuda_call(
                    cudart.cudaMalloc(nbytes), f"cudaMalloc({name})"
                )
                self._host_output_pointers[name] = host_pointer
                self._host_outputs[name] = self._host_array(
                    host_pointer, shape, dtype
                )
                self._device_outputs[name] = device_pointer
                if not self._context.set_tensor_address(name, int(device_pointer)):
                    raise RuntimeError(f"TensorRT rejected output address {name!r}.")
            version = str(trt.__version__)
            self.method = f"{METHOD_PREFIX}trt-{version}/cuda:0"
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
                "Raw four-channel logits are forbidden in the optimized rust plan."
            )
        if tensor_names != expected_names:
            raise RuntimeError(
                "Optimized rust TensorRT I/O names must be exactly "
                f"{sorted(expected_names)}, got {sorted(tensor_names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError("Optimized rust images tensor is not an input.")
        if tuple(engine.get_tensor_shape(INPUT_NAME)) != INPUT_SHAPE:
            raise RuntimeError(f"Optimized rust input shape must be {INPUT_SHAPE}.")
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(INPUT_NAME)))
        if input_dtype != np.dtype(np.float32):
            raise RuntimeError("Optimized rust input dtype must be float32.")
        for name, (shape, dtype) in OUTPUT_CONTRACTS.items():
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                raise RuntimeError(f"Optimized rust tensor {name!r} is not an output.")
            actual_shape = tuple(engine.get_tensor_shape(name))
            actual_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if actual_shape != shape or actual_dtype != dtype:
                raise RuntimeError(
                    f"Optimized rust tensor {name!r} must be {dtype} {shape}, "
                    f"got {actual_dtype} {actual_shape}."
                )
        for name in expected_names:
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f"Tensor {name!r} must use LINEAR format.")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(f"Tensor {name!r} must be on the GPU device.")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._context is None or self._stream is None:
            raise RuntimeError("Optimized rust detector is closed.")
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
                "cudaMemcpyAsync(optimized rust input)",
            )
            if not self._context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError("TensorRT optimized rust execute_async_v3 returned false.")
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
                "cudaStreamSynchronize(optimized rust)",
            )
        except Exception as exc:
            raise RuntimeError(f"Optimized rust inference failed: {exc}") from exc
        try:
            return result_from_postprocessed_outputs(self._host_outputs, self.method)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Optimized rust output validation failed: {exc}") from exc

    def _cuda_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA status.")
        error = result[0]
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed with CUDA error {error}.")
        return None if len(result) == 1 else result[1]

    @staticmethod
    def _host_array(
        pointer: Any, shape: tuple[int, ...], dtype: np.dtype
    ) -> np.ndarray:
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)), ctypes.POINTER(ctypes.c_ubyte)
        )
        return (
            np.ctypeslib.as_array(byte_pointer, shape=(byte_count,))
            .view(dtype)
            .reshape(shape)
        )

    def close(self) -> None:
        cudart = self._cudart
        if cudart is not None and self._stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(self._stream),
                    "cudaStreamSynchronize(optimized rust close)",
                )
            except Exception:
                pass
        for pointer in [self._device_input, *self._device_outputs.values()]:
            if pointer is not None and cudart is not None:
                try:
                    self._cuda_call(cudart.cudaFree(pointer), "cudaFree(optimized rust)")
                except Exception:
                    pass
        if self._stream is not None and cudart is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamDestroy(self._stream),
                    "cudaStreamDestroy(optimized rust)",
                )
            except Exception:
                pass
        for pointer in [
            self._host_input_pointer,
            *self._host_output_pointers.values(),
        ]:
            if pointer is not None and cudart is not None:
                try:
                    self._cuda_call(
                        cudart.cudaFreeHost(pointer), "cudaFreeHost(optimized rust)"
                    )
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
