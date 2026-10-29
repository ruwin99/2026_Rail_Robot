from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


INPUT_NAME = "images"
OUTPUT_NAME = "crack_probability"
PROBABILITY_OUTPUT_NAME = OUTPUT_NAME
CAPTURE_PROFILE = "capture"
REALTIME_PROFILE = "realtime"
MODEL_PROFILES = (CAPTURE_PROFILE, REALTIME_PROFILE)
CAPTURE_INPUT_SHAPE = (1, 3, 720, 1280)
CAPTURE_OUTPUT_SHAPE = (1, 1, 720, 1280)
REALTIME_INPUT_SHAPE = (1, 3, 128, 1280)
REALTIME_OUTPUT_SHAPE = (1, 1, 128, 1280)
PROFILE_SHAPES = {
    CAPTURE_PROFILE: (CAPTURE_INPUT_SHAPE, CAPTURE_OUTPUT_SHAPE),
    REALTIME_PROFILE: (REALTIME_INPUT_SHAPE, REALTIME_OUTPUT_SHAPE),
}
# Backwards-compatible names identify the realtime contract.
INPUT_SHAPE = REALTIME_INPUT_SHAPE
OUTPUT_SHAPE = REALTIME_OUTPUT_SHAPE
TENSORRT_VERSION_PREFIX = "10.3."
DETECTOR_PREFIX = "bgcrack-tensorrt/"
CAPTURE_DETECTOR_PREFIX = f"{DETECTOR_PREFIX}{CAPTURE_PROFILE}/"
REALTIME_DETECTOR_PREFIX = f"{DETECTOR_PREFIX}{REALTIME_PROFILE}/"
DEFAULT_PROBABILITY_THRESHOLD = 0.5
DEFAULT_MIN_COMPONENT_PIXELS = 20
CRACK_COLOR_BGR = (255, 0, 255)


@dataclass
class CrackDetectionResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    crack_pixels: int
    inspected_pixels: int
    crack_ratio: float
    detected: bool
    method: str
    probability_threshold: float
    status: str = "ready"


def preprocess_frame(
    frame: np.ndarray,
    profile: str = REALTIME_PROFILE,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Apply a native static-shape BGCrack input contract."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be an HxWx3 NumPy array.")
    if frame.dtype != np.uint8 or not np.all(np.isfinite(frame)):
        raise ValueError("Camera frame must contain finite uint8 BGR pixels.")
    if profile not in MODEL_PROFILES:
        raise ValueError(f"Unknown crack TensorRT profile: {profile}")
    original_height, original_width = frame.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError("Camera frame must not be empty.")

    input_shape, _ = PROFILE_SHAPES[profile]
    target_height, target_width = input_shape[2:]
    if (original_height, original_width) != (target_height, target_width):
        raise ValueError(
            f"Crack TensorRT {profile} input frame must be "
            f"{target_width}x{target_height}; got "
            f"{original_width}x{original_height}. Native engines do not resize."
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = rgb / 127.5 - 1.0
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
    return tensor, (original_height, original_width)


def restore_probability_map(
    output: np.ndarray,
    original_shape: tuple[int, int],
    output_name: str = OUTPUT_NAME,
) -> np.ndarray:
    if output_name != OUTPUT_NAME:
        raise ValueError(f"Unknown crack TensorRT output name: {output_name!r}.")

    output = np.asarray(output)
    if output.dtype != np.float32:
        raise ValueError(
            f"Crack TensorRT output must have dtype float32, got {output.dtype}."
        )
    if (
        not isinstance(original_shape, tuple)
        or len(original_shape) != 2
        or any(not isinstance(value, int) for value in original_shape)
        or any(value <= 0 for value in original_shape)
    ):
        raise ValueError("original_shape must contain positive integer height and width.")
    expected_shape = (1, 1, *original_shape)
    if output.shape != expected_shape:
        raise ValueError(
            f"TensorRT output must have shape {expected_shape}, got {output.shape}."
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Crack TensorRT output contains NaN or infinity.")

    output_float = output[0, 0].astype(np.float32, copy=False)
    if np.any(output_float < 0.0) or np.any(output_float > 1.0):
        raise ValueError(
            "Crack probability output values must be between zero and one."
        )
    return output_float


def result_from_output(
    output: np.ndarray,
    original_shape: tuple[int, int],
    method: str,
    output_name: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
) -> CrackDetectionResult:
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be between zero and one.")
    if (
        not isinstance(min_component_pixels, int)
        or isinstance(min_component_pixels, bool)
        or min_component_pixels <= 0
    ):
        raise ValueError("min_component_pixels must be a positive integer.")

    probability = restore_probability_map(output, original_shape, output_name)
    candidate = (probability >= probability_threshold).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    filtered = np.zeros(candidate.shape, dtype=np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        filtered[labels == component_id] = 255
        boxes.append(
            (
                int(stats[component_id, cv2.CC_STAT_LEFT]),
                int(stats[component_id, cv2.CC_STAT_TOP]),
                int(stats[component_id, cv2.CC_STAT_WIDTH]),
                int(stats[component_id, cv2.CC_STAT_HEIGHT]),
            )
        )

    crack_pixels = int(np.count_nonzero(filtered))
    inspected_pixels = int(filtered.size)
    return CrackDetectionResult(
        mask=filtered,
        boxes=boxes,
        crack_pixels=crack_pixels,
        inspected_pixels=inspected_pixels,
        crack_ratio=crack_pixels / inspected_pixels,
        detected=crack_pixels > 0,
        method=str(method),
        probability_threshold=float(probability_threshold),
    )


def result_from_logits(
    logits: np.ndarray,
    original_shape: tuple[int, int],
    method: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
) -> CrackDetectionResult:
    """Compatibility wrapper; native models already output probabilities."""

    return result_from_output(
        logits,
        original_shape,
        method,
        PROBABILITY_OUTPUT_NAME,
        probability_threshold,
        min_component_pixels,
    )


class CrackDetector:
    """Synchronous TensorRT runtime for a BGCrack binary segmentation model."""

    def __init__(
        self,
        engine_path: Path,
        profile: str = REALTIME_PROFILE,
        expected_sha256: str | None = None,
        probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
        min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
    ) -> None:
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Unknown crack TensorRT profile: {profile}")
        if not 0.0 < probability_threshold < 1.0:
            raise ValueError("probability_threshold must be between zero and one.")
        if (
            not isinstance(min_component_pixels, int)
            or isinstance(min_component_pixels, bool)
            or min_component_pixels <= 0
        ):
            raise ValueError("min_component_pixels must be a positive integer.")
        if expected_sha256 is not None:
            expected_sha256 = str(expected_sha256).lower()
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            ):
                raise ValueError(
                    "expected_sha256 must be a 64-character hexadecimal digest."
                )

        self.profile = profile
        self.input_shape, self.output_shape = PROFILE_SHAPES[profile]
        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(f"Crack TensorRT engine was not found: {self.engine_path}")
        self.probability_threshold = float(probability_threshold)
        self.min_component_pixels = min_component_pixels
        self.engine_sha256: str | None = None
        self.method = f"{DETECTOR_PREFIX}{profile}/cuda:0"

        self._trt: Any = None
        self._cudart: Any = None
        self._logger: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._host_input_pointer: Any = None
        self._host_output_pointer: Any = None
        self._host_input: np.ndarray | None = None
        self._host_output: np.ndarray | None = None
        self._device_input: Any = None
        self._device_output: Any = None
        self._output_name: str | None = None

        try:
            import tensorrt as trt
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT and NVIDIA CUDA Python are required. "
                "Run scripts/requirement.sh first."
            ) from exc

        self._trt = trt
        self._cudart = cudart
        try:
            tensorrt_version = str(getattr(trt, "__version__", ""))
            if not tensorrt_version.startswith(TENSORRT_VERSION_PREFIX):
                raise RuntimeError(
                    "This plan requires TensorRT 10.3.x, got "
                    f"{tensorrt_version or 'an unknown version'}. Rebuild the plan "
                    "after changing TensorRT."
                )
            self.method = (
                f"{DETECTOR_PREFIX}{profile}/trt-{tensorrt_version}/cuda:0"
            )
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            try:
                serialized_engine = self.engine_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read crack TensorRT engine: {self.engine_path}"
                ) from exc
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if expected_sha256 is not None and self.engine_sha256 != expected_sha256:
                raise RuntimeError(
                    "Crack TensorRT engine changed after preflight SHA-256 "
                    f"validation: expected {expected_sha256}, got "
                    f"{self.engine_sha256}."
                )
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT could not deserialize the crack engine. Verify the "
                    "Jetson, TensorRT version, and engine SHA-256."
                )
            self._output_name = self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context.")

            input_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(INPUT_NAME)))
            output_dtype = np.dtype(
                trt.nptype(self._engine.get_tensor_dtype(self._output_name))
            )
            if input_dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"TensorRT input dtype must be float32, got {input_dtype}."
                )
            if output_dtype != np.dtype(np.float32):
                raise RuntimeError(
                    "TensorRT crack output dtype must be float32, got "
                    f"{output_dtype}."
                )

            input_nbytes = int(np.prod(self.input_shape)) * input_dtype.itemsize
            output_nbytes = int(np.prod(self.output_shape)) * output_dtype.itemsize
            self._host_input_pointer = self._cuda_call(
                cudart.cudaHostAlloc(input_nbytes, 0),
                "cudaHostAlloc(crack input)",
            )
            self._host_output_pointer = self._cuda_call(
                cudart.cudaHostAlloc(output_nbytes, 0),
                "cudaHostAlloc(crack output)",
            )
            self._host_input = self._host_array(
                self._host_input_pointer,
                self.input_shape,
                input_dtype,
            )
            self._host_output = self._host_array(
                self._host_output_pointer,
                self.output_shape,
                output_dtype,
            )
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes),
                "cudaMalloc(crack input)",
            )
            self._device_output = self._cuda_call(
                cudart.cudaMalloc(output_nbytes),
                "cudaMalloc(crack output)",
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(),
                "cudaStreamCreate(crack)",
            )
            if not self._context.set_tensor_address(
                INPUT_NAME, int(self._device_input)
            ):
                raise RuntimeError("TensorRT rejected the crack input buffer address.")
            if not self._context.set_tensor_address(
                self._output_name, int(self._device_output)
            ):
                raise RuntimeError("TensorRT rejected the crack output buffer address.")
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> str:
        trt = self._trt
        engine = self._engine
        tensor_names = [
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        ]
        if (
            len(tensor_names) != 2
            or len(set(tensor_names)) != 2
            or INPUT_NAME not in tensor_names
            or OUTPUT_NAME not in tensor_names
        ):
            raise RuntimeError(
                "Crack TensorRT engine I/O names must be exactly "
                f"{INPUT_NAME!r} and {OUTPUT_NAME!r}, got "
                f"{sorted(tensor_names)}."
            )
        output_name = OUTPUT_NAME
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Tensor {INPUT_NAME!r} is not an engine input.")
        if engine.get_tensor_mode(output_name) != trt.TensorIOMode.OUTPUT:
            raise RuntimeError(f"Tensor {output_name!r} is not an engine output.")
        input_shape = tuple(engine.get_tensor_shape(INPUT_NAME))
        output_shape = tuple(engine.get_tensor_shape(output_name))
        if input_shape != self.input_shape:
            raise RuntimeError(
                f"TensorRT crack {self.profile} input shape must be "
                f"{self.input_shape}, got {input_shape}."
            )
        if output_shape != self.output_shape:
            raise RuntimeError(
                f"TensorRT crack {self.profile} output shape must be "
                f"{self.output_shape}, got {output_shape}."
            )
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(INPUT_NAME)))
        output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(output_name)))
        if input_dtype != np.dtype(np.float32):
            raise RuntimeError(
                f"TensorRT input dtype must be float32, got {input_dtype}."
            )
        if output_dtype != np.dtype(np.float32):
            raise RuntimeError(
                "TensorRT crack output dtype must be float32, got "
                f"{output_dtype}."
            )
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            if engine.get_tensor_format(tensor_name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(
                    f"Tensor {tensor_name!r} must use LINEAR/CHW format."
                )
            if engine.get_tensor_location(tensor_name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(
                    f"Tensor {tensor_name!r} must be located on the GPU device."
                )
        return output_name

    def _cuda_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA status.")
        error = result[0]
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed with CUDA error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    @staticmethod
    def _host_array(
        pointer: Any,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
    ) -> np.ndarray:
        dtype = np.dtype(dtype)
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)),
            ctypes.POINTER(ctypes.c_ubyte),
        )
        byte_array = np.ctypeslib.as_array(byte_pointer, shape=(byte_count,))
        return byte_array.view(dtype).reshape(shape)

    def detect(self, frame: np.ndarray) -> CrackDetectionResult:
        if self._context is None or self._stream is None:
            raise RuntimeError("Crack TensorRT detector is closed.")
        tensor, original_shape = preprocess_frame(frame, self.profile)
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
                "cudaMemcpyAsync(crack input)",
            )
            executed = self._context.execute_async_v3(
                stream_handle=int(self._stream)
            )
            if not executed:
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
            self._cuda_call(
                self._cudart.cudaMemcpyAsync(
                    self._host_output.ctypes.data,
                    self._device_output,
                    self._host_output.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self._stream,
                ),
                "cudaMemcpyAsync(crack output)",
            )
            self._cuda_call(
                self._cudart.cudaStreamSynchronize(self._stream),
                "cudaStreamSynchronize(crack)",
            )
        except Exception as exc:
            raise RuntimeError(f"Crack TensorRT inference failed: {exc}") from exc

        try:
            return result_from_output(
                self._host_output.reshape(self.output_shape),
                original_shape,
                self.method,
                self._output_name,
                self.probability_threshold,
                self.min_component_pixels,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Crack TensorRT output validation failed: {exc}"
            ) from exc

    def close(self) -> None:
        cudart = self._cudart
        stream = self._stream
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(crack close)",
                )
            except Exception:
                pass
        for attribute in ("_device_input", "_device_output"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(
                            cudart.cudaFree(allocation),
                            f"cudaFree({attribute})",
                        )
                except Exception:
                    pass
                setattr(self, attribute, None)
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamDestroy(stream),
                    "cudaStreamDestroy(crack)",
                )
            except Exception:
                pass
        for attribute in ("_host_input_pointer", "_host_output_pointer"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(
                            cudart.cudaFreeHost(allocation),
                            f"cudaFreeHost({attribute})",
                        )
                except Exception:
                    pass
                setattr(self, attribute, None)
        self._stream = None
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None
        self._host_input = None
        self._host_output = None
        self._output_name = None
        self._cudart = None


def annotate_cracks(
    frame: np.ndarray,
    result: CrackDetectionResult,
    zone_number: int | None,
) -> np.ndarray:
    if result.mask.shape != frame.shape[:2]:
        raise ValueError("Crack mask shape must match the frame shape.")

    output = frame.copy()
    overlay = output.copy()
    overlay[result.mask > 0] = CRACK_COLOR_BGR
    output = cv2.addWeighted(overlay, 0.4, output, 0.6, 0)
    color = (0, 0, 255) if result.detected else (60, 180, 75)
    for x, y, width, height in result.boxes:
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)

    zone_label = (
        "MANUAL" if zone_number is None else f"RAIL SECTION {zone_number}"
    )
    state_label = "CRACK CANDIDATE" if result.detected else "NO CRACK CANDIDATE"
    label = (
        f"{state_label}  {zone_label}  "
        f"ratio={result.crack_ratio * 100:.3f}%"
    )
    right = max(13, min(output.shape[1] - 12, 760))
    cv2.rectangle(output, (12, 84), (right, 120), (20, 35, 50), -1)
    cv2.putText(
        output,
        label,
        (24, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return output
