from __future__ import annotations

import ctypes
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# 부식 추론의 읽는 순서: profile 입력 계약 → 전처리 → TensorRT 엔진 실행 → class_map → 비율·박스.
# .plan에는 학습된 가중치와 최적화된 연산 구조가 들어 있다. 이 파일은 학습을 수행하거나
# 가중치를 Python 변수로 하나씩 바꾸지 않고, 준비한 입력 버퍼로 이미 빌드된 엔진을 실행한다.
# GPU 전처리는 cuda_rust_preprocess.py, GPU 클래스 선택은 cuda_argmax.py에서 이어서 확인한다.
INPUT_NAME = "images"
OUTPUT_NAME = "logits"
CLASS_NAMES = ("Good", "Fair", "Poor", "Severe")
TENSORRT_VERSION_PREFIX = "10.3."
STUDENT_PROFILE = "student"
TEACHER_PROFILE = "teacher"
MODEL_PROFILES = (STUDENT_PROFILE, TEACHER_PROFILE)
STUDENT_INPUT_SHAPE = (1, 3, 240, 1280)
STUDENT_OUTPUT_SHAPE = (1, 4, 240, 1280)
TEACHER_INPUT_SHAPE = (1, 3, 720, 1280)
TEACHER_OUTPUT_SHAPE = (1, 4, 720, 1280)
PROFILE_SHAPES = {
    STUDENT_PROFILE: (STUDENT_INPUT_SHAPE, STUDENT_OUTPUT_SHAPE),
    TEACHER_PROFILE: (TEACHER_INPUT_SHAPE, TEACHER_OUTPUT_SHAPE),
}
# Backwards-compatible names identify the realtime/student contract.
INPUT_SHAPE = STUDENT_INPUT_SHAPE
OUTPUT_SHAPE = STUDENT_OUTPUT_SHAPE
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
CLASS_COLORS_BGR = {
    1: (0, 215, 255),
    2: (0, 128, 255),
    3: (0, 0, 255),
}
MIN_BOX_AREA = 250.0


@dataclass(frozen=True)
class LetterboxTransform:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int


@dataclass
class DetectionResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    rust_ratio: float
    method: str
    status: str = "ready"
    class_map: np.ndarray | None = None
    class_ratios: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RustStageTiming:
    preprocess_cpu_ms: float
    input_u8_h2d_ms: float
    gpu_preprocess_ms: float
    trt_inference_ms: float
    gpu_argmax_ms: float
    class_map_d2h_ms: float
    cpu_postprocess_ms: float
    total_ms: float


def validate_native_frame(
    frame: np.ndarray,
    profile: str = STUDENT_PROFILE,
) -> LetterboxTransform:
    """Validate a native static-shape input and describe its identity mapping."""

    # student는 1280×240 ROI, teacher는 1280×720 원본을 그대로 받아야 한다.
    # 이름은 LetterboxTransform이지만 여기서는 리사이즈·패딩 없이 같은 좌표를 반환한다.
    # shape 오류는 모델 안에서 보정하지 않으므로 호출부의 ROI와 선택 profile을 먼저 확인한다.
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be an HxWx3 NumPy array.")
    # A valid uint8 array cannot represent NaN or infinity; rejecting every
    # other dtype preserves the finite-pixel contract without a full-frame scan.
    if frame.dtype != np.uint8:
        raise ValueError("Camera frame must contain finite uint8 BGR pixels.")
    if profile not in MODEL_PROFILES:
        raise ValueError(f"Unknown TensorRT preprocessing profile: {profile}")
    original_height, original_width = frame.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError("Camera frame must not be empty.")

    input_shape, _ = PROFILE_SHAPES[profile]
    target_height, target_width = input_shape[2:]
    if (original_height, original_width) != (target_height, target_width):
        raise ValueError(
            f"TensorRT {profile} input frame must be "
            f"{target_width}x{target_height}; got "
            f"{original_width}x{original_height}. Native engines do not resize, "
            "letterbox, or pad frames."
        )
    return LetterboxTransform(
        original_height=original_height,
        original_width=original_width,
        resized_height=original_height,
        resized_width=original_width,
        top=0,
        left=0,
    )


def preprocess_frame(
    frame: np.ndarray,
    profile: str = STUDENT_PROFILE,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Apply the CPU reference native teacher or student input contract."""

    # CPU 전처리의 두 계약은 서로 다르다. student는 BGR→RGB 후 채널별
    # (픽셀/255 - mean)/std 정규화, teacher는 BGR 0~255 값을 FP32로만 바꾼다.
    # 마지막 transpose와 [None]은 HWC를 배치 1의 NCHW로 바꾸며 연속 메모리를 만든다.
    # 이 mean/std와 profile 선택은 추론 입력 설정이고 엔진 내부의 학습 가중치가 아니다.
    transform = validate_native_frame(frame, profile)
    if profile == STUDENT_PROFILE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        model_input = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    else:
        model_input = frame.astype(np.float32)

    tensor = np.ascontiguousarray(model_input.transpose(2, 0, 1)[None])
    return tensor, transform


def restore_class_map(
    logits: np.ndarray,
    transform: LetterboxTransform,
) -> np.ndarray:
    """Map native logits one-to-one to the input frame."""

    logits = np.asarray(logits)
    if logits.dtype != np.float32:
        raise ValueError(
            f"TensorRT logits must have dtype float32, got {logits.dtype}."
        )
    expected_shape = (
        1,
        len(CLASS_NAMES),
        transform.original_height,
        transform.original_width,
    )
    if logits.shape != expected_shape:
        raise ValueError(
            f"TensorRT output must have shape {expected_shape}, got {logits.shape}."
        )
    # 픽셀마다 네 클래스의 logit 중 가장 큰 값의 '번호'를 고른다. softmax 확률이나
    # confidence를 반환하는 계산이 아니다. 예: [0.1, 0.4, 2.0, -1.0] → 2(Poor).
    # 결과는 H×W uint8 class_map이며 입력 좌표와 일대일로 대응한다.
    return np.argmax(logits[0], axis=0).astype(np.uint8)


def result_from_logits(
    logits: np.ndarray,
    transform: LetterboxTransform,
    method: str,
) -> DetectionResult:
    if not np.all(np.isfinite(logits)):
        raise ValueError("Segmentation logits contain NaN or infinity.")
    class_map = restore_class_map(logits, transform)
    return result_from_class_map(class_map, transform, method)


def result_from_class_map(
    class_map: np.ndarray,
    transform: LetterboxTransform,
    method: str,
) -> DetectionResult:
    class_map = np.asarray(class_map)
    expected_shape = (transform.original_height, transform.original_width)
    if class_map.dtype != np.uint8:
        raise ValueError(
            f"Segmentation class map must have dtype uint8, got {class_map.dtype}."
        )
    if class_map.shape != expected_shape:
        raise ValueError(
            f"Segmentation class map must have shape {expected_shape}, "
            f"got {class_map.shape}."
        )
    if np.any(class_map >= len(CLASS_NAMES)):
        raise ValueError("Segmentation class map contains an invalid class ID.")
    # 0=Good, 1~3=Fair/Poor/Severe로 해석하고 세 비정상 클래스를 합쳐 부식 마스크를 만든다.
    # MIN_BOX_AREA는 작은 윤곽의 '표시 박스'만 걸러내며 마스크나 비율에서 픽셀을 빼지는 않는다.
    # 제어용 최고 등급은 이 결과의 class_map을 받은 node_runtime/control.py에서 따로 계산한다.
    mask = np.where(class_map > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) >= MIN_BOX_AREA
    ]
    # 비율의 분모는 이 함수에 들어온 이미지 전체 픽셀 수다. student면 ROI 면적,
    # teacher면 전체 입력 면적이므로 서로 다른 화각·ROI의 비율을 그대로 비교하면 안 된다.
    # 예: 비정상 1,000픽셀/입력 100,000픽셀=0.01(1%); 신뢰도 1%라는 뜻이 아니다.
    inspected_pixels = max(class_map.size, 1)
    class_ratios = {
        class_name: float(np.count_nonzero(class_map == class_id) / inspected_pixels)
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    rust_ratio = float(np.count_nonzero(mask) / inspected_pixels)
    return DetectionResult(
        mask=mask,
        boxes=boxes,
        rust_ratio=rust_ratio,
        method=method,
        class_map=class_map,
        class_ratios=class_ratios,
    )


class RustDetector:
    """Synchronous TensorRT 10 runtime for a DeepLabV3+ engine."""

    def __init__(
        self,
        engine_path: Path,
        profile: str = STUDENT_PROFILE,
        expected_sha256: str | None = None,
        *,
        gpu_argmax: bool = False,
        stage_timing: bool = False,
    ) -> None:
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Unknown TensorRT preprocessing profile: {profile}")
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
        self.stage_timing = bool(stage_timing)
        self.last_stage_timing: RustStageTiming | None = None
        self.input_shape, self.output_shape = PROFILE_SHAPES[profile]
        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(f"TensorRT engine was not found: {self.engine_path}")

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
        self._gpu_preprocessor: Any = None
        self._gpu_postprocessor: Any = None
        self._timing_events: tuple[Any, ...] = ()
        self.engine_sha256: str | None = None
        self.method = f"deeplabv3plus-tensorrt/{profile}/cuda:0"
        if gpu_argmax and profile != STUDENT_PROFILE:
            raise ValueError("GPU rust argmax is supported only for the student profile.")
        if self.stage_timing and not gpu_argmax:
            raise ValueError("Rust stage timing requires GPU argmax.")

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
                f"deeplabv3plus-tensorrt/{profile}/"
                f"trt-{tensorrt_version}/cuda:0"
            )
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            try:
                serialized_engine = self.engine_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read TensorRT engine: {self.engine_path}"
                ) from exc
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if (
                expected_sha256 is not None
                and self.engine_sha256 != expected_sha256
            ):
                raise RuntimeError(
                    "TensorRT engine changed after preflight SHA-256 validation: "
                    f"expected {expected_sha256}, got {self.engine_sha256}."
                )
            # 파일 바이트를 TensorRT 실행 엔진으로 복원한다. SHA-256은 기대한 엔진 파일인지
            # 확인하는 값이며 모델 정확도를 검증하는 지표는 아니다. 복원 실패 시 기기 호환성,
            # TensorRT 10.3.x 여부, engine_path와 engine_sha256부터 확인한다.
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT could not deserialize the engine. Verify the Jetson, "
                    "TensorRT version, and engine SHA-256."
                )
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context.")

            input_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(INPUT_NAME)))
            output_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(OUTPUT_NAME)))
            if input_dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"TensorRT input dtype must be float32, got {input_dtype}."
                )
            if output_dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"TensorRT output dtype must be float32, got {output_dtype}."
                )

            input_nbytes = int(np.prod(self.input_shape)) * input_dtype.itemsize
            output_nbytes = int(np.prod(self.output_shape)) * output_dtype.itemsize
            if not gpu_argmax:
                self._host_input_pointer = self._cuda_call(
                    cudart.cudaHostAlloc(input_nbytes, 0),
                    "cudaHostAlloc(input)",
                )
                self._host_input = self._host_array(
                    self._host_input_pointer,
                    self.input_shape,
                    input_dtype,
                )
            if not gpu_argmax:
                self._host_output_pointer = self._cuda_call(
                    cudart.cudaHostAlloc(output_nbytes, 0),
                    "cudaHostAlloc(output)",
                )
                self._host_output = self._host_array(
                    self._host_output_pointer,
                    self.output_shape,
                    output_dtype,
                )
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes),
                "cudaMalloc(input)",
            )
            self._device_output = self._cuda_call(
                cudart.cudaMalloc(output_nbytes),
                "cudaMalloc(output)",
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(),
                "cudaStreamCreate",
            )
            # 학습 가중치를 전달하는 코드가 아니라 images/logits가 읽고 쓸 GPU 주소를 등록한다.
            # 엔진 내부 FP16 사용 여부와 별개로 이 런타임의 입출력 dtype 계약은 둘 다 FP32다.
            if not self._context.set_tensor_address(
                INPUT_NAME, int(self._device_input)
            ):
                raise RuntimeError("TensorRT rejected the input buffer address.")
            if not self._context.set_tensor_address(
                OUTPUT_NAME, int(self._device_output)
            ):
                raise RuntimeError("TensorRT rejected the output buffer address.")
            if gpu_argmax:
                from cuda_argmax import CudaArgmaxPostprocessor
                from cuda_rust_preprocess import CudaRustPreprocessor

                self._gpu_preprocessor = CudaRustPreprocessor(
                    cudart,
                    self.input_shape,
                    measure_timing=self.stage_timing,
                )
                self._gpu_postprocessor = CudaArgmaxPostprocessor(
                    cudart,
                    self.output_shape,
                    measure_timing=self.stage_timing,
                )
                self.method += "/gpu-preprocess/gpu-argmax"
            if self.stage_timing:
                self._timing_events = tuple(
                    self._cuda_call(
                        cudart.cudaEventCreate(),
                        f"cudaEventCreate(Rust {name})",
                    )
                    for name in ("TRT start", "TRT end")
                )
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> None:
        # 이름·입출력 방향·고정 shape·FP32·LINEAR·GPU 배치를 모두 검사한다.
        # student/teacher 엔진을 바꿔 끼우거나 다른 export를 쓰면 여기서 중단하는 것이 정상이다.
        trt = self._trt
        engine = self._engine
        tensor_names = {
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        }
        if tensor_names != {INPUT_NAME, OUTPUT_NAME}:
            raise RuntimeError(
                "TensorRT engine I/O names must be exactly "
                f"{INPUT_NAME!r} and {OUTPUT_NAME!r}, got {sorted(tensor_names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Tensor {INPUT_NAME!r} is not an engine input.")
        if engine.get_tensor_mode(OUTPUT_NAME) != trt.TensorIOMode.OUTPUT:
            raise RuntimeError(f"Tensor {OUTPUT_NAME!r} is not an engine output.")
        input_shape = tuple(engine.get_tensor_shape(INPUT_NAME))
        output_shape = tuple(engine.get_tensor_shape(OUTPUT_NAME))
        if input_shape != self.input_shape:
            raise RuntimeError(
                f"TensorRT {self.profile} input shape must be "
                f"{self.input_shape}, got {input_shape}."
            )
        if output_shape != self.output_shape:
            raise RuntimeError(
                f"TensorRT {self.profile} output shape must be "
                f"{self.output_shape}, got {output_shape}."
            )
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(INPUT_NAME)))
        output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(OUTPUT_NAME)))
        if input_dtype != np.dtype(np.float32):
            raise RuntimeError(
                f"TensorRT input dtype must be float32, got {input_dtype}."
            )
        if output_dtype != np.dtype(np.float32):
            raise RuntimeError(
                f"TensorRT output dtype must be float32, got {output_dtype}."
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

    def detect(self, frame: np.ndarray) -> DetectionResult:
        # gpu_argmax 옵션은 student의 GPU 전처리와 GPU 후처리를 함께 선택한다.
        # 꺼져 있으면 CPU에서 FP32 NCHW를 만들고, 켜져 있으면 uint8 원본을 준비한 뒤
        # CUDA에서 정규화해 같은 _device_input에 쓴다. 모델 자체의 추론은 두 경로 모두 GPU다.
        if self._context is None or self._stream is None:
            raise RuntimeError("TensorRT detector is closed.")
        self.last_stage_timing = None
        total_started = time.perf_counter() if self.stage_timing else None
        preprocess_started = time.perf_counter() if self.stage_timing else None
        if self._gpu_preprocessor is not None:
            transform = validate_native_frame(frame, self.profile)
            self._gpu_preprocessor.stage(frame)
            tensor = None
        else:
            tensor, transform = preprocess_frame(frame, self.profile)
        try:
            if tensor is not None:
                np.copyto(self._host_input, tensor, casting="no")
            preprocess_ms = (
                (time.perf_counter() - preprocess_started) * 1000.0
                if preprocess_started is not None
                else None
            )
            if self._gpu_preprocessor is not None:
                self._gpu_preprocessor.enqueue(self._device_input, self._stream)
            else:
                self._cuda_call(
                    self._cudart.cudaMemcpyAsync(
                        self._device_input,
                        self._host_input.ctypes.data,
                        self._host_input.nbytes,
                        self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                        self._stream,
                    ),
                    "cudaMemcpyAsync(input)",
                )
            if self._timing_events:
                self._cuda_call(
                    self._cudart.cudaEventRecord(
                        self._timing_events[0], self._stream
                    ),
                    "cudaEventRecord(Rust TRT start)",
                )
            # 같은 CUDA stream에 입력 준비→엔진 연산을 순서대로 예약한다. 이 호출이 실제로
            # 엔진 속 가중치를 입력에 적용해 logits를 계산하게 하는 경계이며 학습 업데이트는 없다.
            # async는 GPU 예약 방식이다. detect()는 아래 결과 회수를 마친 뒤에야 반환한다.
            executed = self._context.execute_async_v3(
                stream_handle=int(self._stream)
            )
            if not executed:
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
            if self._timing_events:
                self._cuda_call(
                    self._cudart.cudaEventRecord(
                        self._timing_events[1], self._stream
                    ),
                    "cudaEventRecord(Rust TRT end)",
                )
            if self._gpu_postprocessor is not None:
                class_map = self._gpu_postprocessor.process(
                    self._device_output,
                    self._stream,
                )
            else:
                self._cuda_call(
                    self._cudart.cudaMemcpyAsync(
                        self._host_output.ctypes.data,
                        self._device_output,
                        self._host_output.nbytes,
                        self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self._stream,
                    ),
                    "cudaMemcpyAsync(output)",
                )
                self._cuda_call(
                    self._cudart.cudaStreamSynchronize(self._stream),
                    "cudaStreamSynchronize",
                )
        except Exception as exc:
            raise RuntimeError(f"TensorRT inference failed: {exc}") from exc

        try:
            postprocess_started = time.perf_counter() if self.stage_timing else None
            # GPU 경로는 4채널 FP32 logits 대신 argmax 후의 1채널 uint8 지도만 CPU로 가져온다.
            # CPU 경로는 logits 전체를 복사하고 stream 완료를 기다린 뒤 NumPy argmax를 수행한다.
            # 두 경로 모두 최종 박스·비율 계산은 result_from_class_map의 CPU 처리로 합류한다.
            if self._gpu_postprocessor is not None:
                result = result_from_class_map(class_map, transform, self.method)
            else:
                logits = self._host_output.reshape(self.output_shape)
                result = result_from_logits(logits, transform, self.method)
            cpu_postprocess_ms = (
                (time.perf_counter() - postprocess_started) * 1000.0
                if postprocess_started is not None
                else None
            )
            if self.stage_timing:
                if (
                    preprocess_ms is None
                    or cpu_postprocess_ms is None
                    or total_started is None
                ):
                    raise RuntimeError("Rust CPU stage timing was not recorded.")
                gpu_timing = self._gpu_postprocessor.last_stage_timing
                if gpu_timing is None:
                    raise RuntimeError("Rust CUDA stage timing was not recorded.")
                preprocess_timing = self._gpu_preprocessor.timing()
                trt_inference_ms = float(
                    self._cuda_call(
                        self._cudart.cudaEventElapsedTime(
                            self._timing_events[0], self._timing_events[1]
                        ),
                        "cudaEventElapsedTime(Rust TRT)",
                    )
                )
                self.last_stage_timing = RustStageTiming(
                    preprocess_cpu_ms=preprocess_ms,
                    input_u8_h2d_ms=preprocess_timing.input_u8_h2d_ms,
                    gpu_preprocess_ms=preprocess_timing.gpu_preprocess_ms,
                    trt_inference_ms=trt_inference_ms,
                    gpu_argmax_ms=gpu_timing.gpu_argmax_ms,
                    class_map_d2h_ms=gpu_timing.class_map_d2h_ms,
                    cpu_postprocess_ms=cpu_postprocess_ms,
                    total_ms=(time.perf_counter() - total_started) * 1000.0,
                )
            return result
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"TensorRT output validation failed: {exc}") from exc

    def close(self) -> None:
        cudart = self._cudart
        stream = self._stream
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(close)",
                )
            except Exception:
                pass
        if self._gpu_postprocessor is not None:
            try:
                self._gpu_postprocessor.close()
            except Exception:
                pass
            self._gpu_postprocessor = None
        if self._gpu_preprocessor is not None:
            try:
                self._gpu_preprocessor.close()
            except Exception:
                pass
            self._gpu_preprocessor = None
        for event in self._timing_events:
            try:
                if cudart is not None:
                    self._cuda_call(
                        cudart.cudaEventDestroy(event),
                        "cudaEventDestroy(Rust)",
                    )
            except Exception:
                pass
        self._timing_events = ()
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
                    "cudaStreamDestroy",
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
        self._cudart = None


def annotate(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    if result.class_map is not None and result.class_map.shape == output.shape[:2]:
        for class_id, color in CLASS_COLORS_BGR.items():
            overlay[result.class_map == class_id] = color
    else:
        overlay[result.mask > 0] = (0, 95, 255)
    output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

    fair_ratio = result.class_ratios.get("Fair", 0.0)
    poor_ratio = result.class_ratios.get("Poor", 0.0)
    severe_ratio = result.class_ratios.get("Severe", 0.0)
    if severe_ratio > 0.0:
        color = CLASS_COLORS_BGR[3]
    elif poor_ratio > 0.0:
        color = CLASS_COLORS_BGR[2]
    elif fair_ratio > 0.0:
        color = CLASS_COLORS_BGR[1]
    else:
        color = (60, 180, 75)

    for x, y, width, height in result.boxes:
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)

    ratio_label = (
        f"rust={result.rust_ratio * 100:.2f}%  "
        f"Fair={fair_ratio * 100:.2f}%  Poor={poor_ratio * 100:.2f}%  "
        f"Severe={severe_ratio * 100:.2f}%"
    )
    method_label = result.method
    if result.status != "ready":
        method_label += f"  ({result.status})"
    cv2.rectangle(
        output,
        (12, 12),
        (min(output.shape[1] - 12, 920), 76),
        (20, 35, 50),
        -1,
    )
    cv2.putText(
        output,
        ratio_label,
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        method_label,
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return output
