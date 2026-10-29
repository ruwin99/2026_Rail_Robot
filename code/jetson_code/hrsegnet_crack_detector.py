from __future__ import annotations

import ctypes
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crack_detector import CrackDetectionResult


# 균열 추론: BGR 원본 → CPU RGB 정규화 → TensorRT 두 채널 logits → GPU 후보 마스크
# → CPU 연결 성분 제거 → 최종 균열 면적 비율. 실시간 ROI와 정지 촬영 원본의 엔진은 다르다.
# .plan의 가중치·연산 구조는 deserialize/execute로 사용하며 Python에서 재학습하지 않는다.
# GPU 임계값 처리·마스크 회수는 cuda_logit_margin.py, 결과 전달은 ROS vision/inspection 쪽을 본다.
INPUT_NAME = "images"
OUTPUT_NAME = "crack_logits"
INPUT_SHAPE = (1, 3, 128, 1280)
OUTPUT_SHAPE = (1, 2, 128, 1280)
REALTIME_ROI_HEIGHT = 180
CAPTURE_INPUT_SHAPE = (1, 3, 720, 1280)
CAPTURE_OUTPUT_SHAPE = (1, 2, 720, 1280)
TENSORRT_VERSION_PREFIX = "10.3."
DETECTOR_PREFIX = "hrsegnet-b32-tensorrt/realtime/crack/"
CAPTURE_DETECTOR_PREFIX = "hrsegnet-b32-tensorrt/capture/crack/"
DEFAULT_PROBABILITY_THRESHOLD = 0.55
DEFAULT_MIN_COMPONENT_PIXELS = 20

ROLE_CONTRACTS = {
    "realtime": (INPUT_SHAPE, OUTPUT_SHAPE, DETECTOR_PREFIX),
    "capture": (
        CAPTURE_INPUT_SHAPE,
        CAPTURE_OUTPUT_SHAPE,
        CAPTURE_DETECTOR_PREFIX,
    ),
}


@dataclass
class HrSegNetCrackDetectionResult(CrackDetectionResult):
    """Control result plus read-only diagnostics from the same raw logits."""

    max_crack_probability: float = 0.0
    candidate_pixels: int = 0
    filtered_pixels: int = 0
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS


@dataclass(frozen=True)
class HrSegNetStageTiming:
    trt_inference_ms: float
    gpu_logit_margin_threshold_ms: float
    candidate_mask_d2h_ms: float
    cpu_8ccl_filter_ms: float
    detector_total_ms: float


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def probability_threshold_to_logit_margin(probability_threshold: float) -> float:
    # 두 클래스 softmax의 균열 확률은 sigmoid(균열 logit - 배경 logit)와 같다.
    # 그래서 확률 p 대신 log(p/(1-p))와 비교한다. 기본 p=0.55면 약 0.2007이다.
    # p는 출력 후보를 고르는 후처리 설정이며 모델이 학습한 가중치를 바꾸는 파라미터가 아니다.
    probability_threshold = float(probability_threshold)
    if not math.isfinite(probability_threshold) or not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be finite and between zero and one.")
    return math.log(probability_threshold) - math.log1p(-probability_threshold)


_NORMALIZE_SCALE = np.float32(127.5)
_NORMALIZE_OFFSET = np.float32(1.0)
# 남길 성분이 이보다 적으면 성분별로 칠하고, 많으면 라벨 전체를 한 번에 표로 바꾼다.
# Jetson 실측: 표 변환은 성분 수와 무관하게 약 0.6 ms, 성분별 칠하기는 성분당 약 0.1 ms다.
_COMPONENT_PAINT_LIMIT = 6


def _validate_frame(
    frame: np.ndarray,
    input_shape: tuple[int, int, int, int],
) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be an HxWx3 NumPy array.")
    if frame.dtype != np.uint8:
        raise ValueError("Camera frame must contain uint8 BGR pixels.")
    expected_height, expected_width = input_shape[2:]
    if frame.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            "HrSegNet TensorRT input frame must be "
            f"{expected_width}x{expected_height}; got {frame.shape[1]}x{frame.shape[0]}. "
            "The role-specific engine does not resize or pad."
        )


def preprocess_into(frame: np.ndarray, tensor: np.ndarray) -> None:
    """Write the normalized NCHW tensor straight into a caller buffer (pinned host memory)."""
    # 역할별 입력 크기를 검사한 다음 CPU에서 BGR→RGB, 0~255→-1~1로 정규화한다.
    # NCHW FP32는 realtime=(1,3,128,1280), capture=(1,3,720,1280)이며
    # 이 함수는 크기를 늘리거나 줄이지 않는다. shape 오류면 상위의 ROI 선택을 확인한다.
    # 채널 순서를 뒤집은 view를 float32로 나눈 뒤 1을 빼므로 x/127.5-1과 비트 단위로 같다.
    # 중간 배열과 별도 복사가 없어 Jetson에서 realtime 5.4 ms → 1.3 ms, capture 18.6 ms → 7.1 ms다.
    _validate_frame(frame, tensor.shape)
    if tensor.dtype != np.float32 or not tensor.flags.c_contiguous:
        raise ValueError("HrSegNet input tensor must be a contiguous float32 buffer.")
    np.divide(
        frame[:, :, ::-1].transpose(2, 0, 1),
        _NORMALIZE_SCALE,
        out=tensor[0],
        dtype=np.float32,
    )
    np.subtract(tensor, _NORMALIZE_OFFSET, out=tensor)


def preprocess_frame(
    frame: np.ndarray,
    input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
) -> np.ndarray:
    tensor = np.empty(input_shape, dtype=np.float32)
    preprocess_into(frame, tensor)
    return tensor


def resize_realtime_roi(frame: np.ndarray) -> np.ndarray:
    """INTER_AREA 180→128 rows; the width is unchanged, so rows can be resized as one channel."""
    # 가로 배율이 1이라 각 열은 독립적이다. HxWx3을 Hx(W*3) 한 채널로 보고 줄이면
    # 3채널 INTER_AREA와 비트 단위로 같은 결과를 Jetson에서 3.4 ms → 1.2 ms에 얻는다.
    height, width = INPUT_SHAPE[2], INPUT_SHAPE[3]
    flat = np.ascontiguousarray(frame).reshape(frame.shape[0], width * 3)
    return cv2.resize(flat, (width * 3, height), interpolation=cv2.INTER_AREA).reshape(
        height, width, 3
    )


def result_from_logits(
    logits: np.ndarray,
    method: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
    output_shape: tuple[int, int, int, int] = OUTPUT_SHAPE,
) -> HrSegNetCrackDetectionResult:
    # logits를 NumPy로 검증하는 CPU 기준 경로다. 실제 detect()는 아래 계산의 CUDA 버전을
    # 사용해 candidate_map을 받은 뒤 result_from_candidate_map으로 바로 들어간다.
    margin_threshold = np.float32(
        probability_threshold_to_logit_margin(probability_threshold)
    )
    if (
        not isinstance(min_component_pixels, int)
        or isinstance(min_component_pixels, bool)
        or min_component_pixels <= 0
    ):
        raise ValueError("min_component_pixels must be a positive integer.")

    logits = np.asarray(logits)
    if logits.dtype != np.float32:
        raise ValueError(f"HrSegNet logits must have dtype float32, got {logits.dtype}.")
    if logits.shape != output_shape:
        raise ValueError(
            f"HrSegNet logits must have shape {output_shape}, got {logits.shape}."
        )
    if not np.all(np.isfinite(logits)):
        raise ValueError("HrSegNet logits contain NaN or infinity.")

    # logit 자체는 확률이 아니다. 채널 1(균열)-채널 0(배경)의 차이가 약 0.2007 이상이면
    # 기본 임계값을 통과한다. 예: 차이 0.3 → sigmoid≈0.574 → 후보 픽셀 1.
    # max_crack_probability는 필터링 전 가장 높은 한 픽셀의 값이며 최종 면적 비율과 다르다.
    margin = logits[0, 1] - logits[0, 0]
    if not np.all(np.isfinite(margin)):
        raise ValueError("HrSegNet logit margin contains NaN or infinity.")
    candidate = (margin >= margin_threshold).astype(np.uint8)
    max_crack_probability = _stable_sigmoid(float(np.max(margin)))
    return result_from_candidate_map(
        candidate,
        max_crack_probability,
        method,
        probability_threshold,
        min_component_pixels,
    )


def result_from_candidate_map(
    candidate: np.ndarray,
    max_crack_probability: float,
    method: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
    *,
    ccl_timing: list[float] | None = None,
) -> HrSegNetCrackDetectionResult:
    probability_threshold_to_logit_margin(probability_threshold)
    if (
        not isinstance(min_component_pixels, int)
        or isinstance(min_component_pixels, bool)
        or min_component_pixels <= 0
    ):
        raise ValueError("min_component_pixels must be a positive integer.")
    candidate = np.asarray(candidate)
    if candidate.dtype != np.uint8 or candidate.ndim != 2:
        raise ValueError("HrSegNet candidate map must be a two-dimensional uint8 array.")
    if candidate.size == 0 or np.any(candidate > 1):
        raise ValueError("HrSegNet candidate map must contain only zero and one.")
    max_crack_probability = float(max_crack_probability)
    if not math.isfinite(max_crack_probability) or not 0.0 <= max_crack_probability <= 1.0:
        raise ValueError("max_crack_probability must be finite and between zero and one.")

    # 임계값만 넘은 candidate(0/1)를 CPU의 8방향 연결 성분으로 묶는다.
    # 기본 min_component_pixels=20이면 연결된 19픽셀은 제거하고 20픽셀은 남긴다.
    # candidate_pixels와 filtered_pixels 차이는 이 후처리에서 제거된 작은 영역을 뜻한다.
    candidate_pixels = int(np.count_nonzero(candidate))
    ccl_started = time.perf_counter()
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    # 0번(배경)을 뺀 성분 중 면적 기준을 넘는 것만 남긴다. Python 반복은 남는 성분에만 한다.
    # 잡음이 많은 프레임(성분 수천 개)에서도 시간이 거의 늘지 않는다(Jetson 58.7 ms → 4.5 ms).
    kept = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_component_pixels) + 1
    if kept.size == 0:
        filtered = np.zeros(candidate.shape, dtype=np.uint8)
    elif kept.size < _COMPONENT_PAINT_LIMIT:
        filtered = np.zeros(candidate.shape, dtype=np.uint8)
        for component_id in kept:
            filtered[labels == component_id] = 255
    else:
        lookup = np.zeros(component_count, dtype=np.uint8)
        lookup[kept] = 255
        filtered = lookup[labels]
    boxes: list[tuple[int, int, int, int]] = [
        (int(left), int(top), int(width), int(height))
        for left, top, width, height in stats[kept][
            :, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
        ]
    ]
    if ccl_timing is not None:
        ccl_timing.append((time.perf_counter() - ccl_started) * 1000.0)

    # 최종 마스크는 0/255, 박스는 (x,y,width,height)다. ratio는 남은 균열 픽셀/입력 면적이며
    # 분모는 복원된 원본 ROI 면적이다(실시간 하단 ROI는 180×1280).
    # detected는 남은 픽셀이 하나라도 있는지이며 모터 차단 임계값 판단은 상위 제어 로직의 몫이다.
    crack_pixels = int(np.count_nonzero(filtered))
    inspected_pixels = int(filtered.size)
    return HrSegNetCrackDetectionResult(
        mask=filtered,
        boxes=boxes,
        crack_pixels=crack_pixels,
        inspected_pixels=inspected_pixels,
        crack_ratio=crack_pixels / inspected_pixels,
        detected=crack_pixels > 0,
        method=str(method),
        probability_threshold=float(probability_threshold),
        max_crack_probability=max_crack_probability,
        candidate_pixels=candidate_pixels,
        filtered_pixels=crack_pixels,
        min_component_pixels=min_component_pixels,
    )


class HrSegNetCrackDetector:
    """Synchronous TensorRT runtime for the approved HrSegNet-B32 baseline."""

    def __init__(
        self,
        engine_path: Path,
        expected_sha256: str,
        probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
        min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
        *,
        role: str,
        stage_timing: bool = False,
        gpu_preprocess: bool = False,
        linear_restore: bool = False,
    ) -> None:
        probability_threshold_to_logit_margin(probability_threshold)
        if (
            not isinstance(min_component_pixels, int)
            or isinstance(min_component_pixels, bool)
            or min_component_pixels <= 0
        ):
            raise ValueError("min_component_pixels must be a positive integer.")
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        if role not in ROLE_CONTRACTS:
            raise ValueError("role must be exactly 'capture' or 'realtime'.")

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(f"HrSegNet TensorRT engine was not found: {self.engine_path}")
        self.probability_threshold = float(probability_threshold)
        self.min_component_pixels = min_component_pixels
        self.role = role
        self.stage_timing = bool(stage_timing)
        # 180행 ROI 결과를 0/1 NEAREST 대신 logit 차이 선형 보간으로 복원한다(GPU, realtime 전용).
        self.linear_restore = bool(linear_restore) and role == "realtime"
        self.last_stage_timing: HrSegNetStageTiming | None = None
        self.input_shape, self.output_shape, detector_prefix = ROLE_CONTRACTS[role]
        self.engine_sha256: str | None = None
        self.method = f"{detector_prefix}cuda:0/logit-margin"

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
        self._gpu_postprocessor: Any = None
        self._gpu_preprocessor: Any = None
        self._trt_timing_events: tuple[Any, ...] = ()

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
            version = str(getattr(trt, "__version__", ""))
            if not version.startswith(TENSORRT_VERSION_PREFIX):
                raise RuntimeError(
                    "This HrSegNet plan requires TensorRT 10.3.x, got "
                    f"{version or 'an unknown version'}."
                )
            self.method = f"{detector_prefix}trt-{version}/cuda:0/logit-margin"
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            try:
                serialized_engine = self.engine_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read HrSegNet TensorRT engine: {self.engine_path}"
                ) from exc
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if self.engine_sha256 != expected_sha256:
                raise RuntimeError(
                    "HrSegNet TensorRT engine changed after preflight SHA-256 "
                    f"validation: expected {expected_sha256}, got {self.engine_sha256}."
                )
            # SHA-256으로 지정된 .plan을 확인한 뒤 학습 가중치와 연산이 포함된 엔진을 복원한다.
            # probability_threshold와 min_component_pixels는 엔진 밖의 후처리 값으로 따로 유지된다.
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the HrSegNet engine.")
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an HrSegNet execution context.")

            input_nbytes = int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
            output_nbytes = int(np.prod(self.output_shape)) * np.dtype(np.float32).itemsize
            self._host_input_pointer = self._cuda_call(
                cudart.cudaHostAlloc(input_nbytes, 0), "cudaHostAlloc(HrSegNet input)"
            )
            self._host_input = self._host_array(
                self._host_input_pointer, self.input_shape, np.dtype(np.float32)
            )
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes), "cudaMalloc(HrSegNet input)"
            )
            self._device_output = self._cuda_call(
                cudart.cudaMalloc(output_nbytes), "cudaMalloc(HrSegNet output)"
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(), "cudaStreamCreate(HrSegNet)"
            )
            # images/crack_logits라는 엔진 입출력에 GPU 메모리 주소를 바인딩한다.
            # 이 주소는 프레임과 결과 버퍼이며 학습 가중치의 주소를 Python에서 설정하는 것이 아니다.
            if not self._context.set_tensor_address(INPUT_NAME, int(self._device_input)):
                raise RuntimeError("TensorRT rejected the HrSegNet input buffer address.")
            if not self._context.set_tensor_address(OUTPUT_NAME, int(self._device_output)):
                raise RuntimeError("TensorRT rejected the HrSegNet output buffer address.")
            from cuda_logit_margin import CudaLogitMarginPostprocessor

            self._gpu_postprocessor = CudaLogitMarginPostprocessor(
                cudart,
                self.output_shape,
                probability_threshold_to_logit_margin(self.probability_threshold),
                measure_timing=self.stage_timing,
                restore_height=REALTIME_ROI_HEIGHT if self.linear_restore else None,
            )
            if self.stage_timing:
                self._trt_timing_events = (
                    self._cuda_call(cudart.cudaEventCreate(), "cudaEventCreate(TRT start)"),
                    self._cuda_call(cudart.cudaEventCreate(), "cudaEventCreate(TRT end)"),
                )
            if gpu_preprocess and role == "realtime":
                # 180행 ROI의 INTER_AREA·정규화를 GPU에서 한다. CPU 경로와 비트 단위로 같다.
                from cuda_crack_preprocess import CudaCrackPreprocessor

                self._gpu_preprocessor = CudaCrackPreprocessor(cudart)
                self.method += "/gpu-area-preprocess"
            if self.linear_restore:
                self.method += "/gpu-linear-restore"
            self.method += "/gpu-threshold/cpu-8ccl"
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> None:
        # realtime/capture 역할에 맞는 고정 shape, 텐서 이름, FP32, LINEAR GPU 입출력을 검증한다.
        # 엔진을 교체한 뒤 실패하면 role·input_shape·output_shape와 실제 export 계약을 비교한다.
        trt = self._trt
        engine = self._engine
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        if len(names) != 2 or set(names) != {INPUT_NAME, OUTPUT_NAME}:
            raise RuntimeError(
                "HrSegNet TensorRT engine I/O names must be exactly "
                f"{INPUT_NAME!r} and {OUTPUT_NAME!r}, got {sorted(names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Tensor {INPUT_NAME!r} is not an engine input.")
        if engine.get_tensor_mode(OUTPUT_NAME) != trt.TensorIOMode.OUTPUT:
            raise RuntimeError(f"Tensor {OUTPUT_NAME!r} is not an engine output.")
        for name, shape in (
            (INPUT_NAME, self.input_shape),
            (OUTPUT_NAME, self.output_shape),
        ):
            actual_shape = tuple(engine.get_tensor_shape(name))
            if actual_shape != shape:
                raise RuntimeError(
                    f"HrSegNet TensorRT tensor {name!r} shape must be {shape}, "
                    f"got {actual_shape}."
                )
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"HrSegNet TensorRT tensor {name!r} dtype must be float32, got {dtype}."
                )
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f"Tensor {name!r} must use LINEAR/CHW format.")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(f"Tensor {name!r} must be located on the GPU device.")

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
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)), ctypes.POINTER(ctypes.c_ubyte)
        )
        byte_array = np.ctypeslib.as_array(byte_pointer, shape=(byte_count,))
        return byte_array.view(dtype).reshape(shape)

    def detect(self, frame: np.ndarray) -> CrackDetectionResult:
        # 프레임 전처리는 CPU에서 수행하고 고정 host 버퍼에 복사한 다음 GPU로 전송한다.
        # 전송과 엔진 실행과 후보화는 같은 stream 순서를 따른다. 이 메서드는 결과를 회수하는
        # 동기 API이므로 execute_async_v3라는 이름만 보고 호출 즉시 반환한다고 보면 안 된다.
        if self._context is None or self._stream is None:
            raise RuntimeError("HrSegNet TensorRT detector is closed.")
        self.last_stage_timing = None
        total_started = time.perf_counter()
        restore_size = None
        if (
            self.role == "realtime"
            and isinstance(frame, np.ndarray)
            and frame.shape == (REALTIME_ROI_HEIGHT, INPUT_SHAPE[3], 3)
            and frame.dtype == np.uint8
        ):
            # Adapt only the realtime ROI; keep the existing static engine.
            restore_size = (frame.shape[1], frame.shape[0])
            if self._gpu_preprocessor is not None:
                # 원본 ROI만 고정 버퍼에 복사하고 리사이즈·정규화는 아래 stream에서 GPU가 한다.
                self._gpu_preprocessor.stage(frame)
            else:
                frame = resize_realtime_roi(frame)
        gpu_preprocess = restore_size is not None and self._gpu_preprocessor is not None
        if not gpu_preprocess:
            # 역할의 입력 계약으로 먼저 검사하고, 정규화 결과를 고정(pinned) host 버퍼에 바로 쓴다.
            # 크기 오류는 GPU 작업 전에 ValueError다.
            _validate_frame(frame, self.input_shape)
            preprocess_into(frame, self._host_input)
        try:
            if gpu_preprocess:
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
                    "cudaMemcpyAsync(HrSegNet input)",
                )
            if self._trt_timing_events:
                self._cuda_call(
                    self._cudart.cudaEventRecord(
                        self._trt_timing_events[0], self._stream
                    ),
                    "cudaEventRecord(HrSegNet TRT start)",
                )
            # 여기서 엔진의 저장된 가중치로 입력을 계산하여 배경/균열 두 채널 logits를 만든다.
            # 반환 false는 추론 실행 실패로 처리하며 정상 균열 없음 결과로 바꾸지 않는다.
            if not self._context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
            if self._trt_timing_events:
                self._cuda_call(
                    self._cudart.cudaEventRecord(
                        self._trt_timing_events[1], self._stream
                    ),
                    "cudaEventRecord(HrSegNet TRT end)",
                )
            # GPU에서 logit 차이 비교·최대 차이 계산 후 후보 마스크만 CPU로 회수한다.
            # 작은 성분 제거는 여기서 끝나지 않으며 다음 result_from_candidate_map에서 수행한다.
            restored_on_gpu = restore_size is not None and getattr(self, "linear_restore", False)
            if restored_on_gpu:
                gpu_output = self._gpu_postprocessor.process(
                    self._device_output, self._stream, restore=True
                )
            else:
                gpu_output = self._gpu_postprocessor.process(
                    self._device_output,
                    self._stream,
                )
        except Exception as exc:
            raise RuntimeError(f"HrSegNet TensorRT inference failed: {exc}") from exc

        try:
            ccl_timing: list[float] = []
            candidate = gpu_output.candidate_map
            if restore_size is not None and not restored_on_gpu:
                # Restore binary IDs before CCL so areas and boxes use ROI pixels.
                candidate = cv2.resize(
                    candidate, restore_size, interpolation=cv2.INTER_NEAREST
                )
            result = result_from_candidate_map(
                candidate,
                _stable_sigmoid(gpu_output.max_margin),
                self.method,
                self.probability_threshold,
                self.min_component_pixels,
                ccl_timing=ccl_timing,
            )
            # stage_timing은 진단 계측 옵션이다. TRT 시간, GPU 후보화, 전송, CPU 성분 제거를
            # 분리해서 보며 detector_total_ms는 프레임 준비부터 최종 결과까지의 이 호출 시간이다.
            if self.stage_timing:
                trt_ms = float(
                    self._cuda_call(
                        self._cudart.cudaEventElapsedTime(
                            self._trt_timing_events[0], self._trt_timing_events[1]
                        ),
                        "cudaEventElapsedTime(HrSegNet TRT)",
                    )
                )
                if gpu_output.threshold_ms is None or gpu_output.candidate_d2h_ms is None:
                    raise RuntimeError("HrSegNet CUDA stage timing was not recorded.")
                self.last_stage_timing = HrSegNetStageTiming(
                    trt_inference_ms=trt_ms,
                    gpu_logit_margin_threshold_ms=gpu_output.threshold_ms,
                    candidate_mask_d2h_ms=gpu_output.candidate_d2h_ms,
                    cpu_8ccl_filter_ms=ccl_timing[0],
                    detector_total_ms=(time.perf_counter() - total_started) * 1000.0,
                )
            return result
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"HrSegNet TensorRT output validation failed: {exc}"
            ) from exc

    def close(self) -> None:
        cudart = self._cudart
        stream = self._stream
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(HrSegNet close)",
                )
            except Exception:
                pass
        for event in self._trt_timing_events:
            try:
                if cudart is not None:
                    self._cuda_call(cudart.cudaEventDestroy(event), "cudaEventDestroy(HrSegNet)")
            except Exception:
                pass
        self._trt_timing_events = ()
        if getattr(self, "_gpu_preprocessor", None) is not None:
            try:
                self._gpu_preprocessor.close()
            except Exception:
                pass
            self._gpu_preprocessor = None
        if self._gpu_postprocessor is not None:
            try:
                self._gpu_postprocessor.close()
            except Exception:
                pass
            self._gpu_postprocessor = None
        for attribute in ("_device_input", "_device_output"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(cudart.cudaFree(allocation), f"cudaFree({attribute})")
                except Exception:
                    pass
                setattr(self, attribute, None)
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy(HrSegNet)")
            except Exception:
                pass
        for attribute in ("_host_input_pointer", "_host_output_pointer"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(cudart.cudaFreeHost(allocation), f"cudaFreeHost({attribute})")
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
