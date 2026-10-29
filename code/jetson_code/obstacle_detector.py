from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cuda_obstacle_pipeline import (
    CONTROL_ROI_Y_END,
    CONTROL_ROI_Y_START,
    MODEL_INPUT_NBYTES,
    OUTPUT_ROWS,
    CudaObstaclePipeline,
)


# 장애물 추론: 전방 전체 원본 → GPU 패딩·RGB 정규화 → TensorRT → GPU 후보 필터·좌표 복원
# → CPU 검증·결과 객체. 실제 detect()의 CUDA 단계는 cuda_obstacle_pipeline.py에서 확인한다.
# .plan이 가진 학습 가중치·연산 구조를 실행하며, confidence_threshold는 출력 선택 기준이다.
# 이 파일의 CPU preprocess_frame/result_from_output은 기준 구현이고 실제 detect 경로와 구분한다.
INPUT_NAME = "images"
OUTPUT_NAME = "output0"
CAMERA_FRAME_SHAPE = (720, 1280, 3)
INPUT_SHAPE = (1, 3, 736, 1280)
OUTPUT_SHAPE = (1, 300, 6)
PAD_TOP = 8
PAD_BOTTOM = 8
PAD_VALUE = 114
DEFAULT_CONFIDENCE_THRESHOLD = 0.424
# 이 값은 export 메타데이터이고 여기서 별도 NMS를 실행하는 임계값으로 사용하지 않는다.
METADATA_NMS_IOU = 0.70
CLASS_NAMES = ("obstacle",)
SOURCE_ONNX_SHA256 = (
    "c55fb45efd33345d31ec99c4b2ff81c5ccc7973d5810c88ac51210310a681341"
)
TENSORRT_VERSION_PREFIX = "10.3."
DETECTOR_PREFIX = "yolo26n-tensorrt/realtime/foreign-matter/"


@dataclass(frozen=True)
class ObstacleDetection:
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


@dataclass(frozen=True)
class ObstacleDetectionResult:
    detections: tuple[ObstacleDetection, ...]
    method: str
    confidence_threshold: float
    control_roi_detected: bool

    @property
    def detected(self) -> bool:
        return bool(self.detections)

    @property
    def boxes(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(detection.box_xyxy for detection in self.detections)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(detection.confidence for detection in self.detections)

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(detection.class_id for detection in self.detections)

    @property
    def status(self) -> str:
        return "ready"


@dataclass(frozen=True)
class ObstacleStageTiming:
    preprocess_cpu_ms: float
    input_u8_h2d_ms: float
    gpu_preprocess_ms: float
    trt_inference_ms: float
    gpu_postprocess_ms: float
    compact_d2h_ms: float
    cpu_result_ms: float
    total_ms: float


def _validated_confidence_threshold(confidence_threshold: float) -> float:
    # 0~1 사이 출력 점수의 통과 기준이다. 값을 바꾸면 남는 후보가 달라질 뿐,
    # TensorRT 안의 학습 가중치나 입력 영상 자체가 바뀌지는 않는다.
    confidence_threshold = float(confidence_threshold)
    if not math.isfinite(confidence_threshold) or not 0.0 < confidence_threshold < 1.0:
        raise ValueError(
            "confidence_threshold must be finite and between zero and one."
        )
    return confidence_threshold


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert the fixed full top-camera frame to padded RGB FP32."""

    # CPU 기준 전처리: 1280×720 BGR uint8를 RGB/255 FP32 NCHW로 바꾸고
    # 위아래 8픽셀씩 값 114/255로 채워 (1,3,736,1280)을 만든다. 리사이즈는 하지 않는다.
    # 실제 장치 실행에서는 upload_and_preprocess의 CUDA 커널이 이 계약을 구현한다.
    if not isinstance(frame, np.ndarray) or frame.shape != CAMERA_FRAME_SHAPE:
        shape = getattr(frame, "shape", None)
        raise ValueError(
            "Obstacle TensorRT input must be the fixed 1280x720 HxWx3 top-camera "
            f"frame; got {shape}. The detector does not resize the frame."
        )
    if frame.dtype != np.uint8:
        raise ValueError("Obstacle camera input must contain uint8 BGR pixels.")

    tensor = np.full(
        INPUT_SHAPE,
        np.float32(PAD_VALUE / 255.0),
        dtype=np.float32,
    )
    rgb_chw = frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    tensor[0, :, PAD_TOP : PAD_TOP + CAMERA_FRAME_SHAPE[0], :] = rgb_chw / 255.0
    return np.ascontiguousarray(tensor)


def result_from_output(
    output: np.ndarray,
    method: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ObstacleDetectionResult:
    """Decode the exported one-to-one TopK rows without applying extra NMS."""

    # CPU 기준 후처리의 입력은 (1,300,6)이다. 각 행은 x1,y1,x2,y2,score,class_id이며
    # 일반 YOLO의 원시 feature map처럼 다시 sigmoid·anchor 해독·NMS를 적용하지 않는다.
    # 허용 클래스는 0(obstacle)뿐이고 점수·좌표·클래스 형식이 잘못되면 즉시 오류로 처리한다.
    confidence_threshold = _validated_confidence_threshold(confidence_threshold)
    output = np.asarray(output)
    if output.dtype != np.float32:
        raise ValueError(
            f"Obstacle output must have dtype float32, got {output.dtype}."
        )
    if output.shape != OUTPUT_SHAPE:
        raise ValueError(
            f"Obstacle output must have shape {OUTPUT_SHAPE}, got {output.shape}."
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Obstacle output contains NaN or infinity.")

    rows = output[0]
    boxes = rows[:, :4]
    scores = rows[:, 4]
    class_values = rows[:, 5]
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("Obstacle confidence values must be from zero through one.")
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError("Obstacle output contains a reversed xyxy box.")

    rounded_classes = np.rint(class_values)
    if not np.array_equal(class_values, rounded_classes):
        raise ValueError("Obstacle class IDs must be exact integer-valued floats.")
    class_ids = rounded_classes.astype(np.int64)
    if np.any(class_ids != 0):
        raise ValueError("Obstacle output contains a class ID other than 0.")

    frame_height, frame_width = CAMERA_FRAME_SHAPE[:2]
    detections: list[ObstacleDetection] = []
    # 기본 score>=0.424인 행을 남긴다. 예: 0.50은 통과, 0.40은 제외한다.
    # y에서 PAD_TOP=8을 빼 원본 좌표로 복원하고 경계를 잘라낸 뒤 면적이 없는 박스는 버린다.
    # score는 엔진이 낸 후보 점수이며 부식의 argmax 클래스 번호나 면적 비율과 다른 값이다.
    for index in np.flatnonzero(scores >= confidence_threshold):
        x1, y1, x2, y2 = (float(value) for value in boxes[index])
        x1 = float(np.clip(x1, 0.0, float(frame_width)))
        x2 = float(np.clip(x2, 0.0, float(frame_width)))
        y1 = float(np.clip(y1 - PAD_TOP, 0.0, float(frame_height)))
        y2 = float(np.clip(y2 - PAD_TOP, 0.0, float(frame_height)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            ObstacleDetection(
                box_xyxy=(x1, y1, x2, y2),
                confidence=float(scores[index]),
                class_id=int(class_ids[index]),
            )
        )

    # 전체 화면의 검출 존재와 제어 ROI의 검출 존재를 나눠 전달한다. 예: y=300~400인
    # 박스만 있으면 detected=True여도 control_roi_detected=False다(제어 ROI는 y=480~720).
    # 이 플래그가 곧 모터 명령은 아니며 상위에서 임무 상태·균열 결과와 함께 판단한다.
    return ObstacleDetectionResult(
        detections=tuple(detections),
        method=str(method),
        confidence_threshold=confidence_threshold,
        control_roi_detected=any(
            detection.box_xyxy[1] < CONTROL_ROI_Y_END
            and detection.box_xyxy[3] > CONTROL_ROI_Y_START
            for detection in detections
        ),
    )


def result_from_compact_output(
    compact: np.ndarray,
    control_roi_detected: bool,
    method: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ObstacleDetectionResult:
    """Validate GPU-compacted rows without re-decoding raw TensorRT output."""

    # 실제 GPU 경로에서는 필터링·패딩 제거가 끝난 N×6 후보만 CPU로 가져온다.
    # 여기서는 다시 y-8을 하거나 별도 NMS를 하지 않고 값·원본 경계·ROI 플래그 일치만 검사한다.
    # compact 좌표와 control_roi_detected가 다르면 CUDA 후처리 계약이 깨진 것으로 처리한다.
    confidence_threshold = _validated_confidence_threshold(confidence_threshold)
    compact = np.asarray(compact)
    if compact.dtype != np.float32:
        raise ValueError(
            f"Obstacle compact output must have dtype float32, got {compact.dtype}."
        )
    if (
        compact.ndim != 2
        or compact.shape[1:] != (OUTPUT_SHAPE[2],)
        or compact.shape[0] > OUTPUT_ROWS
    ):
        raise ValueError(
            "Obstacle compact output must have shape (N, 6), where 0 <= N <= "
            f"{OUTPUT_ROWS}; got {compact.shape}."
        )
    if not isinstance(control_roi_detected, (bool, np.bool_)):
        raise ValueError("Obstacle GPU control ROI flag must be boolean.")
    if not np.all(np.isfinite(compact)):
        raise ValueError("Obstacle compact output contains NaN or infinity.")

    boxes = compact[:, :4]
    scores = compact[:, 4]
    class_values = compact[:, 5]
    frame_height, frame_width = CAMERA_FRAME_SHAPE[:2]
    if np.any(scores < confidence_threshold) or np.any(scores > 1.0):
        raise ValueError(
            "Obstacle compact confidence values violate the locked threshold."
        )
    if np.any(class_values != 0.0):
        raise ValueError("Obstacle compact output contains a class other than 0.")
    if (
        np.any(boxes[:, 0] < 0.0)
        or np.any(boxes[:, 2] > frame_width)
        or np.any(boxes[:, 1] < 0.0)
        or np.any(boxes[:, 3] > frame_height)
        or np.any(boxes[:, 2] <= boxes[:, 0])
        or np.any(boxes[:, 3] <= boxes[:, 1])
    ):
        raise ValueError("Obstacle compact output contains an invalid xyxy box.")

    expected_control = bool(
        np.any((boxes[:, 1] < CONTROL_ROI_Y_END) & (boxes[:, 3] > CONTROL_ROI_Y_START))
    )
    if bool(control_roi_detected) != expected_control:
        raise ValueError(
            "Obstacle GPU control ROI flag disagrees with compact detections."
        )
    detections = tuple(
        ObstacleDetection(
            box_xyxy=tuple(float(value) for value in row[:4]),
            confidence=float(row[4]),
            class_id=0,
        )
        for row in compact
    )
    return ObstacleDetectionResult(
        detections=detections,
        method=str(method),
        confidence_threshold=confidence_threshold,
        control_roi_detected=bool(control_roi_detected),
    )


class ObstacleDetector:
    """Synchronous TensorRT 10.3 runtime for the pinned YOLO26n export."""

    def __init__(
        self,
        engine_path: Path,
        expected_sha256: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        *,
        stage_timing: bool = False,
    ) -> None:
        confidence_threshold = _validated_confidence_threshold(confidence_threshold)
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(
                "expected_sha256 must be a 64-character hexadecimal digest."
            )

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(
                f"Obstacle TensorRT engine was not found: {self.engine_path}"
            )
        self.confidence_threshold = confidence_threshold
        self.engine_sha256: str | None = None
        self.method = (
            f"{DETECTOR_PREFIX}cuda:0/end2end-topk/no-nms/"
            "roi-y480-720/gpu-preprocess/gpu-compact"
        )

        self._trt: Any = None
        self._cudart: Any = None
        self._logger: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._device_input: Any = None
        self._device_output: Any = None
        self._gpu_pipeline: CudaObstaclePipeline | None = None
        self._stage_timing = bool(stage_timing)
        self._timing_events: list[Any] = []
        self.last_stage_timing: ObstacleStageTiming | None = None

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
                    "This obstacle plan requires TensorRT 10.3.x, got "
                    f"{version or 'an unknown version'}."
                )
            self.method = (
                f"{DETECTOR_PREFIX}trt-{version}/cuda:0/end2end-topk/no-nms/"
                "roi-y480-720/gpu-preprocess/gpu-compact"
            )
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            try:
                serialized_engine = self.engine_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read obstacle TensorRT engine: {self.engine_path}"
                ) from exc
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if self.engine_sha256 != expected_sha256:
                raise RuntimeError(
                    "Obstacle TensorRT engine changed after preflight SHA-256 "
                    f"validation: expected {expected_sha256}, got "
                    f"{self.engine_sha256}."
                )
            # SHA-256으로 지정된 엔진인지 확인한 뒤 저장된 가중치와 연산을 실행 가능한 객체로 복원한다.
            # 파일 복원과 I/O 계약 검증에 실패하면 engine_path, SHA-256, TensorRT 버전부터 확인한다.
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT could not deserialize the obstacle engine."
                )
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(
                    "TensorRT could not create an obstacle execution context."
                )

            input_nbytes = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
            output_nbytes = int(np.prod(OUTPUT_SHAPE)) * np.dtype(np.float32).itemsize
            if input_nbytes != MODEL_INPUT_NBYTES:
                raise RuntimeError("Obstacle CUDA input byte contract changed.")
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes),
                "cudaMalloc(obstacle input)",
            )
            self._device_output = self._cuda_call(
                cudart.cudaMalloc(output_nbytes),
                "cudaMalloc(obstacle output)",
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(),
                "cudaStreamCreate(obstacle)",
            )
            if self._stage_timing:
                for _ in range(6):
                    self._timing_events.append(
                        self._cuda_call(
                            cudart.cudaEventCreate(),
                            "cudaEventCreate(obstacle stage timing)",
                        )
                    )
            # images는 전처리된 프레임 버퍼, output0는 300개 후보 행 버퍼의 GPU 주소다.
            # 학습 가중치를 덮어쓰는 단계가 아니라 엔진 실행에 쓸 입력/출력 메모리를 연결한다.
            if not self._context.set_tensor_address(
                INPUT_NAME, int(self._device_input)
            ):
                raise RuntimeError(
                    "TensorRT rejected the obstacle input buffer address."
                )
            if not self._context.set_tensor_address(
                OUTPUT_NAME, int(self._device_output)
            ):
                raise RuntimeError(
                    "TensorRT rejected the obstacle output buffer address."
                )
            self._gpu_pipeline = CudaObstaclePipeline(
                cudart,
                self.confidence_threshold,
            )
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> None:
        # 이름 images/output0, 고정 shape, FP32 LINEAR GPU 입출력을 강제한다.
        # 비슷한 이름의 다른 YOLO export도 출력 행 구조가 다르면 그대로 사용할 수 없다.
        trt = self._trt
        engine = self._engine
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        if len(names) != 2 or set(names) != {INPUT_NAME, OUTPUT_NAME}:
            raise RuntimeError(
                "Obstacle TensorRT engine I/O names must be exactly "
                f"{INPUT_NAME!r} and {OUTPUT_NAME!r}, got {sorted(names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Tensor {INPUT_NAME!r} is not an engine input.")
        if engine.get_tensor_mode(OUTPUT_NAME) != trt.TensorIOMode.OUTPUT:
            raise RuntimeError(f"Tensor {OUTPUT_NAME!r} is not an engine output.")
        for name, expected_shape in (
            (INPUT_NAME, INPUT_SHAPE),
            (OUTPUT_NAME, OUTPUT_SHAPE),
        ):
            actual_shape = tuple(engine.get_tensor_shape(name))
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Obstacle TensorRT tensor {name!r} shape must be "
                    f"{expected_shape}, got {actual_shape}."
                )
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"Obstacle TensorRT tensor {name!r} dtype must be float32, "
                    f"got {dtype}."
                )
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f"Tensor {name!r} must use LINEAR/CHW format.")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(
                    f"Tensor {name!r} must be located on the GPU device."
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

    def _event_elapsed_ms(self, start: Any, end: Any) -> float:
        return float(
            self._cuda_call(
                self._cudart.cudaEventElapsedTime(start, end),
                "cudaEventElapsedTime(obstacle stage timing)",
            )
        )

    def detect(self, frame: np.ndarray) -> ObstacleDetectionResult:
        # 실제 경로는 CPU에서 원본을 준비해 GPU로 올린 뒤 CUDA가 패딩·채널변환·/255를 처리한다.
        # 아래 엔진 실행과 후보 필터도 같은 stream 순서를 따르며, CPU 기준 함수를 호출하지 않는다.
        if (
            self._context is None
            or self._stream is None
            or self._gpu_pipeline is None
        ):
            raise RuntimeError("Obstacle TensorRT detector is closed.")
        timing_events = self._timing_events if self._stage_timing else None
        if timing_events is not None and len(timing_events) != 6:
            raise RuntimeError("Obstacle stage timing events are unavailable.")
        self.last_stage_timing = None
        total_started = time.perf_counter() if timing_events is not None else None
        try:
            preprocess_cpu_ms = self._gpu_pipeline.upload_and_preprocess(
                frame,
                self._device_input,
                self._stream,
                None
                if timing_events is None
                else (timing_events[0], timing_events[1], timing_events[2]),
            )
            # 엔진에 저장된 가중치로 입력을 계산해 output0의 후보 행을 만든다.
            # async는 GPU 예약 방식이다. 이 detect()는 아래 postprocess가 결과를 회수한 뒤 반환한다.
            if not self._context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
            if timing_events is not None:
                self._cuda_call(
                    self._cudart.cudaEventRecord(timing_events[3], self._stream),
                    "cudaEventRecord(obstacle inference end)",
                )
            # CUDA에서 점수 임계값 적용·y 패딩 제거·경계 제한·빈 박스 제거·ROI 검사를 수행한다.
            # 통과한 records와 ROI 요약만 회수하므로 CPU가 매번 원시 300개 행을 다시 처리하지 않는다.
            compact_output = self._gpu_pipeline.postprocess(
                self._device_output,
                self._stream,
                None
                if timing_events is None
                else (timing_events[4], timing_events[5]),
            )
        except Exception as exc:
            raise RuntimeError(f"Obstacle TensorRT inference failed: {exc}") from exc

        cpu_result_started = (
            time.perf_counter() if timing_events is not None else None
        )
        try:
            result = result_from_compact_output(
                compact_output.records,
                compact_output.control_roi_detected,
                self.method,
                self.confidence_threshold,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Obstacle TensorRT output validation failed: {exc}"
            ) from exc
        # 단계별 지연은 last_stage_timing에서 전처리·추론·후처리·전송으로 나눠 본다.
        # total_ms는 이 detect 호출 전체이고 카메라 촬영이나 WebRTC 전송시간은 포함하지 않는다.
        if timing_events is not None:
            cpu_result_ms = (time.perf_counter() - cpu_result_started) * 1000.0
            self.last_stage_timing = ObstacleStageTiming(
                preprocess_cpu_ms=float(preprocess_cpu_ms),
                input_u8_h2d_ms=self._event_elapsed_ms(
                    timing_events[0], timing_events[1]
                ),
                gpu_preprocess_ms=self._event_elapsed_ms(
                    timing_events[1], timing_events[2]
                ),
                trt_inference_ms=self._event_elapsed_ms(
                    timing_events[2], timing_events[3]
                ),
                gpu_postprocess_ms=self._event_elapsed_ms(
                    timing_events[3], timing_events[4]
                ),
                compact_d2h_ms=self._event_elapsed_ms(
                    timing_events[4], timing_events[5]
                ),
                cpu_result_ms=cpu_result_ms,
                total_ms=(time.perf_counter() - total_started) * 1000.0,
            )
        return result

    def close(self) -> None:
        cudart = self._cudart
        stream = self._stream
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(obstacle close)",
                )
            except Exception:
                pass
        gpu_pipeline = self._gpu_pipeline
        if gpu_pipeline is not None:
            try:
                gpu_pipeline.close()
            except Exception:
                pass
        self._gpu_pipeline = None
        timing_events = getattr(self, "_timing_events", [])
        for event in timing_events:
            try:
                if cudart is not None:
                    self._cuda_call(
                        cudart.cudaEventDestroy(event),
                        "cudaEventDestroy(obstacle stage timing)",
                    )
            except Exception:
                pass
        self._timing_events = []
        self.last_stage_timing = None
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
                    "cudaStreamDestroy(obstacle)",
                )
            except Exception:
                pass
        self._stream = None
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None
        self._cudart = None
