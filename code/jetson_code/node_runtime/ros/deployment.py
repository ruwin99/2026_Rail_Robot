"""Declared ROS deployment parameters and validated, lazy model construction.

Defaults are the approved local plan identities; reading this module does not
open a camera, create a CUDA context, or initialize ROS. Runtime observations
and configured report metadata are deliberately distinguished.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
import os
from pathlib import Path


# [모델/장치 설정] ROS 실행의 기본 카메라 경로, 엔진 파일, 해시, 임계값을 모아둔 곳이다.
# 실시간용 student와 정지 촬영용 teacher는 입력 크기와 목적이 다르므로 구분해서 확인한다.
# 설정/생성 오류는 이 파일에서, 전처리·TensorRT 추론·후처리 오류는 rust_detector.py,
# hrsegnet_crack_detector.py, obstacle_detector.py의 detect()와 그 하위 CUDA 모듈에서 추적한다.
# rebuilt_20260929: 같은 ONNX·같은 trtexec 옵션으로 이 Jetson(Orin Nano Super 25W)에서 다시 빌드한
# 엔진이다. 실제 detect() 경로 A/B에서 출력이 같고(균열 촬영은 비트 동일) 지연 꼬리가 줄어 채택했다.
# 촬영 녹(R101 FP32)은 재빌드 이득이 없어 기존 plan을 유지한다. 원본 plan은 그대로 남아 있다.
DEFAULTS = {
    "side_camera_device": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0",
    "top_camera_device": "/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0",
    "realtime_rust_engine": "/home/bitdol/models/rebuilt_20260929/realtime-rust-mnv2-os16-finetuned-w1280-h240-fp16-rb20260929.plan",
    "realtime_rust_sha256": "efa2c3d9fedbbde88d210d8290b3dde6ede2a00f828667f86e7e52503dba600d",
    "realtime_rust_profile": "student",
    "realtime_crack_engine": "/home/bitdol/models/rebuilt_20260929/hrsegnet-b32-steelcrack-aug400-realtime-w1280-h128-fp16-rb20260929.plan",
    "realtime_crack_sha256": "0ae230c4c9a6cf00de0b108fc69f27e93e9306c457a2c06ffc79fb944896fcd9",
    "realtime_crack_probability_threshold": 0.55,
    "realtime_crack_min_component_pixels": 20,
    "capture_rust_engine": "/home/bitdol/models/capture_r101_hardneg_v6_20260821/plans/capture-rust-r101-os8-hardneg-v6-w1280-h720-fp32-notf32.plan",
    "capture_rust_sha256": "7c442abf7049bcc56d093e0dc7dd2a347caaee74dc8e20006b63ee032b6f22d4",
    "capture_rust_profile": "teacher",
    "capture_crack_engine": "/home/bitdol/models/rebuilt_20260929/hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32-rb20260929.plan",
    "capture_crack_sha256": "72ad6ca53002d781ce750bb7c8857efc424ec44ddf889cf712a4351c3d3d9953",
    "capture_crack_probability_threshold": 0.55,
    "capture_crack_min_component_pixels": 20,
    "obstacle_engine": "/home/bitdol/models/rebuilt_20260929/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-int-h736-fp16-notf32-rb20260929.plan",
    "obstacle_sha256": "280e9669624b67c9eb70c3db9661ccf6ae2a78c978603e7a39223a635cb6a7d6",
    "obstacle_confidence_threshold": 0.424,
    "output_directory": "~/rail_robot_data/captures",
    "report_path": "",
    "training_camera_role": "side",
    "training_output_directory": "~/rail_robot_data/training",
}

CAMERA_PARAMETERS = ("side_camera_device", "top_camera_device")
REALTIME_PARAMETERS = (
    "realtime_rust_engine", "realtime_rust_sha256", "realtime_rust_profile",
    "realtime_crack_engine", "realtime_crack_sha256",
    "realtime_crack_probability_threshold", "realtime_crack_min_component_pixels",
    "obstacle_engine", "obstacle_sha256", "obstacle_confidence_threshold",
)
INSPECTION_PARAMETERS = (
    "capture_rust_engine", "capture_rust_sha256", "capture_rust_profile",
    "capture_crack_engine", "capture_crack_sha256",
    "capture_crack_probability_threshold", "capture_crack_min_component_pixels",
    "realtime_rust_engine", "realtime_rust_sha256", "realtime_rust_profile",
    "realtime_crack_engine", "realtime_crack_sha256",
    "realtime_crack_probability_threshold", "realtime_crack_min_component_pixels",
    "output_directory", "report_path",
)


def read_parameters(node, names):
    """Resolve startup-only settings; dynamic model/device changes are unsupported."""
    # [설정 해석] 이미 선언된 파라미터면 현재 값을 읽고, 아니면 DEFAULTS를 기본값으로 선언한다.
    # launch/YAML에서 넘긴 값은 선언할 때 반영된다. 새 선언은 read_only이므로 실행 중 교체하지 않는다.
    # 잘못된 엔진이 선택되면 환경변수만 보지 말고 해당 노드의 ROS 파라미터와 launch 입력도 확인한다.
    from rcl_interfaces.msg import ParameterDescriptor

    values = {}
    for name in names:
        if node.has_parameter(name):
            value = node.get_parameter(name).value
        else:
            value = node.declare_parameter(
                name, DEFAULTS[name], descriptor=ParameterDescriptor(read_only=True)
            ).value
        values[name] = value
    return values


def require_tf32_baseline():
    """Set the baseline before model construction; reject an explicit conflict."""
    # [정밀도 전제] 모델 생성 전에 TF32 비활성 환경을 맞춘다. 미지정은 0으로 채우지만
    # 사용자가 다른 값을 명시했다면 조용히 덮어쓰지 않고 중단한다. 엔진을 재생성하는 코드는 아니다.
    value = os.environ.get("NVIDIA_TF32_OVERRIDE")
    if value is not None and value != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be exactly 0 for the approved plans.")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"


def _validated_digest(digest):
    digest = str(digest).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Every enabled engine requires a 64-character SHA-256.")
    return digest


def validate_engine(path, digest):
    # 파일명만 같아도 다른 모델일 수 있으므로 실제 바이트의 SHA-256을 확인하고 불일치 시 중단한다.
    digest = _validated_digest(digest)
    path = Path(path).expanduser().resolve()
    calculated = hashlib.sha256()
    # [파일 검증] 빈 파일·잘못된 해시는 추론 전에 거절한다. FileNotFoundError는 경로/배포 문제,
    # SHA-256 mismatch는 지정한 엔진과 실제 파일 내용이 다른 문제로 나눠 확인한다.
    # 해시가 맞다는 사실만으로 현재 Jetson/TensorRT에서 실행 가능하거나 정확도가 검증되지는 않는다.
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            calculated.update(chunk)
            size += len(chunk)
    if not size:
        raise ValueError(f"TensorRT engine is empty: {path}")
    if calculated.hexdigest() != digest:
        raise ValueError(f"TensorRT engine SHA-256 mismatch: {path}")
    return path, digest


def _thresholds(values, role):
    # [판정 설정 검증] 균열 확률 임계값과 최소 연결 성분 픽셀 수를 별도로 검사한다.
    # 마스크가 너무 적거나 많으면 role이 realtime인지 capture인지 먼저 확인한 뒤
    # 해당 detector의 후처리를 본다. 여기서는 수치 범위만 검사하며 마스크를 만들지는 않는다.
    threshold = values[f"{role}_crack_probability_threshold"]
    count = values[f"{role}_crack_min_component_pixels"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Crack probability threshold must be finite and between 0 and 1.")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Crack minimum component pixels must be a positive integer.")
    return float(threshold), count


def _validate_profile(values, role):
    expected = "student" if role == "realtime" else "teacher"
    if values[f"{role}_rust_profile"] != expected:
        raise ValueError(f"{role} rust requires the {expected} profile.")


def close_detectors(detectors):
    """Best-effort cleanup of a partial factory; preserve the original failure."""
    seen = set()
    # [부분 생성 실패 정리] 모델 세 개 중 두 번째가 실패해도 첫 번째 GPU 자원을 정리한다.
    # 같은 객체가 여러 번 들어오면 한 번만 닫고, close 예외로 원래 생성 실패를 가리지 않는다.
    for detector in detectors:
        if detector is None or id(detector) in seen:
            continue
        seen.add(id(detector))
        try:
            detector.close()
        except Exception:
            pass


def create_realtime_detectors(node):
    # vision_node용 녹·균열·장애물 모델 생성. 설정/해시 검증을 마친 뒤 TensorRT 관련 모듈을 읽는다.
    require_tf32_baseline()
    values = read_parameters(node, REALTIME_PARAMETERS)
    _validate_profile(values, "realtime")
    threshold, components = _thresholds(values, "realtime")
    confidence = values["obstacle_confidence_threshold"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Obstacle confidence threshold must be finite and between 0 and 1.")
    plans = {
        role: validate_engine(values[f"{role}_engine"], values[f"{role}_sha256"])
        for role in ("realtime_rust", "realtime_crack", "obstacle")
    }
    # [실시간 모델 생성] 세 파일 검증 뒤 녹(student) → 균열(realtime) → 장애물 순으로 생성한다.
    # 반환 튜플 순서는 vision_node가 받는 순서와 같다. 교대 실행/ROI 선택은 그 노드의 책임이다.
    # 'TensorRT/CUDA required' 예외는 아래 import/생성에서 온 것이므로 먼저 Python 의존 환경을 확인한다.
    from rust_detector import RustDetector, STUDENT_PROFILE
    from hrsegnet_crack_detector import HrSegNetCrackDetector
    from obstacle_detector import ObstacleDetector

    created = []
    try:
        created.append(RustDetector(plans["realtime_rust"][0], STUDENT_PROFILE, plans["realtime_rust"][1], gpu_argmax=True))
        # 180행 ROI의 전처리와 마스크 복원을 GPU에서 한다. 전처리는 CPU와 비트 단위로 같고,
        # 복원은 logit 차이 선형 보간이라 0/1 NEAREST보다 가는 균열 경계가 정확하다.
        created.append(HrSegNetCrackDetector(*plans["realtime_crack"], threshold, components, role="realtime",
                                             gpu_preprocess=True, linear_restore=True))
        created.append(ObstacleDetector(*plans["obstacle"], float(confidence)))
        return tuple(created)
    except Exception:
        close_detectors(created)
        raise


def create_inspection_resources(node):
    """Construct capture engines and a run-specific report without loading realtime engines."""
    # [촬영 검사 자원] teacher 녹과 전체 화면 균열 모델, XLSX 보고서, 출력 경로를 묶어 반환한다.
    # 실시간 모델 정보도 보고서에 필요해 파라미터로 읽지만, 실제 생성하는 모델과 구별해야 한다.
    require_tf32_baseline()
    values = read_parameters(node, INSPECTION_PARAMETERS)
    _validate_profile(values, "capture")
    _validate_profile(values, "realtime")
    threshold, components = _thresholds(values, "capture")
    rt_threshold, rt_components = _thresholds(values, "realtime")
    plans = {
        role: validate_engine(values[f"{role}_engine"], values[f"{role}_sha256"])
        for role in ("capture_rust", "capture_crack")
    }
    # 검사 노드는 촬영용 모델만 생성한다. 보고서의 실시간 모델 정보는 설정값이며,
    # 여기서 그 실시간 엔진을 실행/검증했다는 의미가 아니다.
    for role in ("realtime_rust", "realtime_crack"):
        plans[role] = (
            Path(values[f"{role}_engine"]).expanduser(),
            _validated_digest(values[f"{role}_sha256"]),
        )
    from rust_detector import RustDetector, TEACHER_PROFILE
    from hrsegnet_crack_detector import HrSegNetCrackDetector, DETECTOR_PREFIX
    from inspection_report import InspectionWorkbook
    import numpy as np

    output = Path(values["output_directory"]).expanduser().resolve()
    # [저장 위치] report_path를 지정하지 않으면 촬영 출력 폴더의 부모 아래 reports에 새 이름을 만든다.
    # 촬영 사진을 찾을 때 output_directory, XLSX를 찾을 때 report_path/자동 생성 경로를 따로 확인한다.
    report = Path(values["report_path"]).expanduser().resolve() if values["report_path"] else (
        output.parent / "reports" / f"inspection_{datetime.now():%Y%m%d_%H%M%S_%f}.xlsx"
    )
    created = []
    try:
        created.append(RustDetector(plans["capture_rust"][0], TEACHER_PROFILE, plans["capture_rust"][1]))
        created.append(HrSegNetCrackDetector(*plans["capture_crack"], threshold, components, role="capture"))
        # START 준비 완료 전에 촬영 모델을 예열해 추론 가능 여부를 확인한다.
        # 이 검은 시험 이미지는 검사 사진이나 보고서 측정 결과로 저장하지 않는다.
        warmup = np.zeros((720, 1280, 3), dtype=np.uint8)
        for detector in created:
            result = detector.detect(warmup)
            if result.status != "ready" or result.mask.shape != warmup.shape[:2]:
                raise RuntimeError("Capture model warmup did not return a native ready result.")
        workbook = InspectionWorkbook(
            # [보고서 메타데이터] 엔진 파일/해시/판정 조건을 남겨 나중에 결과의 생성 조건을 추적한다.
            # 실제 사진별 행 추가와 완료 확정은 inspection 작업자와 inspection_report.py에서 처리한다.
            report,
            capture_model_filename=plans["capture_rust"][0].name,
            capture_model_sha256=plans["capture_rust"][1],
            realtime_model_filename=plans["realtime_rust"][0].name,
            realtime_model_sha256=plans["realtime_rust"][1],
            capture_detector=created[0].method,
            realtime_detector="deeplabv3plus-tensorrt/student/configured/gpu-argmax",
            crack_model_filename=plans["capture_crack"][0].name,
            crack_model_sha256=plans["capture_crack"][1],
            crack_detector=created[1].method,
            crack_probability_threshold=threshold,
            capture_crack_min_component_pixels=components,
            realtime_crack_model_filename=plans["realtime_crack"][0].name,
            realtime_crack_model_sha256=plans["realtime_crack"][1],
            realtime_crack_detector=DETECTOR_PREFIX + "configured/logit-margin",
            realtime_crack_probability_threshold=rt_threshold,
            realtime_crack_min_component_pixels=rt_components,
        )
        return created[0], created[1], workbook, output
    except Exception:
        close_detectors(created)
        raise


def create_inspection_obstacle_factory(node):
    """Independent execution context owned by the inspection FIFO worker."""
    # [작업자 전용 장애물 모델] 지금은 설정과 파일을 검증하고 생성 함수(lambda)만 반환한다.
    # 실제 ObstacleDetector 생성은 그 함수를 호출하는 검사 작업자 쪽에서 이루어진다.
    # 실시간 vision 노드의 모델 객체를 공유해 동시에 detect하는 구조가 아니다.
    values = read_parameters(node, (
        "obstacle_engine", "obstacle_sha256", "obstacle_confidence_threshold",
    ))
    path, digest = validate_engine(values["obstacle_engine"], values["obstacle_sha256"])
    confidence = values["obstacle_confidence_threshold"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("Obstacle confidence threshold must be finite and between 0 and 1.")
    from obstacle_detector import ObstacleDetector
    return lambda: ObstacleDetector(path, digest, confidence)
