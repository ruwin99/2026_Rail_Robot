"""ROS adapter for serialized SIDE/TOP realtime model execution."""
from __future__ import annotations

from copy import deepcopy
import threading
import time

import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rail_robot_interfaces.msg import CameraFrame as CameraFrameMessage, DetectionSample, ModelTiming
from std_msgs.msg import String

from ..control import (
    highest_rust_grade,
    obstacle_detected_in_control_roi,
    validated_crack_ratio,
)
from ..settings import REALTIME_INPUT_MAX_AGE_SECONDS, REALTIME_RESULT_MAX_AGE_SECONDS
from ..vision import (
    AlternatingRealtimeInference,
    AlternatingTopRealtimeInference,
    extract_realtime_control_rois,
)
from .common import qos_events, qos_sensor, source_monotonic, stamp_seconds, stamp_from_seconds
from .images import encode_image, message_to_frame
from .deployment import create_realtime_detectors


# 원본 토픽 → 실시간 모델 추론 → vision/results 흐름을 담당한다.
# 이 노드는 판정 자료만 발행하며 모터 명령은 control 노드가 결정한다.
# ROI 좌표·모델 교대 순서는 node_runtime/vision.py, 모델 선택·생성은 ros/deployment.py를 본다.
# 마스크 메시지 변환은 ros/images.py, 부식 등급·균열 비율 판정은 node_runtime/control.py로 이어진다.
class VisionNode(Node):
    """One callback group owns every invocation of the shared crack detector.

    Without injected detectors, startup-only ROS parameters select and validate
    deployment plans. Supplied detectors must already have passed validation.
    """

    def __init__(
        self,
        student_detector=None,
        crack_detector=None,
        obstacle_detector=None,
        *,
        node_name: str = "rail_vision",
        **node_kwargs,
    ) -> None:
        super().__init__(node_name, **node_kwargs)
        self._resource_lock = threading.RLock()
        self._closed = False
        self._stopping = False
        self._failure = None
        self._successful_samples = {}
        self._detectors = ()
        try:
            # 실기기 실행은 deployment에서 세 모델을 함께 생성·검증한다. 주입할 때도
            # 학생 부식·균열·장애물 모델이 모두 필요하므로 일부만 넘기면 시작을 중단한다.
            supplied = (student_detector, crack_detector, obstacle_detector)
            if all(detector is None for detector in supplied):
                supplied = create_realtime_detectors(self)
            elif any(detector is None for detector in supplied):
                raise ValueError("Supply all three realtime detectors or none.")
            self._detectors = supplied
            student_detector, crack_detector, obstacle_detector = supplied
        except Exception:
            super().destroy_node()
            raise
        self._pending = {}
        self._last_keys = {}
        self._streams = {}
        # SIDE/TOP이 같은 균열 모델을 공유하므로 콜백 그룹에서 추론 호출을 직렬화한다.
        # executor 스레드가 여러 개여도 이 그룹의 두 GPU 호출이 동시에 실행되지는 않는다.
        # _pending은 역할별 최신 입력, _inference는 역할별 교대 순서·캐시를 보관한다.
        self._group = MutuallyExclusiveCallbackGroup()
        self._inference = {
            "side": AlternatingRealtimeInference(
                student_detector,
                crack_detector,
                multitask_enabled=False,
                hybrid_enabled=False,
            ),
            "top": AlternatingTopRealtimeInference(obstacle_detector, crack_detector),
        }
        try:
            self._results = self.create_publisher(DetectionSample, "vision/results", qos_events)
            self._timings = self.create_publisher(ModelTiming, "debug/vision/timing", qos_events)
            self._faults = self.create_publisher(String, "runtime/fault", qos_events)
            self._camera_subscriptions = [
                self.create_subscription(
                    CameraFrameMessage,
                    f"camera/{role}",
                    lambda message, role=role: self._receive(role, message),
                    qos_sensor,
                    callback_group=self._group,
                )
                for role in ("side", "top")
            ]
            self._timer = self.create_timer(0.001, self._process_latest, callback_group=self._group)
        except Exception:
            self.destroy_node()
            raise

    def _fault(self, detail: str) -> None:
        # 형식 오류나 모델 예외는 _failure에 남아 이후 추론과 readiness를 막는다.
        # 단순히 오래된 입력을 버리는 warning과 달리 runtime/fault까지 발행되는 실패 경로다.
        self._failure = detail
        message = String()
        message.data = f"vision: {detail}"
        if not self._closed:
            self._faults.publish(message)
        self.get_logger().error(message.data)

    def _receive(self, role, message) -> None:
        with self._resource_lock:
            if self._closed or self._stopping:
                return
            self._receive_locked(role, message)

    def _receive_locked(self, role, message) -> None:
        # 토픽 역할과 메시지 역할이 일치한 뒤 원본 stamp의 나이를 확인한다.
        # Dropped 경고가 반복되면 입력 시각·카메라 주기부터 보고 모델 출력 문제와 구분한다.
        if message.camera_role != role:
            self._fault(f"camera role mismatch on {role} topic")
            return
        try:
            source_monotonic(
                message.header.stamp,
                self.get_clock().now().nanoseconds / 1e9,
                time.monotonic(),
                REALTIME_INPUT_MAX_AGE_SECONDS,
            )
        except ValueError as exc:
            self.get_logger().warning(f"Dropped {role} frame: {exc}")
            return
        try:
            if not message.stream_id:
                raise ValueError("Camera frame has no stream ID")
            key = (stamp_seconds(message.header.stamp), message.sequence)
            previous = self._last_keys.get(role)
            if previous is not None:
                same_stream = self._streams[role] == message.stream_id
                if same_stream and (key[1] <= previous[1] or key[0] < previous[0]):
                    return
                if not same_stream:
                    if key[0] <= previous[0]:
                        return
                    # 카메라 재시작을 감지하면 교대 추론의 캐시와 준비 완료 이력을 비운다.
                    # 새 stream_id라도 시각이 이전 원본보다 과거이면 지연 도착한 데이터로 버린다.
                    self._inference[role].reset()
                    for sample_key in list(self._successful_samples):
                        if sample_key[0] == role:
                            del self._successful_samples[sample_key]
            self._streams[role] = message.stream_id
        except Exception as exc:
            self._fault(f"{role}: {exc}")
            return
        # 역할마다 마지막 메시지 한 개만 덮어써 저장한다. 여기서는 이미지 디코딩을 하지 않는다.
        # 추론 속도보다 입력이 빨라도 누적 영상 큐를 만들지 않고 다음 처리에서 최신 것을 꺼낸다.
        self._last_keys[role] = key
        self._pending[role] = message

    def _process_latest(self) -> None:
        with self._resource_lock:
            if self._closed or self._stopping or self._failure:
                return
            self._process_latest_locked()

    def _process_latest_locked(self) -> None:
        for role in ("side", "top"):
            if self._stopping:
                return
            original = self._pending.pop(role, None)
            if original is None:
                continue
            try:
                frame = message_to_frame(
                    original,
                    ros_now_seconds=self.get_clock().now().nanoseconds / 1e9,
                    monotonic_now=time.monotonic(),
                    max_age=REALTIME_INPUT_MAX_AGE_SECONDS,
                )
            except ValueError as exc:
                # 수신 때 신선했던 입력도 앞 모델의 GPU 처리 동안 만료될 수 있어 재검사한다.
                # 만료 입력은 버리고 다음 입력을 기다린다. 결과가 끊겼을 때의 유지/OFF 판단은
                # control 노드의 감시 로직이 담당하므로 여기서 임의의 정상 결과를 만들지 않는다.
                self.get_logger().warning(f"Dropped {role} frame: {exc}")
                continue
            except Exception as exc:
                self._fault(f"{role}: {exc}")
                continue
            try:
                # SIDE 부식은 첫 ROI, TOP 장애물은 전체 원본을 쓴다. 균열은 별도 ROI이며
                # 각 역할의 process가 주 모델과 균열 모델을 번갈아 실행한다.
                first_roi, crack_roi = extract_realtime_control_rois(frame.frame)
                first_input = first_roi if role == "side" else frame.frame
                outcome = self._inference[role].process(frame, first_input, crack_roi)
                if role == "side":
                    samples = (("rust", outcome.display_rust_result,
                                outcome.rust_start_monotonic_ns, outcome.rust_end_monotonic_ns,
                                outcome.rust_execution_id),
                               ("crack", outcome.display_crack_result,
                                outcome.crack_start_monotonic_ns, outcome.crack_end_monotonic_ns,
                                outcome.crack_execution_id))
                else:
                    samples = (("obstacle", outcome.display_obstacle_result,
                                outcome.obstacle_start_monotonic_ns, outcome.obstacle_end_monotonic_ns,
                                outcome.obstacle_execution_id),
                               ("crack", outcome.display_crack_result,
                                outcome.crack_start_monotonic_ns, outcome.crack_end_monotonic_ns,
                                outcome.crack_execution_id))
                # display_*에는 이번 원본에서 실제 실행한 모델 결과만 들어온다.
                # 교대 실행 캐시를 새 stamp/sequence로 재발행하면 다른 사진에 마스크가 붙으므로
                # 결과가 None인 모델은 건너뛰고 실제 수행한 모델의 시간·판정만 발행한다.
                for detector, result, start_ns, end_ns, execution_id in samples:
                    if result is not None:
                        self._publish_timing(original, detector, start_ns, end_ns, execution_id)
                        self._publish_result(original, detector, result)
            except Exception as exc:
                self._fault(f"{role}: {exc}")

    def _publish_timing(self, original, model, start_ns, end_ns, execution_id) -> None:
        if self._stopping or not execution_id or start_ns <= 0 or end_ns < start_ns:
            return
        timing = ModelTiming()
        # header는 처리 완료 시각, source_stamp는 원본 시각이다.
        # processing_ms는 해당 모델 처리시간으로 영상 전송 지연까지 포함하지 않는다.
        timing.header.stamp = stamp_from_seconds(self.get_clock().now().nanoseconds / 1e9)
        timing.source_stamp = deepcopy(original.header.stamp)
        timing.camera_role = original.camera_role
        timing.stream_id = original.stream_id
        timing.sequence = original.sequence
        timing.model = model
        timing.execution_id = execution_id
        timing.processing_start_monotonic_ns = start_ns
        timing.processing_end_monotonic_ns = end_ns
        timing.processing_ms = (end_ns - start_ns) / 1e6
        self._timings.publish(timing)

    def _publish_result(self, original, detector, result) -> None:
        if self._stopping:
            return
        sample = DetectionSample()
        sample.header = deepcopy(original.header)
        sample.camera_role = original.camera_role
        sample.sequence = original.sequence
        sample.stream_id = original.stream_id
        sample.detector = detector
        sample.detector_method = str(getattr(result, "method", ""))
        sample.rust_grade = -1
        try:
            # 입력이 신선했어도 추론 중 유효시간을 넘을 수 있어 결과 발행 직전에 다시 검사한다.
            # 만료 결과도 valid=False와 error를 담아 알리며 해당 모델의 준비 완료 이력을 지운다.
            source_monotonic(
                original.header.stamp,
                self.get_clock().now().nanoseconds / 1e9,
                time.monotonic(),
                REALTIME_RESULT_MAX_AGE_SECONDS,
            )
        except ValueError as exc:
            sample.valid = False
            sample.error = str(exc)
            self._successful_samples.pop((sample.camera_role, detector), None)
            self.get_logger().warning(f"Expired {sample.camera_role}/{detector}: {exc}")
            self._results.publish(sample)
            return
        try:
            if getattr(result, "status", None) != "ready":
                raise ValueError("Inference result is not ready")
            # 모델별 객체를 공통 DetectionSample로 바꾼다. 부식은 등급·class_map,
            # 균열은 비율·이진 마스크, 장애물은 ROI 판정·박스·점수·클래스를 채운다.
            # 모든 박스는 xyxy로 통일하며 비정상 수치가 있으면 valid=False와 fault를 남긴다.
            if detector == "rust":
                sample.rust_grade = highest_rust_grade(result.class_map)
                sample.mask = encode_image(result.class_map, sample.header)
            elif detector == "crack":
                sample.crack_ratio = validated_crack_ratio(result)
                sample.mask = encode_image(result.mask, sample.header)
            else:
                sample.obstacle_detected = obstacle_detected_in_control_roi(result)
            if detector == "obstacle":
                sample.boxes_xyxy = [float(value) for box in result.boxes for value in box]
                sample.confidences = [float(value) for value in result.scores]
                sample.class_ids = [int(value) for value in result.class_ids]
            else:
                sample.boxes_xyxy = [
                    float(value)
                    for x, y, width, height in result.boxes
                    for value in (x, y, x + width, y + height)
                ]
            if not np.isfinite(sample.boxes_xyxy).all():
                raise ValueError("Detection boxes contain non-finite values")
            sample.valid = True
            sample.error = ""
            self._successful_samples[(sample.camera_role, detector)] = deepcopy(sample.header.stamp)
        except Exception as exc:
            sample.valid = False
            sample.error = str(exc)
            self._fault(f"{sample.camera_role}/{detector}: {exc}")
        self._results.publish(sample)

    def readiness(self):
        # 카메라 두 대의 네 분석 역할이 모두 최근의 유효 결과를 낸 뒤에만 준비 완료다.
        # 시작이 대기하면 _successful_samples의 누락 키와 vision/results의 valid/error를 본다.
        with self._resource_lock:
            if self._closed or self._stopping or self._failure:
                return False, self._failure or "Vision node is closed."
            required = (("side", "rust"), ("side", "crack"), ("top", "obstacle"), ("top", "crack"))
            try:
                for key in required:
                    source_monotonic(
                        self._successful_samples[key],
                        self.get_clock().now().nanoseconds / 1e9,
                        time.monotonic(),
                        REALTIME_RESULT_MAX_AGE_SECONDS,
                    )
            except (KeyError, ValueError):
                return False, "Waiting for fresh results from all four camera/detector roles."
            return True, "All realtime detector roles have fresh validated results."

    def prepare_shutdown(self):
        self._stopping = True

    def destroy_node(self):
        with self._resource_lock:
            return self._destroy_locked()

    def _destroy_locked(self):
        if not self._closed:
            self._closed = True
            seen = set()
            for detector in self._detectors:
                if id(detector) in seen:
                    continue
                seen.add(id(detector))
                close = getattr(detector, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception as exc:
                        self.get_logger().error(f"Detector shutdown failed: {exc}")
        return super().destroy_node()
