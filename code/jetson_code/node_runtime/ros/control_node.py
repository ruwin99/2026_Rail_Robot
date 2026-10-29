"""ROS adapter for validated vision results and temporal actuator decisions."""
from __future__ import annotations

# /mission/state와 /vision/results를 받아 청소 모터·펌프의 /control/decisions를 발행한다.
# 읽는 순서: _enabled(임무 허용) → _on_sample(검증·이력) → _tick(유효기간을 붙여 발행).
# 실제 UART 출력과 만료 차단은 stm_bridge_node.py가 담당한다.
# 판정 기준·이력 개수·일시적 손실 유지 정책은 ../control.py의 History/Hold 구현을 확인한다.
# 시간 변환과 오래된 메시지 판정은 common.py, 장애물 결과 자료형은 obstacle_detector.py에서 가져온다.

import time
import threading
import uuid
from functools import wraps
from types import SimpleNamespace

import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from rail_robot_interfaces.msg import ActuatorDecision, DetectionSample, MissionStatus

from obstacle_detector import ObstacleDetectionResult
from ..control import (
    FrontControlHistory, RealtimeControlHistory, RealtimeActuatorDecisionHold,
    front_outcome_requires_immediate_off, side_outcome_requires_immediate_off,
)
from ..settings import REALTIME_MODE, REALTIME_RESULT_MAX_AGE_SECONDS
from .common import (
    qos_events, qos_state, source_monotonic, stamp_from_seconds, stamp_seconds,
)


def _mono8(message):
    # 마스크를 픽셀 배열로 바꾸기 전에 인코딩·가로/세로·행 간격·바이트 수가 일치하는지 검사한다.
    # width와 step이 다르면 행 끝 패딩을 제외한다. 레이아웃 오류는 모델 결과 자체보다 메시지 생성 쪽을 확인한다.
    if message.encoding != "mono8" or message.height <= 0 or message.width <= 0:
        raise ValueError("Detection mask must be a nonempty mono8 image.")
    if message.step < message.width or len(message.data) != message.step * message.height:
        raise ValueError("Detection mask byte layout is invalid.")
    return np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(
        message.height, message.step,
    )[:, :message.width]


def _serialized(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return invoke


class ControlNode(Node):
    def __init__(self, *, node_name="rail_control", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self._lock = threading.RLock()
        self._closed = False
        self._session = uuid.uuid4().hex
        self._sequence = 0
        self._mode = None
        self._mission_seen = None
        self._mission_stamp = None
        self._failed = False
        self._history = {"side": RealtimeControlHistory(), "front": FrontControlHistory()}
        # history는 판정에 필요한 모델 결과를, hold는 마지막 유효 출력을 유지할 수 있는 시간을 관리한다.
        # 내부 카메라 역할 top은 아래 _on_sample에서 청소 장치 역할 front로 변환한다.
        self._hold = {role: RealtimeActuatorDecisionHold() for role in self._history}
        self._last_sequence = {}
        self._streams = {}
        self._retired_streams = {"side": set(), "front": set()}
        self._last_source_stamp = {}
        self._reason = {role: "not_realtime" for role in self._history}
        self._decisions = self.create_publisher(ActuatorDecision, "control/decisions", qos_events)
        self._faults = self.create_publisher(String, "runtime/fault", qos_events)
        self.create_subscription(DetectionSample, "vision/results", self._on_sample, qos_events)
        self.create_subscription(MissionStatus, "mission/state", self._on_mission, qos_state)
        self.create_subscription(String, "runtime/fault", self._on_fault, qos_events)
        self.create_timer(0.05, self._tick)

    def _ros_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _clear(self, role, reason):
        # 이력과 유지 출력을 함께 지워 이전 모드/카메라의 판단이 새 입력에 섞이지 않게 한다.
        # _reason은 내부 진단 사유다. 실제 OFF 메시지의 reason은 _tick에서 safe_off로 통합된다.
        self._history[role].reset()
        self._hold[role].clear()
        self._reason[role] = reason

    @_serialized
    def readiness(self):
        ready = not self._closed and not self._failed
        return ready, "Control is initialized and inhibited until REALTIME." if ready else "Control is closed or faulted."

    @_serialized
    def _on_fault(self, message):
        if self._closed:
            return
        if message.data:
            self._failed = True
            for role in self._history:
                self._clear(role, "runtime_fault")
            self._tick()

    def _fault(self, reason):
        if not self._failed:
            message = String()
            message.data = f"control: {reason}"
            self._on_fault(message)
            self._faults.publish(message)

    @_serialized
    def _on_mission(self, message):
        if self._closed:
            return
        try:
            # 오래되거나 역순으로 온 임무 상태는 채택하지 않는다. 이 경우 _mission_seen도 갱신하지 않는다.
            # 계속 무시되면 _enabled의 신선도 제한으로 출력이 억제되므로 mode뿐 아니라 header.stamp도 확인한다.
            observed = source_monotonic(message.header.stamp, self._ros_now(), time.monotonic(), 0.8)
            issued = stamp_seconds(message.header.stamp)
        except ValueError:
            return
        if self._mission_stamp is not None and issued < self._mission_stamp:
            return
        self._mission_stamp = issued
        changed = message.mode != self._mode
        self._mode = message.mode
        self._mission_seen = observed
        if message.failed:
            self._failed = True
        if changed or message.mode != REALTIME_MODE or self._failed:
            for role in self._history:
                self._clear(role, "mission_transition")
        self._tick()

    def _enabled(self, now):
        # REALTIME 상태뿐 아니라 최근 0.8초 안의 임무 상태가 필요하다. 상태 통신이 끊기면 출력을 억제한다.
        return (
            not self._failed and self._mode == REALTIME_MODE
            and self._mission_seen is not None and now - self._mission_seen <= 0.8
        )

    @_serialized
    def _on_sample(self, message):
        if self._closed:
            return
        now = time.monotonic()
        if not self._enabled(now):
            return
        role = {"side": "side", "top": "front"}.get(message.camera_role)
        # side는 녹/균열, top(front)은 장애물/균열 조합만 허용한다.
        # 다른 조합은 단순 누락이 아니라 인터페이스 오류이므로 /runtime/fault로 전파한다.
        allowed = {"side": ("rust", "crack"), "front": ("obstacle", "crack")}
        if role is None or message.detector not in allowed[role]:
            self._fault("Unexpected camera/detector combination.")
            return
        try:
            observed = source_monotonic(
                message.header.stamp, self._ros_now(), now,
                REALTIME_RESULT_MAX_AGE_SECONDS,
            )
            issued = stamp_seconds(message.header.stamp)
        except ValueError:
            # 전달받은 결과가 너무 오래되면 그 역할의 기존 유지 출력도 즉시 비운다.
            # 원본 프레임 읽기 완료 시각을 쓰므로 추론이 끝난 시각으로 덮어쓴 메시지는 이 검사를 왜곡한다.
            self._clear(role, "stale_detection")
            self._tick()
            return
        stream = message.stream_id
        if not stream:
            self._fault("Detection is missing its camera stream ID.")
            return
        if stream in self._retired_streams[role]:
            return
        previous_stream = self._streams.get(role)
        if stream != previous_stream:
            # 카메라가 재시작되면 이전 분석 이력을 지운다. 종료된 스트림의 지연 메시지는 다시 채택하지 않는다.
            if issued <= self._last_source_stamp.get(role, float("-inf")):
                return
            if previous_stream is not None:
                self._retired_streams[role].add(previous_stream)
            self._streams[role] = stream
            self._clear(role, "camera_stream_changed")
            self._tick()
        key = (role, stream, message.detector)
        # 검출기별 sequence와 역할별 원본 시각을 모두 검사해 중복·역순 결과를 버린다.
        # 토픽에 메시지가 보이는데 이력이 늘지 않으면 stream_id, detector, sequence, header.stamp를 비교한다.
        if message.sequence <= self._last_sequence.get(key, -1):
            return
        if issued < self._last_source_stamp.get(role, float("-inf")):
            return
        self._last_source_stamp[role] = issued
        self._last_sequence[key] = message.sequence
        if not message.valid:
            self._clear(role, message.error or "invalid_detection")
            self._tick()
            return
        try:
            outcome = SimpleNamespace(
                # ROS 메시지를 기존 제어 로직이 받던 결과 객체로 변환한다. 이 노드에서 모델을 다시 실행하지 않는다.
                # 녹 등급은 마스크 최댓값과 일치해야 하며, 불일치/잘못된 마스크는 아래 fault 경로로 간다.
                ready=True, frame_read_completed_at=observed,
                display_rust_result=None, display_crack_result=None,
                display_obstacle_result=None,
            )
            if message.detector == "rust":
                mask = _mono8(message.mask)
                if int(np.max(mask)) != message.rust_grade:
                    raise ValueError("Rust grade does not match the supplied class map.")
                outcome.display_rust_result = SimpleNamespace(status="ready", class_map=mask)
            elif message.detector == "crack":
                outcome.display_crack_result = SimpleNamespace(
                    status="ready", method=message.detector_method,
                    mask=_mono8(message.mask), crack_ratio=message.crack_ratio,
                )
            else:
                outcome.display_obstacle_result = ObstacleDetectionResult(
                    (), message.detector_method, 0.30, message.obstacle_detected,
                )
            history = self._history[role]
            history.update(outcome)
            # 새 위험은 기존 출력 유지보다 우선한다. 정상 출력은 필요한 모델 이력이 준비된 후에만 갱신한다.
            immediate_off = (
                side_outcome_requires_immediate_off(outcome, history_ready=history.ready)
                if role == "side" else front_outcome_requires_immediate_off(outcome)
            )
            if immediate_off:
                self._hold[role].clear()
                self._reason[role] = "new_hazard"
            elif history.ready:
                self._hold[role].accept(history.decision(), now=now)
                self._reason[role] = "validated_history"
        except (ValueError, TypeError, RuntimeError) as exc:
            self._fault(str(exc))
            return
        self._tick()

    @_serialized
    def _tick(self):
        if self._closed:
            return
        now, ros_now = time.monotonic(), self._ros_now()
        enabled = self._enabled(now)
        for role, history in self._history.items():
            if not enabled:
                self._clear(role, "mission_inhibited")
            hold = self._hold[role]
            decision = None
            if enabled:
                # 주기 발행은 새 분석 결과와 별개다. hold.current/begin_loss가 정한 범위 안에서만 기존 결정을 사용한다.
                # OFF가 계속되면 임무 신선도 → history.ready → hold 유효시간 순으로 확인한다.
                decision = hold.current(now=now) if history.ready else hold.begin_loss(now=now)
            message = ActuatorDecision()
            # ON 조건이 없을 때도 OFF 결정을 발행한다. 이 토픽은 구동 요청이며 STM 실행 확인값이 아니다.
            message.header.stamp = stamp_from_seconds(ros_now)
            message.session_id = self._session
            self._sequence += 1
            message.sequence = self._sequence
            message.role = role
            message.cleaner_duty_percent = 0.0 if decision is None else decision.cleaner_duty_percent
            message.pump_on = False if decision is None else decision.pump_on
            remaining = 0.05 if decision is None else max(0.0, hold.valid_until - now)
            # 주기 발행으로 기존 결정의 수명이 계속 연장되지 않도록 hold의 남은 유효시간을 전달한다.
            message.valid_until = stamp_from_seconds(ros_now + remaining)
            message.reason = self._reason[role] if decision is not None else "safe_off"
            self._decisions.publish(message)

    @_serialized
    def prepare_shutdown(self):
        if not self._closed:
            self._failed = True
            self._tick()
            self._closed = True

    @_serialized
    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
