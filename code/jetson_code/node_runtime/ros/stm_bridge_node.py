"""Exclusive UART owner for ROS requests and expiring actuator decisions.

Future TTS subscribes to ``tts/obstacle_events`` (std_msgs/String, reliable,
volatile event QoS). Data is exactly ``OBSTACLE_HOLD`` or ``OBSTACLE_CLEAR``;
each observed state transition publishes once, with no audio playback here.
Initial CLEAR and consecutive duplicates produce no announcement. Raw UART
tokens remain on stm/events in legacy mode; RR2 mission event names are
normalized to the existing FSM tokens. HOLD clears cleaning targets and sends
the existing four OFF commands; STM independently owns the physical drive stop.
CLEAR removes only this hold, and requires newly issued control decisions plus
the normal mission/STARTED/freshness gates. It never clears a runtime fault or
an explicit SAFE_OFF phase lock. This is an event stream, not latched state;
late subscribers/restarted bridges cannot reconstruct a prior STM HOLD.
tts/obstacle_state additionally retains the last observed state for late audio
subscribers. No initial CLEAR is invented; bridge restart still needs a new UART token.
"""
from __future__ import annotations

# 임무 실행 중 UART를 단독 소유한다. ROS 명령을 STM 프로토콜로 쓰고 STM 이벤트를 토픽으로 돌려준다.
# 읽는 순서: _command(서비스 명령), _on_decision(청소 요청 검증), _tick(UART 수신·출력 만료).
# UART 문자열 분리·전체 쓰기·장치별 재전송은 ../uart.py, 장치 명령 상수와 포트 기본값은 ../settings.py에 있다.
# 청소 쌍의 OFF/갱신 동작은 ../control.py에서 가져오므로 실제 송신 문자열은 이 파일만으로 확정하지 않는다.

from concurrent.futures import TimeoutError as FutureTimeoutError
import math
import threading
import time

import serial
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Bool, String
from rail_robot_interfaces.msg import ActuatorDecision, MissionStatus, StmEvents
from rail_robot_interfaces.srv import UartCommand

from ..control import force_cleaning_pairs_safe_off, update_cleaning_actuators
from ..settings import (
    FRONT_ACTUATOR_COMMANDS, SIDE_ACTUATOR_COMMANDS, REALTIME_MODE, DONE_TRIGGER,
    START_COMMAND, CAPTURE_OK_COMMAND, UART_WRITE_TIMEOUT_SECONDS,
    LINK_OK_COMMAND, LEGACY_LINK_INTERVAL_SECONDS,
    DEFAULT_SERIAL_PORT, DEFAULT_BAUD_RATE,
    AUTOMATIC_RAIL_SECTION_TARGET, DEFAULT_RAIL_SECTION_TARGET,
)
from ..uart import CleanerController, PumpController, read_uart_messages, send_uart_test_command
from ..serial_worker import SerialWorker
from .common import qos_events, qos_state, source_monotonic, stamp_from_seconds, stamp_seconds


# Stable speech-event contract; consumers map tokens to their own spoken text.
TTS_OBSTACLE_EVENTS_TOPIC = "tts/obstacle_events"
OBSTACLE_HOLD_EVENT = "OBSTACLE_HOLD"
OBSTACLE_CLEAR_EVENT = "OBSTACLE_CLEAR"
RR2_EVENT_ALIASES = {"MISSION_STARTED": "STARTED", "MISSION_DONE": "DONE"}
MISSION_PREPARE_COMMAND = b"MISSION_PREPARE\r\n"
MISSION_CANCEL_PREPARE_COMMAND = b"MISSION_CANCEL_PREPARE\r\n"
# STM이 청소 ON을 거절하고 출력을 꺼 둔 NACK 사유다. Jetson이 상태 변화를 늦게 아는 경계에서 생긴다.
#  SAFETY: 초음파 정지 직후(OBSTACLE_HOLD 이벤트 전) 또는 단계 사이 1초 정지(transition pending) 중
#  STATE : 실시간 청소가 끝나 다음 단계로 넘어갔지만 그 시작 이벤트를 아직 처리하기 전
RR2_ON_REFUSAL_PREFIXES = ("STM rejected RR2 command: SAFETY", "STM rejected RR2 command: STATE")
# 거절이 이 시간(1초 단계 정지 + 이벤트 지연 여유) 넘게 이어지면 설명되지 않는 불일치로 보고 fault로 남긴다.
RR2_ON_REFUSAL_RETRY_SECONDS = 3.0
# 거절된 출력의 ON 재전송 간격. 정지 중 매 tick 재전송과 경고 반복을 막는다.
RR2_ON_REFUSAL_BACKOFF_SECONDS = 0.2
# SerialWorker가 SAFE_OFF를 받으면 아직 전송하지 않은 ON 명령의 결과를 이 문구의 실패로 닫는다.
RR2_SAFE_OFF_SUPERSEDED = "Superseded by SAFE_OFF"


class StmBridgeNode(Node):
    def __init__(
        self, serial_port=None, baud_rate=None, *, uart_factory=serial.Serial,
        node_name="rail_stm_bridge", **node_kwargs,
    ):
        super().__init__(node_name, **node_kwargs)
        descriptor = ParameterDescriptor(read_only=True)
        serial_port = self.declare_parameter(
            "serial_port", DEFAULT_SERIAL_PORT if serial_port is None else serial_port,
            descriptor,
        ).value
        baud_rate = self.declare_parameter(
            "baud_rate", DEFAULT_BAUD_RATE if baud_rate is None else baud_rate,
            descriptor,
        ).value
        uart_protocol = self.declare_parameter("uart_protocol", "legacy", descriptor).value
        if not isinstance(serial_port, str) or not serial_port.strip():
            raise ValueError("serial_port must be a nonempty device path.")
        if type(baud_rate) is not int or baud_rate <= 0:
            raise ValueError("baud_rate must be a positive integer.")
        if uart_protocol not in ("legacy", "v2"):
            raise ValueError("uart_protocol must be legacy or v2")
        self._uart_protocol = uart_protocol
        self._lock = threading.RLock()
        self._worker = None
        self._uart = None
        self._safety_future = None
        self._emergency_stop_future = None
        self._emergency_stop_requested = False
        self._v2_applied = {}
        self._v2_last_sent = {}
        self._v2_pending = {}
        self._v2_safety_since = {}
        # RR2 opens, owns and closes serial on its dedicated worker thread.
        # Only the explicit legacy path hands the transport to controllers.
        serial_kwargs = {
            "port": serial_port, "baudrate": baud_rate, "timeout": 0,
            "write_timeout": UART_WRITE_TIMEOUT_SECONDS, "exclusive": True,
        }
        if uart_protocol == "legacy":
            self._uart = uart_factory(**serial_kwargs)
        self._buffer = bytearray()
        controller_uart = self._uart
        self._pairs = {
            "front": (CleanerController(controller_uart, FRONT_ACTUATOR_COMMANDS), PumpController(controller_uart, FRONT_ACTUATOR_COMMANDS)),
            "side": (CleanerController(controller_uart, SIDE_ACTUATOR_COMMANDS), PumpController(controller_uart, SIDE_ACTUATOR_COMMANDS)),
        }
        self._targets = {}
        self._session = None
        self._retired_sessions = set()
        self._last_sequence = {}
        self._last_issued = None
        self._mode = None
        self._mission_seen = None
        self._mission_stamp = None
        self._phase_inhibited = True
        self._failed = False
        self._closed = False
        self._closing = False
        self._start_sent = False
        self._legacy_link_sent_at = None
        self._start_acknowledged = False
        self._mission_done = False
        self._capture_pending = False
        self._obstacle_held = False
        self._obstacle_clear_stamp = None
        self._events = self.create_publisher(StmEvents, "stm/events", qos_events)
        self._obstacle_events = self.create_publisher(String, TTS_OBSTACLE_EVENTS_TOPIC, qos_events)
        self._obstacle_state = self.create_publisher(Bool, "tts/obstacle_state", qos_state)
        self._health = self.create_publisher(String, "runtime/stm_bridge_health", qos_state)
        self._faults = self.create_publisher(String, "runtime/fault", qos_events)
        self.create_subscription(ActuatorDecision, "control/decisions", self._on_decision, qos_events)
        self.create_subscription(MissionStatus, "mission/state", self._on_mission, qos_state)
        self.create_subscription(String, "runtime/fault", self._on_fault, qos_events)
        self.create_service(UartCommand, "stm/command", self._command)
        try:
            if uart_protocol == "v2":
                self._worker = SerialWorker(
                    uart_factory, serial_kwargs=serial_kwargs,
                    ack_timeout=0.12, retry_limit=2,
                )
                # An older STM never receives legacy bytes: READY is mandatory
                # and there is deliberately no automatic fallback.
                if not self._worker.wait_ready(1.0):
                    raise TimeoutError(
                        self._worker.startup_error or "STM RR2 READY timed out"
                    )
                try:
                    safe_result = self._safe_off().result(timeout=1.5)
                except FutureTimeoutError:
                    raise TimeoutError("Startup SAFE_OFF was not acknowledged within 1.5 s") from None
                if not safe_result.success:
                    raise RuntimeError("Startup SAFE_OFF failed: " + safe_result.error)
                now = time.monotonic()
                for role in self._pairs:
                    for device in ("cleaner", "pump"):
                        key = f"{role}_{device}"
                        self._v2_applied[key] = 0.0 if device == "cleaner" else False
                        self._v2_last_sent[key] = now
            else:
                # Connecting a legacy bridge sends only approved OFF lines.
                self._safe_off()
        except Exception:
            if self._worker is not None:
                self._worker.stop(0.2, safe_off=False)
            elif self._uart is not None:
                self._uart.close()
            raise
        self.create_timer(0.02, self._tick)
        self.create_timer(1.0, self._publish_health)

    def _publish_health(self):
        # READY is the RR2 handshake/heartbeat state, not proof that a cleaner,
        # pump, encoder or drive physically works. Legacy UART has no ACK health.
        with self._lock:
            if self._closed or self._closing:
                return
            message = String()
            message.data = ("fault" if self._failed else
                            "ready" if self._worker is not None and self._worker.ready else
                            "unknown")
            try:
                self._health.publish(message)
            except Exception as exc:
                # Optional dashboard observation must not interrupt UART control.
                self.get_logger().warning("STM health telemetry publish failed: " + type(exc).__name__)

    def _ros_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _safe_off(self, *, force=False, fault_latched=False):
        # 여기서 끄는 대상은 앞/옆 청소 모터와 펌프다. 주행 모터 STOP 명령을 보내는 함수는 아니다.
        self._targets.clear()
        if self._worker is not None:
            # Controllers have no serial handle in v2; these calls update only
            # their local mirrors. The worker owns all RR2 OFF transactions.
            force_cleaning_pairs_safe_off(*self._pairs.values())
            if self._emergency_stop_requested:
                # ESTOP stops the drive and all four accessories in the MCU.
                # Do not replace its priority transaction with accessory OFF.
                return self._emergency_stop_future
            future = (
                self._worker.latch_fault(force=force)
                if fault_latched else self._worker.submit("SAFE_OFF", force=force)
            )
            self._safety_future = future
            return future
        force_cleaning_pairs_safe_off(*self._pairs.values())
        return None

    def _escalate_motion_fault(self):
        # START 이후 DONE 전의 fault는 주행 중일 수 있으므로 watchdog(약 0.6 s)을 기다리지 않고
        # ESTOP 최우선 경로로 주행·청소 출력을 멈춘다. STM은 FAULT를 래치하므로 재시작 전 리셋이 필요하다.
        # 정상 DONE 이후나 START 전에는 기존대로 부속장치 OFF와 heartbeat 중단만 한다.
        # 링크가 이미 끊겼으면 emergency_stop은 실패 결과를 돌려주고, STM watchdog이 백업으로 남는다.
        if (self._worker is None or self._emergency_stop_requested
                or not self._start_sent or self._mission_done):
            return
        self._emergency_stop_requested = True
        self._targets.clear()
        self._emergency_stop_future = self._worker.emergency_stop()

    def _on_fault(self, message):
        # 공유 fault가 발생하면 출력 허용을 잠그고 청소 출력을 끈다. 이후 정상 토픽만으로 _failed가 풀리지 않는다.
        # OFF 쓰기 자체의 실패는 로그로 남으므로 fault 수신과 실제 MCU 출력 차단을 동일하게 보지 않는다.
        if message.data:
            with self._lock:
                if self._closed or self._closing:
                    return
                was_failed = self._failed
                self._failed = True
                self._phase_inhibited = True
                try:
                    # 주행 중이면 ESTOP, 아니면 PING 중단 + 부속장치 OFF. 두 경우 모두
                    # worker가 일반 작업을 먼저 취소한 뒤 예약된 안전 transaction을 보낸다.
                    self._escalate_motion_fault()
                    self._safe_off(force=not was_failed, fault_latched=True)
                except RuntimeError as exc:
                    self.get_logger().error(str(exc))

    def _fault(self, reason):
        already_failed = self._failed
        message = String()
        message.data = f"stm_bridge: {reason}"
        self._on_fault(message)
        if not already_failed:
            self._faults.publish(message)

    def _on_mission(self, message):
        with self._lock:
            if self._closed or self._closing:
                return
            try:
                # 오래된/역순 상태는 버려 _mission_seen을 갱신하지 않는다. 다음 _tick이 신선도 만료를 확인한다.
                # 상태가 REALTIME인데 청소가 안 켜지면 _phase_inhibited, STARTED 승인, 시각을 함께 확인한다.
                observed = source_monotonic(message.header.stamp, self._ros_now(), time.monotonic(), 0.8)
                issued = stamp_seconds(message.header.stamp)
            except ValueError:
                return
            if self._mission_stamp is not None and issued < self._mission_stamp:
                return
            self._mission_stamp = issued
            previous_mode = self._mode
            self._mode = message.mode
            self._mission_seen = observed
            fault_transition = bool(message.failed) and not self._failed
            if message.failed:
                self._failed = True
                self._escalate_motion_fault()
            if self._failed or message.mode != REALTIME_MODE:
                self._phase_inhibited = True
                try:
                    self._safe_off(
                        force=fault_transition,
                        fault_latched=bool(message.failed),
                    )
                except RuntimeError as exc:
                    self._fault(str(exc))
            elif previous_mode != REALTIME_MODE:
                # 새로운 REALTIME 진입에서만 단계 잠금을 해제한다. 같은 모드의 반복 발행은 SAFE_OFF 잠금을 풀지 않는다.
                # 모드 전환 전에 mission 노드가 OFF를 보낸 뒤 새 상태를 확정하는 순서와 연결되는 조건이다.
                self._phase_inhibited = False

    def _enabled(self, now):
        # control의 허용 조건을 UART 직전에서 다시 확인하며 STM STARTED 승인도 요구한다.
        return (
            not self._failed and not self._emergency_stop_requested
            and not self._phase_inhibited and not self._obstacle_held
            and self._start_acknowledged
            and self._mode == REALTIME_MODE and self._mission_seen is not None
            and now - self._mission_seen <= 0.8
        )

    def _on_decision(self, message):
        with self._lock:
            if self._closed or self._closing:
                return
            now, ros_now = time.monotonic(), self._ros_now()
            if not self._enabled(now):
                return
            if message.role not in self._pairs or not message.session_id:
                self._fault("Invalid actuator role or session ID.")
                return
            if message.session_id in self._retired_sessions:
                # 종료된 제어 세션이나 이미 처리한 순번은 조용히 버린다. 지연 메시지로 출력이 되살아나지 않게 한다.
                # 결정 토픽 발행만으로 수신 채택을 보장하지 않으므로 session_id/sequence도 비교해야 한다.
                return
            if (
                message.session_id == self._session
                and message.sequence <= self._last_sequence.get(message.role, -1)
            ):
                return
            try:
                issued = stamp_seconds(message.header.stamp)
                # Do not replay decisions queued before/during HOLD after release.
                if self._obstacle_clear_stamp is not None and issued <= self._obstacle_clear_stamp:
                    return
                if self._last_issued is not None and issued < self._last_issued:
                    return
            except ValueError as exc:
                self._fault(str(exc))
                return
            try:
                # 도착 지연이 0.2초를 넘으면 유효기간이 남아 있어도 해당 출력은 OFF로 돌린다.
                source_monotonic(message.header.stamp, ros_now, now, 0.2)
            except ValueError:
                self._targets.pop(message.role, None)
                try:
                    force_cleaning_pairs_safe_off(self._pairs[message.role])
                except RuntimeError as exc:
                    self._fault(str(exc))
                return
            try:
                valid_until = stamp_seconds(message.valid_until)
                # 허용 duty는 정해진 단계값 0/12.5/25이며 임의의 모터 속도 백분율 인터페이스가 아니다.
                # 이미 만료된 결정은 목표에서 제거하고, 비정상 타입·과도한 수명은 fault로 처리한다.
                duty = float(message.cleaner_duty_percent)
                if not math.isfinite(duty) or duty not in (0.0, 12.5, 25.0):
                    raise ValueError("Unapproved cleaner duty.")
                if type(message.pump_on) is not bool:
                    raise ValueError("Pump decision is not a boolean.")
                if not math.isfinite(valid_until) or valid_until <= ros_now:
                    self._targets.pop(message.role, None)
                    return
                if valid_until - issued > 1.2 + 1e-6:
                    raise ValueError("Actuator validity exceeds the 1.2-second limit.")
            except (ValueError, TypeError) as exc:
                self._targets.pop(message.role, None)
                self._fault(str(exc))
                return
            if message.session_id != self._session:
                # control 노드 재시작 시 이전 세션 출력을 먼저 끄고 새 세션의 순번을 받는다.
                if self._last_issued is not None and issued <= self._last_issued:
                    return
                if self._session is not None:
                    self._retired_sessions.add(self._session)
                try:
                    self._safe_off(force=True)
                except RuntimeError as exc:
                    self._fault(str(exc))
                    return
                self._session = message.session_id
                self._last_sequence.clear()
            if message.sequence <= self._last_sequence.get(message.role, -1):
                return
            if self._last_issued is not None and issued < self._last_issued:
                return
            self._last_sequence[message.role] = message.sequence
            self._last_issued = issued
            # ROS 시각의 남은 수명을 로컬 monotonic 기한으로 바꿔 저장한다. 실제 송신은 이후 _tick에서 수행된다.
            # 따라서 _on_decision을 통과했다고 STM에 즉시 전달/실행됐다고 판단할 수는 없다.
            self._targets[message.role] = (message, now + min(1.2, valid_until - ros_now))

    def _command(self, request, response):
        # Never hold the bridge lock while waiting for RR2 ACK. The owner
        # worker continues RX/EVENT_ACK/PING work independently of the ROS
        # service callback; every wait remains explicitly bounded.
        future = None
        wait_seconds = 0.0
        with self._lock:
            if self._closed or self._closing:
                response.success, response.error = False, "STM bridge is closing."
                return response
            try:
                if request.command == "E_STOP":
                    if self._worker is None:
                        raise RuntimeError("E_STOP requires uart_protocol:=v2.")
                    self._emergency_stop_requested = True
                    self._phase_inhibited = True
                    self._targets.clear()
                    force_cleaning_pairs_safe_off(*self._pairs.values())
                    future = self._worker.emergency_stop()
                    self._emergency_stop_future = future
                    wait_seconds = 0.35
                elif request.command == "SAFE_OFF":
                    self._phase_inhibited = True
                    future = self._safe_off(force=True)
                    wait_seconds = 1.5 if self._worker is not None else 0.0
                elif self._emergency_stop_requested:
                    raise RuntimeError("E_STOP is latched; restart the STM before a new mission.")
                elif self._failed:
                    raise RuntimeError("Bridge is faulted; only SAFE_OFF or E_STOP is allowed.")
                elif request.command in ("PREPARE", "CANCEL_PREPARE"):
                    if request.command == "PREPARE" and self._start_sent:
                        raise RuntimeError("Mission START was already sent.")
                    if self._worker is None:
                        # legacy 문자열 경로는 구간 수를 전할 수 없다. 시험 모드(구간 수 변경)는 v2에서만 연다.
                        if AUTOMATIC_RAIL_SECTION_TARGET != DEFAULT_RAIL_SECTION_TARGET:
                            raise RuntimeError(
                                f"{AUTOMATIC_RAIL_SECTION_TARGET}-section test mode requires uart_protocol:=v2."
                            )
                        command = (MISSION_PREPARE_COMMAND if request.command == "PREPARE"
                                   else MISSION_CANCEL_PREPARE_COMMAND)
                        send_uart_test_command(self._uart, command)
                    else:
                        # PREPARE에는 항상 구간 수를 붙인다. 펌웨어 기본 구간 수가 Jetson과 달라도
                        # (예: 4구간 펌웨어에 6구간 Jetson) 조용히 다른 구간 수로 달리지 않고 STM이 거부한다.
                        args = ((str(AUTOMATIC_RAIL_SECTION_TARGET),)
                                if request.command == "PREPARE" else ())
                        future = self._worker.submit(request.command, args)
                        wait_seconds = 0.35
                elif request.command == "START":
                    if self._start_sent:
                        raise RuntimeError("START was already sent.")
                    if self._worker is None:
                        self._uart.reset_input_buffer()
                        self._buffer.clear()
                        send_uart_test_command(self._uart, START_COMMAND)
                        self._start_sent = True
                    else:
                        self._start_sent = True
                        future = self._worker.submit("START")
                        wait_seconds = 0.35
                elif request.command == "CAPTURE_OK":
                    # STM이 촬영을 요청해 _capture_pending인 경우에만 ACK를 허용한다.
                    # ACK 실패 시 이 플래그가 남을 수 있으므로 mission의 촬영 카운터와 함께 확인한다.
                    if not self._start_acknowledged or not self._capture_pending:
                        raise RuntimeError("No started STM capture is awaiting acknowledgement.")
                    if self._worker is None:
                        send_uart_test_command(self._uart, CAPTURE_OK_COMMAND)
                        self._capture_pending = False
                    else:
                        future = self._worker.submit("CAPTURE_OK")
                        wait_seconds = 0.35
                else:
                    raise ValueError("Only START, CAPTURE_OK, SAFE_OFF and E_STOP are supported.")
            except (OSError, RuntimeError, ValueError) as exc:
                response.success, response.error = False, str(exc)
                return response

        if future is None:
            response.success, response.error = True, ""
            return response
        try:
            result = future.result(timeout=wait_seconds)
            if not result.success:
                raise RuntimeError(result.error or "STM did not acknowledge the command")
            with self._lock:
                if request.command == "CAPTURE_OK":
                    self._capture_pending = False
            response.success, response.error = True, ""
            if request.command == "E_STOP":
                self._fault("Operator E_STOP was acknowledged by STM")
        except (TimeoutError, FutureTimeoutError, RuntimeError, OSError, ValueError) as exc:
            # Future.result(timeout)의 예외는 Python 3.10(Jetson)에서 내장 TimeoutError가 아니고 메시지도 비어 있다.
            # 놓치면 서비스 콜백이 예외로 끝나 폴트가 기록되지 않는다.
            detail = str(exc) or f"STM did not acknowledge {request.command} within {wait_seconds:.2f} s"
            response.success, response.error = False, detail
            if request.command in ("START", "CAPTURE_OK", "SAFE_OFF", "E_STOP"):
                self._fault(detail)
        return response

    def _process_tokens(self, tokens):
        # STARTED/CAMERA_CAPTURE are tracked before publishing to mission.
        if tokens:
            for token in tokens:
                if token == "STARTED" and self._start_sent:
                    self._start_acknowledged = True
                elif token == DONE_TRIGGER:
                    self._mission_done = True
                elif token == "CAMERA_CAPTURE":
                    if self._capture_pending:
                        raise RuntimeError("Duplicate STM capture before CAPTURE_OK.")
                    self._capture_pending = True
                elif token in (OBSTACLE_HOLD_EVENT, OBSTACLE_CLEAR_EVENT):
                    held = token == OBSTACLE_HOLD_EVENT
                    # 재생 노드가 늦게 구독해도 마지막 실제 센서 상태를 받는다.
                    self._obstacle_state.publish(Bool(data=held))
                    if held != self._obstacle_held:
                        self._obstacle_held = held
                        # 정지 이벤트가 앞선 SAFETY 거절을 설명했으므로 재시도 시간을 새로 센다.
                        self._v2_safety_since.clear()
                        if held:
                            # Temporary hold: do not set the permanent fault/phase locks.
                            self._safe_off(force=True)
                        else:
                            self._obstacle_clear_stamp = self._ros_now()
                        announcement = String()
                        announcement.data = token
                        self._obstacle_events.publish(announcement)
            event = StmEvents()
            event.header.stamp = stamp_from_seconds(self._ros_now())
            event.tokens = tokens
            self._events.publish(event)

    def _transient_on_refusal(self, key, desired, error, now):
        # STM이 ON을 거절하면 그 출력은 꺼진 상태다. 실시간 청소 끝의 1초 단계 정지와 다음 단계 시작
        # 이벤트 사이, 또는 초음파 정지 이벤트 도착 전의 경합이면 꺼진 상태로 기록하고 잠시 뒤 다시 판단한다.
        # 2026-10-05 실기: 이 거절(SAFETY 4, 이어서 STATE 7) 한 번으로 브리지가 fault가 되어 임무가 종료됐다.
        cleaner = key.endswith("_cleaner")
        is_on = desired != 0.0 if cleaner else bool(desired)
        if not is_on or not error.startswith(RR2_ON_REFUSAL_PREFIXES):
            return False
        first = self._v2_safety_since.get(key)
        if first is None:
            self._v2_safety_since[key] = first = now
            self.get_logger().warning(
                f"STM refused {key} ON ({error.split(': ', 1)[-1]}); output stays off until the STM allows it.")
        if now - first > RR2_ON_REFUSAL_RETRY_SECONDS:
            return False
        self._v2_applied[key] = 0.0 if cleaner else False
        self._v2_last_sent[key] = now
        return True

    def _tick_v2_outputs(self, now):
        for key, (desired, future) in list(self._v2_pending.items()):
            if not future.done():
                continue
            self._v2_pending.pop(key, None)
            result = future.result()
            if not result.success:
                if "ON rejected until SAFE_OFF is acknowledged" in result.error:
                    continue
                if result.error == RR2_SAFE_OFF_SUPERSEDED:
                    # 아직 전송 전이던 ON을 SAFE_OFF가 취소한 정상 경로다. 출력은 OFF로 가며, 적용 상태는
                    # 아래 SAFE_OFF ACK 처리가 OFF로 맞춘다. 단계 전환·장애물 정지와 겹칠 때 생긴다.
                    continue
                if self._transient_on_refusal(key, desired, result.error, now):
                    continue
                self._fault("RR2 actuator command failed: " + result.error)
                return
            self._v2_safety_since.pop(key, None)
            self._v2_applied[key] = desired
            self._v2_last_sent[key] = now

        # SAFE_OFF has reserved priority in SerialWorker. Apply its ACK after
        # any older per-output result handled in this tick, so local state
        # cannot regress to ON after the MCU has acknowledged OFF.
        if self._safety_future is not None and self._safety_future.done():
            result = self._safety_future.result()
            self._safety_future = None
            if not result.success:
                if not self._failed:
                    self._fault("RR2 SAFE_OFF acknowledgement failed: " + result.error)
                return
            for key in self._v2_applied:
                device = "cleaner" if key.endswith("_cleaner") else "pump"
                self._v2_applied[key] = 0.0 if device == "cleaner" else False
                self._v2_last_sent[key] = now

        enabled = self._enabled(now)
        if not enabled:
            self._targets.clear()
        desired_by_key = {}
        if enabled:
            for role, target in list(self._targets.items()):
                message, expiry = target
                if now >= expiry:
                    self._targets.pop(role, None)
                    continue
                desired_by_key[f"{role}_cleaner"] = float(message.cleaner_duty_percent)
                desired_by_key[f"{role}_pump"] = bool(message.pump_on)

        for role in self._pairs:
            for device in ("cleaner", "pump"):
                key = f"{role}_{device}"
                desired = desired_by_key.get(
                    key, 0.0 if device == "cleaner" else False
                )
                if key in self._v2_pending:
                    continue
                current = self._v2_applied.get(key)
                is_on = desired != 0.0 if device == "cleaner" else desired
                if not is_on:
                    # 더 이상 ON을 원하지 않으면 거절 이력을 지워 다음 ON 거절을 새로 센다.
                    self._v2_safety_since.pop(key, None)
                elif (key in self._v2_safety_since and
                      now - self._v2_last_sent.get(key, 0.0) < RR2_ON_REFUSAL_BACKOFF_SECONDS):
                    continue
                refresh_due = is_on and now - self._v2_last_sent.get(key, 0.0) >= 0.2
                if desired == current and not refresh_due:
                    continue
                upper = role.upper()
                if device == "cleaner":
                    suffix = "OFF" if desired == 0.0 else (
                        "PWM_12_5" if desired == 12.5 else "PWM_25"
                    )
                    opcode = f"{upper}_CLEANER_{suffix}"
                else:
                    opcode = f"{upper}_PUMP_{'ON' if desired else 'OFF'}"
                future = self._worker.submit(opcode, coalesce_key=key)
                if future.done():
                    result = future.result()
                    if ("ON rejected until SAFE_OFF is acknowledged" in result.error
                            or result.error == RR2_SAFE_OFF_SUPERSEDED):
                        continue
                    if not result.success:
                        self._fault("RR2 actuator queue rejected command: " + result.error)
                        return
                else:
                    self._v2_pending[key] = (desired, future)

    def _tick(self):
        with self._lock:
            if self._closed or self._closing:
                return
            try:
                now = time.monotonic()
                if self._worker is not None:
                    if not self._worker.ready:
                        self._fault(
                            self._worker.startup_error or "RR2 serial worker lost its session"
                        )
                        return
                    for result in self._worker.drain_results(32):
                        if result.kind == "EVENT" and result.payload:
                            token = RR2_EVENT_ALIASES.get(result.payload[0], result.payload[0])
                            self._process_tokens([token])
                    if not self._emergency_stop_requested:
                        self._tick_v2_outputs(now)
                else:
                    if self._start_sent and not self._failed and (
                            self._legacy_link_sent_at is None
                            or now - self._legacy_link_sent_at >= LEGACY_LINK_INTERVAL_SECONDS):
                        # STM legacy 통신 감시(주행 중 1 s)용 keepalive. 브리지가 fault면 보내지 않으므로
                        # Jetson 쪽 이상도 STM 정지로 이어진다.
                        send_uart_test_command(self._uart, LINK_OK_COMMAND)
                        self._legacy_link_sent_at = now
                    tokens = read_uart_messages(self._uart, self._buffer)
                    self._process_tokens(tokens)
                    enabled = self._enabled(now)
                    if not enabled:
                        self._targets.clear()
                    for role, (cleaner, pump) in self._pairs.items():
                        # 새 토픽이 오지 않아도 20ms 주기 타이머가 만료를 확인해 청소 출력 유지가 무한정 이어지지 않게 한다.
                        target = self._targets.get(role)
                        if target is None or now >= target[1]:
                            if cleaner.is_on:
                                cleaner.force_off()
                            if pump.is_on:
                                pump.force_off()
                            self._targets.pop(role, None)
                        else:
                            # Legacy controller methods are reachable only in explicit legacy mode.
                            update_cleaning_actuators(cleaner, pump, decision=target[0])
            except (OSError, RuntimeError, ValueError) as exc:
                self._fault(str(exc))

    def prepare_shutdown(self):
        with self._lock:
            if not self._closed and not self._closing:
                self._closing = True
                self._phase_inhibited = True
                self._failed = True
                self._safe_off()

    def destroy_node(self):
        with self._lock:
            if not self._closed:
                self._closing = True
                self._closed = True
                if self._worker is None:
                    try:
                        self._safe_off()
                    finally:
                        if self._uart is not None:
                            self._uart.close()
        if self._worker is not None:
            if not self._worker.stop(1.5, safe_off=True):
                self.get_logger().error(
                    "STM serial worker shutdown did not complete before the deadline "
                    "(SAFE_OFF acknowledgement or serial close pending)."
                )
        return super().destroy_node()
