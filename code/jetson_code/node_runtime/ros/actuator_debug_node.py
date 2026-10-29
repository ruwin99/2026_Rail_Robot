"""Exclusive, deadman-controlled bench actuator interface; no mission START."""
from __future__ import annotations

import os
import json
import threading
import time

import serial
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, String
from rail_robot_interfaces.msg import EncoderCommand, EncoderState, MissionStatus

from ..actuator_debug import (ActuatorDebugSession, Rr2ActuatorDebugSession,
                              RR2_CAPABILITIES, CAPABILITIES, DIRECTION_NAMES,
                              DIAGNOSTIC_HINTS)
from ..serial_worker import SerialWorker
from ..settings import DEFAULT_SERIAL_PORT, DEFAULT_BAUD_RATE, UART_WRITE_TIMEOUT_SECONDS
from .common import qos_events, qos_state, stamp_from_seconds


def _exclusive_device(uart):
    # pyserial's exclusive flock protects cooperating owners; TIOCEXCL also
    # prevents later noncooperating opens on the Linux Jetson device.
    if os.name == "posix":
        import fcntl
        import termios
        fcntl.ioctl(uart.fileno(), termios.TIOCEXCL)


class ActuatorDebugNode(Node):
    def __init__(self, serial_port=None, baud_rate=None, debug_capability=None,
                 input_timeout_seconds=None, uart_protocol=None, *, uart_factory=serial.Serial,
                 exclusive_device=_exclusive_device, clock=time.monotonic,
                 node_name="rail_actuator_debug", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        descriptor = ParameterDescriptor(read_only=True)
        def parameter(name, supplied, default):
            return self.declare_parameter(name, default if supplied is None else supplied, descriptor).value
        port = parameter("serial_port", serial_port, DEFAULT_SERIAL_PORT)
        baud = parameter("baud_rate", baud_rate, DEFAULT_BAUD_RATE)
        capability = parameter("debug_capability", debug_capability, "actuators")
        timeout = parameter("input_timeout_seconds", input_timeout_seconds, 1.0)
        protocol = parameter("uart_protocol", uart_protocol, "v2")
        if not isinstance(port, str) or not port.strip():
            raise ValueError("serial_port must be a nonempty device path.")
        if type(baud) is not int or baud <= 0:
            raise ValueError("baud_rate must be a positive integer.")
        if capability not in CAPABILITIES:
            raise ValueError("Unsupported debug_capability.")
        if protocol not in ("v2", "legacy"):
            raise ValueError("uart_protocol must be v2 or legacy.")
        if self._mission_present():
            raise RuntimeError("Actuator debug cannot share a running STM bridge or mission.")
        self._lock = threading.RLock()
        self._protocol = protocol
        self._worker = None
        self._uart = None
        self._monotonic_clock = clock
        self._closed = self._closing = self._failed = False
        self._error = ""
        self._last_feedback = None
        self._last_diagnostics = None
        self._last_control_telemetry = None
        self._last_drive_request = None
        self._reported_reason = None
        self._reported_ready = False
        self._last_graph_check = clock()
        self._feedback_publisher = self.create_publisher(EncoderState, "debug/encoder/state", qos_state)
        self._diagnostic_publisher = self.create_publisher(String, "debug/encoder/diagnostics", qos_state)
        self._faults = self.create_publisher(String, "runtime/fault", qos_events)
        input_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(EncoderCommand, "debug/encoder/command", self._on_encoder, input_qos)
        for channel, topic in (
            ("front_dc", "debug/front/dc/set"), ("side_dc", "debug/side/dc/set"),
            ("front_pump", "debug/front/pump/set"), ("side_pump", "debug/side/pump/set"),
        ):
            self.create_subscription(Bool, topic, lambda message, key=channel: self._on_channel(key, message), input_qos)
        self.create_subscription(MissionStatus, "mission/state", self._on_mission, qos_state)
        self.create_subscription(String, "runtime/fault", self._on_fault, qos_events)
        serial_kwargs = dict(port=port, baudrate=baud, timeout=0,
                             write_timeout=UART_WRITE_TIMEOUT_SECONDS, exclusive=True)
        if protocol == "legacy":
            self._uart = uart_factory(**serial_kwargs)
            try:
                exclusive_device(self._uart)
                self.session = ActuatorDebugSession(self._uart, capability, timeout, clock=clock)
            except Exception:
                self._uart.close()
                raise
        else:
            def owned_serial(**kwargs):
                uart = uart_factory(**kwargs)
                try:
                    exclusive_device(uart)
                except Exception:
                    uart.close()
                    raise
                return uart
            self._worker = SerialWorker(
                owned_serial, serial_kwargs=serial_kwargs,
                capability=RR2_CAPABILITIES[capability])
            self.session = Rr2ActuatorDebugSession(
                self._worker, capability, timeout, clock=clock)
        self.create_timer(0.02, self._tick)

    def _mission_present(self):
        if self.count_publishers("mission/state"):
            return True
        bridge_service = self.resolve_service_name("stm/command")
        return any(name == bridge_service for name, _ in self.get_service_names_and_types())

    def _fault(self, reason, *, publish=True):
        with self._lock:
            if self._failed or self._closed:
                return
            self._failed, self._error = True, reason
            try:
                self.session.fail(reason)
            except Exception as exc:
                self.get_logger().error(f"Debug stop write failed: {exc}")
            self._publish_feedback(valid=False)
            self._publish_diagnostics()
            if publish:
                message = String()
                message.data = f"actuator_debug: {reason}"
                self._faults.publish(message)

    def _on_fault(self, message):
        if message.data:
            self._fault(message.data, publish=False)

    def _on_mission(self, message):
        self._fault("A mission publisher appeared while debug exclusively owned UART.")

    def _on_encoder(self, message):
        with self._lock:
            if self._closing or self._closed or self._failed:
                return
            if not self.session.ready:
                self.get_logger().warning("Encoder input ignored until debug handshake completes.")
                return
            try:
                if not self.session.command_encoder(message.direction, message.speed_percent):
                    return
                self._last_drive_request = dict(direction=message.direction, speed_percent=message.speed_percent,
                                                receipt_stamp=self.get_clock().now().nanoseconds / 1e9)
            except (TypeError, ValueError) as exc:
                self._fault(str(exc))

    def _on_channel(self, channel, message):
        with self._lock:
            if self._closing or self._closed or self._failed:
                return
            if not self.session.ready:
                self.get_logger().warning("Actuator input ignored until debug handshake completes.")
                return
            try:
                self.session.command_channel(channel, message.data)
            except (TypeError, ValueError) as exc:
                self._fault(str(exc))

    def _publish_feedback(self, *, valid):
        message = EncoderState()
        message.header.stamp = stamp_from_seconds(self.get_clock().now().nanoseconds / 1e9)
        message.session_id = self.session.session_id
        message.left_counter_modulus = self.session.left_counter_modulus
        message.right_counter_modulus = self.session.right_counter_modulus
        message.commanded_direction = DIRECTION_NAMES[self.session.commanded_direction]
        message.commanded_speed_percent = self.session.commanded_percent / 100.0
        message.feedback_valid = valid
        message.error = self._error
        feedback = self.session.feedback
        if feedback is not None:
            message.sequence, message.mcu_millis = feedback.sequence, feedback.mcu_millis
            message.left_raw_count, message.right_raw_count = feedback.left_raw_count, feedback.right_raw_count
            message.applied_direction = DIRECTION_NAMES[feedback.direction]
            message.applied_speed_percent = feedback.percent_basis_points / 100.0
            message.left_pwm_compare, message.right_pwm_compare = feedback.left_pwm_compare, feedback.right_pwm_compare
            message.active_output_mask = feedback.output_mask
        else:
            message.applied_direction = "stop"
        self._feedback_publisher.publish(message)

    def _publish_diagnostics(self):
        # 기존 EncoderState 메시지를 변경하지 않고 추가 JSON 토픽에 진단 정보를 제공한다.
        # 마지막 요청/이전 PWM은 이력이다. 정지 후 이 값을 현재 구동 출력으로 표시하지 않는다.
        # READY의 기존 trim 필드는 전진 기준이다. 후진 기준값은 UART로 받지 않으므로 추정하지 않는다.
        data = dict(self.session.diagnostics or {})
        received = data.pop('received_at', None)
        age_ms = None if received is None else max(0, self._monotonic_clock() - received) * 1000
        reason = data.get('reason', 'UNAVAILABLE')
        historical = (self._failed or self.session.closed or self._closing or self._closed
                      or reason != 'RUNNING' or age_ms is None or age_ms >= 600)
        control = dict(self.session.control_telemetry or {})
        control_received = control.pop('received_at', None)
        if control:
            control_age_ms = (None if control_received is None else
                              max(0, self._monotonic_clock() - control_received) * 1000)
            control.update(
                sample_period_ms=control.pop('elapsed_ms', None),
                direction=DIRECTION_NAMES[control['direction']],
                physical_right_delta=control.pop('right_normalized_delta', None),
                physical_left_delta=control.pop('left_normalized_delta', None),
                sample_age_ms=control_age_ms,
                historical=historical or control_age_ms is None or control_age_ms >= 600)
        data.update(session_id=self.session.session_id, reason=reason,
                    hint=DIAGNOSTIC_HINTS.get(reason, '진단 프레임 미수신. STM 펌웨어 반영 여부 확인'),
                    receipt_stamp=self.get_clock().now().nanoseconds / 1e9,
                    sample_age_ms=age_ms, historical=historical,
                    session_active=self.session.ready and not (self._failed or self._closing or self._closed),
                    last_drive_request=self._last_drive_request, last_nonzero_pwm=self.session.last_nonzero_pwm,
                    registered_drive_pwm=self.session.registered_drive_pwm,
                    forward_right_trim_percent=self.session.right_trim_percent,
                    reverse_right_trim_percent=None,
                    right_trim_percent=self.session.right_trim_percent,
                    left_trim_percent=self.session.left_trim_percent,
                    control_loop=control or None, error=self._error)
        message = String()
        message.data = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        self._diagnostic_publisher.publish(message)
        if reason != self._reported_reason:
            self._reported_reason = reason
            self.get_logger().info(f"STM diagnostic {reason}: {data['hint']}" + self.session.diagnostic_summary())

    def _tick(self):
        with self._lock:
            if self._closing or self._closed or self._failed:
                return
            try:
                now = self._monotonic_clock()
                if now - self._last_graph_check >= 0.2:
                    self._last_graph_check = now
                    if self._mission_present():
                        self._fault("STM bridge or mission conflicts with actuator debug.")
                        return
                self.session.tick()
                if self.session.ready and not self._reported_ready:
                    self._reported_ready = True
                    if self._protocol == "legacy":
                        self.get_logger().info(
                            f"STM debug ready: registered PWM={self.session.registered_drive_pwm}, "
                            f"forward right trim={self.session.right_trim_percent:g}% (reverse trim not reported), "
                            f"encoder modulus={self.session.left_counter_modulus}/{self.session.right_counter_modulus}. "
                            "100% means the registered PWM reference, not full timer duty.")
                    else:
                        self.get_logger().info(
                            "STM RR2 debug ready; READY omits PWM reference, trim, and encoder modulus metadata.")
                if self.session.failed:
                    self._fault(self.session.error)
                else:
                    if self.session.feedback is not None and self.session.feedback is not self._last_feedback:
                        self._last_feedback = self.session.feedback
                        self._publish_feedback(valid=True)
                    diagnostics_changed = (self.session.diagnostics is not None
                                           and self.session.diagnostics is not self._last_diagnostics)
                    control_changed = (self.session.control_telemetry is not None
                                       and self.session.control_telemetry is not self._last_control_telemetry)
                    if diagnostics_changed:
                        self._last_diagnostics = self.session.diagnostics
                    if control_changed:
                        self._last_control_telemetry = self.session.control_telemetry
                    # DIAG와 CTRL이 같은 UART read에서 함께 들어와도 JSON은 한 번만 발행한다.
                    if diagnostics_changed or control_changed:
                        self._publish_diagnostics()
            except Exception as exc:
                self._fault(str(exc))

    def readiness(self):
        return (self.session.ready and not self._failed and not self._closing,
                self._error or (f"{self._protocol.upper()} acknowledged" if self.session.ready
                                else f"Waiting for compatible {self._protocol.upper()} firmware"))

    def prepare_shutdown(self):
        with self._lock:
            if self._closed or self._closing:
                return
            self._closing = True
            for timer in self.timers:
                timer.cancel()
            try:
                self.session.close()
            except Exception as exc:
                self.get_logger().error(f"Debug shutdown write failed: {exc}")
            self._error = self._error or "Debug session closing; last MCU sample is historical, STOP is not acknowledged."
            self._publish_feedback(valid=False)
            self._publish_diagnostics()

    def destroy_node(self):
        self.prepare_shutdown()
        with self._lock:
            if self._closed:
                return None
            self._closed = True
            if self._uart is not None:
                self._uart.close()
        if self._worker is not None and not self._worker.stop(1.5, safe_off=True):
            self.get_logger().error("RR2 debug STOP acknowledgement or serial close timed out.")
        return super().destroy_node()
