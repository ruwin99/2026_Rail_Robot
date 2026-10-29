"""하나의 UART 소유자로 LED 및 초음파만 시험한다. 임무/구동 명령은 보내지 않는다."""
import json
import os
import threading
import time

import serial
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from ..sensor_debug import SensorDebugSession, Rr2SensorDebugSession
from ..serial_worker import SerialWorker
from ..settings import DEFAULT_SERIAL_PORT, DEFAULT_BAUD_RATE, UART_WRITE_TIMEOUT_SECONDS
from .common import qos_state


def _exclusive_device(uart):
    if os.name == 'posix':
        import fcntl
        import termios
        fcntl.ioctl(uart.fileno(), termios.TIOCEXCL)


class SensorDebugNode(Node):
    def __init__(self, *, uart_protocol=None, uart_factory=serial.Serial, exclusive_device=_exclusive_device,
                 clock=time.monotonic, node_name='rail_sensor_debug', **kwargs):
        super().__init__(node_name, **kwargs)
        descriptor = ParameterDescriptor(read_only=True)
        port = self.declare_parameter('serial_port', DEFAULT_SERIAL_PORT, descriptor).value
        baud = self.declare_parameter('baud_rate', DEFAULT_BAUD_RATE, descriptor).value
        protocol = self.declare_parameter('uart_protocol',
                                          'v2' if uart_protocol is None else uart_protocol,
                                          descriptor).value
        if not isinstance(port, str) or not port.strip() or type(baud) is not int or baud <= 0:
            raise ValueError('serial_port/baud_rate are invalid')
        if protocol not in ('v2', 'legacy'):
            raise ValueError('uart_protocol must be v2 or legacy')
        if self._mission_present():
            raise RuntimeError('Stop mission/STM bridge before starting sensor debug')
        self._lock = threading.RLock()
        self._protocol = protocol
        self._worker = None
        self._uart = None
        # rclpy.Node._clock는 ROS 타이머가 사용하는 Clock 객체이므로 덮어쓰지 않는다.
        self._monotonic_clock = clock
        self._last_graph = clock()
        self._closed = self._failed = False
        self._error = ''
        self._last_feedback = None
        self._range = self.create_publisher(String, 'debug/ultrasonic/state', qos_state)
        self._led = self.create_publisher(String, 'debug/led/state', qos_state)
        command_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(String, 'debug/led/set', self._on_led, command_qos)
        serial_kwargs = dict(port=port, baudrate=baud, timeout=0,
                             write_timeout=UART_WRITE_TIMEOUT_SECONDS, exclusive=True)
        if protocol == 'legacy':
            self._uart = uart_factory(**serial_kwargs)
            try:
                exclusive_device(self._uart)
                self.session = SensorDebugSession(self._uart, clock=clock)
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
            self._worker = SerialWorker(owned_serial, serial_kwargs=serial_kwargs,
                                        capability='SENSOR_DEBUG')
            self.session = Rr2SensorDebugSession(self._worker, clock=clock)
        self.create_timer(0.02, self._tick)
        self.get_logger().info('Sensor debug only: LED + ultrasonic; no motor/pump authority')

    def _mission_present(self):
        return bool(self.count_publishers('mission/state') or any(
            name == self.resolve_service_name('stm/command')
            for name, _ in self.get_service_names_and_types()))

    def _on_led(self, message):
        with self._lock:
            if self._closed or self._failed:
                return
            try:
                self.session.command_led(message.data)
            except ValueError as exc:
                self.get_logger().warning(str(exc))

    def _publish(self, valid):
        # receipt_stamp는 ROS 수신 시각이다. MCU 시각(mcu_millis)과 같은 시계로 취급하지 않는다.
        # 센서 전용 세션은 근거리 감지를 보고만 하고 LED를 덮어쓰지 않는다.
        # fault 빨강 경고는 STM32 내부에서 여전히 우선한다.
        data = dict(self.session.feedback or {})
        common = dict(session_id=self.session.session_id, feedback_valid=valid, error=self._error,
                      receipt_stamp=self.get_clock().now().nanoseconds / 1e9,
                      sequence=data.get('sequence'), mcu_millis=data.get('mcu_millis'))
        distance = dict(common, distance_mm=data.get('distance_mm'), valid=bool(valid and data.get('valid')),
                        detected=data.get('detected'), held=data.get('held'),
                        range_status=data.get('range_status', 'WAITING'))
        led = dict(common, auto=data.get('led_auto'), requested_rgb=data.get('requested_rgb'),
                   obstacle_override=False)
        for publisher, payload in ((self._range, distance), (self._led, led)):
            message = String()
            message.data = json.dumps(payload, separators=(',', ':'))
            publisher.publish(message)

    def _tick(self):
        with self._lock:
            if self._closed or self._failed:
                return
            try:
                if self._monotonic_clock() - self._last_graph >= 0.2:
                    self._last_graph = self._monotonic_clock()
                    if self._mission_present():
                        raise RuntimeError('Mission/STM bridge conflicts with sensor debug')
                self.session.tick()
                if self.session.failed:
                    raise RuntimeError(self.session.error)
                if self.session.feedback is not None and self.session.feedback is not self._last_feedback:
                    self._last_feedback = self.session.feedback
                    self._publish(True)
            except Exception as exc:
                self._failed, self._error = True, str(exc)
                try:
                    if self._worker is not None:
                        self.session.fail(str(exc))
                    else:
                        self.session.close()
                finally:
                    self._publish(False)

    def prepare_shutdown(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for timer in self.timers:
                timer.cancel()
            try:
                self.session.close()
            finally:
                self._error = self._error or 'Sensor session closed; last values are historical'
                self._publish(False)
                if self._uart is not None:
                    self._uart.close()
        if self._worker is not None and not self._worker.stop(1.5, safe_off=True):
            self.get_logger().error('RR2 sensor STOP acknowledgement or serial close timed out')

    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
