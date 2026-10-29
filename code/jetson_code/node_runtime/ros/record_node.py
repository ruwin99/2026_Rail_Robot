"""경량 주행 상태만 관찰한다. 모터/UART/영상/클라우드 인터페이스를 소유하지 않는다."""
from __future__ import annotations

import os
from pathlib import Path
import time

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.utilities import get_message

from ..telemetry_record import TelemetryWriter


# 상대 이름이므로 --ros-args -r __ns:=/bench 에서 같은 namespace 장치를 관찰한다.
# WebRTC/Firebase JSON에는 인증/시그널링 정보가 들어갈 수 있어 기록 대상에서 제외한다.
TOPICS = {
    'debug/encoder/state': 'rail_robot_interfaces/msg/EncoderState',
    'debug/encoder/diagnostics': 'std_msgs/msg/String',
    'debug/encoder/command': 'rail_robot_interfaces/msg/EncoderCommand',
    'debug/ultrasonic/state': 'std_msgs/msg/String',
    'debug/led/state': 'std_msgs/msg/String',
    'runtime/fault': 'std_msgs/msg/String',
    'mission/state': 'rail_robot_interfaces/msg/MissionStatus',
    'control/decisions': 'rail_robot_interfaces/msg/ActuatorDecision',
    'stm/events': 'rail_robot_interfaces/msg/StmEvents',
    'debug/front/dc/set': 'std_msgs/msg/Bool',
    'debug/side/dc/set': 'std_msgs/msg/Bool',
    'debug/front/pump/set': 'std_msgs/msg/Bool',
    'debug/side/pump/set': 'std_msgs/msg/Bool',
    'tts/obstacle_state': 'std_msgs/msg/Bool',
}


class RecordNode(Node):
    def __init__(self, *, node_name='rail_record', **kwargs):
        super().__init__(node_name, **kwargs)
        self._closed = self._closing = False
        self.writer = None
        try:
            default = Path(os.environ.get('RAIL_ROBOT_HOME', str(Path.cwd()))) / 'records'
            directory = self.declare_parameter('output_directory', str(default)).value
            if not isinstance(directory, str) or not directory.strip():
                raise ValueError('output_directory must be a nonempty path')
            resolved = {name: self.resolve_topic_name(name) for name in TOPICS}
            self.writer = TelemetryWriter(directory,
                topics={resolved[name]: typename for name, typename in TOPICS.items()},
                namespace=self.get_namespace())
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=50,
                reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
            self._listeners = [self.create_subscription(get_message(typename), name,
                lambda message, topic=resolved[name]: self._receive(topic, message), qos)
                for name, typename in TOPICS.items()]
            print(f'RAIL RECORD recording: {self.writer.path}', flush=True)
        except BaseException:
            if self.writer is not None:
                self.writer.abort()
            super().destroy_node()
            raise

    @property
    def _failure(self):
        return self.writer.error if self.writer is not None else ''

    def shutdown_requested(self):
        return self.writer is not None and self.writer.limit_reached

    def _receive(self, topic, message):
        if self._closing or self._closed:
            return
        wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
        try:
            data = message_to_ordereddict(message)
            self.writer.submit(topic, data, wall_ns=wall_ns, monotonic_ns=mono_ns)
        except (AttributeError, TypeError, ValueError, OverflowError, RecursionError):
            self.writer.note_serialization_error()

    def prepare_shutdown(self):
        self._closing = True
        if self.writer is not None:
            self.writer.stop_accepting()

    def destroy_node(self):
        if self._closed:
            return True
        self.prepare_shutdown()
        success = self.writer.close() if self.writer is not None else False
        self._closed = True
        result = super().destroy_node()
        if not success:
            path = self.writer.path if self.writer is not None else '(not opened)'
            reason = self.writer.error if self.writer is not None else 'Recorder unavailable'
            raise OSError(f'RAIL RECORD incomplete: {path}: {reason}')
        print(f'RAIL RECORD saved: {self.writer.path} | samples={self.writer._samples} '
              f'loss={self.writer._loss} limit_reached={self.writer.limit_reached}', flush=True)
        return result
