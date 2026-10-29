"""ROS 요청/완료 토픽과 고정 WAV worker를 연결한다. 주행 명령을 보내지 않는다."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from ..audio_alerts import AudioAlerts
from .common import qos_events, qos_state


class AudioAlertNode(Node):
    def __init__(self, *, backend_factory=AudioAlerts, node_name="rail_audio_alert", **kwargs):
        super().__init__(node_name, **kwargs)
        descriptor = ParameterDescriptor(read_only=True)
        defaults = dict(audio_directory=str(Path.home() / ".local/share/rail_robot/audio"),
                        audio_device="default", playback_timeout_sec=30.0, repeat_gap_sec=0.5)
        config = {name: self.declare_parameter(name, value, descriptor).value
                  for name, value in defaults.items()}
        self._backend = backend_factory(**config)
        self._closed = False
        self._backend_closed = False
        self._result = self.create_publisher(String, "tts/result", qos_events)
        self._status = self.create_publisher(String, "tts/status", qos_state)
        self.create_subscription(String, "tts/request", self._on_request, qos_events)
        self.create_subscription(String, "tts/debug", self._on_debug, qos_events)
        self.create_subscription(Bool, "tts/obstacle_state", self._on_obstacle, qos_state)
        self.create_subscription(String, "runtime/fault", self._on_fault, qos_events)
        self.create_service(Trigger, "tts/ready", self._ready)
        self.create_timer(0.05, self._tick)

    def _on_request(self, message):
        if self._closed:
            return
        try:
            if len(message.data) > 1024:
                raise ValueError("Audio request exceeds 1024 characters")
            request = json.loads(message.data)
            self._backend.submit(request, self.get_clock().now().nanoseconds / 1e9)
        except (ValueError, TypeError) as exc:
            # 잘못된 메시지는 ID를 신뢰할 수 없으므로 승인/실패 응답을 꾸며내지 않는다.
            self.get_logger().warning(f"Rejected audio request: {exc}")

    def _on_debug(self, message):
        if self._closed:
            return
        cue = message.data
        if not isinstance(cue, str) or len(cue) > 32 or cue not in ("mission_started", "mission_completed"):
            self.get_logger().warning("Rejected TTS debug cue")
            return
        now = self.get_clock().now().nanoseconds / 1e9
        request = {"request_id": str(uuid.uuid4()), "cue": cue, "expires_at": now + 40.0}
        try:
            self._backend.submit(request, now)
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"Rejected TTS debug request: {exc}")

    def _on_obstacle(self, message):
        if not self._closed:
            self._backend.set_obstacle(message.data)

    def _on_fault(self, message):
        if not self._closed and message.data:
            self._backend.cancel_all("Runtime fault cancelled audio")

    def _ready(self, request, response):
        response.success, response.message = self._backend.readiness()
        return response

    def _tick(self):
        if self._closed:
            return
        # worker에서는 ROS API를 호출하지 않는다. 완료 결과만 executor로 회수하여 발행한다.
        for result in self._backend.poll_results():
            self._result.publish(String(data=json.dumps(result, ensure_ascii=False)))
        self._status.publish(String(data=json.dumps(self._backend.status(), ensure_ascii=False)))

    def prepare_shutdown(self):
        self._closed = True
        if not self._backend_closed:
            self._backend.close()
            self._backend_closed = True

    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
