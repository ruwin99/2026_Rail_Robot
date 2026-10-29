"""Opt-in dashboard command bridge to the existing mission and STM services."""
from __future__ import annotations

import threading
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger
from rail_robot_interfaces.msg import MissionStatus
from rail_robot_interfaces.srv import UartCommand

from ..clock_sync import CLOCK_UNSYNCED_MESSAGE
from ..remote_control import RemoteControlWorker
from ..settings import CAPTURE_SCAN_MODE
from .common import qos_state


class RemoteControlNode(Node):
    def __init__(self, **kwargs):
        super().__init__("rail_remote_control", **kwargs)
        self._lock = threading.Lock()
        self._processing_lock = threading.Lock()
        self._mission = None
        self._mission_seen = 0.0
        self._stop_requested = False
        self._pending = {}
        self._closed = False
        self._last_error = None
        group = ReentrantCallbackGroup()
        self._start = self.create_client(Trigger, "mission/start", callback_group=group)
        self._stm = self.create_client(UartCommand, "stm/command", callback_group=group)
        self.create_subscription(MissionStatus, "mission/state", self._on_mission, qos_state,
                                 callback_group=group)
        self._worker = RemoteControlWorker(self._mission_ready)
        self.create_timer(0.05, self._tick)

    def _on_mission(self, message):
        with self._lock:
            self._mission = message
            self._mission_seen = time.monotonic()

    def _mission_ready(self):
        with self._lock:
            message = self._mission
            return bool(
                not self._closed and not self._stop_requested
                and message is not None and time.monotonic() - self._mission_seen <= 1.0
                and message.mode == CAPTURE_SCAN_MODE
                and message.initial_count == 0 and message.rescan_count == 0
                and not message.failed
            )

    def _dispatch(self, command_id, action):
        if action == "E_STOP":
            with self._lock:
                self._stop_requested = True
            if not self._stm.service_is_ready():
                self._worker.complete(command_id, False, "estop_unavailable")
                return
            request = UartCommand.Request()
            request.command = "E_STOP"
            future = self._stm.call_async(request)
            self._pending[command_id] = (action, future, time.monotonic() + 3.0)
            return
        if not self._mission_ready():
            self._worker.complete(command_id, False, "remote_stop_requested" if self._stop_requested else "mission_not_ready")
            return
        if not self._start.service_is_ready():
            self._worker.complete(command_id, False, "start_unavailable")
            return
        future = self._start.call_async(Trigger.Request())
        # Startup audio may intentionally take 40 seconds; bound the full reply.
        self._pending[command_id] = (action, future, time.monotonic() + 60.0)

    def _tick(self):
        with self._processing_lock:
            self._process()

    def _process(self):
        if self._closed:
            return
        # Drain all newly accepted commands before checking slow START replies.
        while True:
            command = self._worker.next_request()
            if command is None:
                break
            self._dispatch(*command)
        now = time.monotonic()
        for command_id, (action, future, deadline) in list(self._pending.items()):
            if not future.done() and now < deadline:
                continue
            del self._pending[command_id]
            if not future.done():
                future.cancel()
                code = "estop_unconfirmed" if action == "E_STOP" else "start_timeout"
                self._worker.complete(command_id, False, code)
                continue
            success, code = self._outcome(action, future)
            self._worker.complete(command_id, success, code)
        error = self._worker.last_error
        if error and error != self._last_error:
            self.get_logger().warning("Dashboard control unavailable: " + error)
        self._last_error = error

    @staticmethod
    def _outcome(action, future):
        message = ""
        try:
            response = future.result()
            success = bool(response.success)
            message = str(getattr(response, "message", "") or "")
        except Exception:
            success = False
        code = ("estop_ack" if success else "estop_rejected") if action == "E_STOP" else (
            "start_sent" if success else
            # 대시보드가 "시계 동기화 대기"를 따로 안내할 수 있게 원인을 구분한다.
            "clock_not_synced" if message == CLOCK_UNSYNCED_MESSAGE else "start_rejected")
        return success, code

    def prepare_shutdown(self):
        with self._processing_lock:
            if self._closed:
                return
            with self._lock:
                self._closed = True
            for command_id, (action, future, _) in list(self._pending.items()):
                if future.done():
                    success, code = self._outcome(action, future)
                else:
                    future.cancel()
                    success, code = False, ("estop_unconfirmed" if action == "E_STOP"
                                            else "start_outcome_unknown")
                self._worker.complete(command_id, success, code)
            self._pending.clear()
            while True:
                command = self._worker.next_request()
                if command is None:
                    break
                command_id, action = command
                code = "estop_unavailable" if action == "E_STOP" else "start_unavailable"
                self._worker.complete(command_id, False, code)
        self._worker.close()

    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
