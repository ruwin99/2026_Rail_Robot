"""Asynchronous ROS coordinator for the existing STM-led inspection FSM."""
from __future__ import annotations

# ROS 임무 조정자: /stm/events를 FSM에 전달하고 촬영·UART 서비스를 비동기로 연결한다.
# 읽는 순서: _start → _on_events/_process_messages → _begin_capture 또는 _begin_transition.
# 상태 순서·카운터 검증은 ../mission.py, 트리거 상수는 ../settings.py를 따라간다.
# 서비스 응답 필드는 rail_robot_interfaces의 CaptureInspection/DrainInspection/UartCommand 정의로 확인한다.

import asyncio
import json
import math
import threading
import time
import uuid
from collections import deque
from functools import wraps

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from rail_robot_interfaces.msg import MissionStatus, StmEvents
from rail_robot_interfaces.srv import CaptureInspection, DrainInspection, UartCommand

from ..clock_sync import CLOCK_UNSYNCED_MESSAGE, clock_synchronized
from ..mission import MissionStateMachine
from ..settings import (
    CAPTURE_TRIGGER, STARTED_TRIGGER, REALTIME_START_TRIGGER,
    RESCAN_RETURN_START_TRIGGER, DONE_TRIGGER, COMPLETE_MODE,
)
from .common import qos_events, qos_state, source_monotonic, stamp_from_seconds


def _serialized(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return invoke


class MissionNode(Node):
    def __init__(self, *, node_name="rail_mission", **node_kwargs):
        super().__init__(node_name, **node_kwargs)
        self._lock = threading.RLock()
        self.state = MissionStateMachine()
        self._failed = False
        self._closed = False
        self._shutdown_prepared = False
        self._error = ""
        self._busy = False
        self._start_pending = False
        self._start_command_requested = False
        self._mcu_prepared = False
        self._preflight_error = None
        self._messages = deque()
        self._pending_since = None
        self._pending_future = None
        # 시작 음성이 성공적으로 끝난 경우에만 UART START를 허용한다.
        # WAV 파일이 없으면 audio_enabled:=false를 명시한 시험만 음성을 생략한다.
        descriptor = ParameterDescriptor(read_only=True)
        self._audio_enabled = self.declare_parameter("audio_enabled", True, descriptor).value
        self._audio_timeout = float(self.declare_parameter("audio_start_timeout_sec", 40.0, descriptor).value)
        if type(self._audio_enabled) is not bool or not math.isfinite(self._audio_timeout) or not 0 < self._audio_timeout <= 300:
            raise ValueError("Invalid audio_enabled or audio_start_timeout_sec.")
        # RTC 배터리가 없으면 부팅 직후 시계가 과거에 머물다 NTP가 붙을 때 크게 점프한다.
        # 임무 도중 점프하면 ROS 신선도 검사가 fault를 내므로 동기화 전에는 START하지 않는다.
        # 시간 동기화 수단이 없는 시험장에서만 require_clock_sync:=false로 끈다.
        self._require_clock_sync = self.declare_parameter("require_clock_sync", True, descriptor).value
        if type(self._require_clock_sync) is not bool:
            raise ValueError("require_clock_sync must be a boolean.")
        # 임무가 정상 완료(COMPLETE)되면 완료 상태를 잠시 보여 준 뒤 스택을 스스로 종료한다. 웹 "프로그램 실행"
        # 유닛이 inactive로 돌아가 대시보드에서 실행 버튼을 처음부터 다시 누를 수 있다. 업로드가 켜져 있으면
        # 업로드 결과(성공·실패)를 기다리되 upload_wait_seconds를 넘기면 기다리지 않는다. 실패했으면 종료하지 않는다.
        self._exit_after_complete = self.declare_parameter("exit_after_complete", True, descriptor).value
        self._exit_hold = float(self.declare_parameter("exit_hold_seconds", 20.0, descriptor).value)
        self._wait_for_upload = self.declare_parameter("wait_for_upload", False, descriptor).value
        self._upload_wait = float(self.declare_parameter("upload_wait_seconds", 180.0, descriptor).value)
        if (type(self._exit_after_complete) is not bool or type(self._wait_for_upload) is not bool
                or not math.isfinite(self._exit_hold) or not 0 <= self._exit_hold <= 600
                or not math.isfinite(self._upload_wait) or not 0 <= self._upload_wait <= 900):
            raise ValueError("Invalid exit_after_complete, exit_hold_seconds, wait_for_upload or upload_wait_seconds.")
        self._complete_since = None
        self._upload_status = None
        self._exit_logged = False
        self._audio_waiting = False
        self._audio_request = None
        self._completion_request = None
        self._audio_last_sent = 0.0
        self._completion_last_sent = 0.0
        self._obstacle_held = False
        # 비동기 응답을 처리할 콜백 그룹은 재진입을 허용하되 공유 상태 변경은 _lock으로 직렬화한다.
        # _busy는 작업 순서를, _pending_future/_pending_since는 현재 응답과 시간 제한을 추적한다.
        self._callback_group = ReentrantCallbackGroup()
        self._uart = self.create_client(UartCommand, "stm/command", callback_group=self._callback_group)
        self._capture = self.create_client(CaptureInspection, "inspection/capture", callback_group=self._callback_group)
        self._drain = self.create_client(DrainInspection, "inspection/drain", callback_group=self._callback_group)
        self._ready_clients = {
            role: self.create_client(Trigger, f"{role}/ready", callback_group=self._callback_group)
            for role in ("camera", "inspection", "control", "vision")
        }
        self._status = self.create_publisher(MissionStatus, "mission/state", qos_state)
        if self._audio_enabled:
            self._ready_clients["tts"] = self.create_client(Trigger, "tts/ready", callback_group=self._callback_group)
        self._audio_requests = self.create_publisher(String, "tts/request", qos_events)
        self.create_subscription(String, "tts/result", self._on_audio_result, qos_events,
                                 callback_group=self._callback_group)
        self.create_subscription(Bool, "tts/obstacle_state", self._on_obstacle, qos_state,
                                 callback_group=self._callback_group)
        self.create_subscription(String, "inspection/upload_status", self._on_upload_status, qos_events,
                                 callback_group=self._callback_group)
        self._faults = self.create_publisher(String, "runtime/fault", qos_events)
        self.create_subscription(StmEvents, "stm/events", self._on_events, qos_events)
        self.create_subscription(String, "runtime/fault", self._on_fault, qos_events)
        self.create_service(Trigger, "mission/start", self._start, callback_group=self._callback_group)
        self.create_timer(0.2, self._tick)
        self._publish_status()

    def _ros_now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _clock_unsynchronized(self):
        # None(상태를 읽을 수 없는 비Linux 시험 환경)은 막지 않는다.
        return self._require_clock_sync and clock_synchronized() is False

    def _make_audio_request(self, cue, timeout):
        return dict(request_id=str(uuid.uuid4()), cue=cue, expires_at=self._ros_now() + timeout)

    @_serialized
    def _on_obstacle(self, message):
        self._obstacle_held = bool(message.data)

    @_serialized
    def _on_upload_status(self, message):
        # 완료 보고서 업로드 노드의 결과("uploaded: ..." 또는 "upload failed: ..."). 완료 뒤 첫 결과만 본다.
        if self._complete_since is not None and self._upload_status is None and len(message.data) <= 4096:
            self._upload_status = message.data
            if message.data.startswith("upload failed"):
                self.get_logger().warning("Dashboard report upload failed: " + message.data[:300])

    @_serialized
    def shutdown_requested(self):
        # run_node 루프가 0.1초마다 묻는다. True가 되면 이 프로세스가 정상 종료하고, 필수 노드 종료로 launch 묶음 전체가 내려간다.
        if (not self._exit_after_complete or self._complete_since is None or self._failed or self._closed
                or self._completion_request is not None):
            return False
        elapsed = time.monotonic() - self._complete_since
        if elapsed < self._exit_hold:
            return False
        if self._wait_for_upload and self._upload_status is None and elapsed < self._upload_wait:
            return False
        if not self._exit_logged:
            self._exit_logged = True
            self.get_logger().info("Mission complete; shutting down so the dashboard can launch the stack again.")
        return True

    def _send_audio(self, request):
        self._audio_requests.publish(String(data=json.dumps(request)))

    @_serialized
    def _on_audio_result(self, message):
        # 요청 ID와 cue가 일치하는 재생 종료 응답만 인정한다. 늦은 응답은 START를 풀 수 없다.
        if self._closed or len(message.data) > 4096:
            return
        try:
            result = json.loads(message.data)
            if not isinstance(result, dict) or type(result.get("success")) is not bool:
                return
        except (ValueError, TypeError):
            return
        for name in ("_audio_request", "_completion_request"):
            request = getattr(self, name)
            if request is None or any(result.get(key) != request[key] for key in ("request_id", "cue")):
                continue
            if self._ros_now() > request["expires_at"]:
                result = dict(result, success=False, error="Audio result arrived after its deadline.")
            if name == "_audio_request":
                future = self._pending_future
                if self._audio_waiting and future is not None and not future.done() and not future.cancelled():
                    future.set_result(result)
            else:
                if not result["success"]:
                    self.get_logger().warning("Completion audio failed: " + str(result.get("error", ""))[:300])
                self._completion_request = None
            return

    @_serialized
    def _publish_status(self):
        # /mission/state에서 mode, initial_count, rescan_count, failed/error를 함께 읽어 진행 지점을 찾는다.
        # 메시지 발행 시각은 임무 노드의 생존 확인용이며 촬영 시각이나 STM의 실행 완료 시각이 아니다.
        if self._closed:
            return
        message = MissionStatus()
        message.header.stamp = stamp_from_seconds(self._ros_now())
        message.mode = self.state.mode
        message.initial_count = self.state.initial_rail_section_count
        message.rescan_count = self.state.rescan_rail_section_count
        message.failed = self._failed
        message.error = self._error
        self._status.publish(message)

    @_serialized
    def _tick(self):
        # Service futures may never complete after a peer disappears. Keep
        # publishing state while bounding pending work; no ACK on timeout.
        # 준비 확인은 5초, 시작 음성은 별도 audio_start_timeout_sec 후 취소한다.
        # 일반 서비스 보류는 10초 후 fault로 고정하며 음성 대기에는 이 제한을 적용하지 않는다.
        # 작업 중에도 0.2초마다 상태를 발행하므로 토픽이 살아 있다는 사실만으로 서비스 완료를 판단하면 안 된다.
        if (
            self._start_pending and not self._start_command_requested
            and self._pending_since is not None
            and time.monotonic() - self._pending_since > (self._audio_timeout if self._audio_waiting else 5.0)
        ):
            self._preflight_error = ("Start audio timed out; START was not sent." if self._audio_waiting
                                     else "Readiness checks exceeded 5 seconds; START was not sent.")
            pending = self._pending_future
            self._pending_future = None
            self._pending_since = None
            if pending is not None and not pending.done() and not pending.cancelled():
                pending.cancel()
        elif (
            not self._failed and not self._audio_waiting and self._pending_since is not None
            and time.monotonic() - self._pending_since > 10.0
        ):
            self._fault("Pending inspection/UART service exceeded 10 seconds.")
        now = time.monotonic()
        # 발견/응답 손실 시 같은 ID만 재전송한다. 음성 노드는 같은 ID를 두 번 재생하지 않는다.
        for name, last_name in (("_audio_request", "_audio_last_sent"), ("_completion_request", "_completion_last_sent")):
            request = getattr(self, name)
            if request is None or self._failed or self._closed or self._preflight_error:
                continue
            if self._ros_now() >= request["expires_at"]:
                if name == "_completion_request":
                    self.get_logger().warning("Completion audio acknowledgement timed out.")
                    self._completion_request = None
                continue
            if now - getattr(self, last_name) >= 0.5:
                self._send_audio(request)
                setattr(self, last_name, now)
        self._publish_status()

    @_serialized
    def _on_fault(self, message):
        # 첫 오류를 _error에 남기고 대기 이벤트/응답을 버린다. 실패 후 늦게 도착한 응답으로 임무를 재개하지 않는다.
        # SAFE_OFF는 청소 출력 해제 요청이다. 실제 UART 처리는 stm_bridge 노드이며 이 호출은 응답을 기다리지 않는다.
        if message.data and not self._failed:
            self._failed = True
            self._error = message.data
            self._messages.clear()
            self._pending_since = None
            if self.state.pending_capture is not None and self.state.failure is None:
                self.state.fail_capture(self.state.pending_capture, message.data)
            pending = self._pending_future
            self._pending_future = None
            if pending is not None and not pending.done():
                pending.cancel()
            if self._uart.service_is_ready():
                request = UartCommand.Request()
                request.command = "SAFE_OFF"
                self._uart.call_async(request)
            self._publish_status()

    @_serialized
    def _fault(self, reason):
        if not self._failed:
            message = String()
            message.data = f"mission: {reason}"
            self._on_fault(message)
            self._faults.publish(message)

    async def _start(self, request, response):
        # launch는 준비 상태까지만 만든다. /mission/start에서 각 노드 준비를 확인한 뒤 START를 보낸다.
        with self._lock:
            if self._failed or self._busy or self._start_pending or self.state.start_command_sent:
                response.success, response.message = False, "Mission is busy, started or faulted."
                return response
            clients = (self._uart, self._capture, self._drain, *self._ready_clients.values())
            if self._obstacle_held:
                response.success, response.message = False, "Ultrasonic obstacle hold is active."
                return response
            if self._clock_unsynchronized():
                response.success, response.message = False, CLOCK_UNSYNCED_MESSAGE
                return response
            if not self._uart.service_is_ready():
                response.success, response.message = False, "STM command service must be available to prepare the mission."
                return response
            self._start_pending = True
            self._start_command_requested = False
            self._preflight_error = None
            self._pending_since = time.monotonic()
            prepare_command = UartCommand.Request()
            prepare_command.command = "PREPARE"
            command = UartCommand.Request()
            command.command = "START"
        try:
            # PREPARE를 받으면 STM LED가 빠른 파란 깜빡임(시작 직전)으로 바뀐다.
            # 준비 확인과 시작 음성이 도는 몇 초 동안 현장에서 곧 움직인다는 것을 알 수 있게 한다.
            with self._lock:
                if self._failed or self._closed or self._preflight_error:
                    raise RuntimeError(self._error or self._preflight_error or "Mission is closing.")
                self._pending_future = self._uart.call_async(prepare_command)
                prepare_future = self._pending_future
            result = await prepare_future
            with self._lock:
                if prepare_future.cancelled() or self._preflight_error:
                    raise RuntimeError(self._preflight_error or "STM PREPARE was cancelled.")
                if self._failed or self._closed:
                    raise RuntimeError(self._error or "Mission is closing.")
                if not result.success:
                    raise RuntimeError(result.error or "STM rejected PREPARE.")
                self._mcu_prepared = True
                self._pending_future = None
                self._pending_since = None

            with self._lock:
                if not all(client.service_is_ready() for client in clients):
                    raise RuntimeError("Required mission services (including tts/ready when audio is enabled) must be available.")

            for role, client in self._ready_clients.items():
                # 서비스가 보이는 것과 장치/모델 준비 성공은 다르므로 각 ready 응답을 순차 확인한다.
                # await 동안 lock을 놓아 상태 타이머와 오류 콜백이 실행될 수 있게 한다.
                with self._lock:
                    if self._failed or self._closed or self._preflight_error:
                        raise RuntimeError(self._error or self._preflight_error or "Mission is closing.")
                    self._pending_future = client.call_async(Trigger.Request())
                    self._pending_since = time.monotonic()
                    readiness = self._pending_future
                result = await readiness
                with self._lock:
                    if readiness.cancelled() or self._preflight_error:
                        raise RuntimeError(self._preflight_error or f"{role} readiness was cancelled.")
                    if self._failed or self._closed:
                        raise RuntimeError(self._error or "Mission is closing.")
                    if not result.success:
                        raise RuntimeError(f"{role} is not ready: {result.message}")
            if self._audio_enabled:
                with self._lock:
                    if self._failed or self._closed or self._preflight_error:
                        raise RuntimeError(self._error or self._preflight_error or "Mission is closing.")
                    self._audio_waiting = True
                    self._pending_since = time.monotonic()
                    self._pending_future = Future()
                    audio_future = self._pending_future
                    self._audio_request = self._make_audio_request("mission_started", self._audio_timeout)
                    self._send_audio(self._audio_request)
                    self._audio_last_sent = time.monotonic()
                result = await audio_future
                with self._lock:
                    self._audio_waiting = False
                    self._audio_request = None
                    if audio_future.cancelled() or self._failed or self._closed or self._preflight_error:
                        raise RuntimeError(self._error or self._preflight_error or "Start audio was cancelled.")
                    if not result or not result["success"]:
                        raise RuntimeError(str((result or {}).get("error") or "Start audio failed."))
                    if not all(client.service_is_ready() for client in clients):
                        raise RuntimeError("Required service disappeared during start audio.")
            with self._lock:
                if self._failed or self._closed or self._preflight_error:
                    raise RuntimeError(self._error or self._preflight_error or "Mission is closing.")
                if self._obstacle_held:
                    raise RuntimeError("Ultrasonic obstacle hold is active; START was not sent.")
                if self._clock_unsynchronized():
                    raise RuntimeError(CLOCK_UNSYNCED_MESSAGE)
                self._start_command_requested = True
                self._pending_since = time.monotonic()
                self._pending_future = self._uart.call_async(command)
                future = self._pending_future
            result = await future
            with self._lock:
                if self._failed:
                    raise RuntimeError(self._error)
                if future.cancelled():
                    raise RuntimeError("START service request was cancelled.")
                if not result.success:
                    raise RuntimeError(result.error)
                self.state.mark_start_sent()
                self._mcu_prepared = False
                # 서비스 성공은 UART 전송 성공이다. 실제 시작 승인은 이후 STM의 STARTED 이벤트로 받는다.
                response.success, response.message = True, "START sent; awaiting STM STARTED."
        except (Exception, asyncio.CancelledError) as exc:
            with self._lock:
                if self._start_command_requested:
                    self._fault(str(exc))
                response.success, response.message = False, self._error or self._preflight_error or str(exc)
                cancel_prepared = self._mcu_prepared and not self._start_command_requested and not self._closed
            if cancel_prepared and self._uart.service_is_ready():
                cancel_command = UartCommand.Request()
                cancel_command.command = "CANCEL_PREPARE"
                try:
                    with self._lock:
                        self._pending_future = self._uart.call_async(cancel_command)
                        cancel_future = self._pending_future
                        self._pending_since = time.monotonic()
                    cancel_result = await asyncio.wait_for(cancel_future, timeout=1.0)
                    if cancel_future.cancelled() or not cancel_result.success:
                        raise RuntimeError(getattr(cancel_result, "error", "") or "STM rejected CANCEL_PREPARE.")
                    with self._lock:
                        self._mcu_prepared = False
                except (Exception, asyncio.CancelledError) as cancel_error:
                    with self._lock:
                        if not self._closed:
                            self._fault(f"Could not clear STM pre-start state: {cancel_error}")
        finally:
            # 시작 서비스가 끝나야 큐에 쌓인 STARTED 등의 이벤트를 처리한다. UART 응답이 빨리 와도 순서가 보존된다.
            # 시작 실패 시 response.message가 준비 오류인지, /mission/state.error가 운용 오류인지 구분해 확인한다.
            with self._lock:
                self._start_pending = False
                self._audio_waiting = False
                self._audio_request = None
                self._start_command_requested = False
                self._pending_since = None
                self._pending_future = None
                self._publish_status()
                self._process_messages()
        return response

    @_serialized
    def _on_events(self, message):
        if self._failed:
            return
        try:
            # 0.8초 이내의 이벤트인지와 동일 배치 모순을 검사한 뒤 최대 32개까지만 대기시킨다.
            # backlog 오류는 STM 요청이 처리보다 빠르다는 뜻이므로 촬영/분석 서비스 지연부터 확인한다.
            source_monotonic(message.header.stamp, self._ros_now(), time.monotonic(), 0.8)
            tokens = list(message.tokens)
            self.state.validate_batch(tokens)
            if len(self._messages) + len(tokens) > 32:
                raise RuntimeError("STM event backlog exceeds 32 messages.")
            self._messages.extend(tokens)
            self._process_messages()
        except (RuntimeError, ValueError) as exc:
            self._fault(str(exc))

    @_serialized
    def _process_messages(self):
        # 서비스 처리 중의 STM 이벤트는 순서대로 보관하고, 완료 후 다음 이벤트를 처리한다.
        while self._messages and not (self._busy or self._start_pending or self._failed):
            message = self._messages.popleft()
            try:
                self.state.require_started(message)
                if message == STARTED_TRIGGER:
                    self.state.acknowledge_started()
                elif message == CAPTURE_TRIGGER:
                    self._begin_capture()
                else:
                    transition = self.state.prepare_transition(message)
                    if transition is not None:
                        self._begin_transition(transition)
                self._publish_status()
            except (RuntimeError, ValueError) as exc:
                self._fault(str(exc))

    def _call(self, client, request, on_success):
        # 응답을 여기서 기다리지 않고 완료 콜백으로 다음 단계를 잇는다. 타이머는 상태 발행·시간 제한을 계속 감시한다.
        if self._failed:
            return
        if not client.service_is_ready():
            self._fault("Required service is unavailable.")
            return
        self._busy = True
        self._pending_since = time.monotonic()
        try:
            future = client.call_async(request)
            self._pending_future = future
        except Exception as exc:
            self._fault(str(exc))
            return

        def completed(done):
            # 콜백 실패도 같은 fault 경로로 보낸다. 촬영 중이면 FSM에 실패를 기록해 CAPTURE_OK를 재시도하지 않는다.
            # on_success가 다음 서비스를 연결할 수 있으므로 여기서 무조건 _busy를 풀지 않는다.
            with self._lock:
                if self._failed:
                    return
                if self._pending_future is done:
                    self._pending_future = None
                try:
                    if done.cancelled():
                        raise RuntimeError("Pending service request was cancelled.")
                    on_success(done.result())
                except Exception as exc:
                    if self.state.pending_capture is not None and self.state.failure is None:
                        self.state.fail_capture(self.state.pending_capture, str(exc))
                    self._fault(str(exc))

        future.add_done_callback(completed)

    def _finish_work(self):
        self._busy = False
        self._pending_since = None
        self._publish_status()
        self._process_messages()

    def _begin_capture(self):
        pending = self.state.begin_capture()
        request = CaptureInspection.Request()
        request.request_id = uuid.uuid4().hex
        request.phase, request.phase_sequence = pending.phase, pending.phase_sequence
        request.trigger = CAPTURE_TRIGGER

        def accepted(result):
            # accepted는 필요한 원본 저장과 분석 큐 접수 완료를 뜻한다. 추론·업로드 완료를 기다리는 ACK가 아니다.
            if not result.accepted:
                raise RuntimeError(result.error or "Inspection capture was rejected.")
            self.state.accept_capture(pending)
            command = UartCommand.Request()
            command.command = "CAPTURE_OK"

            def acknowledged(reply):
                # ACK 전체 쓰기 성공 후에만 구간 수를 올리고 다음 STM 이벤트를 처리한다.
                # accepted 이후 여기서 실패하면 원본은 존재할 수 있어도 구간은 완료로 계산되지 않는다.
                if not reply.success:
                    raise RuntimeError(reply.error or "CAPTURE_OK was not completely sent.")
                self.state.acknowledge_capture(pending)
                self._finish_work()

            self._call(self._uart, command, acknowledged)

        self._call(self._capture, request, accepted)

    def _begin_transition(self, transition):
        # Jetson 상태 확정 전에 청소 출력 OFF 및 필요한 분석 작업 소진을 확인한다.
        # STM의 물리적 상태 전이를 이 함수가 새로 명령하는 것은 아니다.
        def commit():
            self.state.complete_transition(transition)
            if self._audio_enabled and self.state.mode == COMPLETE_MODE:
                # DONE 수신만으로 말하지 않는다. SAFE_OFF와 최종 분석 drain 성공 후에 요청한다.
                self._completion_request = self._make_audio_request("mission_completed", 120.0)
                self._send_audio(self._completion_request)
                self._completion_last_sent = time.monotonic()
            if self.state.mode == COMPLETE_MODE:
                self._complete_since = time.monotonic()
            self._finish_work()

        def after_off(result):
            if not result.success:
                raise RuntimeError(result.error or "SAFE_OFF failed.")
            if transition.trigger in (REALTIME_START_TRIGGER, DONE_TRIGGER):
                # REALTIME 진입과 DONE에서는 접수된 분석 작업이 모두 끝나야 상태를 확정한다.
                # DONE의 finalize=True는 검사 종료 처리도 요청한다. 대기 시 inspection/drain 응답과 해당 노드 로그를 본다.
                request = DrainInspection.Request()
                request.finalize = transition.trigger == DONE_TRIGGER

                def drained(reply):
                    if not reply.success:
                        raise RuntimeError(reply.error or "Capture analysis drain failed.")
                    commit()

                self._call(self._drain, request, drained)
            else:
                commit()

        if transition.trigger in (REALTIME_START_TRIGGER, RESCAN_RETURN_START_TRIGGER, DONE_TRIGGER):
            request = UartCommand.Request()
            request.command = "SAFE_OFF"
            self._call(self._uart, request, after_off)
        else:
            commit()

    @_serialized
    def prepare_shutdown(self):
        # Best-effort request; bridge status timeout also independently expires
        # outputs if shutdown prevents a response from reaching this node.
        if self._shutdown_prepared:
            return
        self._shutdown_prepared = True
        cancel_prepared = self._mcu_prepared or (self._start_pending and not self._start_command_requested)
        if not (self.state.mode == COMPLETE_MODE and not self._failed):
            # 정상 완료 뒤의 종료는 실패가 아니다. 마지막 상태가 FAILED로 대시보드에 남지 않게 한다.
            self._failed = True
            self._error = self._error or "Mission node is shutting down."
        self._messages.clear()
        self._pending_since = None
        pending = self._pending_future
        self._pending_future = None
        if pending is not None and not pending.done() and not pending.cancelled():
            pending.cancel()
        self._publish_status()
        if self._uart.service_is_ready():
            if cancel_prepared:
                cancel = UartCommand.Request()
                cancel.command = "CANCEL_PREPARE"
                self._uart.call_async(cancel)
            request = UartCommand.Request()
            request.command = "SAFE_OFF"
            self._uart.call_async(request)
        self._closed = True

    @_serialized
    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
