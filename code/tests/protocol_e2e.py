"""웹 임무 시작부터 완료까지의 프로토콜 통합 시험 도구.

웹 대시보드 명령이 닿는 지점(RemoteControlNode)부터 실제 코드를 그대로 연결한다.
  RemoteControlNode → /mission/start → MissionNode(실제 FSM) → StmBridgeNode → SerialWorker
  → RR2 UART 프레임 → 컴파일된 STM 펌웨어(host HAL, rr2_stdio)
ROS 통신(DDS)과 Firestore, 카메라·모델, 실제 모터·센서는 쓰지 않는다. 사진 저장·분석과 음성 재생은
요청 내용만 기록하는 대역이고, 엔코더·초음파는 시험이 물리 자극으로 주입한다.
시험 시간과 펌웨어 시계는 1:1로 흘러 UART 감시(100 ms PING, 600 ms)가 실제와 같게 동작한다.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

# STM MissionState 열거 번호(RR2 응답 마지막 필드)
(IDLE, INITIAL_SCAN, WAIT_CAPTURE_ACK, RETURNING, REALTIME_CLEANING, ACTUATOR_TEST, DEBUG_TEST,
 RESCAN_RETURN, RESCAN, MISSION_DONE, MISSION_FAULT, FINAL_RETURN) = range(12)
STATE_NAMES = {value: name for name, value in dict(
    IDLE=0, INITIAL_SCAN=1, WAIT_CAPTURE_ACK=2, RETURNING=3, REALTIME_CLEANING=4, ACTUATOR_TEST=5,
    DEBUG_TEST=6, RESCAN_RETURN=7, RESCAN=8, MISSION_DONE=9, MISSION_FAULT=10, FINAL_RETURN=11).items()}
ACCESSORY_OPCODES = ("FRONT_CLEANER_PWM_12_5", "FRONT_CLEANER_PWM_25", "FRONT_PUMP_ON",
                     "SIDE_CLEANER_PWM_12_5", "SIDE_CLEANER_PWM_25", "SIDE_PUMP_ON")


def install(executable_builder):
    """가짜 ROS와 UART·제어 대역을 설치하고 호스트 펌웨어를 빌드한다(자식 프로세스 전용)."""
    from test_ros_robotics import FakeNode, install_fake_ros

    executable = executable_builder()
    install_fake_ros()
    FakeNode.count_publishers = lambda self, _topic: 0
    FakeNode.resolve_service_name = lambda self, name: "/" + name
    FakeNode.get_service_names_and_types = lambda self: []

    class Output:
        def __init__(self, *_args):
            self.uart = None
            self.is_on = False

        def force_off(self):
            self.is_on = False

    uart = types.ModuleType("node_runtime.uart")
    uart.CleanerController = uart.PumpController = Output
    uart.read_uart_messages = lambda *_args: []
    uart.send_uart_test_command = lambda *_args: None
    sys.modules["node_runtime.uart"] = uart
    control = types.ModuleType("node_runtime.control")
    control.force_cleaning_pairs_safe_off = lambda *_pairs: None
    control.update_cleaning_actuators = lambda *_args, **_kwargs: None
    sys.modules["node_runtime.control"] = control
    return executable


class HostSerial:
    """UART 대역: 컴파일된 펌웨어 프로세스에 RR2 줄을 넣고 STM이 보낸 바이트를 그대로 돌려준다."""
    executable = None
    instances = []
    keep_open = False  # True: 시리얼 포트만 닫히고 STM(펌웨어 프로세스)은 계속 켜져 있다

    def __init__(self, **_kwargs):
        self.process = subprocess.Popen(
            [str(HostSerial.executable), "rr2_stdio"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.pending = bytearray()
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.closed = False
        self.writes = []      # Jetson → STM 원문(줄 단위)
        self.fw_ms = 0        # 펌웨어 시계 추정: !TIME 합 + RR2 프레임당 10 ms(호스트 브리지 규칙)
        self.rx = bytearray()  # STM → Jetson 원문 전체
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        HostSerial.instances.append(self)

    def _read_stdout(self):
        while True:
            chunk = self.process.stdout.read(1)
            if not chunk:
                return
            with self.lock:
                self.pending.extend(chunk)
                self.rx.extend(chunk)

    @property
    def in_waiting(self):
        with self.lock:
            return len(self.pending)

    def read(self, amount):
        with self.lock:
            data = bytes(self.pending[:amount])
            del self.pending[:amount]
            return data

    def _send(self, data):
        if data.startswith(b"RR2 ") and not data.startswith(b"RR2 HELLO "):
            self.fw_ms += 10
        elif data.startswith(b"!TIME "):
            self.fw_ms += int(data.split()[1])
        with self.write_lock:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        return len(data)

    def write(self, data):
        self.writes.append(data)
        return self._send(data)

    def encoder(self, coast_count=None):
        # 물리 엔코더 경계 인터럽트. coast_count가 있으면 정지 직후 관성으로 TIM3가 그 값까지 더 굴러간 상태다.
        self._send(b"!ENCODER\n" if coast_count is None else f"!ENCODER {int(coast_count)}\n".encode("ascii"))

    def advance_ms(self, amount):
        assert 0 < amount <= 200
        self._send(f"!TIME {amount}\n".encode("ascii"))

    def echo(self, echo_us):
        self._send(f"!ULTRASONIC {echo_us}\n".encode("ascii"))

    def close(self):
        if self.closed or HostSerial.keep_open:
            return
        self.closed = True
        self.process.stdin.close()
        self.process.wait(timeout=2)
        self.reader.join(timeout=1)
        error = self.process.stderr.read().decode(errors="replace")
        self.process.stdout.close()
        self.process.stderr.close()
        if self.process.returncode:
            raise AssertionError(error)

    def rx_frames(self):
        from node_runtime.uart_v2 import decode_frame
        frames = []
        with self.lock:
            data = bytes(self.rx)
        # 읽기 스레드가 아직 받는 중인 마지막 줄(줄바꿈 전)은 제외하고 완성된 줄만 해석한다.
        data = data[:data.rfind(b"\n") + 1]
        for raw in data.split(b"\n"):
            if raw.startswith(b"RR2 "):
                frames.append(decode_frame(raw + b"\n"))
        return frames

    def tx_commands(self):
        """Jetson이 보낸 CMD 프레임의 (순번, 오피코드, 인자) 목록."""
        from node_runtime.uart_v2 import decode_frame
        commands = []
        for wire in self.writes:
            fields = decode_frame(wire)
            if fields[0] == "CMD":
                commands.append((int(fields[2]), fields[3], tuple(fields[4:])))
        return commands


class FakeRemoteWorker:
    """Firestore 폴링 스레드 대역. 웹 명령 문서가 수락된 뒤의 (id, action) 큐와 완료 보고만 흉내 낸다."""

    def __init__(self, _ready):
        self.queue, self.results, self.last_error = [], {}, None

    def next_request(self):
        return self.queue.pop(0) if self.queue else None

    def complete(self, command_id, success, code):
        self.results[command_id] = (bool(success), code)

    def close(self):
        pass


class Plant:
    def __init__(self, executable, *, coast=0, sections=6, decisions=False, exit_hold=None, wait_for_upload=False):
        self.executable = executable
        self.coast = coast
        self.sections = sections
        self.decisions = decisions
        self.exit_hold, self.wait_for_upload = exit_hold, wait_for_upload
        self.nodes = ()
        self.peer = None
        self.t0 = time.monotonic()
        self.fake_ms = 0
        self.timeline = []          # (라벨 또는 이벤트 토큰) 발생 순서
        self.events = []            # Jetson 임무 노드가 받은 STM 이벤트 토큰
        self.faults = []
        self.stm_commands = []      # stm/command 서비스 요청 이름
        self.capture_requests = []  # (phase, phase_sequence)
        self.drain_requests = []    # finalize 값
        self.tts_requests = []
        self._cursor = {}
        self._tts_replied = set()
        self._started = 0
        self._decision_sequence = {"front": 0, "side": 0}
        self._last_decision = 0.0
        self.last_dir, self.last_coast = 0, 0
        self.last_stop_ms = 0
        self.frozen = False       # True: Jetson 쪽 전체가 멈춘 것처럼 아무 노드도 돌리지 않는다

    # ---- 구성 -----------------------------------------------------------------------------
    def boot(self):
        """스택을 새로 띄운다. 이미 켜진 STM(self.peer)이 있으면 같은 STM에 새 UART 세션으로 붙는다."""
        import node_runtime.ros.remote_control_node as remote_module
        from node_runtime.ros.mission_node import MissionNode
        from node_runtime.ros.remote_control_node import RemoteControlNode
        from node_runtime.ros.stm_bridge_node import StmBridgeNode
        remote_module.RemoteControlWorker = FakeRemoteWorker
        HostSerial.executable = self.executable
        if self.peer is None:
            factory = HostSerial
        else:
            self.peer.pending.clear()  # 새 프로세스가 포트를 열면 이전 세션의 남은 바이트는 없다
            factory = lambda **_kwargs: self.peer
        self.bridge = StmBridgeNode(
            "fake", uart_factory=factory,
            parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
        if self.peer is None:
            self.peer = HostSerial.instances[-1]
        overrides = [SimpleNamespace(name="require_clock_sync", value=False),
                     SimpleNamespace(name="wait_for_upload", value=self.wait_for_upload)]
        if self.exit_hold is not None:
            overrides.append(SimpleNamespace(name="exit_hold_seconds", value=self.exit_hold))
        self.mission = MissionNode(parameter_overrides=overrides)
        self.remote = RemoteControlNode()
        self._cursor.clear()
        self._started = 0
        self.nodes = (self.bridge, self.mission, self.remote)
        bridge_command = self.bridge.services["stm/command"].callback

        def stm_call(request):
            self.stm_commands.append(request.command)
            self.timeline.append("cmd:" + request.command)
            return bridge_command(request, SimpleNamespace())

        def capture(request):
            self.capture_requests.append((request.phase, request.phase_sequence))
            return SimpleNamespace(accepted=True, error="")

        def drain(request):
            self.drain_requests.append(bool(request.finalize))
            self.timeline.append("drain:" + ("finalize" if request.finalize else "plain"))
            return SimpleNamespace(success=True, error="", manifest_path="")

        self.mission._uart.responder = stm_call
        self.mission._capture.responder = capture
        self.mission._drain.responder = drain
        self.remote._stm.responder = stm_call
        self.remote._start.responder = lambda _request: None  # 코루틴이 완료하면 pump가 채운다
        # 부팅 직후 STM이 시작 승인 조건(20회 중 원거리 16회)을 갖추도록 센서 표본을 준다.
        for _ in range(20):
            self.peer.echo(2000)

    def close(self):
        nodes, self.nodes = self.nodes, ()
        for node in reversed(nodes):
            node.destroy_node()

    def stop_stack_keep_stm(self):
        """프로세스 종료를 흉내 낸다: 노드를 닫고 시리얼만 놓는다. STM은 그대로 켜져 있다."""
        worker = self.bridge._worker
        HostSerial.keep_open = True
        try:
            self.close()
        finally:
            HostSerial.keep_open = False
        # 실제로는 이전 프로세스가 완전히 끝난 뒤에야 새 프로세스가 포트를 연다. 이전 직렬 스레드가
        # 같은 대역 포트를 계속 읽으면 새 세션의 READY를 가로챌 수 있으므로 끝날 때까지 기다린다.
        worker._thread.join(5.0)
        assert not worker._thread.is_alive(), "previous serial worker is still reading the port"

    def begin_next_mission(self):
        self.events, self.timeline, self.faults = [], [], []
        self.stm_commands, self.capture_requests = [], []
        self.drain_requests, self.tts_requests = [], []
        self._tts_replied.clear()
        self.last_dir, self.last_coast = 0, 0
        self.wire_mark = len(self.peer.tx_commands())
        self._decision_sequence = {"front": 0, "side": 0}

    # ---- 노드 사이 메시지 전달 ---------------------------------------------------------------
    def _fresh(self, node, topic):
        key = (id(node), topic)
        messages = node.publishers[topic].messages
        start = self._cursor.get(key, 0)
        self._cursor[key] = len(messages)
        return messages[start:]

    def _forward(self):
        from std_msgs.msg import String
        for message in self._fresh(self.bridge, "stm/events"):
            self.events.extend(message.tokens)
            for token in message.tokens:
                self.timeline.append(token)
            self.mission.subscriptions["stm/events"].callback(message)
        for message in self._fresh(self.bridge, "tts/obstacle_state"):
            self.mission.subscriptions["tts/obstacle_state"].callback(message)
        for message in self._fresh(self.bridge, "runtime/fault"):
            self.faults.append(message.data)
            self.mission.subscriptions["runtime/fault"].callback(message)
        for message in self._fresh(self.mission, "runtime/fault"):
            self.faults.append(message.data)
            self.bridge.subscriptions["runtime/fault"].callback(message)
        for message in self._fresh(self.mission, "mission/state"):
            self.bridge.subscriptions["mission/state"].callback(message)
            self.remote.subscriptions["mission/state"].callback(message)
        for message in self._fresh(self.mission, "tts/request"):
            request = json.loads(message.data)
            if request["request_id"] in self._tts_replied:
                continue
            self._tts_replied.add(request["request_id"])
            self.tts_requests.append(request["cue"])
            self.timeline.append("tts:" + request["cue"])
            result = {"request_id": request["request_id"], "cue": request["cue"], "success": True, "error": ""}
            self.mission.subscriptions["tts/result"].callback(String(data=json.dumps(result)))

    def _feed_decisions(self, ros_now):
        # 제어 노드가 REALTIME 동안 두 장치에 계속 보내는 청소 결정(전면 25%+펌프, 측면 12.5%+펌프).
        from node_runtime.ros.common import stamp_from_seconds
        from rail_robot_interfaces.msg import ActuatorDecision
        if not self.decisions or self.mission.state.mode != "REALTIME":
            return
        if time.monotonic() - self._last_decision < 0.1:
            return
        self._last_decision = time.monotonic()
        for role, duty in (("front", 25.0), ("side", 12.5)):
            self._decision_sequence[role] += 1
            message = ActuatorDecision(
                session_id="control-a", sequence=self._decision_sequence[role], role=role,
                cleaner_duty_percent=duty, pump_on=True,
                valid_until=stamp_from_seconds(ros_now + 0.6))
            message.header.stamp = stamp_from_seconds(ros_now)
            self.bridge.subscriptions["control/decisions"].callback(message)

    def _start_coroutines(self):
        futures = self.remote._start.futures
        while self._started < len(futures):
            future = futures[self._started]
            self._started += 1

            async def run(pending=future):
                response = SimpleNamespace()
                await self.mission._start(SimpleNamespace(), response)
                pending.set_result(response)
            asyncio.get_running_loop().create_task(run())

    def pump(self):
        if self.frozen or not self.nodes:
            return
        ros_now = 1000.0 + (time.monotonic() - self.t0)
        for node in self.nodes:
            node._ros_seconds = ros_now
        self.bridge._tick()
        self._forward()
        self._feed_decisions(ros_now)
        self.remote._tick()
        self._start_coroutines()
        self.mission._tick()
        self._forward()

    # ---- 시간 --------------------------------------------------------------------------------
    async def step(self, fake_ms=50):
        # 펌웨어 시계는 최대 50 ms씩 올린다. 100 ms PING 사이 간격을 실제 운용과 비슷하게 유지한다.
        while fake_ms > 0:
            chunk = min(fake_ms, 50)
            fake_ms -= chunk
            self.peer.advance_ms(chunk)
            self.fake_ms += chunk
            end = time.monotonic() + chunk / 1000.0
            while True:
                self.pump()
                if time.monotonic() >= end:
                    break
                await asyncio.sleep(0.005)

    async def wait(self, condition, what, fake_ms=4000):
        waited = 0
        while not condition():
            if waited >= fake_ms:
                raise AssertionError(f"timeout waiting for {what}; events={self.events} faults={self.faults} "
                                     f"stm_commands={self.stm_commands[-6:]}")
            await self.step(25)
            waited += 25
        return waited

    # ---- 물리 시나리오 ------------------------------------------------------------------------
    def _coast_count(self, direction):
        if self.coast == 0:
            return None
        return self.coast if direction > 0 else 600 - self.coast

    async def _begin_phase(self, direction):
        # 방향이 바뀌고 이전 정지에서 더 굴러갔다면, 되돌아가며 옛 경계를 먼저 지나는 보정 이벤트가 한 번 온다.
        if self.last_dir == -direction and self.last_coast > 0:
            await self.step(50)
            self.peer.encoder()
            await self.step(25)

    def _end_phase(self, direction):
        self.last_dir, self.last_coast = direction, self.coast

    async def stop_boundaries(self, direction, count, after_each=None):
        """경계마다 정지하는 구간(촬영 구간, 일반 복귀)."""
        await self._begin_phase(direction)
        for index in range(1, count + 1):
            await self.step(100)
            self.timeline.append(f"boundary:{direction}:{index}")
            self.last_stop_ms = self.peer.fw_ms
            self.peer.encoder(self._coast_count(direction))
            await self.step(25)
            if after_each is not None:
                await after_each(index)
        self._end_phase(direction)

    async def continuous_boundaries(self, direction, count, hook=None):
        """중간 경계는 통과하고 마지막 경계에서만 정지하는 연속 주행(실시간 청소, 재검사 복귀, 출발점 복귀)."""
        await self._begin_phase(direction)
        for index in range(1, count + 1):
            await self.step(100)
            if hook is not None:
                await hook(index)
            last = index == count
            self.timeline.append(f"boundary:{direction}:{index}" + (":last" if last else ""))
            if last:
                self.last_stop_ms = self.peer.fw_ms
            self.peer.encoder(self._coast_count(direction) if last else None)
            await self.step(25)
        self._end_phase(direction)

    async def pause_until(self, condition, what):
        # 단계 사이 1초 정지. 마지막 정지 뒤 그보다 일찍 다음 단계가 시작되면 프로토콜 위반이다.
        # 펌웨어 시계로 재야 한다(RR2 프레임마다 10 ms씩 더 간다). 아직 처리되지 않은 프레임 몇 개의 오차는 허용한다.
        await self.wait(condition, what, fake_ms=3000)
        paused = self.peer.fw_ms - self.last_stop_ms
        assert paused >= 950, f"{what} arrived after only {paused} ms of the 1 s pause"

    async def ultrasonic(self, echo_us, count):
        for _ in range(count):
            self.peer.echo(echo_us)
            await self.step(10)

    async def web_command(self, command_id, action):
        self.remote._worker.queue.append((command_id, action))
        await self.wait(lambda: command_id in self.remote._worker.results, f"web {action} result", 70000)
        return self.remote._worker.results[command_id]

    def firmware_states(self):
        """PING·명령 응답의 마지막 필드로 본 펌웨어 상태 번호의 변화(연속 중복 제거)."""
        states = []
        for frame in self.peer.rx_frames():
            if frame[0] in ("ACK", "NACK") and frame[-1].isdigit():
                value = int(frame[-1])
                if not states or states[-1] != value:
                    states.append(value)
        return states

    def nacks(self):
        return [frame for frame in self.peer.rx_frames() if frame[0] == "NACK"]

    async def run_mission(self, *, start_id="web-start-1", after_start=None, realtime_hook=None,
                          final_hook=None, final_return_hook=None):
        """웹 START부터 DONE까지. 반환 시 STM은 IDLE로 돌아와 있어야 한다."""
        n = self.sections
        mission = self.mission
        await self.step(300)  # 임무 노드 상태가 웹 명령 노드에 도달

        # 1) 웹 START
        result = await self.web_command(start_id, "START")
        assert result == (True, "start_sent"), result
        await self.wait(lambda: mission.state.start_acknowledged, "STARTED")
        if after_start is not None:
            await after_start(self)

        # 2) 초기 촬영: 경계마다 정지, CAMERA_CAPTURE → 사진 접수 → CAPTURE_OK
        async def captured_initial(index):
            await self.wait(lambda: mission.state.initial_rail_section_count == index, f"initial capture {index}")
        await self.stop_boundaries(1, n, captured_initial)
        await self.pause_until(lambda: mission.state.mode == "RETURN", "RETURN_START")

        # 3) 복귀: 경계마다 정지하지만 Jetson 이벤트는 없다
        await self.stop_boundaries(-1, n)
        await self.pause_until(lambda: mission.state.mode == "REALTIME", "REALTIME_START")

        # 4) 실시간 청소(연속)
        await self.continuous_boundaries(1, n, realtime_hook)
        await self.pause_until(lambda: mission.state.mode == "RESCAN_RETURN", "RESCAN_RETURN_START")

        # 5) 재검사 복귀(연속)
        await self.continuous_boundaries(-1, n)
        await self.pause_until(lambda: mission.state.mode == "RESCAN", "RESCAN_START")

        # 6) 재검사 촬영
        async def captured_rescan(index):
            await self.wait(lambda: mission.state.rescan_rail_section_count == index, f"rescan capture {index}")
        await self.stop_boundaries(1, n, captured_rescan)
        if final_hook is not None:
            await final_hook()

        # 7) 재검사 종료 → 출발점 복귀(연속) → DONE
        await self.pause_until(lambda: mission.state.mode == "RESCAN_DONE", "RESCAN_DONE")
        assert "DONE" not in self.events, "DONE arrived before the robot returned to the start"
        await self.continuous_boundaries(-1, n, final_return_hook)
        await self.pause_until(lambda: mission.state.mode == "COMPLETE", "DONE")
        await self.wait(lambda: "mission_completed" in self.tts_requests, "completion audio")
        await self.step(300)
