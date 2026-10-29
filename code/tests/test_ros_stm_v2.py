"""Exercise the actual ROS bridge v2 handshake and execution ACK path in isolation."""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


def child_checks():
    from test_ros_robotics import install_fake_ros

    install_fake_ros()
    serial_module = types.ModuleType("serial")
    serial_module.SerialException = OSError
    serial_module.Serial = object
    sys.modules["serial"] = serial_module
    settings = types.ModuleType("node_runtime.settings")
    for name, value in {
        "FRONT_ACTUATOR_COMMANDS": object(), "SIDE_ACTUATOR_COMMANDS": object(),
        "REALTIME_MODE": "REALTIME", "START_COMMAND": b"START\r\n",
        "CAPTURE_OK_COMMAND": b"CAPTURE_OK\r\n", "UART_WRITE_TIMEOUT_SECONDS": 0.2,
        "DEFAULT_SERIAL_PORT": "/dev/fake", "DEFAULT_BAUD_RATE": 115200,
        "DONE_TRIGGER": "DONE",
        "LINK_OK_COMMAND": b"LINK_OK\r\n", "LEGACY_LINK_INTERVAL_SECONDS": 0.2, "AUTOMATIC_RAIL_SECTION_TARGET": 4, "DEFAULT_RAIL_SECTION_TARGET": 4,
    }.items():
        setattr(settings, name, value)
    sys.modules["node_runtime.settings"] = settings

    class Output:
        def __init__(self, *_args):
            self.uart = None
            self.is_on = False

        def force_off(self):
            self.is_on = False

    uart = types.ModuleType("node_runtime.uart")
    uart.CleanerController = uart.PumpController = Output
    uart.read_uart_messages = lambda _serial, _pending: []
    uart.send_uart_test_command = lambda _serial, _command: None
    sys.modules["node_runtime.uart"] = uart
    control = types.ModuleType("node_runtime.control")
    control.force_cleaning_pairs_safe_off = lambda *_pairs: None
    control.update_cleaning_actuators = lambda *_args, **_kwargs: None
    sys.modules["node_runtime.control"] = control

    from node_runtime.uart_v2 import decode_frame, encode_frame
    from node_runtime.serial_worker import CommandResult
    from node_runtime.ros.stm_bridge_node import StmBridgeNode
    from node_runtime.ros.stm_bridge_node import RR2_ON_REFUSAL_BACKOFF_SECONDS as RR2_BACKOFF
    from concurrent.futures import Future

    class RespondingSerial:
        last = None
        instances = []

        def __init__(self, **_kwargs):
            self.pending = bytearray()
            self.writes = []
            self.closed = False
            self.close_count = 0
            self.io_threads = set()
            self.start_written = threading.Event()
            self.ping_written = threading.Event()
            self.start_ack_written = threading.Event()
            self.ping_during_start = threading.Event()
            self.event_ack_count = 0
            self._start_ack_timer_started = False
            self.ping_at = None
            self.start_ack_at = None
            self.reject_opcode = None
            self.ping_count = 0
            self.last_rr2_activity = time.monotonic()
            self.drive_active = False
            self.estop_count = 0
            RespondingSerial.last = self
            RespondingSerial.instances.append(self)

        @property
        def in_waiting(self):
            self.io_threads.add(threading.get_ident())
            return len(self.pending)

        def read(self, size):
            self.io_threads.add(threading.get_ident())
            data = bytes(self.pending[:size])
            del self.pending[:size]
            return data

        def write(self, data):
            self.io_threads.add(threading.get_ident())
            self.writes.append(data)
            fields = decode_frame(data)
            if fields[0] == "HELLO":
                self.pending.extend(encode_frame("READY", fields[1], "2"))
            elif fields[0] == "CMD":
                if fields[3] == self.reject_opcode:
                    self.pending.extend(encode_frame(
                        "NACK", fields[1], fields[2], fields[3], "INVALID_STATE"
                    ))
                elif fields[3] == "START":
                    self.drive_active = True
                    self.start_written.set()
                    if not self._start_ack_timer_started:
                        self._start_ack_timer_started = True
                        self.pending.extend(encode_frame("EVENT", fields[1], "1", "MISSION_STARTED"))
                        self.pending.extend(encode_frame("EVENT", fields[1], "2", "CAMERA_CAPTURE"))

                        def delayed_ack():
                            self.start_ack_at = time.monotonic()
                            self.last_rr2_activity = self.start_ack_at
                            self.pending.extend(encode_frame(
                                "ACK", fields[1], fields[2], fields[3], "ACCEPTED", "1"
                            ))
                            self.start_ack_written.set()
                        threading.Timer(0.17, delayed_ack).start()
                elif fields[3] == "ESTOP":
                    # Modeled MCU: ESTOP stops the drive before its ACK.
                    self.drive_active = False
                    self.estop_count += 1
                    self.last_rr2_activity = time.monotonic()
                    self.pending.extend(encode_frame(
                        "ACK", fields[1], fields[2], fields[3], "ACCEPTED", "1"
                    ))
                else:
                    self.last_rr2_activity = time.monotonic()
                    self.pending.extend(encode_frame(
                        "ACK", fields[1], fields[2], fields[3], "ACCEPTED", "1"
                    ))
            elif fields[0] == "PING":
                self.ping_at = time.monotonic()
                self.ping_count += 1
                self.last_rr2_activity = self.ping_at
                if self.start_written.is_set() and not self.start_ack_written.is_set():
                    self.ping_during_start.set()
                self.ping_written.set()
                self.pending.extend(encode_frame("ACK", fields[1], fields[2], "PING", "ACCEPTED", "1"))
            elif fields[0] == "EVENT_ACK":
                self.event_ack_count += 1
            return len(data)

        def close(self):
            self.io_threads.add(threading.get_ident())
            self.close_count += 1
            self.closed = True

    node = StmBridgeNode("fake", uart_factory=RespondingSerial,
                         parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    assert node._uart is None
    assert all(cleaner.uart is None and pump.uart is None
               for cleaner, pump in node._pairs.values())

    # If an OFF transaction and an older pending ON finish in the same ROS
    # tick, the OFF ACK must remain the final applied state.
    applied_on = Future()
    applied_on.set_result(CommandResult(True, "a" * 32, 1))
    applied_off = Future()
    applied_off.set_result(CommandResult(True, "a" * 32, 2))
    node._v2_pending["front_cleaner"] = (12.5, applied_on)
    node._safety_future = applied_off
    node._tick_v2_outputs(time.monotonic())
    assert node._v2_applied["front_cleaner"] == 0.0

    # 2026-10-05 실기: 실시간 청소 끝의 1초 단계 정지 동안 ON 재전송이 SAFETY 4, 다음 단계 시작 직후
    # STATE 7로 거절되어(초음파 정지 이벤트 전 경합도 같은 SAFETY) 브리지가 fault가 됐다.
    # STM이 출력을 꺼 둔 거절이므로 꺼진 상태로 기록하고 잠시 뒤 다시 판단한다.
    def finished(success, error=""):
        future = Future()
        future.set_result(CommandResult(success, "a" * 32, 3, error))
        return future
    safety = "STM rejected RR2 command: SAFETY 4"
    state = "STM rejected RR2 command: STATE 7"

    def realtime_bridge(now):
        # control이 SIDE 청소기 25%를 계속 원하는 REALTIME 상태의 브리지.
        bridge = StmBridgeNode("fake", uart_factory=RespondingSerial,
                               parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
        bridge._mode, bridge._start_acknowledged, bridge._phase_inhibited = "REALTIME", True, False
        bridge._mission_seen = now + 3.5
        bridge._targets["side"] = (SimpleNamespace(cleaner_duty_percent=25.0, pump_on=False), now + 10)
        assert bridge._enabled(now + 3.1)
        return bridge

    def refuse(bridge, error, at):
        bridge._v2_pending["side_cleaner"] = (25.0, finished(False, error))
        bridge._tick_v2_outputs(at)

    started = time.monotonic()
    probe = realtime_bridge(started)
    refuse(probe, safety, started)                 # 단계 정지 시작: SAFETY 4
    assert not probe._failed and probe._v2_applied["side_cleaner"] == 0.0
    # 거절된 출력은 매 tick이 아니라 0.2초 간격으로만 다시 보낸다(정지 중 재전송·경고 폭주 방지).
    probe._tick_v2_outputs(started + 0.1)
    assert "side_cleaner" not in probe._v2_pending
    probe._tick_v2_outputs(started + RR2_BACKOFF + 0.01)
    assert "side_cleaner" in probe._v2_pending
    refuse(probe, state, started + 1.2)            # 다음 단계 시작 직후: STATE 7
    refuse(probe, state, started + 2.9)
    assert not probe._failed
    # 거절이 3초 넘게 이어지면(이벤트 유실·상태 불일치) fault로 남긴다.
    refuse(probe, state, started + 3.1)
    assert probe._failed
    probe.destroy_node()
    # ACK가 오면 그 출력의 거절 이력을 지운다.
    node._v2_applied["side_pump"] = True
    node._v2_pending["side_pump"] = (True, finished(False, safety))
    node._tick_v2_outputs(started)
    assert not node._failed and node._v2_applied["side_pump"] is False
    node._v2_pending["side_pump"] = (True, finished(True))
    node._tick_v2_outputs(started + 0.3)
    assert node._v2_applied["side_pump"] is True and "side_pump" not in node._v2_safety_since
    # 아직 전송 전이던 ON을 SAFE_OFF가 취소한 결과는 정상이다(단계 전환·장애물 정지와 겹칠 때 생긴다).
    # 2026-10-05 통합 시험: 이 결과를 fault로 보면 REALTIME→재검사 복귀 경계에서 임무가 내려갈 수 있었다.
    node._v2_pending["side_cleaner"] = (25.0, finished(False, "Superseded by SAFE_OFF"))
    node._tick_v2_outputs(started + 0.4)
    assert not node._failed and "side_cleaner" not in node._v2_pending
    # 다른 취소 사유(예: ESTOP)나 OFF의 거절은 여전히 fault다.

    for key, desired, error in (("front_pump", False, safety),
                                ("front_pump", True, "STM rejected RR2 command: ARGS 4")):
        probe = StmBridgeNode("fake", uart_factory=RespondingSerial,
                              parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
        probe._v2_pending[key] = (desired, finished(False, error))
        probe._tick_v2_outputs(time.monotonic())
        assert probe._failed, (desired, error)
        probe.destroy_node()
    node.destroy_node()
    node = StmBridgeNode("fake", uart_factory=RespondingSerial,
                         parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    for command in ("PREPARE", "CANCEL_PREPARE"):
        response = node._command(SimpleNamespace(command=command), SimpleNamespace())
        assert response.success, response.error
    emitted = [decode_frame(frame)[3] for frame in RespondingSerial.last.writes
               if decode_frame(frame)[0] == "CMD"]
    assert emitted[-2:] == ["PREPARE", "CANCEL_PREPARE"], emitted
    response_box = []
    service_thread = threading.Thread(target=lambda: response_box.append(
        node._command(SimpleNamespace(command="START"), SimpleNamespace())
    ))
    service_thread.start()
    assert RespondingSerial.last.start_written.wait(0.3)
    # A ROS subscription callback can acquire bridge state while the service
    # waits for the delayed MCU ACK; it never performs or waits on UART I/O.
    callback_started = time.monotonic()
    node._on_decision(SimpleNamespace())
    assert time.monotonic() - callback_started < 0.03
    assert RespondingSerial.last.ping_written.wait(0.3)
    service_thread.join(0.5)
    assert not service_thread.is_alive()
    response = response_box[0]
    assert response.success, response.error
    assert RespondingSerial.last.start_ack_written.is_set()
    assert RespondingSerial.last.ping_during_start.is_set()
    assert RespondingSerial.last.event_ack_count == 2
    node._tick()
    assert node._start_acknowledged and node._capture_pending
    assert node.publishers["stm/events"].messages[0].tokens == ["STARTED"]
    assert node.publishers["stm/events"].messages[-1].tokens == ["CAMERA_CAPTURE"]
    RespondingSerial.last.pending.extend(encode_frame(
        "EVENT", node._worker.session_id, "3", "MISSION_DONE"))
    time.sleep(0.02)
    node._tick()
    assert node.publishers["stm/events"].messages[-1].tokens == ["DONE"]
    assert not any(frame == b"START\r\n" for frame in RespondingSerial.last.writes)
    node.destroy_node()
    assert RespondingSerial.last.closed
    assert RespondingSerial.last.close_count == 1
    assert len(RespondingSerial.last.io_threads) == 1
    assert any(decode_frame(frame)[0] == "EVENT_ACK" for frame in RespondingSerial.last.writes)

    class NoCommandAckSerial(RespondingSerial):
        def write(self, data):
            fields = decode_frame(data)
            if fields[0] == "CMD" and fields[3] == "START":
                self.io_threads.add(threading.get_ident())
                self.writes.append(data)
                self.start_written.set()
                return len(data)
            return super().write(data)

    no_ack = StmBridgeNode("fake", uart_factory=NoCommandAckSerial,
                           parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    denied = no_ack._command(SimpleNamespace(command="START"), SimpleNamespace())
    assert not denied.success
    assert no_ack._failed
    command_frames = [frame for frame in NoCommandAckSerial.last.writes
                      if decode_frame(frame)[0] == "CMD" and decode_frame(frame)[3] == "START"]
    assert len(command_frames) == 2
    assert command_frames[0] == command_frames[1]
    assert b"START\r\n" not in NoCommandAckSerial.last.writes
    no_ack.destroy_node()

    racing = StmBridgeNode("fake", uart_factory=NoCommandAckSerial,
                           parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    raced_responses = []
    service = threading.Thread(target=lambda: raced_responses.append(
        racing._command(SimpleNamespace(command="START"), SimpleNamespace())
    ))
    service.start()
    assert NoCommandAckSerial.last.start_written.wait(0.3)
    close_started = time.monotonic()
    racing.destroy_node()
    service.join(0.5)
    assert not service.is_alive(), "service future was stranded during bridge shutdown"
    assert len(raced_responses) == 1 and not raced_responses[0].success
    assert time.monotonic() - close_started < 0.5
    assert NoCommandAckSerial.last.close_count == 1
    assert len(NoCommandAckSerial.last.io_threads) == 1

    off_node = StmBridgeNode("fake", uart_factory=RespondingSerial,
                             parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    RespondingSerial.last.reject_opcode = "FRONT_PUMP_OFF"
    off_response = off_node._command(
        SimpleNamespace(command="SAFE_OFF"), SimpleNamespace()
    )
    assert not off_response.success
    assert off_node._failed
    off_node.destroy_node()

    # A runtime fault after START and before DONE may happen while driving: it
    # escalates to the priority ESTOP instead of waiting for the 600 ms watchdog.
    fault_node = StmBridgeNode("fake", uart_factory=RespondingSerial,
                               parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    drive_started = fault_node._command(
        SimpleNamespace(command="START"), SimpleNamespace()
    )
    assert drive_started.success
    fault_serial = RespondingSerial.last
    assert fault_serial.drive_active
    fault_node._on_fault(SimpleNamespace(data="runtime fault"))
    estop_result = fault_node._emergency_stop_future.result(timeout=1.0)
    assert estop_result.success
    assert not fault_serial.drive_active, "ESTOP, not the watchdog, stops the drive"
    assert fault_serial.estop_count == 1
    ping_count_at_fault_off = fault_serial.ping_count
    time.sleep(0.25)
    assert fault_serial.ping_count == ping_count_at_fault_off
    # A repeated fault joins the same ESTOP transaction.
    fault_node._on_fault(SimpleNamespace(data="second fault"))
    assert fault_serial.estop_count == 1
    fault_node.destroy_node()

    # After DONE the STM is already idle: keep accessory OFF + heartbeat stop only.
    done_node = StmBridgeNode("fake", uart_factory=RespondingSerial,
                              parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    assert done_node._command(SimpleNamespace(command="START"), SimpleNamespace()).success
    done_serial = RespondingSerial.last
    with done_node._lock:
        done_node._process_tokens(["DONE"])
    done_node._on_fault(SimpleNamespace(data="post-mission fault"))
    assert done_node._safety_future.result(timeout=1.0).success
    assert done_serial.estop_count == 0
    assert done_node._worker.ready, "intentional heartbeat suspension must not self-fault the worker"
    done_node.destroy_node()

    # Before START nothing can be driving: the fault also keeps the accessory OFF path.
    idle_node = StmBridgeNode("fake", uart_factory=RespondingSerial,
                              parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    idle_serial = RespondingSerial.last
    idle_node._on_fault(SimpleNamespace(data="pre-start fault"))
    assert idle_node._safety_future.result(timeout=1.0).success
    assert idle_serial.estop_count == 0
    idle_node.destroy_node()

    # A normal non-REALTIME mission SAFE_OFF is not a fault and keeps PING alive.
    normal_node = StmBridgeNode("fake", uart_factory=RespondingSerial,
                                parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    normal_serial = RespondingSerial.last
    normal_now = normal_node._ros_now()
    normal_node._on_mission(SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(
            sec=int(normal_now), nanosec=int((normal_now % 1.0) * 1_000_000_000)
        )),
        mode="IDLE", failed=False,
    ))
    ping_count_before_idle = normal_serial.ping_count
    time.sleep(0.25)
    assert normal_serial.ping_count > ping_count_before_idle
    assert normal_node._worker.ready
    normal_node.destroy_node()

    # Obstacle hold is a temporary SAFE_OFF, not a runtime fault; it must keep
    # PING alive so a held mission does not expire the STM session watchdog.
    obstacle_node = StmBridgeNode(
        "fake", uart_factory=RespondingSerial,
        parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")],
    )
    obstacle_serial = RespondingSerial.last
    obstacle_node._process_tokens(["OBSTACLE_HOLD"])
    obstacle_off = obstacle_node._safety_future.result(timeout=1.0)
    assert obstacle_off.success
    assert not obstacle_node._failed
    obstacle_ping_count = obstacle_serial.ping_count
    time.sleep(0.25)
    assert obstacle_serial.ping_count > obstacle_ping_count
    assert obstacle_node._worker.ready
    obstacle_node.destroy_node()

    # A failed mission status is the same runtime fault path: acknowledge
    # SAFE_OFF, suspend heartbeat, and leave worker RX/close service alive.
    mission_fault_node = StmBridgeNode(
        "fake", uart_factory=RespondingSerial,
        parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")],
    )
    mission_fault_serial = RespondingSerial.last
    mission_now = mission_fault_node._ros_now()
    mission_fault_node._on_mission(SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(
            sec=int(mission_now), nanosec=int((mission_now % 1.0) * 1_000_000_000)
        )),
        mode="IDLE", failed=True,
    ))
    mission_fault_off = mission_fault_node._safety_future.result(timeout=1.0)
    assert mission_fault_off.success
    mission_ping_count = mission_fault_serial.ping_count
    time.sleep(0.25)
    assert mission_fault_serial.ping_count == mission_ping_count
    assert mission_fault_node._worker.ready
    mission_fault_node.destroy_node()

    close_release = threading.Event()
    class StuckCloseSerial(RespondingSerial):
        def close(self):
            self.io_threads.add(threading.get_ident())
            self.close_count += 1
            close_release.wait(1.0)
            self.closed = True

    close_node = StmBridgeNode("fake", uart_factory=StuckCloseSerial,
                               parameter_overrides=[SimpleNamespace(name="uart_protocol", value="v2")])
    original_stop = close_node._worker.stop
    close_node._worker.stop = lambda _timeout, safe_off=True: original_stop(
        0.03, safe_off=safe_off
    )
    close_node.destroy_node()
    assert any("serial worker" in message.lower() and "shutdown" in message.lower()
               for message in close_node.log_messages)
    close_release.set()
    close_node._worker._thread.join(0.5)
    assert close_node._worker._stopped_event.is_set()
    assert StuckCloseSerial.last.close_count == 1


class RosStmV2Tests(unittest.TestCase):
    def test_bridge_ack_and_heartbeat_failure(self):
        result = subprocess.run([sys.executable, __file__, "--child"],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--child" in sys.argv:
        child_checks()
    else:
        unittest.main()
