"""Virtual ROS callback ↔ RR2 serial-peer checks; no ROS, STM, or Jetson hardware."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


def child_checks(case):
    import json
    import threading
    import time
    import types
    from types import SimpleNamespace

    from test_ros_robotics import FakeNode, install_fake_ros

    install_fake_ros()
    FakeNode.count_publishers = lambda self, _topic: 0
    FakeNode.resolve_service_name = lambda self, name: "/" + name
    FakeNode.get_service_names_and_types = lambda self: []

    settings = types.ModuleType("node_runtime.settings")
    for name, value in {
        "FRONT_ACTUATOR_COMMANDS": object(), "SIDE_ACTUATOR_COMMANDS": object(),
        "REALTIME_MODE": "REALTIME", "START_COMMAND": b"START\r\n",
        "CAPTURE_OK_COMMAND": b"CAPTURE_OK\r\n", "UART_WRITE_TIMEOUT_SECONDS": .2,
        "DEFAULT_SERIAL_PORT": "/dev/fake", "DEFAULT_BAUD_RATE": 115200,
        "DONE_TRIGGER": "DONE",
        "LINK_OK_COMMAND": b"LINK_OK\r\n", "LEGACY_LINK_INTERVAL_SECONDS": 0.2, "AUTOMATIC_RAIL_SECTION_TARGET": 4, "DEFAULT_RAIL_SECTION_TARGET": 4,
    }.items():
        setattr(settings, name, value)
    sys.modules["node_runtime.settings"] = settings
    uart = types.ModuleType("node_runtime.uart")

    class Output:
        def __init__(self, *_args):
            self.uart = None
            self.is_on = False

        def force_off(self):
            self.is_on = False

    uart.CleanerController = uart.PumpController = Output
    uart.read_uart_messages = lambda *_args: []
    uart.send_uart_test_command = lambda *_args: None
    sys.modules["node_runtime.uart"] = uart
    control = types.ModuleType("node_runtime.control")
    control.force_cleaning_pairs_safe_off = lambda *_pairs: None
    control.update_cleaning_actuators = lambda *_args, **_kwargs: None
    sys.modules["node_runtime.control"] = control

    import rail_robot_interfaces.msg as messages
    from node_runtime.rr2_protocol import Rr2Frame, decode_rr2, encode_rr2
    from node_runtime.ros.common import stamp_from_seconds

    class EncoderState:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    messages.EncoderCommand = SimpleNamespace
    messages.EncoderState = EncoderState
    from node_runtime.ros.stm_bridge_node import StmBridgeNode
    from node_runtime.ros.actuator_debug_node import ActuatorDebugNode
    from node_runtime.ros.sensor_debug_node import SensorDebugNode

    class Peer:
        """A protocol double; records accepted commands and sends valid RR2 frames."""

        instances = []

        def __init__(self, **_kwargs):
            self.lock = threading.RLock()
            self.pending = bytearray()
            self.writes = []
            self.accepted = []
            self.event_acks = []
            self.seen = set()
            self.session = None
            self.capability = None
            self.closed = False
            self.tx_sequence = 0
            self.tick = 100
            self.drop_first_start_ack = False
            self.start_attempts = 0
            self.rebooted = False
            Peer.instances.append(self)

        @property
        def in_waiting(self):
            with self.lock:
                return len(self.pending)

        def read(self, amount):
            with self.lock:
                data = bytes(self.pending[:amount])
                del self.pending[:amount]
                return data

        def queue(self, frame, *, corrupt=False):
            wire = encode_rr2(frame)
            if corrupt:
                tail = b"0" if wire[-6:-5] != b"0" else b"1"
                wire = wire[:-6] + tail + wire[-5:]
            with self.lock:
                self.pending.extend(wire)

        def emit(self, kind, *payload, session=None, sequence=None, corrupt=False):
            with self.lock:
                if sequence is None:
                    self.tx_sequence += 1
                    sequence = self.tx_sequence
                self.queue(Rr2Frame(kind, session or self.session, sequence, payload),
                           corrupt=corrupt)
                return sequence

        def _initial_status(self):
            if self.capability == "MISSION":
                return
            if self.capability == "SENSOR_DEBUG":
                self.emit("STATUS", "ULTRASONIC", "100", "180", "1", "1", "0",
                          "1", "0", "0", "0", "ECHO")
            else:
                self.emit("STATUS", "ENCODER", "100", "599", "4", "0", "0", "0",
                          "0", "0")

        def write(self, data):
            with self.lock:
                frame = decode_rr2(data)
                self.writes.append(frame)
                if self.rebooted:
                    # MCU power loss discards its session. A live Jetson owner
                    # may not silently reuse the old START/session afterward.
                    return len(data)
                if frame.kind == "HELLO":
                    self.session, self.capability = frame.session, frame.payload[0]
                    self.queue(Rr2Frame("READY", self.session, None,
                                        ("2", self.capability)))
                    self._initial_status()
                elif frame.kind == "PING":
                    self.queue(Rr2Frame("ACK", frame.session, frame.sequence,
                                        ("PING", "ACCEPTED", "0")))
                elif frame.kind == "EVENT_ACK":
                    self.event_acks.append(frame.sequence)
                elif frame.kind == "CMD":
                    opcode = frame.payload[0]
                    key = (frame.session, frame.sequence, frame.payload)
                    if key not in self.seen:
                        self.seen.add(key)
                        self.accepted.append(frame.payload)
                        if opcode == "START":
                            self.emit("EVENT", "MISSION_STARTED")
                            self.emit("EVENT", "CAMERA_CAPTURE")
                        elif opcode == "SET":
                            self.tick += 1
                            self.emit("STATUS", "ENCODER", str(self.tick), "600", "5",
                                      frame.payload[1], frame.payload[2],
                                      frame.payload[3], "80", "80")
                        elif opcode == "LED":
                            self.tick += 1
                            self.emit("STATUS", "ULTRASONIC", str(self.tick), "180",
                                      "1", "1", "0", "0", *frame.payload[2:],
                                      "ECHO")
                    if opcode == "START":
                        self.start_attempts += 1
                        if self.drop_first_start_ack and self.start_attempts == 1:
                            return len(data)
                    self.queue(Rr2Frame("ACK", frame.session, frame.sequence,
                                        (opcode, "ACCEPTED", "0")))
                return len(data)

        def close(self):
            self.closed = True

    def until(predicate, node=None, seconds=.5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if node is not None:
                node._tick()
            if predicate():
                return
            time.sleep(.003)
        assert predicate(), "virtual peer response timed out"

    if case == "mission":
        # The service waits for an ACK. The peer intentionally drops the first
        # START ACK, so the worker must retry the identical frame only once.
        class RetryStartPeer(Peer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.drop_first_start_ack = True

        node = StmBridgeNode("fake", uart_factory=RetryStartPeer,
                             parameter_overrides=[
                                 SimpleNamespace(name="uart_protocol", value="v2")])
        peer = Peer.instances[-1]
        try:
            response = node.services["stm/command"].callback(
                SimpleNamespace(command="START"), SimpleNamespace())
            assert response.success, response.error
            starts = [frame for frame in peer.writes
                      if frame.kind == "CMD" and frame.payload == ("START",)]
            assert len(starts) == 2 and starts[0] == starts[1]
            assert peer.accepted.count(("START",)) == 1
            until(lambda: node._start_acknowledged and node._capture_pending,
                  node)
            assert [msg.tokens for msg in node.publishers["stm/events"].messages] == [
                ["STARTED"], ["CAMERA_CAPTURE"]]
            assert peer.event_acks[:2] == [1, 2]
            response = node.services["stm/command"].callback(
                SimpleNamespace(command="CAPTURE_OK"), SimpleNamespace())
            assert response.success and not node._capture_pending
            assert ("CAPTURE_OK",) in peer.accepted

            node.subscriptions["mission/state"].callback(SimpleNamespace(
                header=SimpleNamespace(stamp=stamp_from_seconds(100.0)),
                mode="REALTIME", failed=False))
            node.subscriptions["control/decisions"].callback(SimpleNamespace(
                header=SimpleNamespace(stamp=stamp_from_seconds(100.0)),
                valid_until=stamp_from_seconds(100.6),
                session_id="control-a", sequence=1, role="side",
                cleaner_duty_percent=12.5, pump_on=True))
            until(lambda: ("SIDE_CLEANER_PWM_12_5",) in peer.accepted
                  and ("SIDE_PUMP_ON",) in peer.accepted, node)
            cleaner_off_before = peer.accepted.count(("SIDE_CLEANER_OFF",))
            pump_off_before = peer.accepted.count(("SIDE_PUMP_OFF",))
            node._ros_seconds = 100.1
            node.subscriptions["control/decisions"].callback(SimpleNamespace(
                header=SimpleNamespace(stamp=stamp_from_seconds(100.1)),
                valid_until=stamp_from_seconds(100.7),
                session_id="control-a", sequence=2, role="side",
                cleaner_duty_percent=0.0, pump_on=False))
            until(lambda: peer.accepted.count(("SIDE_CLEANER_OFF",)) > cleaner_off_before
                  and peer.accepted.count(("SIDE_PUMP_OFF",)) > pump_off_before, node)

            # A damaged event has no effect. The same valid frame then enters
            # the existing obstacle and TTS topics exactly once.
            next_seq = peer.tx_sequence + 1
            peer.emit("EVENT", "OBSTACLE_HOLD", sequence=next_seq, corrupt=True)
            time.sleep(.02)
            node._tick()
            assert not node._obstacle_held and next_seq not in peer.event_acks
            peer.emit("EVENT", "OBSTACLE_HOLD", sequence=next_seq)
            until(lambda: node._obstacle_held, node)
            count = len(node.publishers["tts/obstacle_events"].messages)
            peer.emit("EVENT", "OBSTACLE_HOLD", sequence=next_seq)
            until(lambda: peer.event_acks.count(next_seq) == 2, node)
            assert len(node.publishers["tts/obstacle_events"].messages) == count
            assert node.publishers["tts/obstacle_state"].messages[-1].data
        finally:
            node.destroy_node()
        assert peer.closed

        # A new owner negotiates a new session. Delayed prior-session frames
        # cannot make the new node think it has STARTED or captured an image.
        restarted = StmBridgeNode("fake", uart_factory=Peer,
                                  parameter_overrides=[
                                      SimpleNamespace(name="uart_protocol", value="v2")])
        new_peer = Peer.instances[-1]
        try:
            assert new_peer.session != peer.session
            new_peer.emit("EVENT", "STARTED", session=peer.session)
            new_peer.emit("EVENT", "CAMERA_CAPTURE", session=peer.session)
            time.sleep(.02)
            restarted._tick()
            assert not restarted._start_acknowledged
            assert not restarted._capture_pending
            assert not restarted.publishers["stm/events"].messages
        finally:
            restarted.destroy_node()

    elif case == "actuator":
        node = ActuatorDebugNode("fake", uart_protocol="v2", uart_factory=Peer,
                                 exclusive_device=lambda _uart: None)
        peer = Peer.instances[-1]
        try:
            until(lambda: node.session.feedback is not None, node)
            node.subscriptions["debug/encoder/command"].callback(
                SimpleNamespace(direction="forward", speed_percent=20.0))
            until(lambda: ("SET", "1", "2000", "0") in peer.accepted, node)
            node.subscriptions["debug/front/dc/set"].callback(SimpleNamespace(data=True))
            node.subscriptions["debug/side/pump/set"].callback(SimpleNamespace(data=True))
            until(lambda: ("SET", "1", "2000", "9") in peer.accepted, node)
            until(lambda: bool(node.publishers["debug/encoder/state"].messages)
                  and node.publishers["debug/encoder/state"].messages[-1].active_output_mask == 9,
                  node)
            state = node.publishers["debug/encoder/state"].messages[-1]
            assert state.feedback_valid and state.applied_direction == "forward"
            assert state.active_output_mask == 9
            off_before = peer.accepted.count(("SET", "0", "0", "0"))
            peer.emit("STATUS", "DIAG", "110", "RUNNING", "0", "0", "0",
                      "600", "5", "80", "80")
            until(lambda: bool(node.publishers["debug/encoder/diagnostics"].messages)
                  and json.loads(node.publishers["debug/encoder/diagnostics"].messages[-1].data)
                  ["reason"] == "RUNNING", node)
            node.subscriptions["debug/encoder/command"].callback(
                SimpleNamespace(direction="stop", speed_percent=0.0))
            node.subscriptions["debug/front/dc/set"].callback(SimpleNamespace(data=False))
            node.subscriptions["debug/side/pump/set"].callback(SimpleNamespace(data=False))
            until(lambda: peer.accepted.count(("SET", "0", "0", "0")) > off_before, node)
            assert all(frame.kind != "CMD" or frame.payload[0] != "START"
                       for frame in peer.writes)
        finally:
            node.destroy_node()
        assert peer.closed and ("STOP",) in peer.accepted

    elif case == "sensor":
        node = SensorDebugNode(uart_protocol="v2", uart_factory=Peer,
                                exclusive_device=lambda _uart: None)
        peer = Peer.instances[-1]
        try:
            until(lambda: node.session.feedback is not None, node)
            state = json.loads(node.publishers["debug/ultrasonic/state"].messages[-1].data)
            assert state["feedback_valid"] and state["distance_mm"] == 180
            node.subscriptions["debug/led/set"].callback(SimpleNamespace(data="green"))
            until(lambda: ("LED", "breathe", "0", "255", "0") in peer.accepted, node)
            until(lambda: bool(node.publishers["debug/led/state"].messages)
                  and json.loads(node.publishers["debug/led/state"].messages[-1].data)
                  ["requested_rgb"] == [0, 255, 0], node)
            led = json.loads(node.publishers["debug/led/state"].messages[-1].data)
            assert led["requested_rgb"] == [0, 255, 0]
            assert all(frame.kind != "CMD" or frame.payload[0] not in ("START", "SET")
                       for frame in peer.writes)
        finally:
            node.destroy_node()
        assert peer.closed and ("STOP",) in peer.accepted
    elif case == "reboot":
        node = StmBridgeNode("fake", uart_factory=Peer,
                             parameter_overrides=[
                                 SimpleNamespace(name="uart_protocol", value="v2")])
        peer = Peer.instances[-1]
        try:
            response = node.services["stm/command"].callback(
                SimpleNamespace(command="START"), SimpleNamespace())
            assert response.success
            start_count = peer.accepted.count(("START",))
            peer.rebooted = True
            peer.session = None
            peer.seen.clear()
            until(lambda: not node._worker.ready, seconds=1.2)
            node._tick()
            assert node._failed
            assert peer.accepted.count(("START",)) == start_count
            assert not any(frame.kind == "HELLO" for frame in peer.writes[1:])
        finally:
            node.destroy_node()
    else:
        raise ValueError(case)


class VirtualRosRr2PeerTests(unittest.TestCase):
    def _run_case(self, case):
        result = subprocess.run(
            [sys.executable, __file__, "--child", case],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mission_service_events_crc_retry_dedup_and_restart(self):
        self._run_case("mission")

    def test_actuator_topics_encoder_dc_pump_status_and_stop(self):
        self._run_case("actuator")

    def test_sensor_led_ultrasonic_status_and_stop(self):
        self._run_case("sensor")

    def test_mcu_reboot_during_active_session_fails_without_restarting_drive(self):
        self._run_case("reboot")


if __name__ == "__main__":
    if "--child" in sys.argv:
        child_checks(sys.argv[-1])
    else:
        unittest.main()
