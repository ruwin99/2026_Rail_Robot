"""ROS callback ↔ compiled STM application C round trips via a host HAL.

This is a process-backed UART double. It runs production STM translation units
with the test HAL, not a board, ROS DDS, or physical outputs.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


def child_checks():
    import json
    import threading
    import time
    import types
    from types import SimpleNamespace

    from stm32_host import build_host
    from test_ros_robotics import FakeNode, install_fake_ros

    executable = build_host()
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
    from node_runtime.ros.common import stamp_from_seconds

    class EncoderState:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    messages.EncoderCommand = SimpleNamespace
    messages.EncoderState = EncoderState
    from node_runtime.ros.stm_bridge_node import StmBridgeNode
    from node_runtime.ros.actuator_debug_node import ActuatorDebugNode
    from node_runtime.ros.sensor_debug_node import SensorDebugNode

    class HostSerial:
        instances = []

        def __init__(self, **_kwargs):
            self.process = subprocess.Popen(
                [str(executable), "rr2_stdio"], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            self.pending = bytearray()
            self.lock = threading.RLock()
            self.write_lock = threading.Lock()
            self.closed = False
            self.writes = []
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
            with self.write_lock:
                self.process.stdin.write(data)
                self.process.stdin.flush()
            return len(data)

        def write(self, data):
            self.writes.append(data)
            return self._send(data)

        def pulse_encoder(self):
            # Test-only physical stimulus to the compiled C host, not UART data.
            self._send(b"!ENCODER\n")

        def advance_ms(self, amount):
            # Advance the fake HAL clock so firmware periodic STATUS executes.
            assert 0 < amount <= 200
            self._send(f"!TIME {amount}\n".encode("ascii"))

        def provide_ultrasonic_echo(self, echo_us):
            # Physical range samples after the mission handshake; pre-session
            # samples are intentionally discarded by the STM firmware.
            self._send(f"!ULTRASONIC {echo_us}\n".encode("ascii"))

        def close(self):
            if self.closed:
                return
            self.closed = True
            self.process.stdin.close()
            self.process.wait(timeout=2)
            self.reader.join(timeout=1)
            if self.process.returncode:
                raise AssertionError(self.process.stderr.read().decode(errors="replace"))
            self.process.stdout.close()
            self.process.stderr.close()

    def until(predicate, node, seconds=1.5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            node._tick()
            if predicate():
                return
            time.sleep(.005)
        assert predicate(), "compiled STM C response timed out"

    mission = StmBridgeNode("fake", uart_factory=HostSerial,
                            parameter_overrides=[
                                SimpleNamespace(name="uart_protocol", value="v2")])
    mission_peer = HostSerial.instances[-1]
    try:
        for _ in range(20):
            mission_peer.provide_ultrasonic_echo(2000)
        start = mission.services["stm/command"].callback(
            SimpleNamespace(command="START"), SimpleNamespace())
        assert start.success, start.error
        until(lambda: mission._start_acknowledged, mission)
        mission_peer.pulse_encoder()
        until(lambda: mission._capture_pending, mission)
        assert any("CAMERA_CAPTURE" in message.tokens
                   for message in mission.publishers["stm/events"].messages)
        capture = mission.services["stm/command"].callback(
            SimpleNamespace(command="CAPTURE_OK"), SimpleNamespace())
        assert capture.success and not mission._capture_pending, capture.error
    finally:
        mission.destroy_node()
    assert mission_peer.closed and mission_peer.process.returncode == 0

    actuator = ActuatorDebugNode("fake", uart_protocol="v2", uart_factory=HostSerial,
                                  exclusive_device=lambda _uart: None)
    until(lambda: len(HostSerial.instances) >= 2, actuator)
    actuator_peer = HostSerial.instances[-1]
    try:
        actuator_peer.advance_ms(100)
        until(lambda: actuator.session.feedback is not None, actuator)
        actuator.subscriptions["debug/encoder/command"].callback(
            SimpleNamespace(direction="forward", speed_percent=20.0))
        until(lambda: any(message.commanded_speed_percent == 20.0
                          and message.feedback_valid
                          for message in actuator.publishers["debug/encoder/state"].messages),
              actuator)
        until(lambda: any(b" SET 1 2000 0 " in wire for wire in actuator_peer.writes),
              actuator)
        actuator_peer.advance_ms(100)
        until(lambda: actuator.session.feedback is not None
              and actuator.session.feedback.direction == 1
              and actuator.session.feedback.percent_basis_points == 2000, actuator)
    finally:
        actuator.destroy_node()
    assert actuator_peer.closed and actuator_peer.process.returncode == 0

    sensor = SensorDebugNode(uart_protocol="v2", uart_factory=HostSerial,
                              exclusive_device=lambda _uart: None)
    until(lambda: len(HostSerial.instances) >= 3, sensor)
    sensor_peer = HostSerial.instances[-1]
    try:
        sensor_peer.advance_ms(100)
        until(lambda: sensor.session.feedback is not None, sensor)
        sensor.subscriptions["debug/led/set"].callback(SimpleNamespace(data="green"))
        until(lambda: bool(sensor.publishers["debug/led/state"].messages)
              and json.loads(sensor.publishers["debug/led/state"].messages[-1].data)
              ["requested_rgb"] == [0, 255, 0], sensor)
        assert any(b" LED breathe 0 255 0 " in wire for wire in sensor_peer.writes)
        state = json.loads(sensor.publishers["debug/ultrasonic/state"].messages[-1].data)
        assert state["feedback_valid"]
    finally:
        sensor.destroy_node()
    assert sensor_peer.closed and sensor_peer.process.returncode == 0


class RosCompiledStmPeerTests(unittest.TestCase):
    def test_ros_callbacks_round_trip_through_compiled_stm_app_and_fake_hal(self):
        result = subprocess.run(
            [sys.executable, __file__, "--child"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=40,
        )
        if result.returncode == 0 and "SKIP " in result.stdout:
            self.skipTest(result.stdout.split("SKIP ", 1)[1].strip())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--child" in sys.argv:
        try:
            child_checks()
        except unittest.SkipTest as skipped:  # STM 소스가 없는 Jetson
            print("SKIP", skipped)
    else:
        unittest.main()
