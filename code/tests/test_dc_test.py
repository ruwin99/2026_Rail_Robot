from __future__ import annotations

import io
import sys
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "jetson_code"))

import dc_test


class FakeUart:
    def __init__(self, ready: bool = True) -> None:
        self.pending = bytearray(b"ACTUATOR_TEST_READY\r\n" if ready else b"")
        self.writes: list[bytes] = []
        self._lock = threading.Lock()

    @property
    def in_waiting(self) -> int:
        return len(self.pending)

    def read(self, size: int) -> bytes:
        data = bytes(self.pending[:size])
        del self.pending[:size]
        return data

    def reset_input_buffer(self) -> None:
        # READY is injected after START by write(), matching the real handshake.
        self.pending.clear()

    def write(self, command: bytes) -> int:
        with self._lock:
            self.writes.append(command)
            if command == dc_test.ACTUATOR_TEST_START:
                self.pending.extend(b"ACTUATOR_TEST_READY\r\n")
        return len(command)


class DcTestTests(unittest.TestCase):
    def test_handshake_is_required_before_session_starts(self) -> None:
        uart = FakeUart()
        session = dc_test.ActuatorTestSession(uart, heartbeat_seconds=0.01)
        with redirect_stdout(io.StringIO()):
            session.start()
            session.close()

        self.assertEqual(uart.writes[0], dc_test.ACTUATOR_TEST_START)
        self.assertEqual(
            uart.writes[-5:],
            [
                dc_test.FRONT_CLEANER_OFF,
                dc_test.FRONT_PUMP_OFF,
                dc_test.SIDE_CLEANER_OFF,
                dc_test.SIDE_PUMP_OFF,
                dc_test.ACTUATOR_TEST_STOP,
            ],
        )

    def test_front_dc_keeps_only_front_cleaner_alive(self) -> None:
        uart = FakeUart()
        session = dc_test.ActuatorTestSession(uart, heartbeat_seconds=0.01)
        with redirect_stdout(io.StringIO()):
            session.start()
            session.select("front_dc")
            deadline = time.monotonic() + 0.2
            while (
                uart.writes.count(dc_test.FRONT_CLEANER_PWM_12_5) < 3
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            session.close()

        selection = uart.writes.index(dc_test.FRONT_CLEANER_OFF)
        self.assertEqual(
            uart.writes[selection : selection + 5],
            [*dc_test.ALL_OUTPUTS_OFF, dc_test.FRONT_CLEANER_PWM_12_5],
        )
        heartbeat_writes = uart.writes[selection + 5 : -5]
        self.assertGreaterEqual(
            heartbeat_writes.count(dc_test.FRONT_CLEANER_PWM_12_5), 2
        )
        self.assertNotIn(dc_test.FRONT_PUMP_ON, uart.writes)
        self.assertNotIn(dc_test.SIDE_CLEANER_PWM_12_5, uart.writes)
        self.assertNotIn(dc_test.SIDE_PUMP_ON, uart.writes)

    def test_front_pump_keeps_only_front_pump_alive(self) -> None:
        uart = FakeUart()
        session = dc_test.ActuatorTestSession(uart, heartbeat_seconds=0.01)
        with redirect_stdout(io.StringIO()):
            session.start()
            session.select("front_pump")
            deadline = time.monotonic() + 0.2
            while (
                uart.writes.count(dc_test.FRONT_PUMP_ON) < 3
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            session.close()

        selection = uart.writes.index(dc_test.FRONT_CLEANER_OFF)
        self.assertEqual(
            uart.writes[selection : selection + 5],
            [*dc_test.ALL_OUTPUTS_OFF, dc_test.FRONT_PUMP_ON],
        )
        heartbeat_writes = uart.writes[selection + 5 : -5]
        self.assertGreaterEqual(heartbeat_writes.count(dc_test.FRONT_PUMP_ON), 2)
        self.assertNotIn(dc_test.FRONT_CLEANER_PWM_12_5, uart.writes)
        self.assertNotIn(dc_test.SIDE_CLEANER_PWM_12_5, uart.writes)
        self.assertNotIn(dc_test.SIDE_PUMP_ON, uart.writes)

    def test_side_dc_keeps_only_cleaner_alive(self) -> None:
        uart = FakeUart()
        session = dc_test.ActuatorTestSession(uart, heartbeat_seconds=0.01)
        with redirect_stdout(io.StringIO()):
            session.start()
            session.select("side_dc")
            deadline = time.monotonic() + 0.2
            while (
                uart.writes.count(dc_test.SIDE_CLEANER_PWM_12_5) < 3
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            session.close()

        selection = uart.writes.index(dc_test.FRONT_CLEANER_OFF)
        self.assertEqual(
            uart.writes[selection : selection + 5],
            [*dc_test.ALL_OUTPUTS_OFF, dc_test.SIDE_CLEANER_PWM_12_5],
        )
        heartbeat_writes = uart.writes[selection + 5 : -5]
        self.assertGreaterEqual(heartbeat_writes.count(dc_test.SIDE_CLEANER_PWM_12_5), 2)
        self.assertNotIn(dc_test.FRONT_CLEANER_PWM_12_5, uart.writes)
        self.assertNotIn(dc_test.FRONT_PUMP_ON, uart.writes)
        self.assertNotIn(dc_test.SIDE_PUMP_ON, uart.writes)

    def test_side_pump_keeps_only_pump_alive(self) -> None:
        uart = FakeUart()
        session = dc_test.ActuatorTestSession(uart, heartbeat_seconds=0.01)
        with redirect_stdout(io.StringIO()):
            session.start()
            session.select("side_pump")
            deadline = time.monotonic() + 0.2
            while (
                uart.writes.count(dc_test.SIDE_PUMP_ON) < 3
                and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            session.close()

        selection = uart.writes.index(dc_test.FRONT_CLEANER_OFF)
        self.assertEqual(
            uart.writes[selection : selection + 5],
            [*dc_test.ALL_OUTPUTS_OFF, dc_test.SIDE_PUMP_ON],
        )
        heartbeat_writes = uart.writes[selection + 5 : -5]
        self.assertGreaterEqual(heartbeat_writes.count(dc_test.SIDE_PUMP_ON), 2)
        self.assertNotIn(dc_test.FRONT_CLEANER_PWM_12_5, uart.writes)
        self.assertNotIn(dc_test.FRONT_PUMP_ON, uart.writes)
        self.assertNotIn(dc_test.SIDE_CLEANER_PWM_12_5, uart.writes)

    def test_stop_turns_all_four_outputs_off(self) -> None:
        uart = FakeUart()
        session = dc_test.ActuatorTestSession(uart, heartbeat_seconds=0.01)
        with redirect_stdout(io.StringIO()):
            session.start()
            session.select("side_dc")
            session.select("stop")
            writes_after_stop = len(uart.writes)
            time.sleep(0.03)
            session.close()

        self.assertEqual(
            uart.writes[writes_after_stop - 4 : writes_after_stop],
            list(dc_test.ALL_OUTPUTS_OFF),
        )
        self.assertEqual(len(uart.writes), writes_after_stop + 5)

    def test_script_uses_project_virtual_environment(self) -> None:
        source = (CODE_ROOT / "scripts" / "dc_test.sh").read_text(encoding="utf-8")
        self.assertIn('RAIL_ROBOT_VENV', source)
        self.assertIn('jetson_code/dc_test.py', source)
        self.assertIn('import serial', source)


if __name__ == "__main__":
    unittest.main()
