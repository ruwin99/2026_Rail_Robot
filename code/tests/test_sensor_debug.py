"""Sensor-only UART contract: no ROS installation or hardware required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson_code'))
from node_runtime.sensor_debug import SensorDebugSession

SESSION = '0123456789abcdef0123456789abcdef'


class Uart:
    def __init__(self):
        self.pending = bytearray()
        self.writes = []

    @property
    def in_waiting(self):
        return len(self.pending)

    def read(self, size):
        result = bytes(self.pending[:size])
        del self.pending[:size]
        return result

    def write(self, data):
        self.writes.append(data)
        return len(data)


class SensorSessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.uart = Uart()
        self.session = SensorDebugSession(self.uart, clock=lambda: self.now, session_id=SESSION)

    def feed(self, line):
        self.uart.pending.extend((line + '\r\n').encode())

    def state(self, tick=10000, seq=0, details='150 1 1 1 1 0 0 0 ECHO'):
        self.feed(f'DBG1_SENSOR {SESSION} {seq} {tick} {details}')

    def ready(self):
        self.feed(f'DBG1_READY {SESSION} 8 1600 106 1200 3000 601 601')
        self.state()
        self.session.tick()

    def test_handshake_and_heartbeat_never_enable_outputs(self):
        self.assertEqual(self.uart.writes[0], f'DBG1_HELLO {SESSION} 8\r\n'.encode())
        self.ready()
        self.assertTrue(self.session.ready)
        self.assertTrue(self.uart.writes[-1].endswith(b' 1 0 0 0\r\n'))
        self.assertEqual(self.session.feedback['distance_mm'], 150)
        self.assertTrue(self.session.feedback['detected'])

    def test_sensor_accepts_explicit_bp_and_legacy_permille_ready(self):
        for trim_and_unit in (('9921', ' BP'), ('992', '')):
            self.setUp()
            trim, unit = trim_and_unit
            self.feed(f'DBG1_READY {SESSION} 8 1600 {trim} 1200 3000 601 601{unit}')
            self.state()
            self.session.tick()
            self.assertTrue(self.session.ready, self.session.error)
            self.assertFalse(self.session.failed)

    def test_sensor_rejects_malformed_ready_metadata_and_units(self):
        for tail in ('9920 1200 3000 601 601', '9920 1200 3000 601 601 XX',
                     '9920 1200 3000 601 601 BP extra', '0 1200 3000 601 601 BP',
                     '20001 1200 3000 601 601 BP', '9920 BAD 3000 601 601 BP'):
            self.setUp()
            self.feed(f'DBG1_READY {SESSION} 8 1600 {tail}')
            self.session.tick()
            self.assertTrue(self.session.failed, tail)
            self.assertFalse(any(b'DBG1_SET' in frame for frame in self.uart.writes))

    def test_latest_led_command_is_bounded_and_sent_instead_of_heartbeat(self):
        self.ready()
        for _ in range(100):
            self.session.command_led('red')
        self.session.command_led('12,34,56')
        before = len(self.uart.writes)
        self.now += 0.21
        self.session.tick()
        self.assertEqual(len(self.uart.writes), before + 1)
        self.assertTrue(self.uart.writes[-1].endswith(b' 2 manual 12 34 56\r\n'))

    def test_primary_colors_request_repeating_operational_effects(self):
        self.ready()
        for command, expected in (('blue', b'breathe 0 0 255'),
                                  ('green', b'breathe 0 255 0'),
                                  ('red', b'blink 255 0 0')):
            self.session.command_led(command)
            self.now += 0.21
            self.state(tick=int(self.now * 1000), seq=self.session.sequence)
            self.session.tick()
            self.assertTrue(self.uart.writes[-1].endswith(expected + b'\r\n'))
        self.session.command_led('off')
        self.now += 0.21
        self.state(tick=int(self.now * 1000), seq=self.session.sequence)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b'manual 0 0 0\r\n'))

    def test_invalid_commands_do_not_queue_or_write(self):
        self.ready()
        for command in ('256,0,0', '-1,0,0', '1,2', 'START', 'red\nSTART', ''):
            with self.assertRaises(ValueError):
                self.session.command_led(command)

    def test_no_echo_is_invalid_not_zero_distance_measurement(self):
        self.ready()
        self.state(10200, 1, '0 0 0 0 1 0 0 0 NO_ECHO')
        self.session.tick()
        self.assertFalse(self.session.feedback['valid'])
        self.assertIsNone(self.session.feedback['distance_mm'])

    def test_stale_feedback_does_not_extend_deadline(self):
        self.ready()
        self.now += 0.61
        self.state()
        self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertIn('feedback', self.session.error)
        self.assertTrue(self.uart.writes[-1].startswith(b'DBG1_CLOSE'))

    def test_old_firmware_times_out_without_commands(self):
        self.now += 2.1
        self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertIn('firmware', self.session.error)
        self.assertFalse(any(b'DBG1_SET' in w for w in self.uart.writes))

    def test_wrong_session_ignored_and_malformed_own_feedback_fails(self):
        self.feed('DBG1_SENSOR other bad')
        self.session.tick()
        self.assertFalse(self.session.failed)
        self.ready()
        self.state(10200, 1, '150 2 0 0 1 0 0 0 ECHO')
        self.session.tick()
        self.assertTrue(self.session.failed)

    def test_shutdown_single_close_and_no_later_commands(self):
        self.ready()
        self.session.close()
        before = len(self.uart.writes)
        self.session.close()
        self.session.tick()
        self.assertEqual(len(self.uart.writes), before)
        with self.assertRaises(ValueError):
            self.session.command_led('auto')


if __name__ == '__main__':
    unittest.main()
