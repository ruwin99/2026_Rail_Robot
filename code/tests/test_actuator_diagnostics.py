"""Fault and hold diagnostics must explain stops without weakening deadman checks."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson_code'))
from node_runtime.actuator_debug import ActuatorDebugSession
from test_sensor_debug import Uart, SESSION


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.uart = Uart()
        self.session = ActuatorDebugSession(self.uart, 'motor_encoder', clock=lambda: self.now, session_id=SESSION)
        self.feed('READY', f'{self.session.capability} 1600 106 1200 3000 601 601')
        self.feed('STATE', '0 10000 10 20 0 0 0 0 0')
        self.session.tick()

    def feed(self, kind, fields):
        self.uart.pending.extend(f'DBG1_{kind} {SESSION} {fields}\n'.encode())

    def test_stall_frame_is_preserved_and_error_is_not_generic_timeout(self):
        self.session.command_encoder('forward', 100.0)
        self.session.tick()
        seq = self.session.sequence
        self.now += .1
        self.feed('STATE', f'{seq} 10100 10 20 1 10000 0 1587 1600')
        self.session.tick()
        self.now += .1
        self.feed('DIAG', f'{seq} 10200 DRIVE_STALL 4 0 0 10 20 0 0')
        self.feed('ERROR', 'DRIVE_STALL')
        self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertIn('DRIVE_STALL', self.session.error)
        self.assertIn('counts=10/20', self.session.error)
        self.assertIn('PWM=0/0', self.session.error)
        self.assertIn('last_nonzero_PWM=1587/1600', self.session.error)
        self.assertEqual(self.session.diagnostics['fault_code'], 4)

    def test_hold_is_not_fault_when_wall_clock_state_keeps_arriving(self):
        for offset in range(1, 11):
            self.now += .1
            seq, tick = self.session.sequence, 10000 + offset * 100
            self.feed('STATE', f'{seq} {tick} 10 20 0 0 0 0 0')
            self.feed('DIAG', f'{seq} {tick} ULTRASONIC_HOLD 0 1 1 10 20 0 0')
            self.session.tick()
        self.assertFalse(self.session.failed, self.session.error)
        self.assertTrue(self.session.diagnostics['held'])

    def test_hold_discards_prior_drive_intent_and_requires_fresh_input_after_clear(self):
        self.session.command_encoder('forward', 100.0)
        self.now += .05
        self.feed('STATE', '1 10050 10 20 0 0 0 0 0')
        self.feed('DIAG', '1 10050 ULTRASONIC_HOLD 0 1 1 10 20 0 0')
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b' 0 0 0\r\n'))
        self.assertFalse(self.session.command_encoder('forward', 100.0))
        self.now += .1
        seq = self.session.sequence
        self.feed('STATE', f'{seq} 10150 10 20 0 0 0 0 0')
        self.feed('DIAG', f'{seq} 10150 READY 0 0 0 10 20 0 0')
        self.session.tick()
        self.now += .11
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b' 0 0 0\r\n'))
        self.session.command_encoder('forward', 100.0)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b' 1 10000 0\r\n'))

    def test_diagnostics_alone_cannot_keep_motor_feedback_alive(self):
        for offset in range(1, 8):
            self.now += .1
            self.feed('DIAG', f'1 {10000 + offset * 100} READY 0 0 0 10 20 0 0')
            self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertIn('feedback timed out', self.session.error)

    def test_malformed_diagnostics_fail_closed(self):
        self.feed('DIAG', '1 10100 DRIVE_STALL 0 0 0 10 20 0 0')
        self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertIn('diagnostic', self.session.error.lower())

    def test_range_fault_reports_the_out_of_range_raw_count(self):
        self.feed('DIAG', '1 10100 DRIVE_ENCODER_RANGE 3 0 0 601 65535 0 0')
        self.feed('ERROR', 'DRIVE_ENCODER_RANGE')
        self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertIn('DRIVE_ENCODER_RANGE', self.session.error)
        self.assertIn('counts=601/65535', self.session.error)

    def test_wrong_session_and_replayed_diagnostics_cannot_replace_current(self):
        self.feed('DIAG', '1 10100 READY 0 0 0 10 20 0 0')
        self.session.tick()
        first = self.session.diagnostics
        self.uart.pending.extend(b'DBG1_DIAG other invalid\n')
        self.feed('DIAG', '1 10000 RUNNING 0 0 0 11 21 1 1')
        self.session.tick()
        self.assertIs(self.session.diagnostics, first)


if __name__ == '__main__':
    unittest.main()
