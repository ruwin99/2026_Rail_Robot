"""Debug diagnostics from real C firmware, without a physical board."""
import unittest
from stm32_host import run_scenario


class DebugDiagnosticTests(unittest.TestCase):
    def test_stall_report_precedes_host_timeout_and_explains_busy(self):
        self.assertIn('PASS debug_diag_stall', run_scenario('debug_diag_stall'))

    def test_held_motor_commands_stop_outputs_but_keep_feedback_alive(self):
        self.assertIn('PASS debug_diag_hold', run_scenario('debug_diag_hold'))

    def test_fault_survives_close_and_deactivation_without_isr_uart(self):
        self.assertIn('PASS debug_diag_pending', run_scenario('debug_diag_pending'))

    def test_hold_heartbeat_with_pid_disabled(self):
        self.assertIn('PASS debug_diag_hold', run_scenario('debug_diag_hold', pid_enabled=False))

    def test_fault_reasons_include_actual_out_of_range_counts(self):
        self.assertIn('PASS debug_diag_faults', run_scenario('debug_diag_faults'))
