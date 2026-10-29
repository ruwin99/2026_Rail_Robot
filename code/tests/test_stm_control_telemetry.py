"""Actual STM control samples and opt-in, rate-limited UART telemetry."""
import unittest
from stm32_host import run_scenario


class ControlTelemetryTests(unittest.TestCase):
    def test_negotiation_and_legacy_wire_compatibility(self):
        self.assertIn('PASS control_negotiation', run_scenario('control_negotiation'))

    def test_real_samples_reverse_pi_enabled_and_lifecycle(self):
        self.assertIn('PASS control_samples', run_scenario('control_samples'))

    def test_generation_stop_does_not_report_calculation_as_applied(self):
        self.assertIn('PASS control_preempt', run_scenario('control_preempt'))

    def test_global_disabled_reports_unavailable(self):
        self.assertIn('PASS control_disabled', run_scenario('control_disabled', pid_enabled=False))

    def test_actual_elapsed_normalization_wrap_and_fault_invalidity(self):
        self.assertIn('PASS control_elapsed_wrap_fault', run_scenario('control_elapsed_wrap_fault'))
