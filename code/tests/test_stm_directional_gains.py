"""Directional tuning through actual firmware scheduling and PWM output."""
import unittest
from stm32_host import run_scenario


class DirectionalGainTests(unittest.TestCase):
    def test_reverse_half_percent_target_preserves_fractional_bias(self):
        self.assertIn('PASS reverse_half_percent_target', run_scenario(
            'reverse_half_percent_target', forward_target_bp=None, reverse_target_bp=9950))

    def test_forward_reverse_gains_caps_and_transition_reset(self):
        self.assertIn('PASS directional_gains', run_scenario(
            'directional_gains', forward_target_bp=10000, reverse_target_bp=10000))

    def test_reverse_disabled_preserves_protection_and_fixed_output(self):
        self.assertIn('PASS directional_disabled', run_scenario(
            'directional_disabled', reverse_pid_enabled=False,
            forward_target_bp=10000, reverse_target_bp=10000))
