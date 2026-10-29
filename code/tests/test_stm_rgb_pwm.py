import unittest
from stm32_host import CORE, run_scenario

class RGBPWMTests(unittest.TestCase):
    def test_real_rgb_pwm(self):
        self.assertTrue((CORE / 'Src/rgb_pwm.c').exists(), 'RGB PWM production module missing')
        self.assertIn('PASS rgb_pwm', run_scenario('rgb_pwm'))

    def test_active_low(self):
        self.assertTrue((CORE / 'Src/rgb_pwm.c').exists(), 'RGB PWM production module missing')
        self.assertIn('PASS rgb_pwm', run_scenario('rgb_pwm', rgb_active_low=True))

    def test_mission_status(self):
        self.assertIn('PASS rgb_status', run_scenario('rgb_status'))

    def test_mission_status_active_low(self):
        self.assertIn('PASS rgb_status', run_scenario('rgb_status', rgb_active_low=True))

    def test_rgb_stays_off_until_mission_session_except_for_latched_fault(self):
        self.assertIn('PASS rgb_mission_gate', run_scenario('rgb_mission_gate'))

    def test_connected_idle_mission_session_shows_blue_breathing(self):
        self.assertIn('PASS rgb_idle_session', run_scenario('rgb_idle_session'))

    def test_full_mission_colours(self):
        self.assertIn('PASS rgb_full_mission', run_scenario('rgb_full_mission'))

    def test_full_mission_colours_active_low(self):
        self.assertIn('PASS rgb_full_mission', run_scenario('rgb_full_mission', rgb_active_low=True))
