"""Compile the actual firmware modules; no board access or flashing."""
import unittest
from stm32_host import run_scenario


class SensorDebugTests(unittest.TestCase):
    def test_range_and_no_echo(self):
        self.assertIn("PASS sensor_debug_range", run_scenario("sensor_debug_range"))

    def test_led_and_obstacle_priority(self):
        self.assertIn("PASS sensor_debug_led", run_scenario("sensor_debug_led"))

    def test_led_operational_effects_repeat(self):
        self.assertIn("PASS sensor_debug_effects", run_scenario("sensor_debug_effects"))

    def test_sensor_debug_ignores_near_readings_for_led(self):
        self.assertIn("PASS sensor_debug_prestart", run_scenario("sensor_debug_prestart"))

    def test_timeout_while_operational_clock_is_frozen(self):
        self.assertIn("PASS sensor_debug_timeout", run_scenario("sensor_debug_timeout"))

    def test_reject_motors_malformed_and_uart_error(self):
        self.assertIn("PASS sensor_debug_reject", run_scenario("sensor_debug_reject"))

    def test_active_low_led(self):
        self.assertIn("PASS sensor_debug_led", run_scenario("sensor_debug_led", rgb_active_low=True))
        self.assertIn("PASS sensor_debug_effects", run_scenario("sensor_debug_effects", rgb_active_low=True))

    def test_no_pid_sensor_session(self):
        self.assertIn("PASS sensor_debug_range", run_scenario("sensor_debug_range", pid_enabled=False))
