"""Role protocol/output tests compile real firmware, without source-string checks."""
import unittest
from stm32_host import run_scenario


class StmFrontSideActuatorProtocolTests(unittest.TestCase):
    def scenario(self, name):
        self.assertIn("PASS " + name, run_scenario(name))

    def test_role_mapping_calibration_gpio_and_independent_off(self):
        self.scenario("actuator_roles")

    def test_explicit_ready_handshake_and_fail_safe_test_stop(self):
        self.scenario("actuator_handshake")

    def test_legacy_aliases_do_not_energize_outputs(self):
        self.scenario("actuator_legacy")

    def test_actuator_state_gate_and_encoder_boundary(self):
        self.scenario("actuator_gate")

    def test_role_isolated_off_priority_latest_pwm_and_irq_mask(self):
        self.scenario("actuator_off")

    def test_four_independent_watchdogs_refresh_and_tick_wrap(self):
        self.scenario("actuator_watchdogs")

    def test_uart_error_bad_rx_and_rearm_failure_stop_four_channels(self):
        self.scenario("actuator_uart")


if __name__ == "__main__":
    unittest.main()
