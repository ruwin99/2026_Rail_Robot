"""Public Robot API integration checks using linked production modules."""
import unittest
from stm32_host import run_scenario


class StmModuleBootTests(unittest.TestCase):
    def test_robot_init_starts_with_all_pwm_zero(self):
        self.assertIn("PASS boot", run_scenario("boot"))

    def test_fatal_stop_before_peripheral_initialization(self):
        self.assertIn("PASS fatal_early", run_scenario("fatal_early"))

    def test_fatal_stop_after_only_accessory_initialization(self):
        self.assertIn("PASS fatal_partial", run_scenario("fatal_partial"))

    def test_fatal_stop_with_all_outputs_energized(self):
        self.assertIn("PASS fatal_initialized", run_scenario("fatal_initialized"))
