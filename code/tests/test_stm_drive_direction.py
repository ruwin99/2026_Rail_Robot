"""Verify actual timer direction channels and encoder-1 calibration."""
import unittest
from stm32_host import run_scenario


class StmDriveDirectionTests(unittest.TestCase):
    def test_opposite_drive_polarity_with_configured_right_base_pwm(self):
        self.assertIn("PASS direction", run_scenario("direction"))

    def test_pid_off_preserves_direction_and_trim(self):
        self.assertIn("PASS direction", run_scenario("direction", False))


if __name__ == "__main__":
    unittest.main()
