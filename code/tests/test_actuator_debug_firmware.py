"""DBG1 behavior through the real separately compiled firmware modules."""
import unittest
from stm32_host import run_scenario


class FirmwareTests(unittest.TestCase):
    def test_debug_protocol_with_default_pid_firmware(self):
        self._protocol(None)

    def test_debug_protocol_with_explicit_pid_enabled(self):
        self._protocol(True)

    def test_debug_protocol_with_pid_disabled(self):
        self._protocol(False)

    def _protocol(self, enabled):
        for case in ("pwm", "reverse", "expiry", "replay", "malformed", "framing"):
            with self.subTest(case=case, pid=enabled):
                self.assertIn("PASS debug_" + case, run_scenario("debug_" + case, enabled))


if __name__ == "__main__":
    unittest.main()
