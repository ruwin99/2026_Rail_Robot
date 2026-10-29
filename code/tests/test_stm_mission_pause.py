"""Mission lifecycle/pauses exercised through actual Robot and UART APIs."""
import unittest
from stm32_host import run_scenario


class MissionPauseTests(unittest.TestCase):
    def test_four_sections_all_five_pauses_capture_return_realtime_and_rescan(self):
        self.assertIn("PASS mission_full", run_scenario("mission_full"))

    def test_coast_carries_within_phase_and_resets_at_each_phase_start(self):
        # 4구간 뒤 단계가 바뀌면 구간 수 0, CNT 0/600에서 다시 센다. 관성 이월은 같은 단계 안에서만 쓴다.
        self.assertIn("PASS mission_coast_phase_reset", run_scenario("mission_coast_phase_reset"))

    def test_two_section_test_mode_runs_every_phase_with_two_sections(self):
        self.assertIn("PASS mission_two_sections", run_scenario("mission_two_sections"))

    def test_transition_discards_on_start_and_duplicate_ack(self):
        self.assertIn("PASS mission_pending", run_scenario("mission_pending"))

    def test_faults_cancel_transition_and_preserve_irq_mask(self):
        self.assertIn("PASS mission_faults", run_scenario("mission_faults"))

    def test_legacy_text_link_watchdog_stops_silent_drive(self):
        self.assertIn("PASS mission_legacy_link", run_scenario("mission_legacy_link"))

    def test_pause_deadline_survives_tick_wrap(self):
        self.assertIn("PASS mission_wrap", run_scenario("mission_wrap"))

    def test_lifecycle_with_pid_disabled_preserves_policy(self):
        self.assertIn("PASS mission_full", run_scenario("mission_full", False))


if __name__ == "__main__":
    unittest.main()
