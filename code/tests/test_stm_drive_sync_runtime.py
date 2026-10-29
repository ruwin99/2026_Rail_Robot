"""Execute linked production PID scheduling with fake HAL/DWT and preemption."""
import unittest
from stm32_host import run_scenario, build_host


class StmDriveSyncRuntimeTests(unittest.TestCase):
    def scenario(self, name):
        self.assertIn("PASS " + name, run_scenario(name))

    def test_default_pid_applies_requested_forward_feedback_polarity(self):
        self.scenario("normal")

    def test_current_header_and_extended_target_range_start_without_bad_config(self):
        self.assertIn("PASS target_start", run_scenario("target_start",forward_target_bp=None))
        for target in (9700,9800,10100):
            with self.subTest(target=target):
                self.assertIn(f"TARGET {target} ACTIVE",run_scenario("target_start",forward_target_bp=target))

    def test_target_outside_shared_bounds_cannot_compile(self):
        for target in (9299,10501):
            with self.subTest(target=target), self.assertRaisesRegex(AssertionError,"MIN_BP..MAX_BP"):
                build_host(forward_target_bp=target)

    def test_explicit_reverse_left_target_bias_and_direction_reset(self):
        self.assertIn("PASS reverse_target_bias",run_scenario("reverse_target_bias",
                      forward_target_bp=9800,reverse_target_bp=9900))

    def test_reverse_target_compile_and_runtime_bounds_match(self):
        for target in (9700,9900,10100):
            with self.subTest(target=target):
                self.assertIn("PASS reverse_target_start",run_scenario("reverse_target_start",reverse_target_bp=target))
        for target in (9699,10101):
            with self.subTest(target=target),self.assertRaisesRegex(AssertionError,"MIN_BP..MAX_BP"):
                build_host(reverse_target_bp=target)

    def test_default_fractional_forward_target_applies_and_resets_on_reverse(self):
        self.scenario("forward_target_bias")

    def test_equal_physical_speed_preserves_feedforward(self):
        self.scenario("equal")

    def test_reverse_encoder_feedback(self):
        self.scenario("reverse")

    def test_reverse_fixed_pwm_and_forward_pi_restore(self):
        self.assertIn("PASS reverse_fixed_restore", run_scenario("reverse_fixed_restore", reverse_pid_enabled=False))

    def test_reverse_pi_enabled_by_default_applies_inverted_coupling_and_resets(self):
        self.scenario("reverse_pi_enabled")

    def test_reverse_protections_remain_active_with_pi_correction(self):
        for name in ("reverse_stall", "reverse_wrong_direction", "reverse_range",
                     "reverse_late", "reverse_main_timeout", "reverse_stop"):
            with self.subTest(name=name):
                self.scenario(name)

    def test_missing_second_encoder_fault_latches_and_rejects_late_on(self):
        self.scenario("missing")

    def test_restart_rebases_encoder_counters(self):
        self.scenario("restart")

    def test_stop_preempts_pid_calculation(self):
        self.scenario("stop")

    def test_delayed_commit_latches_fault(self):
        self.scenario("delay")

    def test_exact_twenty_ms_commit_boundary(self):
        self.scenario("boundary")

    def test_main_freshness_at_entry_and_commit(self):
        self.scenario("freshness")

    def test_same_target_preserves_output_and_sample_cadence(self):
        self.scenario("heartbeat")

    def test_disabled_pid_tick_is_noop(self):
        self.assertIn("PASS disabled", run_scenario("disabled", False))

    def test_measured_forward_sign_reverse_wrap_and_actual_reverse_fault(self):
        self.scenario("measured_encoder_sign")


if __name__ == "__main__":
    unittest.main()
