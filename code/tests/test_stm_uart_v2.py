import unittest
from stm32_host import run_scenario


class UartV2Tests(unittest.TestCase):
    def test_shared_python_c_vectors(self):
        self.assertIn("PASS", run_scenario("uart_v2_shared_vectors"))

    def test_ping_interleaving_does_not_stale_exact_command_retry(self):
        self.assertIn("PASS", run_scenario("uart_v2_independent_cmd_ping_lanes"))

    def test_duplicate_hello_does_not_refresh_watchdog(self):
        self.assertIn("PASS", run_scenario("uart_v2_hello_retry_not_heartbeat"))

    def test_clean_idle_timeout_closes_session_and_allows_reconnect(self):
        self.assertIn("PASS", run_scenario("uart_v2_idle_close_reconnect"))

    def test_prepared_or_moving_timeout_still_latches_fault(self):
        self.assertIn("PASS", run_scenario("uart_v2_active_timeout_still_faults"))

    def test_debug_sessions_end_in_idle_but_link_errors_still_fault(self):
        self.assertIn("PASS", run_scenario("uart_v2_debug_sessions_end_idle"))

    def test_only_mission_capability_is_accepted(self):
        self.assertIn("PASS", run_scenario("uart_v2_bad_capability"))

    def test_bad_crc_is_not_acknowledged(self):
        self.assertIn("PASS", run_scenario("uart_v2_bad_crc"))

    def test_partial_line_is_buffered_and_oversize_fails_safe(self):
        self.assertIn("PASS", run_scenario("uart_v2_partial_and_oversize"))

    def test_faulted_mission_nacks_start(self):
        self.assertIn("PASS", run_scenario("uart_v2_fault_nack"))

    def test_estop_stops_drive_and_latches_fault_with_replayable_ack(self):
        self.assertIn("PASS", run_scenario("uart_v2_estop"))

    def test_malformed_command_is_cached_and_conflicting_retry_is_rejected(self):
        self.assertIn("PASS", run_scenario("uart_v2_malformed_command"))

    def test_crc(self):
        self.assertIn("PASS", run_scenario("uart_v2_crc"))

    def test_hello_requires_idle(self):
        self.assertIn("PASS", run_scenario("uart_v2_busy_hello"))

    def test_prepare_section_argument_is_validated_and_latched_at_start(self):
        for name in ("uart_v2_prepare_sections", "uart_v2_prepare_default_after_test"):
            with self.subTest(name=name):
                self.assertIn("PASS " + name, run_scenario(name))

    def test_mission_prepare_cancel_and_start(self):
        self.assertIn("PASS", run_scenario("uart_v2_mission_prepare"))

    def test_legacy_start_blocked_after_hello(self):
        self.assertIn("PASS", run_scenario("uart_v2_legacy_start"))


    def test_duplicate(self):
        self.assertIn("PASS", run_scenario("uart_v2_duplicate"))

    def test_capture_ack_and_stale_rejection(self):
        self.assertIn("PASS", run_scenario("uart_v2_capture"))

    def test_heartbeat_and_command_burst(self):
        self.assertIn("PASS", run_scenario("uart_v2_burst"))

    def test_late_start_after_liveness_timeout_cannot_move(self):
        self.assertIn("PASS", run_scenario("uart_v2_late_command"))

    def test_uart_fault_stops_drive(self):
        self.assertIn("PASS", run_scenario("uart_v2_uart_fault"))

    def test_uart_fault_mid_robot_update_latches_all_outputs_off(self):
        self.assertIn("PASS", run_scenario("uart_v2_fault_mid_robot_update"))

    def test_pending_uart_fault_after_tx_dequeue_discards_stale_queue(self):
        self.assertIn("PASS", run_scenario("uart_v2_tx_fault_race"))

    def test_active_rr2_rejects_legacy_accessory_test_and_dbg1_parsers(self):
        self.assertIn("PASS", run_scenario("uart_v2_legacy_isolation"))

    def test_rr2_mission_capability_maps_individual_accessory_commands(self):
        self.assertIn("PASS", run_scenario("uart_v2_mission_accessories"))

    def test_actuator_debug_capability_maps_set_and_stop(self):
        self.assertIn("PASS", run_scenario("uart_v2_debug_actuator"))

    def test_actuator_debug_capabilities_reject_other_actuator_groups(self):
        self.assertIn("PASS", run_scenario("uart_v2_debug_capability_gates"))

    def test_sensor_debug_capability_maps_led_and_periodic_status(self):
        self.assertIn("PASS", run_scenario("uart_v2_sensor_status"))

    def test_reliable_event_retries_same_sequence_until_ack(self):
        self.assertIn("PASS", run_scenario("uart_v2_event_retry_ack"))

    def test_obstacle_hold_and_clear_are_reliable_events(self):
        self.assertIn("PASS", run_scenario("uart_v2_obstacle_event"))

    def test_event_retry_exhaustion_and_queue_overflow_fail_safe(self):
        self.assertIn("PASS", run_scenario("uart_v2_event_fail_safe"))


    def test_heartbeat(self):
        self.assertIn("PASS", run_scenario("uart_v2_heartbeat"))


    def test_capture_timeout(self):
        self.assertIn("PASS", run_scenario("uart_v2_capture_timeout"))
