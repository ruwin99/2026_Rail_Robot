from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

# The protocol tests use fake transports; no serial device or pyserial driver
# is needed on the Windows development host.
serial_stub = ModuleType("serial")
serial_stub.SerialException = OSError
serial_stub.Serial = object
sys.modules.setdefault("serial", serial_stub)

JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from node_runtime import control, uart
from node_runtime.mission import CaptureRequest, MissionStateMachine
from node_runtime.settings import (
    AUTOMATIC_RAIL_SECTION_TARGET,
    CAPTURE_OK_COMMAND,
    CAPTURE_SCAN_MODE,
    CAPTURE_TRIGGER,
    COMPLETE_MODE,
    DONE_TRIGGER,
    FRAME_WIDTH,
    REALTIME_CRACK_ROI_BOTTOM,
    REALTIME_CRACK_ROI_TOP,
    REALTIME_MODE,
    REALTIME_START_TRIGGER,
    RESCAN_DONE_MODE,
    RESCAN_DONE_TRIGGER,
    RESCAN_MODE,
    RESCAN_RETURN_START_TRIGGER,
    RESCAN_START_TRIGGER,
    RETURN_MODE,
    RETURN_START_TRIGGER,
)


class MissionNodeCoreTests(unittest.TestCase):
    def setUp(self):
        self.mission = MissionStateMachine()

    def start(self):
        self.mission.mark_start_sent()
        self.assertTrue(self.mission.acknowledge_started())

    def capture(self):
        request = self.mission.begin_capture()
        self.mission.accept_capture(request)
        self.mission.acknowledge_capture(request)
        return request

    def transition(self, trigger):
        transition = self.mission.prepare_transition(trigger)
        self.assertIsNotNone(transition)
        self.mission.complete_transition(transition)

    def reach_realtime(self):
        self.start()
        for _ in range(AUTOMATIC_RAIL_SECTION_TARGET):
            self.capture()
        self.transition(RETURN_START_TRIGGER)
        self.transition(REALTIME_START_TRIGGER)

    def test_every_initial_and_rescan_section_then_done(self):
        self.reach_realtime()
        self.assertEqual(self.mission.mode, REALTIME_MODE)
        self.transition(RESCAN_RETURN_START_TRIGGER)
        self.transition(RESCAN_START_TRIGGER)
        self.assertEqual(self.mission.mode, RESCAN_MODE)
        for sequence in range(1, AUTOMATIC_RAIL_SECTION_TARGET + 1):
            request = self.capture()
            self.assertEqual(request.phase_sequence, sequence)
        self.transition(RESCAN_DONE_TRIGGER)
        self.assertEqual(self.mission.mode, RESCAN_DONE_MODE)
        transition = self.mission.prepare_transition(DONE_TRIGGER)
        self.assertEqual(self.mission.mode, RESCAN_DONE_MODE)
        self.mission.complete_transition(transition)
        self.assertEqual(self.mission.mode, COMPLETE_MODE)
        self.assertEqual(self.mission.initial_rail_section_count, AUTOMATIC_RAIL_SECTION_TARGET)
        self.assertEqual(self.mission.rescan_rail_section_count, AUTOMATIC_RAIL_SECTION_TARGET)

    def test_capture_requires_started_acknowledgement(self):
        self.assertFalse(self.mission.acknowledge_started())
        self.mission.mark_start_sent()
        with self.assertRaisesRegex(RuntimeError, "before STARTED"):
            self.mission.begin_capture()
        self.assertTrue(self.mission.acknowledge_started())
        self.assertFalse(self.mission.acknowledge_started())

    def test_capture_counter_advances_only_after_acceptance_and_ack(self):
        self.start()
        request = self.mission.begin_capture()
        with self.assertRaisesRegex(RuntimeError, "accepted raw capture"):
            self.mission.acknowledge_capture(request)
        self.assertEqual(self.mission.initial_rail_section_count, 0)
        self.mission.accept_capture(request)
        self.assertEqual(self.mission.initial_rail_section_count, 0)
        self.mission.acknowledge_capture(request)
        self.assertEqual(self.mission.initial_rail_section_count, 1)
        self.assertIsNone(self.mission.pending_capture)
        with self.assertRaisesRegex(RuntimeError, "pending request"):
            self.mission.acknowledge_capture(request)

    def test_partial_dual_capture_queue_failure_never_acknowledges(self):
        self.start()
        request = self.mission.begin_capture()
        # One camera may already be saved/queued when the other fails. Only
        # acceptance of the full pair permits the coordinator to write ACK.
        self.mission.fail_capture(request, "TOP queue is full")
        with self.assertRaisesRegex(RuntimeError, "TOP queue is full"):
            self.mission.accept_capture(request)
        with self.assertRaises(RuntimeError):
            self.mission.acknowledge_capture(request)
        self.assertEqual(self.mission.initial_rail_section_count, 0)
        self.assertFalse(self.mission.capture_accepted)

    def test_short_ack_write_latches_failure_without_advancing(self):
        self.start()
        request = self.mission.begin_capture()
        self.mission.accept_capture(request)
        transport = Mock()
        transport.write.return_value = len(CAPTURE_OK_COMMAND) - 1
        try:
            uart.send_capture_ok(transport)
        except RuntimeError as exc:
            self.mission.fail_capture(request, str(exc))
        self.assertEqual(self.mission.initial_rail_section_count, 0)
        self.assertIn("complete CAPTURE_OK", self.mission.failure)
        with self.assertRaises(RuntimeError):
            self.mission.begin_capture()

    def test_pending_capture_blocks_next_capture_and_transition(self):
        self.start()
        request = self.mission.begin_capture()
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            self.mission.begin_capture()
        with self.assertRaisesRegex(RuntimeError, "already pending"):
            self.mission.prepare_transition(RETURN_START_TRIGGER)
        replacement = CaptureRequest(request.phase, request.phase_sequence)
        with self.assertRaisesRegex(RuntimeError, "pending request"):
            self.mission.accept_capture(replacement)

    def test_early_return_is_ignored_and_extra_capture_rejected(self):
        self.start()
        self.assertIsNone(self.mission.prepare_transition(RETURN_START_TRIGGER))
        self.assertIsNone(self.mission.prepare_transition(REALTIME_START_TRIGGER))
        for _ in range(AUTOMATIC_RAIL_SECTION_TARGET):
            self.capture()
        with self.assertRaisesRegex(RuntimeError, "extra STM"):
            self.mission.begin_capture()
        self.assertEqual(self.mission.mode, CAPTURE_SCAN_MODE)

    def test_transition_does_not_commit_before_external_preparation(self):
        self.start()
        for _ in range(AUTOMATIC_RAIL_SECTION_TARGET):
            self.capture()
        transition = self.mission.prepare_transition(RETURN_START_TRIGGER)
        self.assertEqual(self.mission.mode, CAPTURE_SCAN_MODE)
        self.mission.complete_transition(transition)
        realtime = self.mission.prepare_transition(REALTIME_START_TRIGGER)
        self.assertEqual(self.mission.mode, RETURN_MODE)
        # Leaving the token pending when drain/OFF fails cannot claim REALTIME.
        self.assertEqual(realtime.next_mode, REALTIME_MODE)

    def test_invalid_rescan_and_done_order_is_rejected(self):
        self.start()
        for trigger in (
            RESCAN_RETURN_START_TRIGGER, RESCAN_START_TRIGGER,
            RESCAN_DONE_TRIGGER, DONE_TRIGGER,
        ):
            with self.subTest(trigger=trigger), self.assertRaises(RuntimeError):
                self.mission.prepare_transition(trigger)

    def test_ambiguous_uart_batches_are_rejected_before_any_mutation(self):
        for messages in (
            [CAPTURE_TRIGGER, CAPTURE_TRIGGER],
            [REALTIME_START_TRIGGER, RESCAN_RETURN_START_TRIGGER],
        ):
            with self.subTest(messages=messages), self.assertRaises(RuntimeError):
                self.mission.validate_batch(messages)
        self.assertEqual(self.mission.initial_rail_section_count, 0)


class ControlAndUartNodeCoreTests(unittest.TestCase):
    def test_uart_retains_unterminated_protocol_line(self):
        transport = Mock(in_waiting=7)
        transport.read.return_value = b"STARTE"
        pending = bytearray()
        self.assertEqual(uart.read_uart_messages(transport, pending), [])
        transport.read.return_value = b"D\r\nCAMERA_CAPTURE\n"
        self.assertEqual(
            uart.read_uart_messages(transport, pending),
            ["STARTED", "CAMERA_CAPTURE"],
        )
        self.assertEqual(pending, bytearray())

    def test_cleaner_heartbeat_preserves_approved_pwm_and_state_on_failure(self):
        transport = Mock()
        transport.write.side_effect = len
        cleaner = uart.CleanerController(transport)
        with patch.object(uart.time, "monotonic", side_effect=[10.0, 10.1, 10.3]):
            cleaner.set_duty_percent(12.5)
            cleaner.set_duty_percent(12.5)
            cleaner.set_duty_percent(12.5)
        self.assertEqual(transport.write.call_count, 2)
        with self.assertRaises(ValueError):
            cleaner.set_duty_percent(50.0)
        transport.write.side_effect = OSError("disconnected")
        with self.assertRaises(RuntimeError):
            cleaner.force_off()
        self.assertFalse(cleaner.is_on)
        self.assertEqual(cleaner.duty_percent, 0.0)

    def test_safe_off_attempts_every_output_even_if_first_write_fails(self):
        outputs = [Mock() for _ in range(4)]
        outputs[0].force_off.side_effect = RuntimeError("UART disconnected")
        with self.assertRaisesRegex(RuntimeError, "UART disconnected"):
            control.force_cleaning_pairs_safe_off(
                (outputs[0], outputs[1]), (outputs[2], outputs[3]),
            )
        for output in outputs:
            output.force_off.assert_called_once_with()

    def test_hold_cannot_resurrect_expired_decision(self):
        hold = control.RealtimeActuatorDecisionHold(hold_seconds=1.2)
        decision = control.FrontControlDecision(12.5, True, False, 0.0)
        hold.accept(decision, now=10.0)
        self.assertIsNone(hold.begin_loss(now=11.21))
        self.assertFalse(hold.has_decision)

    def test_new_front_crack_stops_before_four_result_window_is_ready(self):
        mask = np.zeros(
            (REALTIME_CRACK_ROI_BOTTOM - REALTIME_CRACK_ROI_TOP, FRAME_WIDTH),
            dtype=np.uint8,
        )
        # 180행 ROI에서 0.05% 기준(약 115px)을 확실히 넘는 새 균열이다.
        mask[0, :200] = 255
        result = SimpleNamespace(
            status="ready", method=control.REALTIME_CRACK_DETECTOR_PREFIX,
            mask=mask, crack_ratio=200 / mask.size,
        )
        outcome = SimpleNamespace(display_crack_result=result)
        self.assertTrue(control.front_outcome_requires_immediate_off(outcome))
        history = control.FrontControlHistory()
        history.update_crack(result)
        self.assertFalse(history.ready)


if __name__ == "__main__":
    unittest.main()
