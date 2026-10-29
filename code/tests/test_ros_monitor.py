"""Read-only terminal contracts with ROS doubles; these are not DDS tests."""
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch


class FakeNode:
    def __init__(self, *args, **kwargs):
        self.timers = []
        self.subscribed = []

    def declare_parameter(self, name, value):
        return NS(value=value)

    def create_timer(self, period, callback):
        self.timers.append(NS(cancel=lambda: None))

    def get_topic_names_and_types(self):
        return [("/debug/side/dc/set", ["std_msgs/msg/Bool"]),
                ("/camera/side", ["rail_robot_interfaces/msg/CameraFrame"])]

    def get_node_names(self):
        return ["rail_monitor", "rail_actuator_debug"]

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscribed.append((topic, callback, qos))
        return object()


def load_monitor():
    modules = {name: ModuleType(name) for name in (
        "rclpy", "rclpy.node", "rclpy.qos", "rosidl_runtime_py", "rosidl_runtime_py.utilities")}
    modules["rclpy.node"].Node = FakeNode
    qos = modules["rclpy.qos"]
    qos.QoSProfile = lambda **kwargs: NS(**kwargs)
    qos.ReliabilityPolicy = NS(BEST_EFFORT="best_effort", RELIABLE="reliable")
    qos.DurabilityPolicy = NS(VOLATILE="volatile", TRANSIENT_LOCAL="transient_local")
    qos.HistoryPolicy = NS(KEEP_LAST="keep_last")
    modules["rosidl_runtime_py.utilities"].get_message = lambda name: NS
    path = Path(__file__).resolve().parents[1] / "jetson_code/node_runtime/ros/monitor_node.py"
    spec = importlib.util.spec_from_file_location("isolated_monitor", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.module = load_monitor()
        self.now = 10.0
        self.node = self.module.MonitorNode(clock=lambda: self.now, stream=io.StringIO())

    def encoder(self, **changes):
        fields = dict(feedback_valid=True, applied_direction='forward', applied_speed_percent=100.0,
                      left_raw_count=4, right_raw_count=594, left_counter_modulus=601,
                      right_counter_modulus=601, left_pwm_compare=1587, right_pwm_compare=1600,
                      error='', active_output_mask=0, sequence=12, mcu_millis=1200)
        fields.update(changes)
        self.node._receive('/debug/encoder/state', NS(**fields))

    def test_physical_right_and_left_encoders_visible_at_top_in_small_terminal(self):
        self.encoder()
        lines = self.node.lines(width=80, height=24)
        top = '\n'.join(lines[:8])
        self.assertIn('ENCODER LIVE', top)
        self.assertIn('RIGHT ENC1/TIM3 CNT=4/601 PWM=1587', top)
        self.assertIn('LEFT  ENC2/TIM4 CNT=594/601 PWM=1600', top)
        self.assertIn('MCU=1200ms', top)
        self.assertTrue(all(len(line) <= 80 for line in lines))

    def test_encoder_panel_marks_old_and_invalid_values_historical(self):
        self.encoder()
        self.now += .7
        self.assertIn('ENCODER OLD/HISTORICAL', '\n'.join(self.node.lines()))
        self.encoder(feedback_valid=False, error='DRIVE_STALL')
        self.assertIn('ENCODER INVALID/HISTORICAL', '\n'.join(self.node.lines()))

    def test_encoder_missing_and_wrap_are_not_reported_as_speed(self):
        self.assertIn('ENCODER WAIT', '\n'.join(self.node.lines()))
        self.encoder(left_raw_count=599, right_raw_count=2)
        self.now += .1
        self.encoder(left_raw_count=3, right_raw_count=597, mcu_millis=1300)
        top = '\n'.join(self.node.lines(width=120)[:8])
        self.assertIn('RIGHT ENC1/TIM3 CNT=3/601', top)
        self.assertIn('LEFT  ENC2/TIM4 CNT=597/601', top)
        self.assertNotIn('RPM=', top)
        self.assertNotIn('speed_ratio=', top)
        self.assertEqual(self.node._hz, 5.0)

    def test_malformed_encoder_sample_does_not_keep_live_panel(self):
        self.encoder()
        self.node._receive('/debug/encoder/state', NS())
        self.assertIn('ENCODER INVALID', '\n'.join(self.node.lines()))

    def test_only_known_lightweight_topics_subscribed_with_bounded_qos(self):
        self.assertEqual([entry[0] for entry in self.node.subscribed], ["/debug/side/dc/set"])
        qos = self.node.subscribed[0][2]
        self.assertEqual((qos.depth, qos.reliability, qos.durability), (1, "best_effort", "volatile"))
        self.node._discover()
        self.assertEqual(len(self.node.subscribed), 1)

    def test_sanitize_terminal_control_chars_and_bound_output(self):
        self.assertNotIn("\x1b", self.module.safe_text("\x1b[2J\n\tbad"))
        self.node._receive("/runtime/fault", NS(data="\x1b[2J\n" + "x" * 20000))
        lines = self.node.lines(width=60, height=8)
        self.assertLessEqual(len(lines), 8)
        self.assertTrue(all(len(line) <= 60 and "\x1b" not in line for line in lines))

    def test_wait_then_fresh_then_historical(self):
        self.assertIn("WAIT", "\n".join(self.node.lines()))
        self.node._receive("/debug/side/dc/set", NS(data=True))
        self.assertIn("REQUEST ON", "\n".join(self.node.lines()))
        self.now += 4
        line = next(line for line in self.node.lines() if "/debug/side/dc/set:" in line)
        self.assertIn("OLD", line)
        self.assertIn("4.0s", line)

    def test_motor_diagnostic_row_distinguishes_fault_and_applied_pwm(self):
        import json
        self.node._receive('/debug/encoder/diagnostics', NS(data=json.dumps(dict(
            reason='DRIVE_STALL', fault_code=4, held=False, left_raw_count=12,
            right_raw_count=23, left_pwm_compare=0, right_pwm_compare=0, historical=False))))
        text = '\n'.join(self.node.lines(width=240, height=100))
        self.assertIn('DRIVE_STALL', text)
        self.assertIn('CNT=12/23 PWM=0/0', text)

    def test_top_panel_shows_latest_ten_ms_pi_controller_sample(self):
        import json
        self.encoder()
        self.node._receive('/debug/encoder/diagnostics', NS(data=json.dumps(dict(
            reason='RUNNING', fault_code=0, held=False, left_raw_count=12,
            right_raw_count=23, left_pwm_compare=1587, right_pwm_compare=1600,
            historical=False, control_loop=dict(
                sample_sequence=41, sample_period_ms=10, sample_age_ms=8.0,
                direction='forward', pi_enabled=True, valid=True, applied=True,
                physical_right_delta=12, physical_left_delta=11, speed_error=1,
                integral_q8=384, integral_pwm=1.5, correction_pwm=9)))))
        top = '\n'.join(self.node.lines(width=120, height=30)[:10])
        self.assertIn('CONTROL LIVE PI=ON dt=10ms sample=41 age=8ms', top)
        self.assertIn('DELTA(dt=10ms) RIGHT=12 LEFT=11 error=1', top)
        self.assertIn('integral=1.5 PWM correction=9 PWM', top)

    def test_control_panel_marks_invalid_sample_historical(self):
        import json
        self.node._receive('/debug/encoder/diagnostics', NS(data=json.dumps(dict(
            reason='READY', historical=False, control_loop=dict(
                sample_sequence=42, sample_period_ms=10, sample_age_ms=5.0,
                direction='stop', pi_enabled=False, valid=False, applied=False,
                physical_right_delta=0, physical_left_delta=0, speed_error=0,
                integral_q8=0, integral_pwm=0.0, correction_pwm=0)))))
        self.assertIn('CONTROL INVALID/HISTORICAL PI=OFF', '\n'.join(self.node.lines(width=120)))

    def test_control_panel_does_not_show_closed_session_sample_as_live(self):
        import json
        self.node._receive('/debug/encoder/diagnostics', NS(data=json.dumps(dict(
            reason='RUNNING', historical=True, control_loop=dict(
                sample_sequence=43, sample_period_ms=10, sample_age_ms=5.0,
                direction='forward', pi_enabled=True, valid=True, applied=True,
                physical_right_delta=10, physical_left_delta=10, speed_error=0,
                integral_q8=0, integral_pwm=0.0, correction_pwm=0)))))
        self.assertIn('CONTROL OLD/HISTORICAL PI=ON', '\n'.join(self.node.lines(width=120)))

    def test_delta_label_uses_actual_delayed_sample_period(self):
        import json
        self.node._receive('/debug/encoder/diagnostics', NS(data=json.dumps(dict(
            reason='RUNNING', historical=False, control_loop=dict(
                sample_sequence=44, sample_period_ms=15, sample_age_ms=3.0,
                direction='forward', pi_enabled=True, valid=True, applied=True,
                physical_right_delta=20, physical_left_delta=10, speed_error=6,
                integral_q8=1152, integral_pwm=4.5, correction_pwm=12)))))
        text = '\n'.join(self.node.lines(width=120))
        self.assertIn('DELTA(dt=15ms) RIGHT=20 LEFT=10 error=6', text)
        self.assertNotIn('10ms DELTA', text)

    def test_ultrasonic_row_requests_last_observed_state_for_late_monitor(self):
        self.node.get_topic_names_and_types = lambda: [("/tts/obstacle_state", ["std_msgs/msg/Bool"])]
        self.node._discover()
        qos = next(item[2] for item in self.node.subscribed if item[0] == '/tts/obstacle_state')
        self.assertEqual(qos.durability, 'transient_local')
        self.node._receive('/tts/obstacle_state', NS(data=True))
        self.assertIn('ULTRASONIC HOLD', '\n'.join(self.node.lines(height=100)))

    def test_interactive_screen_replaces_frame_and_restores_terminal(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        self.node._stream = stream
        with patch.object(self.module.shutil, 'get_terminal_size', return_value=NS(columns=80, lines=24)):
            self.node._render()
            self.node._render()
        output = stream.getvalue()
        self.assertEqual(output.count('\x1b[?1049h'), 1)
        self.assertEqual(output.count('\x1b[H\x1b[2J'), 2)
        self.assertFalse(output.endswith('\n'))
        self.node.prepare_shutdown()
        self.assertTrue(stream.getvalue().endswith('\x1b[?25h\x1b[?1049l'))

    def test_piped_stdout_uses_controlling_terminal(self):
        stream = io.StringIO()
        with patch.object(self.module.sys, 'stdout', io.StringIO()), patch('builtins.open', return_value=stream):
            node = self.module.MonitorNode()
        self.assertIs(node._stream, stream)
        self.assertTrue(node._owns_stream)
        node.prepare_shutdown()
        self.assertTrue(stream.closed)

    def test_requested_motion_does_not_claim_applied_state(self):
        self.node._receive("/debug/encoder/command", NS(direction="forward", speed_percent=30.0))
        output = "\n".join(self.node.lines(width=240))
        self.assertIn("REQUEST forward 30.0%", output)
        self.assertNotIn("applied=forward", output)
        self.node._receive("/debug/encoder/state", NS(
            feedback_valid=True, applied_direction="stop", applied_speed_percent=0.0,
            left_raw_count=12, right_raw_count=18, error="", active_output_mask=9,
            left_pwm_compare=0, right_pwm_compare=0))
        output = "\n".join(self.node.lines(width=240))
        self.assertIn("applied=stop", output)
        self.assertIn("DC(F/S)=1/0 pump(F/S)=0/1", output)

    def test_bad_json_and_runtime_fault_do_not_stop_monitor(self):
        self.node._receive("/rail_robot/webrtc/status", NS(data="{broken"))
        self.node._receive("/runtime/fault", NS(data="stm fault"))
        self.assertIn("INVALID TELEMETRY", "\n".join(self.node.lines(width=240)))
        self.assertFalse(self.node.shutdown_requested())
        self.node._render()
        self.assertIn("stm fault", self.node._stream.getvalue())

    def test_dynamic_model_names_cannot_grow_storage_forever(self):
        for index in range(100):
            self.node._receive("/debug/vision/timing", NS(camera_role="side", model=str(index), processing_ms=12.0))
        self.assertLessEqual(len(self.node._rows), 48)


if __name__ == "__main__":
    unittest.main()
