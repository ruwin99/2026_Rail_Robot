"""Pure protocol and actual ROS callbacks with UART/ROS doubles; no devices."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
from node_runtime.actuator_debug import ActuatorDebugSession
from test_ros_robotics import FakeUart

SESSION = "0123456789abcdef0123456789abcdef"


class Rr2SessionTests(unittest.TestCase):
    def test_rr2_status_preserves_feedback_and_deadman_refreshes_set(self):
        from concurrent.futures import Future
        from node_runtime.actuator_debug import Rr2ActuatorDebugSession
        from node_runtime.serial_worker import SerialResult, CommandResult

        class Worker:
            ready = True
            startup_error = ""
            session_id = SESSION

            def __init__(self):
                self.sent = []
                self.results = []

            def submit(self, opcode, args=(), **kwargs):
                self.sent.append((opcode, args, kwargs))
                future = Future()
                future.set_result(CommandResult(True, SESSION, len(self.sent), state="0"))
                return future

            def drain_results(self, maximum):
                result, self.results = self.results[:maximum], self.results[maximum:]
                return result

            def latch_fault(self):
                return self.submit("STOP")

        worker = Worker()
        now = [10.0]
        session = Rr2ActuatorDebugSession(worker, clock=lambda: now[0])
        worker.results.append(SerialResult("STATUS", 1,
            ("ENCODER", "10000", "599", "4", "0", "0", "0", "0", "0"), SESSION))
        session.tick()
        self.assertTrue(session.ready)
        self.assertEqual(session.feedback.left_raw_count, 599)
        self.assertTrue(session.command_encoder("forward", 12.5))
        session.tick()
        self.assertIn(("SET", ("1", "1250", "0"), {"coalesce_key": "debug_snapshot"}), worker.sent)
        now[0] += .21
        session.tick()
        self.assertEqual(worker.sent[-1][1], ("1", "1250", "0"))
        now[0] += 1.01
        worker.results.append(SerialResult("STATUS", 2,
            ("ENCODER", "11220", "600", "5", "1", "1250", "0", "100", "100"), SESSION))
        session.tick()
        self.assertEqual(worker.sent[-1][1], ("0", "0", "0"))
        self.assertFalse(session.failed, session.error)
        session.close()
        self.assertEqual(worker.sent[-1][0], "STOP")

    def test_rr2_capability_and_bad_status_are_rejected(self):
        from node_runtime.actuator_debug import Rr2ActuatorDebugSession
        from node_runtime.serial_worker import SerialResult

        class Worker:
            ready, startup_error, session_id = True, "", SESSION
            def __init__(self):
                self.results = []
                self.stopped = False
            def drain_results(self, _maximum):
                result, self.results = self.results, []
                return result
            def latch_fault(self):
                self.stopped = True

        worker = Worker()
        session = Rr2ActuatorDebugSession(worker, "dc_motor")
        with self.assertRaises(ValueError):
            session.command_encoder("forward", 10)
        with self.assertRaises(ValueError):
            session.command_channel("front_pump", True)
        worker.results.append(SerialResult("STATUS", 1,
            ("ENCODER", "1", "0", "0", "0", "0", "16", "0", "0"), SESSION))
        session.tick()
        self.assertTrue(session.failed)
        self.assertTrue(worker.stopped)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.uart = FakeUart()
        self.session = ActuatorDebugSession(self.uart, clock=lambda: self.now, session_id=SESSION)

    def state(self, direction=0, percent=0, mask=0, seq=None):
        seq = self.session.sequence if seq is None else seq
        self.uart.pending.extend(
            f"DBG1_STATE {SESSION} {seq} {round(self.now * 1000)} 599 4 {direction} {percent} {mask} 0 0\r\n".encode())

    def ready(self):
        self.uart.pending.extend(
            f"DBG1_READY {SESSION} {self.session.capability} 1600 106 1200 3000 601 601\r\n".encode())
        self.state()
        self.session.tick()

    def test_handshake_starts_no_movement_and_uses_actual_firmware_metadata(self):
        self.assertEqual(self.uart.writes, [f"DBG1_HELLO {SESSION} 23\r\n".encode()])
        self.ready()
        self.assertTrue(self.session.ready)
        self.assertEqual(self.session.registered_drive_pwm, 1600)
        self.assertEqual(self.session.left_trim_percent, 106)
        self.assertEqual(self.session.left_counter_modulus, 601)
        self.assertTrue(self.uart.writes[-1].endswith(b" 1 0 0 0\r\n"))

    def test_control_frame_preserves_latest_ten_ms_pi_sample(self):
        self.ready()
        self.uart.pending.extend(
            f"DBG1_CTRL {SESSION} 1 37 10120 10 1 7 6 -11 12 11 1 384 9\r\n".encode())
        self.session.tick()
        self.assertFalse(self.session.failed, self.session.error)
        self.assertEqual(self.session.control_telemetry, dict(
            command_sequence=1, sample_sequence=37, sample_millis=10120,
            elapsed_ms=10, direction=1, pi_enabled=True, valid=True,
            applied=True, right_raw_delta=6, left_raw_delta=-11,
            right_normalized_delta=12, left_normalized_delta=11,
            speed_error=1, integral_q8=384, integral_pwm=1.5,
            correction_pwm=9, received_at=self.now))

    def test_initial_unavailable_control_snapshot_allows_zero_elapsed_time(self):
        self.ready()
        self.uart.pending.extend(
            f"DBG1_CTRL {SESSION} 0 0 0 0 0 0 0 0 0 0 0 0 0\r\n".encode())
        self.session.tick()
        self.assertFalse(self.session.failed, self.session.error)
        self.assertFalse(self.session.control_telemetry['valid'])
        self.assertFalse(self.session.control_telemetry['applied'])
        self.assertEqual(self.session.control_telemetry['elapsed_ms'], 0)

    def test_control_frame_rejects_unknown_flags_and_replayed_sample(self):
        self.ready()
        self.uart.pending.extend(
            f"DBG1_CTRL {SESSION} 0 2 10010 10 1 6 5 -10 10 10 0 0 0\r\n".encode())
        self.session.tick()
        first = self.session.control_telemetry
        self.uart.pending.extend(
            f"DBG1_CTRL {SESSION} 0 1 10000 10 1 6 4 -8 8 8 0 0 0\r\n".encode())
        self.session.tick()
        self.assertIs(self.session.control_telemetry, first)
        self.uart.pending.extend(
            f"DBG1_CTRL {SESSION} 0 3 10020 10 1 8 5 -10 10 10 0 0 0\r\n".encode())
        self.session.tick()
        self.assertTrue(self.session.failed)

    def test_stop_invalidation_with_same_sample_sequence_replaces_live_sample(self):
        self.ready()
        self.uart.pending.extend(
            f"DBG1_CTRL {SESSION} 1 37 10120 10 1 7 6 -11 12 11 1 384 9\r\n".encode())
        self.session.tick()
        self.session.command_encoder('stop', 0)
        self.session.tick()
        self.assertEqual(self.session.sequence, 2)
        self.uart.pending.extend(
            f"DBG1_CTRL {SESSION} 2 37 10120 10 1 1 6 -11 12 11 1 384 9\r\n".encode())
        self.session.tick()
        self.assertFalse(self.session.failed, self.session.error)
        self.assertFalse(self.session.control_telemetry['valid'])
        self.assertFalse(self.session.control_telemetry['applied'])
        self.assertEqual(self.session.control_telemetry['command_sequence'], 2)

    def test_handshake_decodes_ninety_nine_point_two_percent_for_physical_right_encoder(self):
        self.uart.pending.extend(
            f"DBG1_READY {SESSION} {self.session.capability} 1600 992 1200 3000 601 601\r\n".encode())
        self.state()
        self.session.tick()
        self.assertEqual(self.session.right_trim_percent, 99.2)
        self.assertEqual(self.session.left_trim_percent, 99.2)  # legacy compatibility alias

    def test_explicit_bp_ready_preserves_fraction_and_never_guesses_unit(self):
        for trim, expected in ((9920, 99.2), (9950, 99.5), (9921, 99.21), (10000, 100.0), (500, 5.0)):
            with self.subTest(trim=trim):
                self.setUp()
                self.uart.pending.extend(
                    f"DBG1_READY {SESSION} {self.session.capability} 1600 {trim} 1200 3000 601 601 BP\r\n".encode())
                self.state()
                self.session.tick()
                self.assertFalse(self.session.failed, self.session.error)
                self.assertEqual(self.session.right_trim_percent, expected)

    def test_ready_rejects_missing_wrong_extra_units_and_invalid_bp(self):
        for suffix in ('9920 1200 3000 601 601', '9920 1200 3000 601 601 PP',
                       '9920 1200 3000 601 601 BP extra', '0 1200 3000 601 601 BP',
                       '20001 1200 3000 601 601 BP', '-1 1200 3000 601 601 BP'):
            with self.subTest(suffix=suffix):
                self.setUp()
                self.uart.pending.extend(f"DBG1_READY {SESSION} 7 1600 {suffix}\r\n".encode())
                self.session.tick()
                self.assertTrue(self.session.failed)
                self.assertFalse(any(b'DBG1_SET' in frame for frame in self.uart.writes))

    def test_full_percent_is_registered_pwm_scalar_not_timer_duty(self):
        self.ready()
        self.session.command_encoder("forward", 100.0)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b" 1 10000 0\r\n"))
        self.assertNotIn(b"START\r\n", b"".join(self.uart.writes))

    def test_invalid_numeric_direction_and_stop_values(self):
        self.ready()
        for value in (-1, 100.01, float("nan"), float("inf"), True, "50"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.session.command_encoder("forward", value)
        for direction, percent in (("left", 50), ("stop", 10)):
            with self.assertRaises(ValueError):
                self.session.command_encoder(direction, percent)

    def test_channels_are_independent_and_expire_without_cancelling_fresh_peer(self):
        self.ready()
        self.session.command_channel("front_dc", True)
        self.session.command_channel("side_pump", True)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b" 0 0 9\r\n"))
        self.now += 0.5
        self.state(mask=9)
        self.session.command_channel("side_pump", True)
        self.session.tick()
        self.now += 0.51
        self.state(mask=9)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b" 0 0 8\r\n"))

    def test_reverse_waits_for_stop_feedback_and_dwell_preserving_pump(self):
        self.ready()
        self.session.command_channel("side_pump", True)
        self.session.command_encoder("forward", 100)
        self.session.tick()
        self.now += 0.01
        self.state(1, 10000, 8)
        self.session.command_encoder("reverse", 50)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b" 0 0 8\r\n"))
        self.now += 0.21
        self.state(1, 10000, 8, seq=self.session.sequence - 1)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b" 0 0 8\r\n"))
        self.now += 0.21
        self.state(0, 0, 8)
        self.session.tick()
        self.assertTrue(self.uart.writes[-1].endswith(b" -1 5000 8\r\n"))

    def test_timeout_and_late_commands_close_and_do_not_resume(self):
        self.ready()
        self.session.command_encoder("forward", 10)
        self.session.tick()
        self.now += 0.61
        self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertEqual(self.uart.writes[-1], f"DBG1_CLOSE {SESSION}\r\n".encode())
        before = len(self.uart.writes)
        with self.assertRaises(ValueError):
            self.session.command_encoder("forward", 100)
        self.session.tick()
        self.assertEqual(len(self.uart.writes), before)

    def test_old_firmware_or_busy_firmware_never_receives_on(self):
        self.now += 2.01
        self.session.tick()
        self.assertTrue(self.session.failed)
        self.assertIn("compatible firmware", self.session.error)
        self.assertFalse(any(b"DBG1_SET" in line for line in self.uart.writes))

    def test_wrong_session_and_replayed_feedback_do_not_extend_deadman(self):
        self.ready()
        self.now += 0.4
        self.uart.pending.extend(f"DBG1_STATE {'f'*32} 0 10001 0 0 0 0 0 0 0\n".encode())
        self.uart.pending.extend(f"DBG1_STATE {SESSION} 0 10000 0 0 0 0 0 0 0\n".encode())
        self.session.tick()
        self.now += 0.21
        self.session.tick()
        self.assertTrue(self.session.failed)

    def test_capability_limits_and_close_are_enforced(self):
        session = ActuatorDebugSession(FakeUart(), "dc_motor", clock=lambda: self.now)
        session.ready = True
        with self.assertRaises(ValueError):
            session.command_encoder("forward", 20)
        with self.assertRaises(ValueError):
            session.command_channel("front_pump", True)
        session.command_channel("front_dc", True)
        session.close()
        count = len(session.uart.writes)
        session.close()
        self.assertEqual(len(session.uart.writes), count)


def run_node_tests():
    from test_ros_robotics import install_fake_ros, FakeNode
    fake = install_fake_ros()
    import rail_robot_interfaces.msg as messages
    import std_msgs.msg as std_messages
    # Read actual IDL field declarations so typos fail instead of being silently
    # accepted by the usual loose callback test messages.
    def message_type(name):
        source = Path(__file__).resolve().parents[1] / "ros2" / "rail_robot_interfaces" / "msg" / (name + ".msg")
        fields = [line.split() for line in source.read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
        class Message:
            __slots__ = tuple(name for _, name in fields)
            def __init__(self, **kwargs):
                for kind, name in fields:
                    value = (SimpleNamespace(stamp=fake.Time()) if kind == "std_msgs/Header"
                             else "" if kind == "string" else False if kind == "bool" else 0)
                    setattr(self, name, kwargs.get(name, value))
        return Message
    messages.EncoderCommand, messages.EncoderState = message_type("EncoderCommand"), message_type("EncoderState")
    std_messages.Bool = SimpleNamespace
    FakeNode.count_publishers = lambda self, topic: getattr(self, "mission_count", 0)
    FakeNode.get_service_names_and_types = lambda self: getattr(self, "graph_services", [])
    FakeNode.resolve_service_name = lambda self, name: "/rail/" + name
    from node_runtime.ros.actuator_debug_node import ActuatorDebugNode

    class NodeTests(unittest.TestCase):
        def make_node(self, **kwargs):
            node = ActuatorDebugNode("fake", uart_protocol="legacy", uart_factory=FakeUart,
                                     exclusive_device=lambda uart: None, **kwargs)
            self.addCleanup(node.destroy_node)
            return node

        def ready(self, node):
            session = node.session
            node._uart.pending.extend(f"DBG1_READY {session.session_id} {session.capability} 1600 106 1200 3000 601 601\n".encode())
            node._uart.pending.extend(f"DBG1_STATE {session.session_id} 0 42 599 4 0 0 0 0 0\n".encode())
            node._tick()

        def test_topics_readonly_parameters_and_kernel_exclusivity_request(self):
            node = self.make_node()
            self.assertTrue(node._uart.options["exclusive"])
            self.assertTrue(all(value.read_only for value in node.parameter_descriptors.values()))
            self.assertEqual(node.subscriptions["debug/encoder/command"].qos.depth, 1)
            self.ready(node)
            feedback = node.publishers["debug/encoder/state"].messages[-1]
            self.assertTrue(feedback.feedback_valid)
            self.assertEqual(feedback.left_raw_count, 599)
            self.assertEqual(feedback.left_counter_modulus, 601)
            self.assertEqual(feedback.mcu_millis, 42)
            self.assertEqual(feedback.applied_direction, "stop")

        def test_invalid_input_sends_close_latches_once_and_stops_all(self):
            node = self.make_node()
            self.ready(node)
            node._on_encoder(messages.EncoderCommand(direction="forward", speed_percent=float("nan")))
            self.assertTrue(node._failed)
            self.assertTrue(node._uart.writes[-1].startswith(b"DBG1_CLOSE"))
            self.assertEqual(len(node.publishers["runtime/fault"].messages), 1)
            node._on_fault(SimpleNamespace(data="other fault"))
            self.assertEqual(len(node.publishers["runtime/fault"].messages), 1)
            self.assertFalse(node.publishers["debug/encoder/state"].messages[-1].feedback_valid)

        def test_mission_publication_stops_debug_even_when_mission_idle(self):
            node = self.make_node()
            self.ready(node)
            node._on_mission(SimpleNamespace(mode="IDLE"))
            self.assertTrue(node._failed)
            self.assertTrue(node._uart.writes[-1].startswith(b"DBG1_CLOSE"))

        def test_drive_fault_diagnostics_survive_shutdown_with_requested_speed(self):
            import json
            node = self.make_node(debug_capability="motor_encoder")
            self.ready(node)
            node._on_encoder(messages.EncoderCommand(direction="forward", speed_percent=100.0))
            session = node.session
            node._uart.pending.extend(
                f"DBG1_DIAG {session.session_id} {session.sequence} 43 DRIVE_STALL 4 0 0 599 4 0 0\n"
                f"DBG1_ERROR {session.session_id} DRIVE_STALL\n".encode())
            node._tick()
            self.assertTrue(node._failed)
            self.assertIn('DRIVE_STALL', node._error)
            report = json.loads(node.publishers['debug/encoder/diagnostics'].messages[-1].data)
            self.assertEqual(report['reason'], 'DRIVE_STALL')
            self.assertEqual(report['fault_code'], 4)
            self.assertEqual(report['last_drive_request']['speed_percent'], 100.0)
            self.assertEqual(report['left_raw_count'], 599)
            self.assertEqual(report['left_pwm_compare'], 0)
            self.assertEqual(report['forward_right_trim_percent'], node.session.right_trim_percent)
            self.assertEqual(report['right_trim_percent'], report['forward_right_trim_percent'])
            self.assertIsNone(report['reverse_right_trim_percent'])  # READY에서 받지 않은 값을 추정하지 않는다.
            self.assertFalse(report['session_active'])
            node.prepare_shutdown()
            closed = json.loads(node.publishers['debug/encoder/diagnostics'].messages[-1].data)
            self.assertTrue(closed['historical'])
            self.assertEqual(closed['reason'], 'DRIVE_STALL')

        def test_control_sample_is_exposed_in_diagnostics_json(self):
            import json
            node = self.make_node(debug_capability="motor_encoder")
            self.ready(node)
            session = node.session
            node._uart.pending.extend(
                f"DBG1_CTRL {session.session_id} 0 8 55 10 -1 6 -4 9 8 9 -1 0 0\n".encode())
            node._tick()
            report = json.loads(node.publishers['debug/encoder/diagnostics'].messages[-1].data)
            control = report['control_loop']
            self.assertEqual(control['sample_period_ms'], 10)
            self.assertEqual(control['physical_right_delta'], 8)
            self.assertEqual(control['physical_left_delta'], 9)
            self.assertEqual(control['speed_error'], -1)
            self.assertEqual(control['integral_pwm'], 0.0)
            self.assertEqual(control['correction_pwm'], 0)
            self.assertFalse(control['pi_enabled'])
            self.assertTrue(control['valid'])
            self.assertTrue(control['applied'])

        def test_ready_diagnostic_marks_previous_control_sample_historical(self):
            import json
            node = self.make_node(debug_capability="motor_encoder")
            self.ready(node)
            session = node.session
            node._uart.pending.extend(
                f"DBG1_CTRL {session.session_id} 0 8 55 10 1 7 4 -8 8 8 0 0 0\n"
                f"DBG1_DIAG {session.session_id} 0 56 READY 0 0 0 20 30 0 0\n".encode())
            node._tick()
            report = json.loads(node.publishers['debug/encoder/diagnostics'].messages[-1].data)
            self.assertTrue(report['historical'])
            self.assertTrue(report['control_loop']['historical'])

        def test_first_failure_report_marks_previous_pwm_historical(self):
            import json
            node = self.make_node()
            self.ready(node)
            sid = node.session.session_id
            node._uart.pending.extend(f"DBG1_DIAG {sid} 1 43 RUNNING 0 0 0 599 4 1600 1600\n".encode())
            node._tick()
            node._on_fault(SimpleNamespace(data='external stop'))
            report = json.loads(node.publishers['debug/encoder/diagnostics'].messages[-1].data)
            self.assertTrue(report['historical'])
            self.assertFalse(report['session_active'])

        def test_bridge_graph_conflict_refuses_uart_open(self):
            from unittest.mock import patch
            calls = []
            with patch.object(FakeNode, "get_service_names_and_types", return_value=[("/rail/stm/command", [])]):
                with self.assertRaises(RuntimeError):
                    ActuatorDebugNode("fake", uart_factory=lambda **kw: calls.append(kw))
            self.assertEqual(calls, [])

        def test_shutdown_closes_once_and_late_callback_has_no_effect(self):
            node = self.make_node()
            self.ready(node)
            node.prepare_shutdown()
            state = node.publishers["debug/encoder/state"].messages[-1]
            self.assertFalse(state.feedback_valid)
            self.assertIn("not acknowledged", state.error)
            self.assertEqual(state.commanded_direction, "stop")
            before = len(node._uart.writes)
            node._on_channel("front_dc", SimpleNamespace(data=True))
            node._tick()
            node.destroy_node()
            self.assertTrue(node._uart.closed)
            self.assertEqual(len(node._uart.writes), before)

        def test_other_robot_namespace_does_not_block_a_separate_uart(self):
            from unittest.mock import patch
            with patch.object(FakeNode, "get_service_names_and_types", return_value=[("/other/stm/command", [])]):
                node = self.make_node()
            self.assertFalse(node._failed)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NodeTests))
    return 0 if result.wasSuccessful() else 1


class RosCallbackTests(unittest.TestCase):
    def test_default_rr2_node_owns_serial_in_worker_and_stops_on_shutdown(self):
        script = r'''
import threading, time
from types import SimpleNamespace
from test_ros_robotics import install_fake_ros, FakeNode, FakeUart
install_fake_ros()
FakeNode.count_publishers = lambda self, topic: 0
FakeNode.resolve_service_name = lambda self, name: '/' + name
FakeNode.get_service_names_and_types = lambda self: []
import rail_robot_interfaces.msg as messages
import std_msgs.msg as std_messages
class EncoderState:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None)
messages.EncoderCommand = SimpleNamespace
messages.EncoderState = EncoderState
std_messages.Bool = SimpleNamespace
from node_runtime.rr2_protocol import Rr2Frame, decode_rr2, encode_rr2
from node_runtime.ros.actuator_debug_node import ActuatorDebugNode
io_threads, wires, serials = set(), [], []
class Serial(FakeUart):
    def __init__(self, **kw):
        super().__init__(**kw)
        io_threads.add(threading.get_ident())
        serials.append(self)
    @property
    def in_waiting(self):
        io_threads.add(threading.get_ident())
        return super().in_waiting
    def read(self, length):
        io_threads.add(threading.get_ident())
        return super().read(length)
    def write(self, data):
        io_threads.add(threading.get_ident())
        frame = decode_rr2(data)
        wires.append(frame)
        if frame.kind == 'HELLO':
            self.pending.extend(encode_rr2(Rr2Frame('READY', frame.session, None,
                ('2', 'ACTUATOR_DEBUG_ALL'))))
            self.pending.extend(encode_rr2(Rr2Frame('STATUS', frame.session, 1,
                ('ENCODER','100','599','4','0','0','0','0','0'))))
        elif frame.kind in ('CMD','PING'):
            opcode = frame.payload[0] if frame.kind == 'CMD' else 'PING'
            self.pending.extend(encode_rr2(Rr2Frame('ACK', frame.session, frame.sequence,
                (opcode,'ACCEPTED','0'))))
        return super().write(data)
    def close(self):
        io_threads.add(threading.get_ident())
        super().close()
node = ActuatorDebugNode('fake', uart_protocol='v2', uart_factory=Serial,
    exclusive_device=lambda uart: io_threads.add(threading.get_ident()))
for _ in range(50):
    node._tick()
    if node.session.feedback is not None: break
    time.sleep(.005)
assert node.session.ready and node.session.feedback is not None
assert 'debug/encoder/state' in node.publishers
prior = len([frame for frame in wires if frame.kind == 'CMD' and frame.payload[0] == 'SET'])
node._on_encoder(SimpleNamespace(direction='forward', speed_percent=20.0))
assert len([frame for frame in wires if frame.kind == 'CMD' and frame.payload[0] == 'SET']) == prior
node._tick()
for _ in range(50):
    if any(frame.kind == 'CMD' and frame.payload == ('SET','1','2000','0') for frame in wires): break
    node._tick()
    time.sleep(.005)
assert any(frame.kind == 'CMD' and frame.payload == ('SET','1','2000','0') for frame in wires)
node.destroy_node()
assert serials[0].closed
assert any(frame.kind == 'CMD' and frame.payload == ('STOP',) for frame in wires)
assert len(io_threads) == 1
assert not any(b'DBG1' in data for data in serials[0].writes)
'''
        tests = Path(__file__).resolve().parent
        source = f'import sys; sys.path[:0] = {[str(tests), str(tests.parent / "jetson_code")]!r}\n' + script
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_isolated_ros_callbacks(self):
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--node-tests"],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    if "--node-tests" in sys.argv:
        raise SystemExit(run_node_tests())
    unittest.main()
