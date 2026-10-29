"""Exercise actual node callbacks with fake ROS only inside a child process.

Other ROS adapter tests may import install_fake_ros inside their own subprocess.
Importing this file never replaces modules in the test runner.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


class FakeFuture:
    def __init__(self, *, executor=None):
        self._done = False
        self._cancelled = False
        self._value = None
        self._error = None
        self._callbacks = []

    def done(self):
        return self._done

    def cancelled(self):
        return self._cancelled

    def result(self):
        if self._error is not None:
            raise self._error
        return self._value

    def set_result(self, value):
        self._cancelled = False
        self._value, self._done = value, True
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            callback(self)

    def set_exception(self, error):
        self._error = error
        self.set_result(None)

    def cancel(self):
        if not self._done and not self._cancelled:
            self._cancelled = True
            callbacks, self._callbacks = self._callbacks, []
            for callback in callbacks:
                callback(self)

    def add_done_callback(self, callback):
        if self._done or self._cancelled:
            callback(self)
        else:
            self._callbacks.append(callback)

    def __await__(self):
        async def wait():
            while not self._done and not self._cancelled:
                await asyncio.sleep(0)
            return self.result()
        return wait().__await__()


class FakeClient:
    def __init__(self):
        self.requests, self.futures = [], []
        self.ready = True
        self.responder = None

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        future = FakeFuture()
        self.futures.append(future)
        # responder가 None을 돌려주면 아직 응답하지 않은 호출이다. 시험이 futures로 직접 채운다.
        response = None if self.responder is None else self.responder(request)
        if response is not None:
            future.set_result(response)
        return future


class FakeNode:
    def __init__(self, node_name, **kwargs):
        self.node_name = node_name
        self._ros_seconds = 100.0
        self.publishers, self.clients, self.subscriptions, self.services = {}, {}, {}, {}
        self.timers = []
        self.log_messages = []
        self.parameters = {}
        self.parameter_descriptors = {}
        self.parameter_overrides = {
            value.name: value.value for value in kwargs.get("parameter_overrides", [])
        }

    def create_publisher(self, message_type, topic, qos, **kwargs):
        messages = []
        publisher = SimpleNamespace(messages=messages, publish=messages.append, qos=qos)
        self.publishers[topic] = publisher
        return publisher

    def create_client(self, service_type, name, **kwargs):
        client = FakeClient()
        if name.endswith("/ready"):
            client.responder = lambda request: SimpleNamespace(success=True, message="ready")
        self.clients[name] = client
        return client

    def declare_parameter(self, name, value=None, descriptor=None):
        parameter = SimpleNamespace(value=self.parameter_overrides.get(name, value))
        self.parameters[name] = parameter
        self.parameter_descriptors[name] = descriptor
        return parameter

    def get_parameter(self, name):
        return self.parameters[name]

    def create_subscription(self, message_type, topic, callback, qos, **kwargs):
        subscription = SimpleNamespace(callback=callback, qos=qos)
        self.subscriptions[topic] = subscription
        return subscription

    def create_service(self, service_type, name, callback, **kwargs):
        service = SimpleNamespace(callback=callback)
        self.services[name] = service
        return service

    def create_timer(self, interval, callback, **kwargs):
        timer = SimpleNamespace(interval=interval, callback=callback, cancel=lambda: None)
        self.timers.append(timer)
        return timer

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self._ros_seconds * 1e9)))

    def get_logger(self):
        return SimpleNamespace(error=self.log_messages.append, warning=self.log_messages.append, info=self.log_messages.append)

    def destroy_timer(self, timer):
        if timer in self.timers:
            self.timers.remove(timer)
        return True

    def destroy_node(self):
        self.destroyed = True
        return True


class FakeUart:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.writes = []
        self.pending = bytearray()
        self.closed = False
        self.short_write = False

    @property
    def in_waiting(self):
        return len(self.pending)

    def read(self, length):
        data = bytes(self.pending[:length])
        del self.pending[:length]
        return data

    def write(self, data):
        self.writes.append(data)
        return len(data) - int(self.short_write)

    def reset_input_buffer(self):
        self.pending.clear()

    def close(self):
        self.closed = True


def install_fake_ros():
    """Install a small ROS API double; caller must use process isolation."""
    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        sys.modules[name] = value
        return value

    class Time:
        def __init__(self, sec=0, nanosec=0):
            self.sec, self.nanosec = sec, nanosec

    class Header:
        def __init__(self):
            self.stamp, self.frame_id = Time(), ""

    class Message:
        def __init__(self, **kwargs):
            self.header = Header()
            self.__dict__.update(kwargs)

    class Image(Message):
        def __init__(self, **kwargs):
            super().__init__()
            self.height = self.width = self.step = self.is_bigendian = 0
            self.encoding, self.data = "", b""
            self.__dict__.update(kwargs)

    class Sample(Message):
        def __init__(self, **kwargs):
            super().__init__()
            self.mask, self.image = Image(), Image()
            self.valid_until = Time()
            self.sequence = self.initial_count = self.rescan_count = 0
            self.valid = self.failed = self.pump_on = self.obstacle_detected = False
            self.rust_grade, self.crack_ratio, self.cleaner_duty_percent = 0, 0.0, 0.0
            self.camera_role = self.detector = self.detector_method = self.stream_id = ""
            self.session_id = self.role = self.mode = self.error = self.reason = ""
            self.tokens = []
            self.__dict__.update(kwargs)

    class String:
        def __init__(self, data=""):
            self.data = data

    class Service:
        Request = SimpleNamespace
        Response = SimpleNamespace

    class QoSProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module("rclpy")
    module("rclpy.node", Node=FakeNode)
    module("rcl_interfaces")
    module("rcl_interfaces.msg", ParameterDescriptor=SimpleNamespace)
    module("rclpy.task", Future=FakeFuture)
    module("rclpy.callback_groups", ReentrantCallbackGroup=type("ReentrantCallbackGroup", (), {}), MutuallyExclusiveCallbackGroup=type("MutuallyExclusiveCallbackGroup", (), {}))
    module("rclpy.qos", QoSProfile=QoSProfile,
           ReliabilityPolicy=SimpleNamespace(BEST_EFFORT="best_effort", RELIABLE="reliable"),
           DurabilityPolicy=SimpleNamespace(VOLATILE="volatile", TRANSIENT_LOCAL="transient_local"),
           HistoryPolicy=SimpleNamespace(KEEP_LAST="keep_last"))
    module("builtin_interfaces")
    module("builtin_interfaces.msg", Time=Time)
    module("std_msgs")
    module("std_msgs.msg", Header=Header, String=String, Bool=String)
    module("std_srvs")
    module("std_srvs.srv", Trigger=Service)
    module("sensor_msgs")
    module("sensor_msgs.msg", Image=Image)
    module("rail_robot_interfaces")
    module("rail_robot_interfaces.msg", CameraFrame=Sample, DetectionSample=Sample,
           ActuatorDecision=Sample, MissionStatus=Sample, StmEvents=Sample)
    module("rail_robot_interfaces.srv", UartCommand=Service, CaptureInspection=Service, DrainInspection=Service, CapturePair=Service)
    module("serial", Serial=FakeUart, SerialException=OSError)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
    return SimpleNamespace(Node=FakeNode, Uart=FakeUart, Message=Sample, Time=Time, Image=Image)


def run_callback_tests():
    install_fake_ros()
    import numpy as np
    from unittest.mock import patch
    from node_runtime.ros.common import stamp_from_seconds, qos_events
    from node_runtime.settings import REALTIME_CRACK_ROI_TOP, REALTIME_CRACK_ROI_BOTTOM, FRAME_WIDTH, AUTOMATIC_RAIL_SECTION_TARGET
    from node_runtime.ros.control_node import ControlNode
    from node_runtime.ros.stm_bridge_node import StmBridgeNode
    from node_runtime.ros.mission_node import MissionNode as AudioMissionNode
    from rail_robot_interfaces.msg import ActuatorDecision, DetectionSample, MissionStatus, StmEvents
    from std_msgs.msg import String

    # 기존 FSM 시험은 음성을 명시적으로 끈다. 음성 게이트는 test_ros_mission_audio에서 검증한다.
    def MissionNode():
        return AudioMissionNode(parameter_overrides=[SimpleNamespace(name="audio_enabled", value=False)])

    def stamped(kind, stamp=100.0, **kwargs):
        message = kind(**kwargs)
        message.header.stamp = stamp_from_seconds(stamp)
        return message

    def ready_mission(node):
        node.state.mark_start_sent()
        node.state.acknowledge_started()

    def realtime_bridge():
        node = StmBridgeNode("fake", uart_factory=FakeUart)
        response = node._command(SimpleNamespace(command="START"), SimpleNamespace())
        assert response.success
        node._uart.pending.extend(b"STARTED\n")
        node._tick()
        node._on_mission(stamped(MissionStatus, mode="REALTIME"))
        return node

    def decision(stamp=100.0, expiry=100.6, **kwargs):
        return stamped(ActuatorDecision, stamp, session_id=kwargs.pop("session_id", "control-a"),
                       sequence=kwargs.pop("sequence", 1), role="side",
                       cleaner_duty_percent=12.5, pump_on=True,
                       valid_until=stamp_from_seconds(expiry), **kwargs)

    def crack_sample(stream="camera-a", sequence=1, stamp=100.0, valid=True):
        message = stamped(DetectionSample, stamp, camera_role="side", stream_id=stream,
                          sequence=sequence, detector="crack", valid=valid,
                          detector_method="bgcrack-tensorrt/realtime/", crack_ratio=0.0)
        mask = np.zeros((REALTIME_CRACK_ROI_BOTTOM - REALTIME_CRACK_ROI_TOP, FRAME_WIDTH), dtype=np.uint8)
        message.mask.encoding = "mono8"
        message.mask.height, message.mask.width = mask.shape
        message.mask.step = mask.shape[1]
        message.mask.data = mask.tobytes()
        return message

    class CallbackTests(unittest.TestCase):
        def test_bridge_constructor_does_not_start_motion(self):
            node = StmBridgeNode("fake", uart_factory=FakeUart)
            self.assertEqual(len(node._uart.writes), 4)
            self.assertTrue(all(b"OFF" in item for item in node._uart.writes))
            rejected = node._command(SimpleNamespace(command="SIDE_PUMP_ON"), SimpleNamespace())
            self.assertFalse(rejected.success)
            self.assertNotIn(b"START\r\n", node._uart.writes)

        def test_bridge_legacy_link_keepalive_only_after_start_and_stops_on_fault(self):
            # STM legacy 경로는 주행 중 1 s 동안 명령이 없으면 멈춘다. 브리지는 START 뒤 200 ms마다
            # LINK_OK를 보내고, 브리지 fault 뒤에는 보내지 않아 STM이 스스로 멈추게 둔다.
            node = StmBridgeNode("fake", uart_factory=FakeUart)
            with patch("time.monotonic", return_value=50.0):
                node._tick()
            self.assertNotIn(b"LINK_OK\r\n", node._uart.writes)
            self.assertTrue(node._command(SimpleNamespace(command="START"), SimpleNamespace()).success)
            for now in (50.0, 50.1, 50.2, 50.45):
                with patch("time.monotonic", return_value=now):
                    node._tick()
            self.assertEqual(node._uart.writes.count(b"LINK_OK\r\n"), 3)
            node._fault("test fault")
            with patch("time.monotonic", return_value=51.0):
                node._tick()
            self.assertEqual(node._uart.writes.count(b"LINK_OK\r\n"), 3)

        def test_bridge_standalone_uses_readonly_ros_serial_parameters(self):
            node = StmBridgeNode(parameter_overrides=[
                SimpleNamespace(name="serial_port", value="/dev/test-stm"),
                SimpleNamespace(name="baud_rate", value=57600),
            ])
            self.assertEqual(node._uart.options["port"], "/dev/test-stm")
            self.assertEqual(node._uart.options["baudrate"], 57600)
            self.assertTrue(node.parameter_descriptors["serial_port"].read_only)
            self.assertTrue(node.parameter_descriptors["baud_rate"].read_only)
            node.prepare_shutdown()
            self.assertFalse(node._uart.closed)
            node.destroy_node()
            self.assertTrue(node._uart.closed)

        def test_bridge_restart_requires_its_own_started_handshake(self):
            node = StmBridgeNode("fake", uart_factory=FakeUart)
            node._on_mission(stamped(MissionStatus, mode="REALTIME"))
            node._on_decision(decision())
            node._tick()
            self.assertFalse(node._targets)
            self.assertFalse(node._pairs["side"][0].is_on)

        def test_bridge_capture_ack_requires_stm_request_and_only_once(self):
            node = realtime_bridge()
            rejected = node._command(SimpleNamespace(command="CAPTURE_OK"), SimpleNamespace())
            self.assertFalse(rejected.success)
            node._uart.pending.extend(b"CAMERA_CAPTURE\n")
            node._tick()
            self.assertTrue(node._command(SimpleNamespace(command="CAPTURE_OK"), SimpleNamespace()).success)
            self.assertFalse(node._command(SimpleNamespace(command="CAPTURE_OK"), SimpleNamespace()).success)

        def test_bridge_old_session_and_sequence_cannot_replace_new_target(self):
            node = realtime_bridge()
            node._on_decision(decision(sequence=2))
            node._on_decision(decision(sequence=1))
            self.assertEqual(node._targets["side"][0].sequence, 2)
            node._ros_seconds = 100.1
            node._on_decision(decision(stamp=100.1, expiry=100.7, session_id="control-b"))
            self.assertEqual(node._session, "control-b")
            node._ros_seconds = 100.2
            node._on_decision(decision(stamp=100.2, expiry=100.8, session_id="control-a", sequence=99))
            self.assertEqual(node._session, "control-b")

        def test_bridge_expiry_and_safe_off_do_not_reenable_outputs(self):
            node = realtime_bridge()
            node._on_decision(decision(expiry=100.2))
            deadline = node._targets["side"][1]
            node._tick()
            self.assertTrue(node._pairs["side"][0].is_on)
            with patch("time.monotonic", return_value=deadline + 0.01):
                node._tick()
            self.assertFalse(node._pairs["side"][0].is_on)
            self.assertTrue(node._command(SimpleNamespace(command="SAFE_OFF"), SimpleNamespace()).success)
            node._on_decision(decision(sequence=3))
            self.assertFalse(node._targets)
            self.assertNotIn(b"SAFE_OFF\r\n", node._uart.writes)

        def test_bridge_stale_decision_drops_off_and_fresh_one_can_recover(self):
            node = realtime_bridge()
            node._on_decision(decision())
            node._tick()
            self.assertTrue(node._pairs["side"][0].is_on)
            node._ros_seconds = 100.3
            node._on_decision(decision(sequence=2, stamp=100.0))
            self.assertFalse(node._pairs["side"][0].is_on)
            self.assertFalse(node._failed)
            node._on_decision(decision(sequence=3, stamp=100.3, expiry=100.6))
            node._tick()
            self.assertTrue(node._pairs["side"][0].is_on)

        def test_bridge_missing_mission_heartbeat_expires_active_output(self):
            with patch("time.monotonic", return_value=10.0):
                node = realtime_bridge()
                node._on_decision(decision(expiry=101.2))
                node._tick()
            self.assertTrue(node._pairs["side"][0].is_on)
            with patch("time.monotonic", return_value=10.81):
                node._tick()
            self.assertFalse(node._pairs["side"][0].is_on)

        def test_bridge_obstacle_announcements_are_edges_and_keep_raw_events(self):
            node = realtime_bridge()
            self.assertIn("tts/obstacle_events", node.publishers)
            announcements = node.publishers["tts/obstacle_events"]
            self.assertIs(announcements.qos, qos_events)
            node._uart.pending.extend(b"OBSTACLE_CLEAR\nOBSTACLE_HO")
            node._tick()
            self.assertFalse(announcements.messages)
            node._uart.pending.extend(b"LD\nOBSTACLE_HOLD\n")
            node._tick()
            node._uart.pending.extend(b"OBSTACLE_HOLD\nOBSTACLE_CLEAR\nOBSTACLE_CLEAR\n")
            node._tick()
            node._uart.pending.extend(b"OBSTACLE_HOLD\nOBSTACLE_CLEAR\n")
            node._tick()
            self.assertEqual([m.data for m in announcements.messages], [
                "OBSTACLE_HOLD", "OBSTACLE_CLEAR", "OBSTACLE_HOLD", "OBSTACLE_CLEAR",
            ])
            self.assertEqual([token for event in node._events.messages for token in event.tokens], [
                "STARTED", "OBSTACLE_CLEAR", "OBSTACLE_HOLD", "OBSTACLE_HOLD",
                "OBSTACLE_HOLD", "OBSTACLE_CLEAR", "OBSTACLE_CLEAR", "OBSTACLE_HOLD", "OBSTACLE_CLEAR",
            ])
            self.assertFalse(node._faults.messages)

        def test_bridge_obstacle_hold_stops_outputs_without_latching_fault(self):
            node = realtime_bridge()
            node._on_decision(decision())
            node._tick()
            self.assertTrue(node._pairs["side"][0].is_on)
            before = len(node._uart.writes)
            node._uart.pending.extend(b"OBSTACLE_HOLD\n")
            node._tick()
            self.assertFalse(node._targets)
            self.assertFalse(node._failed)
            self.assertFalse(node._faults.messages)
            self.assertEqual(len(node._uart.writes) - before, 4)
            self.assertTrue(all(b"OFF" in line for line in node._uart.writes[before:]))
            for cleaner, pump in node._pairs.values():
                self.assertFalse(cleaner.is_on or pump.is_on)
            node._on_mission(stamped(MissionStatus, mode="REALTIME"))
            node._on_decision(decision(sequence=2))
            node._uart.pending.extend(b"OBSTACLE_HOLD\n")
            node._tick()
            self.assertFalse(node._targets)
            self.assertEqual(len(node._uart.writes) - before, 4)
            node._uart.pending.extend(b"OBSTACLE_CLEAR\n")
            node._tick()
            self.assertFalse(node._targets)
            self.assertFalse(node._pairs["side"][0].is_on)
            node._ros_seconds = 100.1
            node._on_decision(decision(sequence=3, stamp=100.1, expiry=100.7))
            node._tick()
            self.assertTrue(node._pairs["side"][0].is_on)

        def test_bridge_clear_rejects_delayed_decisions_from_before_release(self):
            node = realtime_bridge()
            node._uart.pending.extend(b"OBSTACLE_HOLD\n")
            node._tick()
            node._ros_seconds = 100.1
            node._uart.pending.extend(b"OBSTACLE_CLEAR\n")
            node._tick()
            node._on_decision(decision(stamp=100.05, expiry=100.7))
            self.assertFalse(node._targets)
            node._on_decision(decision(sequence=2, stamp=100.1, expiry=100.7))
            self.assertFalse(node._targets)
            node._ros_seconds = 100.15
            node._on_decision(decision(sequence=3, stamp=100.15, expiry=100.7))
            self.assertIn("side", node._targets)

        def test_bridge_clear_preserves_other_output_inhibits(self):
            for inhibit in ("fault", "safe_off", "mission_mode", "mission_expired"):
                with self.subTest(inhibit=inhibit):
                    node = realtime_bridge()
                    node._uart.pending.extend(b"OBSTACLE_HOLD\n")
                    node._tick()
                    if inhibit == "fault":
                        node._on_fault(String(data="external failure"))
                    elif inhibit == "safe_off":
                        node._command(SimpleNamespace(command="SAFE_OFF"), SimpleNamespace())
                    elif inhibit == "mission_mode":
                        node._on_mission(stamped(MissionStatus, mode="RETURN"))
                    else:
                        node._mission_seen -= 1.0
                    node._uart.pending.extend(b"OBSTACLE_CLEAR\n")
                    node._tick()
                    node._ros_seconds = 100.1
                    node._on_decision(decision(stamp=100.1, expiry=100.7))
                    node._tick()
                    self.assertFalse(node._targets)
                    self.assertFalse(node._pairs["side"][0].is_on)

        def test_bridge_capture_ack_still_works_during_obstacle_hold(self):
            node = realtime_bridge()
            node._uart.pending.extend(b"OBSTACLE_HOLD\nCAMERA_CAPTURE\n")
            node._tick()
            self.assertTrue(node._command(SimpleNamespace(command="CAPTURE_OK"), SimpleNamespace()).success)
            self.assertFalse(node._command(SimpleNamespace(command="CAPTURE_OK"), SimpleNamespace()).success)
            node._on_decision(decision())
            self.assertFalse(node._targets)

        def test_bridge_obstacle_off_write_failure_remains_a_runtime_fault(self):
            node = realtime_bridge()
            node._uart.short_write = True
            node._uart.pending.extend(b"OBSTACLE_HOLD\n")
            node._tick()
            self.assertTrue(node._failed)
            self.assertTrue(node._faults.messages)

        def test_control_stream_restart_clears_history_and_rejects_retired_stream(self):
            node = ControlNode()
            node._on_mission(stamped(MissionStatus, mode="REALTIME"))
            node._on_sample(crack_sample(sequence=5))
            self.assertEqual(len(node._history["side"]._crack_ratios), 1)
            node._ros_seconds = 100.1
            node._on_sample(crack_sample(stream="camera-b", sequence=0, stamp=100.1))
            self.assertEqual(len(node._history["side"]._crack_ratios), 1)
            node._on_sample(crack_sample(stream="camera-a", sequence=6, stamp=100.1))
            self.assertEqual(node._streams["side"], "camera-b")
            self.assertFalse(node._failed)
            self.assertIs(node.subscriptions["vision/results"].qos, qos_events)

        def test_control_invalid_result_is_recoverable_safe_off(self):
            node = ControlNode()
            node._on_mission(stamped(MissionStatus, mode="REALTIME"))
            node._on_sample(crack_sample(valid=False))
            self.assertFalse(node._failed)
            self.assertFalse(node._hold["side"].has_decision)
            node._on_sample(crack_sample(sequence=2))
            self.assertEqual(len(node._history["side"]._crack_ratios), 1)

        def test_control_timer_does_not_extend_held_decision_deadline(self):
            node = ControlNode()
            with patch("time.monotonic", return_value=10.0):
                node._on_mission(stamped(MissionStatus, mode="REALTIME"))
                node._hold["side"].accept(SimpleNamespace(cleaner_duty_percent=12.5, pump_on=True))
                node._tick()
            first = next(message for message in reversed(node._decisions.messages) if message.role == "side")
            node._ros_seconds = 100.3
            with patch("time.monotonic", return_value=10.3):
                node._tick()
            second = next(message for message in reversed(node._decisions.messages) if message.role == "side")
            first_ns = first.valid_until.sec * 1_000_000_000 + first.valid_until.nanosec
            second_ns = second.valid_until.sec * 1_000_000_000 + second.valid_until.nanosec
            self.assertLessEqual(abs(second_ns - first_ns), 1)
            self.assertGreater(second.sequence, first.sequence)

        def test_mission_capture_acceptance_then_uart_ack_advances_once(self):
            node = MissionNode()
            ready_mission(node)
            node._on_events(stamped(StmEvents, tokens=["CAMERA_CAPTURE"]))
            self.assertFalse(node._uart.requests)
            self.assertEqual(node.state.initial_rail_section_count, 0)
            node._capture.futures[0].set_result(SimpleNamespace(accepted=True, error=""))
            self.assertEqual(node._uart.requests[-1].command, "CAPTURE_OK")
            self.assertEqual(node.state.initial_rail_section_count, 0)
            node._uart.futures[-1].set_result(SimpleNamespace(success=True, error=""))
            self.assertEqual(node.state.initial_rail_section_count, 1)
            self.assertFalse(node._busy)

        def test_mission_queue_failure_never_requests_capture_ack(self):
            node = MissionNode()
            ready_mission(node)
            node._on_events(stamped(StmEvents, tokens=["CAMERA_CAPTURE"]))
            node._capture.futures[0].set_result(SimpleNamespace(accepted=False, error="TOP queue full"))
            self.assertTrue(node._failed)
            self.assertEqual(node.state.initial_rail_section_count, 0)
            self.assertNotIn("CAPTURE_OK", [request.command for request in node._uart.requests])

        def test_mission_realtime_transition_waits_for_off_and_drain(self):
            node = MissionNode()
            ready_mission(node)
            for _ in range(AUTOMATIC_RAIL_SECTION_TARGET):
                capture = node.state.begin_capture()
                node.state.accept_capture(capture)
                node.state.acknowledge_capture(capture)
            node._on_events(stamped(StmEvents, tokens=["RETURN_START", "REALTIME_START"]))
            self.assertEqual(node.state.mode, "RETURN")
            self.assertEqual(node._uart.requests[-1].command, "SAFE_OFF")
            node._uart.futures[-1].set_result(SimpleNamespace(success=True, error=""))
            self.assertEqual(node.state.mode, "RETURN")
            self.assertFalse(node._drain.requests[-1].finalize)
            node._drain.futures[-1].set_result(SimpleNamespace(success=True, error=""))
            self.assertEqual(node.state.mode, "REALTIME")

        def test_mission_start_buffers_started_received_before_service_reply(self):
            node = MissionNode()
            node._uart.responder = lambda request: (
                SimpleNamespace(success=True, error="") if request.command == "PREPARE" else None
            )

            async def start_with_early_event():
                task = asyncio.create_task(node._start(SimpleNamespace(), SimpleNamespace()))
                await asyncio.sleep(0)
                node._on_events(stamped(StmEvents, tokens=["STARTED"]))
                self.assertFalse(node.state.start_acknowledged)
                node._uart.futures[-1].set_result(SimpleNamespace(success=True, error=""))
                return await task

            self.assertTrue(asyncio.run(start_with_early_event()).success)
            self.assertTrue(node.state.start_acknowledged)

        def test_mission_timeout_cancels_pending_work_without_ack(self):
            node = MissionNode()
            ready_mission(node)
            node._on_events(stamped(StmEvents, tokens=["CAMERA_CAPTURE"]))
            pending = node._capture.futures[0]
            with patch("time.monotonic", return_value=node._pending_since + 10.1):
                node._tick()
            self.assertTrue(node._failed)
            self.assertTrue(pending.cancelled())
            self.assertFalse(pending.done())
            self.assertEqual(node.state.initial_rail_section_count, 0)

        def test_mission_start_is_explicit_and_timeout_returns_failure(self):
            node = MissionNode()
            self.assertFalse(node._uart.requests)
            node._uart.responder = lambda request: (
                SimpleNamespace(success=True, error="") if request.command == "PREPARE" else None
            )

            async def start_and_timeout():
                task = asyncio.create_task(node._start(SimpleNamespace(), SimpleNamespace()))
                await asyncio.sleep(0)
                with patch("time.monotonic", return_value=node._pending_since + 10.1):
                    node._tick()
                return await task

            result = asyncio.run(start_and_timeout())
            self.assertFalse(result.success)
            self.assertTrue(node._failed)

        def test_mission_readiness_rejection_sends_no_start_and_allows_retry(self):
            node = MissionNode()
            node._uart.responder = lambda request: SimpleNamespace(success=True, error="")
            node._ready_clients["vision"].responder = lambda request: SimpleNamespace(success=False, message="No fresh model results")
            response = asyncio.run(node._start(SimpleNamespace(), SimpleNamespace()))
            self.assertFalse(response.success)
            self.assertIn("vision", response.message)
            self.assertEqual([request.command for request in node._uart.requests], ["PREPARE", "CANCEL_PREPARE"])
            self.assertFalse(node._failed)
            node._ready_clients["vision"].responder = lambda request: SimpleNamespace(success=True, message="ready")
            node._uart.responder = lambda request: SimpleNamespace(success=True, error="")
            self.assertTrue(asyncio.run(node._start(SimpleNamespace(), SimpleNamespace())).success)
            self.assertEqual(node._uart.requests[-1].command, "START")

        def test_mission_readiness_timeout_cancels_probe_without_start_or_fault(self):
            node = MissionNode()
            node._uart.responder = lambda request: SimpleNamespace(success=True, error="")
            node._ready_clients["camera"].responder = None

            async def probe_and_timeout():
                task = asyncio.create_task(node._start(SimpleNamespace(), SimpleNamespace()))
                await asyncio.sleep(0)
                future = node._pending_future
                with patch("time.monotonic", return_value=node._pending_since + 5.1):
                    node._tick()
                response = await task
                self.assertTrue(future.cancelled())
                return response

            self.assertFalse(asyncio.run(probe_and_timeout()).success)
            self.assertEqual([request.command for request in node._uart.requests], ["PREPARE", "CANCEL_PREPARE"])
            self.assertFalse(node._failed)

        def test_mission_fault_during_final_readiness_never_sends_start(self):
            node = MissionNode()
            node._uart.responder = lambda request: SimpleNamespace(success=True, error="")

            def ready_but_faulted(request):
                node._on_fault(String(data="Camera disconnected"))
                return SimpleNamespace(success=True, message="ready")

            node._ready_clients["vision"].responder = ready_but_faulted
            self.assertFalse(asyncio.run(node._start(SimpleNamespace(), SimpleNamespace())).success)
            self.assertNotIn("START", [request.command for request in node._uart.requests])

        def test_mission_destroy_cancels_start_and_blocks_late_status_publish(self):
            node = MissionNode()

            async def start_and_destroy():
                task = asyncio.create_task(node._start(SimpleNamespace(), SimpleNamespace()))
                await asyncio.sleep(0)
                pending = node._pending_future
                node.destroy_node()
                published_at_close = len(node._status.messages)
                response = await task
                self.assertTrue(pending.cancelled())
                self.assertEqual(len(node._status.messages), published_at_close)
                return response

            self.assertFalse(asyncio.run(start_and_destroy()).success)
            self.assertTrue(node._closed)
            self.assertIsNone(node._pending_future)

        def test_mission_destroy_cancels_capture_without_ack(self):
            node = MissionNode()
            ready_mission(node)
            node._on_events(stamped(StmEvents, tokens=["CAMERA_CAPTURE"]))
            pending = node._pending_future
            node.destroy_node()
            self.assertTrue(pending.cancelled())
            self.assertNotIn("CAPTURE_OK", [request.command for request in node._uart.requests])
            self.assertEqual(node.state.initial_rail_section_count, 0)

        def test_received_fault_does_not_republish_fault_loop(self):
            nodes = [ControlNode(), MissionNode(), StmBridgeNode("fake", uart_factory=FakeUart)]
            for node in nodes:
                node._on_fault(String(data="external failure"))
                self.assertTrue(node._failed)
                self.assertFalse(node._faults.messages)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CallbackTests))
    return 0 if result.wasSuccessful() else 1


class RosRoboticsIsolationTests(unittest.TestCase):
    def test_node_callbacks_in_isolated_process(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(run_callback_tests())
    unittest.main()
