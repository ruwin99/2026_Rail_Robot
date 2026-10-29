"""Standalone executor/dispatch checks with fake ROS in a separate process."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


def run_isolated_tests():
    import signal
    import time
    from unittest.mock import patch
    from test_ros_robotics import install_fake_ros, FakeNode

    install_fake_ros()
    import rclpy
    from node_runtime.ros import entrypoints

    events = []
    context = {"ok": False}
    behavior = {"stop": "signal", "shutdown_ok": True}

    class ExternalShutdownException(Exception):
        pass

    class InvalidHandle(Exception):
        pass

    class Executor:
        def __init__(self, num_threads):
            events.append(("executor", num_threads))
            self.node = None
            self.stopped_once = False

        def add_node(self, node):
            self.node = node

        def spin_once(self, timeout_sec):
            if not self.stopped_once:
                self.stopped_once = True
                if behavior["stop"] == "signal":
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                elif behavior["stop"] == "fault":
                    self.node.subscriptions["runtime/fault"].callback(SimpleNamespace(data="test fault"))
                elif behavior["stop"] == "exception":
                    raise RuntimeError("callback failure")
            else:
                time.sleep(0.001)

        def shutdown(self, timeout_sec):
            events.append(("executor_shutdown", context["ok"]))
            return behavior["shutdown_ok"]

    def init(*, args, signal_handler_options):
        context["ok"] = True
        events.append(("init", args, signal_handler_options))

    def shutdown():
        events.append(("context_shutdown",))
        context["ok"] = False

    rclpy.init, rclpy.ok, rclpy.try_shutdown = init, lambda: context["ok"], shutdown
    for name, values in {
        "rclpy.executors": {"MultiThreadedExecutor": Executor, "ExternalShutdownException": ExternalShutdownException},
        "rclpy.exceptions": {"InvalidHandle": InvalidHandle},
        "rclpy.signals": {"SignalHandlerOptions": SimpleNamespace(NO="none")},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        sys.modules[name] = module

    class Node(FakeNode):
        def __init__(self):
            super().__init__("test_node")
            self._closed = False
            self._failed = False
            self.prepared = False

        def readiness(self):
            return False, "fresh camera required"

        def prepare_shutdown(self):
            self.prepared = True
            events.append(("prepare", context["ok"]))

        def destroy_node(self):
            events.append(("destroy", context["ok"]))
            self._closed = True
            return super().destroy_node()

    class EntrypointTests(unittest.TestCase):
        def setUp(self):
            events.clear()
            context["ok"] = False
            behavior.update(stop="signal", shutdown_ok=True)

        def test_dispatch_imports_only_selected_node(self):
            fake_module = SimpleNamespace(ControlNode=lambda: "control instance")
            with patch.object(entrypoints, "import_module", return_value=fake_module) as load:
                self.assertEqual(entrypoints.create_node("control_node"), "control instance")
            load.assert_called_once_with("node_runtime.ros.control_node")

        def test_sigint_quiesces_callbacks_before_destroy_with_context_alive(self):
            node = Node()
            argv = ["control_node", "--ros-args", "-r", "__ns:=/rail"]
            previous = signal.getsignal(signal.SIGINT)
            with patch.object(entrypoints, "create_node", return_value=node):
                result = entrypoints.run_node("control_node", argv)
            self.assertEqual(result, 0)
            names = [event[0] for event in events]
            self.assertLess(names.index("prepare"), names.index("executor_shutdown"))
            self.assertLess(names.index("executor_shutdown"), names.index("destroy"))
            self.assertLess(names.index("destroy"), names.index("context_shutdown"))
            self.assertIn(("executor", 2), events)
            self.assertIn(("destroy", True), events)
            self.assertEqual(events[0], ("init", argv, "none"))
            self.assertIs(signal.getsignal(signal.SIGINT), previous)

        def test_runtime_fault_is_nonzero_and_still_cleans_up(self):
            behavior["stop"] = "fault"
            with patch.object(entrypoints, "create_node", return_value=Node()):
                self.assertEqual(entrypoints.run_node("control_node", []), 1)
            self.assertIn(("destroy", True), events)
            self.assertFalse(context["ok"])

        def test_monitor_does_not_install_fault_shutdown_subscription(self):
            node = Node()
            with patch.object(entrypoints, "create_node", return_value=node):
                self.assertEqual(entrypoints.run_node("monitor_node", []), 0)
            self.assertNotIn("runtime/fault", node.subscriptions)

        def test_webrtc_sender_survives_other_node_faults(self):
            # 2026-10-05 실기: rail_drive(actuator_debug) 오류 fault마다 따로 띄운 WebRTC 송출이 함께 죽었다.
            node = Node()
            with patch.object(entrypoints, "create_node", return_value=node):
                self.assertEqual(entrypoints.run_node("webrtc_sender_node", []), 0)
            self.assertNotIn("runtime/fault", node.subscriptions)
            self.assertEqual(entrypoints.FAULT_OBSERVER_NODES,
                             {"monitor_node", "record_node", "webrtc_sender_node"})

        def test_recorder_sigint_finalizes_without_fault_shutdown_subscription(self):
            node = Node()
            with patch.object(entrypoints, "create_node", return_value=node):
                self.assertEqual(entrypoints.run_node("record_node", []), 0)
            self.assertNotIn("runtime/fault", node.subscriptions)
            names = [event[0] for event in events]
            self.assertLess(names.index("prepare"), names.index("executor_shutdown"))
            self.assertLess(names.index("executor_shutdown"), names.index("destroy"))

        def test_real_recorder_fault_then_sigint_saves_single_file_with_summary(self):
            import json
            import os
            import tempfile
            from test_ros_record import load_record
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, RAIL_ROBOT_HOME=directory):
                record = load_record().RecordNode()
                callbacks = {topic: callback for _, topic, callback, _ in record.subscriptions}
                def spin_fault_then_stop(executor, timeout_sec):
                    if not executor.stopped_once:
                        executor.stopped_once = True
                        callbacks['runtime/fault'](SimpleNamespace(data='DRIVE_STALL'))
                    else:
                        callbacks['debug/encoder/command'](SimpleNamespace(direction='stop', speed_percent=0))
                        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
                with patch.object(entrypoints, 'create_node', return_value=record), \
                     patch.object(Executor, 'spin_once', spin_fault_then_stop):
                    self.assertEqual(entrypoints.run_node('record_node', []), 0)
                paths = list(Path(directory).rglob('*.jsonl'))
                self.assertEqual(paths, [record.writer.path])
                rows = [json.loads(line) for line in paths[0].read_text(encoding='utf-8').splitlines()]
                self.assertEqual(rows[-1]['event'], 'summary')
                self.assertEqual(rows[-1]['samples_written'], 2)
                self.assertEqual(rows[1]['data']['data'], 'DRIVE_STALL')

        def test_callback_exception_is_nonzero_and_cleans_up(self):
            behavior["stop"] = "exception"
            with patch.object(entrypoints, "create_node", return_value=Node()):
                self.assertEqual(entrypoints.run_node("control_node", []), 1)
            self.assertIn(("destroy", True), events)

        def test_startup_failure_shuts_context_down(self):
            with patch.object(entrypoints, "create_node", side_effect=ValueError("bad model")):
                self.assertEqual(entrypoints.run_node("vision_node", []), 1)
            self.assertFalse(context["ok"])

        def test_active_callback_timeout_does_not_close_its_resources(self):
            behavior["shutdown_ok"] = False
            with patch.object(entrypoints, "create_node", return_value=Node()):
                self.assertEqual(entrypoints.run_node("vision_node", []), 1)
            self.assertNotIn("destroy", [event[0] for event in events])

        def test_readiness_service_uses_node_health(self):
            node = Node()
            entrypoints._add_readiness_service(node, "camera_node")
            response = node.services["camera/ready"].callback(SimpleNamespace(), SimpleNamespace())
            self.assertFalse(response.success)
            self.assertEqual(response.message, "fresh camera required")

        def test_wrapper_preserves_basename_and_returns_exit_status(self):
            argv = ["/tmp/stm_bridge_node", "--ros-args", "-p", "baud_rate:=57600"]
            with patch.object(entrypoints, "run_node", return_value=1) as run:
                with self.assertRaises(SystemExit) as stopped:
                    entrypoints.main(argv)
            self.assertEqual(stopped.exception.code, 1)
            run.assert_called_once_with("stm_bridge_node", ros_args=argv)

        def test_unknown_executable_does_not_initialize_ros(self):
            self.assertEqual(entrypoints.run_node("unknown", []), 2)
            self.assertFalse(events)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EntrypointTests))
    return 0 if result.wasSuccessful() else 1


class RosEntrypointIsolationTests(unittest.TestCase):
    def test_entrypoints_in_isolated_process(self):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--isolated"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--isolated" in sys.argv:
        raise SystemExit(run_isolated_tests())
    unittest.main()
