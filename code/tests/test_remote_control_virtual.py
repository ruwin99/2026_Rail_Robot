"""Hardware-free dashboard command path: Firestore document -> ROS service request.

ROS imports are confined to a child process so discovery cannot contaminate
other tests with fake ROS modules. No network, serial port, or robot is used.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


class VirtualRemoteControlProcessTests(unittest.TestCase):
    def test_virtual_command_path(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Ran 6 tests", result.stderr)


def run_virtual_suite():
    import queue
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch

    from test_ros_robotics import install_fake_ros
    install_fake_ros()

    import time

    from node_runtime.remote_control import RemoteControlWorker
    from node_runtime.ros import remote_control_node as module
    from node_runtime.webrtc_signaling import utc

    robot = {"enabled": True, "remoteControlEnabled": True, "senderId": "sender",
             "operatorUids": ["operator"]}

    def command(action="START", state="requested", ident="c1"):
        import time
        now = time.time()
        return {"path": "robots/r1/commands/" + ident, "updateTime": utc(now), "data": {
            "schemaVersion": 1, "robotId": "r1", "ownerUid": "operator",
            "action": action, "state": state, "createdAt": utc(now),
            "acceptedAt": utc(now) if state == "accepted" else None,
            "finishedAt": None, "resultCode": None,
        }}

    class MemoryStore:
        sender_uid, robot_id = "sender", "r1"

        def __init__(self, documents):
            self.documents = documents
            self.claimed = []
            self.finished = []

        def robot(self):
            return robot

        def now(self):
            return time.time()

        def commands(self):
            return list(self.documents)

        def stop_commands(self):
            return iter(doc for doc in self.documents if doc["data"]["action"] == "E_STOP")

        def claim(self, doc):
            self.claimed.append(doc["path"])
            doc["data"]["state"] = "accepted"
            doc["data"]["acceptedAt"] = utc()
            return doc

        def finish(self, doc, success, code):
            self.finished.append((doc["path"].rsplit("/", 1)[-1], success, code))
            doc["data"]["state"] = "succeeded" if success else "failed"
            doc["data"]["resultCode"] = code

    def make_system(documents, *, ready=True):
        store = MemoryStore(documents)
        worker = object.__new__(RemoteControlWorker)
        worker._store = store
        worker._requests = queue.Queue(maxsize=32)
        worker._results = queue.Queue(maxsize=32)
        worker._inflight = {}
        worker._stop = threading.Event()
        worker.last_error = None
        with patch.object(module, "RemoteControlWorker", return_value=worker):
            node = module.RemoteControlNode()
        worker._ready = node._mission_ready
        if ready:
            node._on_mission(SimpleNamespace(mode=module.CAPTURE_SCAN_MODE,
                                             initial_count=0, rescan_count=0, failed=False))
        return store, worker, node

    class VirtualTests(unittest.TestCase):
        def test_clock_unsynced_start_rejection_has_its_own_result_code(self):
            from node_runtime.clock_sync import CLOCK_UNSYNCED_MESSAGE
            done = lambda **fields: SimpleNamespace(result=lambda: SimpleNamespace(**fields))
            self.assertEqual(module.RemoteControlNode._outcome(
                "START", done(success=False, message=CLOCK_UNSYNCED_MESSAGE)), (False, "clock_not_synced"))
            self.assertEqual(module.RemoteControlNode._outcome(
                "START", done(success=False, message="busy")), (False, "start_rejected"))
            self.assertEqual(module.RemoteControlNode._outcome(
                "START", done(success=True, message="")), (True, "start_sent"))

        def test_ready_start_reaches_mock_mission_service_and_finishes(self):
            store, worker, node = make_system([command()])
            worker._poll()
            node._process()
            self.assertEqual(len(node.clients["mission/start"].requests), 1)
            self.assertEqual(node.clients["stm/command"].requests, [])
            node.clients["mission/start"].futures[0].set_result(SimpleNamespace(success=True))
            node._process()
            worker._finish_results()
            self.assertEqual(store.finished, [("c1", True, "start_sent")])

        def test_start_without_fresh_ready_mission_is_rejected(self):
            store, worker, node = make_system([command()], ready=False)
            worker._poll()
            node._process()
            self.assertEqual(store.finished, [("c1", False, "mission_not_ready")])
            self.assertEqual(node.clients["mission/start"].requests, [])

        def test_stop_is_first_and_blocks_start_in_same_poll(self):
            store, worker, node = make_system([command("START", ident="start"),
                                               command("E_STOP", ident="stop")])
            # The fake STM service converts the requested opcode into an
            # in-memory UART write. It never opens a device or drives motors.
            uart_writes = []

            def mock_stm_service(request):
                uart_writes.append(request.command.encode("ascii"))
                return SimpleNamespace(success=True)

            node.clients["stm/command"].responder = mock_stm_service
            worker._poll()
            node._process()
            self.assertEqual([request.command for request in node.clients["stm/command"].requests],
                             ["E_STOP"])
            self.assertEqual(uart_writes, [b"E_STOP"])
            self.assertEqual(node.clients["mission/start"].requests, [])
            node._process()
            worker._finish_results()
            self.assertIn(("stop", True, "estop_ack"), store.finished)
            self.assertIn(("start", False, "remote_stop_requested"), store.finished)

        def test_accepted_start_is_never_replayed_after_restart(self):
            store, worker, node = make_system([command(state="accepted")])
            worker._poll()
            node._process()
            self.assertEqual(store.claimed, [])
            self.assertEqual(store.finished, [("c1", False, "start_outcome_unknown")])
            self.assertEqual(node.clients["mission/start"].requests, [])

        def test_repeated_poll_does_not_dispatch_inflight_stop_twice(self):
            store, worker, node = make_system([command("E_STOP")])
            worker._poll()
            worker._poll()
            node._process()
            self.assertEqual(len(node.clients["stm/command"].requests), 1)
            self.assertEqual(len(store.claimed), 1)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VirtualTests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--child":
        sys.exit(0 if run_virtual_suite() else 1)
    unittest.main()
