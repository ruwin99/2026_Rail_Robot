"""Remote mission process launch never invokes mission START or hardware."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))

from node_runtime.mission_launcher import (  # noqa: E402
    LaunchRequestStore, MissionLauncher, MissionService, launch_allowed,
    valid_launch_request,
)
from node_runtime.webrtc_signaling import utc  # noqa: E402


ROBOT = {"enabled": True, "remoteLaunchEnabled": True, "senderId": "sender",
         "operatorUids": ["operator"]}


def request(state="requested", age=0, action="LAUNCH_MISSION", legacy_expires=None):
    created = time.time() - age
    data = {
        "schemaVersion": 1, "robotId": "r1", "ownerUid": "operator",
        "action": action, "state": state, "createdAt": utc(created),
        "acceptedAt": None, "finishedAt": None, "resultCode": None,
    }
    if legacy_expires is not None:
        # 전환 기간: 예전 웹은 브라우저 시계로 만든 expiresAt을 함께 보냈다.
        data["expiresAt"] = utc(created + legacy_expires)
    return {"path": "robots/r1/launchRequests/request_1", "updateTime": utc(), "data": data}


class FakeStore:
    robot_id = "r1"
    sender_uid = "sender"

    def __init__(self, docs=(), robot_data=None):
        self.docs = list(docs)
        self.robot_data = ROBOT if robot_data is None else robot_data
        self.claimed = []
        self.finished = []
        self.statuses = []

    def robot(self):
        return self.robot_data

    def now(self):
        return time.time()

    def requests(self):
        return self.docs

    def claim(self, doc):
        self.claimed.append(doc["path"])
        return dict(doc, data=dict(doc["data"], state="accepted"))

    def finish(self, doc, success, code):
        self.finished.append((doc["path"], success, code))

    def status(self, ready, mission_running):
        self.statuses.append((ready, mission_running))


class FakeService:
    def __init__(self, state="inactive", loaded=True, external=False):
        self.state = state
        self.loaded = loaded
        self.external = external
        self.starts = 0

    def inspect(self):
        return self.loaded, self.state

    def start(self):
        self.starts += 1
        self.state = "active"

    def other_mission_running(self):
        return self.external


class MissionLauncherTests(unittest.TestCase):
    def test_validation_requires_exact_shape_freshness_and_distinct_launch_permission(self):
        self.assertTrue(valid_launch_request(request(), "r1", time.time()))
        self.assertTrue(valid_launch_request(request(age=-4), "r1", time.time()))
        self.assertFalse(valid_launch_request(request(age=-6), "r1", time.time()))
        # 새 형식의 수명은 서버 createdAt + 10 s다.
        self.assertTrue(valid_launch_request(request(age=9), "r1", time.time()))
        self.assertFalse(valid_launch_request(request(age=11), "r1", time.time()))
        self.assertFalse(valid_launch_request(request(age=16), "r1", time.time()))
        # 전환 기간이 끝나 예전 expiresAt 형식은 받지 않는다.
        self.assertFalse(valid_launch_request(request(legacy_expires=15), "r1", time.time()))
        self.assertFalse(valid_launch_request(request(action="START"), "r1", time.time()))
        altered = request()
        altered["data"]["extra"] = True
        self.assertFalse(valid_launch_request(altered, "r1", time.time()))
        self.assertTrue(launch_allowed(ROBOT, "sender", "operator"))
        self.assertTrue(launch_allowed(dict(ROBOT, remoteControlEnabled=False), "sender", "operator"))
        self.assertFalse(launch_allowed(dict(ROBOT, remoteLaunchEnabled=False), "sender", "operator"))
        self.assertFalse(launch_allowed(ROBOT, "sender", "viewer"))

    def test_launch_is_claimed_once_and_reports_started_without_sending_uart_start(self):
        store = FakeStore([request()])
        service = FakeService()
        launcher = MissionLauncher(store, service)
        launcher.poll()
        self.assertEqual(service.starts, 1)
        self.assertEqual(len(store.claimed), 1)
        self.assertEqual(store.finished[0][1:], (True, "launch_started"))

    def test_accepted_request_is_not_replayed_after_crash(self):
        store = FakeStore([request(state="accepted")])
        service = FakeService()
        MissionLauncher(store, service).poll()
        self.assertEqual(service.starts, 0)
        self.assertEqual(store.finished[0][2], "launch_outcome_unknown")

    def test_running_unit_cannot_be_launched_again(self):
        store = FakeStore([request()])
        service = FakeService("active")
        MissionLauncher(store, service).poll()
        self.assertEqual(service.starts, 0)
        self.assertEqual(store.finished[0][2], "already_running")
        store = FakeStore([request()])
        service = FakeService("inactive", loaded=False)
        MissionLauncher(store, service).poll()
        self.assertEqual(service.starts, 0)
        self.assertEqual(store.finished[0][2], "unit_unavailable")

    def test_failed_unit_is_launched_after_reset_failed(self):
        # 비정상 종료로 failed가 남아도 다음 원격 실행을 막지 않는다.
        store = FakeStore([request()])
        service = FakeService("failed")
        MissionLauncher(store, service).poll()
        self.assertEqual(service.starts, 1)
        self.assertEqual(store.finished[0][1:], (True, "launch_started"))
        self.assertEqual(MissionService.RESET_FAILED_ARGS,
                         ("systemctl", "--user", "reset-failed", "rail-mission.service"))

    def test_status_only_ready_for_loaded_idle_unit_with_robot_opt_in(self):
        store = FakeStore()
        service = FakeService()
        launcher = MissionLauncher(store, service)
        launcher.publish_status()
        self.assertEqual(store.statuses[-1], (True, False))
        store.robot_data = dict(ROBOT, remoteControlEnabled=False)
        launcher.publish_status()
        self.assertEqual(store.statuses[-1], (True, False))
        service.state = "failed"
        launcher.publish_status()
        self.assertEqual(store.statuses[-1], (True, False))
        service.state = "active"
        launcher.publish_status()
        self.assertEqual(store.statuses[-1], (False, True))
        service.state = "inactive"
        service.external = True
        launcher.publish_status()
        self.assertEqual(store.statuses[-1], (False, True))
        blocked = FakeStore([request()])
        MissionLauncher(blocked, service).poll()
        self.assertEqual(service.starts, 0)
        self.assertEqual(blocked.finished[0][2], "already_running")

    def test_systemctl_uses_fixed_arguments_without_shell(self):
        self.assertEqual(MissionService.START_ARGS,
                         ("systemctl", "--user", "start", "rail-mission.service"))
        self.assertNotIn("shell", MissionService.START_ARGS)

    def test_orphaned_camera_or_manual_robot_launch_blocks_second_stack(self):
        with TemporaryDirectory() as root:
            proc = Path(root)
            pid = proc / "123"
            pid.mkdir()
            (pid / "cmdline").write_bytes(
                b"/home/bitdol/Desktop/rail_robot/code/ros2/install/rail_robot_nodes/"
                b"lib/rail_robot_nodes/camera_node\0--ros-args\0")
            self.assertTrue(MissionService.other_mission_running(proc))
            (pid / "cmdline").write_bytes(
                b"/usr/bin/python3\0/opt/ros/humble/bin/ros2\0launch\0"
                b"rail_robot_nodes\0robot.launch.py\0mode:=dual_camera\0")
            self.assertTrue(MissionService.other_mission_running(proc))
            (pid / "cmdline").write_bytes(
                b"/home/bitdol/.../rail_robot_nodes/webrtc_sender_node\0")
            self.assertFalse(MissionService.other_mission_running(proc))

    def test_store_uses_cas_and_server_timestamps(self):
        class FakeSignaling:
            base = "https://firestore.googleapis.com/v1/projects/p/databases/(default)/documents"

            def __init__(self):
                self.calls = []

            def request(self, method, path, body=None, query=()):
                self.calls.append((method, path, body))
                return {"writeResults": [{"updateTime": utc()}]}

        signal = FakeSignaling()
        store = LaunchRequestStore({"robotId": "r1", "senderUid": "sender"}, signal)
        original = request()
        accepted = store.claim(original)
        store.finish(accepted, True, "launch_started")
        store.status(True, False)
        writes = [call[2]["writes"][0] for call in signal.calls]
        self.assertEqual(writes[0]["currentDocument"], {"updateTime": original["updateTime"]})
        # 조회는 최근 60 s의 요청만 읽는다(단일 필드라 복합 인덱스가 필요 없다).
        signal.calls.clear()
        signal.document = lambda row: row
        signal.request = lambda method, path, body=None, query=(): signal.calls.append(body) or []
        store.requests()
        where = signal.calls[0]["structuredQuery"]["where"]["fieldFilter"]
        self.assertEqual((where["field"]["fieldPath"], where["op"]),
                         ("createdAt", "GREATER_THAN_OR_EQUAL"))
        self.assertEqual(writes[0]["updateTransforms"][0]["fieldPath"], "acceptedAt")
        self.assertEqual(writes[1]["updateTransforms"][0]["fieldPath"], "finishedAt")
        self.assertEqual(writes[2]["updateTransforms"][0]["fieldPath"], "updatedAt")


if __name__ == "__main__":
    unittest.main()
