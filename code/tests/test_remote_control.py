"""Remote dashboard command safety checks; no ROS, Firebase or hardware."""
from pathlib import Path
import queue
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
from node_runtime.remote_control import (  # noqa: E402
    RemoteCommandStore, RemoteControlWorker, robot_allows, valid_command,
)
from node_runtime.webrtc_signaling import FirestoreSignaling, SignalingError, utc  # noqa: E402


def command(action="START", state="requested", age=0, accepted_at=None, command_id="c1",
            legacy_expires=None):
    now = time.time()
    created = now - age
    data = {
        "schemaVersion": 1, "robotId": "r1", "ownerUid": "operator",
        "action": action, "state": state, "createdAt": utc(created),
        "acceptedAt": accepted_at, "finishedAt": None, "resultCode": None,
    }
    if legacy_expires is not None:
        # 전환 기간: 예전 웹은 브라우저 시계로 만든 expiresAt을 함께 보냈다.
        data["expiresAt"] = utc(created + legacy_expires)
    return {"path": "robots/r1/commands/" + command_id, "updateTime": utc(now), "data": data}


ROBOT = {"enabled": True, "remoteControlEnabled": True, "senderId": "sender",
         "operatorUids": ["operator"]}


class FakeSignaling:
    base = "https://firestore.googleapis.com/v1/projects/p/databases/(default)/documents"

    def __init__(self):
        self.calls = []

    def request(self, method, path, body=None, query=()):
        self.calls.append((method, path, body, query))
        return {"writeResults": [{"updateTime": utc()}]}


class FakeStopQuerySignaling(FakeSignaling):
    document = staticmethod(FirestoreSignaling.document)

    def __init__(self):
        super().__init__()
        now = time.time()
        self.rows = [{"name": self.base.split("/v1/", 1)[1] + f"/robots/r1/commands/e{i:03d}",
                      "updateTime": utc(now),
                      "fields": {"createdAt": {"timestampValue": utc(now - i / 1000)},
                                 "action": {"stringValue": "E_STOP"}}}
                     for i in range(101)]

    def request(self, method, path, body=None, query=()):
        self.calls.append((method, path, body, query))
        cursor = body["structuredQuery"].get("startAt")
        if cursor is None:
            return [{"document": row} for row in self.rows[:100]]
        self.cursor = cursor
        return [{"document": self.rows[100]}]


class FakeStore:
    sender_uid = "sender"
    robot_id = "r1"

    def __init__(self, docs, stop_docs=None):
        self.docs = docs
        self.stop_docs = ([doc for doc in docs if doc["data"]["action"] == "E_STOP"]
                          if stop_docs is None else stop_docs)
        self.finished = []
        self.claimed = []

    def robot(self):
        return ROBOT

    def now(self):
        return time.time()

    def commands(self):
        return self.docs[:100]

    def stop_commands(self):
        return iter(self.stop_docs)

    def claim(self, doc):
        self.claimed.append(doc["path"])
        return dict(doc, data=dict(doc["data"], state="accepted", acceptedAt=utc()))

    def finish(self, doc, success, code):
        self.finished.append((doc["path"], success, code))


class RemoteControlTests(unittest.TestCase):
    def test_operator_and_robot_opt_in_are_required(self):
        self.assertTrue(robot_allows(ROBOT, "sender", "operator"))
        self.assertFalse(robot_allows(ROBOT, "sender", "viewer"))
        self.assertFalse(robot_allows(dict(ROBOT, remoteControlEnabled=False), "sender", "operator"))
        self.assertFalse(robot_allows(dict(ROBOT, senderId="other"), "sender", "operator"))

    def test_start_is_short_lived_and_bound_to_robot(self):
        self.assertTrue(valid_command(command(), "r1", time.time()))
        # 새 형식의 수명은 서버 createdAt + 10 s다.
        self.assertTrue(valid_command(command(age=9), "r1", time.time()))
        self.assertFalse(valid_command(command(age=11), "r1", time.time()))
        self.assertFalse(valid_command(command(age=16), "r1", time.time()))
        # 전환 기간이 끝나 예전 expiresAt 형식은 받지 않는다.
        self.assertFalse(valid_command(command(legacy_expires=10), "r1", time.time()))
        self.assertFalse(valid_command(command(), "r2", time.time()))
        doc = command()
        doc["data"]["unexpected"] = True
        self.assertFalse(valid_command(doc, "r1", time.time()))

    def test_accepted_stop_can_recover_but_accepted_start_cannot(self):
        stop = command("E_STOP", "accepted", age=18, accepted_at=utc(time.time() - 10))
        self.assertTrue(valid_command(stop, "r1", time.time(), accepted_stop=True))
        # 수락 후 복구 기간은 createdAt + 10 s + 45 s까지다.
        self.assertTrue(valid_command(command("E_STOP", "accepted", age=54,
                                              accepted_at=utc(time.time() - 50)),
                                      "r1", time.time(), accepted_stop=True))
        self.assertFalse(valid_command(command("E_STOP", "accepted", age=56,
                                               accepted_at=utc(time.time() - 50)),
                                       "r1", time.time(), accepted_stop=True))
        self.assertFalse(valid_command(command("E_STOP", "accepted", age=65,
                                               accepted_at=utc(time.time() - 60)),
                                       "r1", time.time(), accepted_stop=True))
        self.assertFalse(valid_command(command("START", "accepted", accepted_at=utc()),
                                       "r1", time.time(), accepted_stop=True))

    def test_cas_transition_only_updates_status_with_server_time(self):
        signaling = FakeSignaling()
        store = RemoteCommandStore({"robotId": "r1", "senderUid": "sender"}, signaling)
        doc = command()
        accepted = store.claim(doc)
        store.finish(accepted, True, "start_sent")
        first, second = [call[2]["writes"][0] for call in signaling.calls]
        self.assertEqual(first["currentDocument"], {"updateTime": doc["updateTime"]})
        self.assertEqual(first["updateMask"]["fieldPaths"], ["state"])
        self.assertEqual(first["updateTransforms"][0]["fieldPath"], "acceptedAt")
        self.assertEqual(second["updateMask"]["fieldPaths"], ["state", "resultCode"])
        self.assertEqual(second["updateTransforms"][0]["fieldPath"], "finishedAt")

    def test_stop_query_pages_past_100_with_descending_stable_cursor(self):
        signaling = FakeStopQuerySignaling()
        store = RemoteCommandStore({"robotId": "r1", "senderUid": "sender"}, signaling)
        self.assertEqual(len(list(store.stop_commands())), 101)
        first_query = signaling.calls[0][2]["structuredQuery"]
        self.assertEqual([row["direction"] for row in first_query["orderBy"]],
                         ["DESCENDING", "DESCENDING"])
        self.assertEqual(first_query["where"]["compositeFilter"]["filters"][0]
                         ["fieldFilter"]["value"], {"stringValue": "E_STOP"})
        self.assertFalse(signaling.cursor["before"])
        self.assertEqual(signaling.cursor["values"][1]["referenceValue"],
                         signaling.rows[99]["name"])

    def test_stop_is_queued_before_start_and_start_is_not_replayed(self):
        store = FakeStore([command("START"), command("E_STOP", command_id="c2")])
        worker = object.__new__(RemoteControlWorker)
        worker._store = store
        worker._ready = lambda: True
        worker._requests = queue.Queue(maxsize=32)
        worker._inflight = {}
        worker._stop = threading.Event()
        worker._poll()
        self.assertEqual(worker.next_request()[1], "E_STOP")
        self.assertEqual(worker.next_request()[1], "START")

        store = FakeStore([command("START", "accepted", accepted_at=utc())])
        worker._store = store
        worker._inflight = {}
        worker._poll()
        self.assertEqual(store.claimed, [])
        self.assertEqual(store.finished[0][2], "start_outcome_unknown")

    def test_stop_survives_more_than_100_newer_start_commands(self):
        starts = [command("START", command_id=f"s{i:03d}") for i in range(120)]
        stop = command("E_STOP", command_id="urgent")
        store = FakeStore(starts, stop_docs=[stop])
        self.assertNotIn(stop, store.commands())
        worker = object.__new__(RemoteControlWorker)
        worker._store = store
        worker._ready = lambda: True
        worker._requests = queue.Queue(maxsize=32)
        worker._inflight = {}
        worker._stop = threading.Event()
        worker._poll()
        self.assertEqual(worker.next_request(), ("urgent", "E_STOP"))
        self.assertIn(stop["path"], store.claimed)

    def test_stop_is_accepted_when_the_jetson_clock_is_hours_behind(self):
        # createdAt은 서버 시각이다. Jetson 시계 대신 저장소의 서버 기준 now()로 판정한다.
        server_now = time.time() + 3 * 3600
        stop = command("E_STOP", command_id="skewed", age=-3 * 3600)
        store = FakeStore([], stop_docs=[stop])
        store.now = lambda: server_now
        worker = object.__new__(RemoteControlWorker)
        worker._store = store
        worker._ready = lambda: True
        worker._requests = queue.Queue(maxsize=32)
        worker._inflight = {}
        worker._stop = threading.Event()
        worker._poll()
        self.assertEqual(worker.next_request(), ("skewed", "E_STOP"))
        self.assertEqual(store.finished, [])

    def test_result_write_failure_does_not_block_stop_polling(self):
        worker = object.__new__(RemoteControlWorker)
        stop = command("E_STOP", command_id="urgent")

        class FailingFinishStore(FakeStore):
            def finish(self, doc, success, code):
                raise SignalingError("http_503")

            def commands(self):
                # One loop iteration is enough for this check.
                worker._stop.set()
                return []

        store = FailingFinishStore([], stop_docs=[stop])
        worker._store = store
        worker._ready = lambda: True
        worker._requests = queue.Queue(maxsize=32)
        worker._results = queue.Queue(maxsize=32)
        worker._inflight = {"old": command("START", "accepted", command_id="old")}
        worker._stop = threading.Event()
        worker.last_error = None
        worker.complete("old", True, "start_sent")
        worker._run()
        # The failing result write is retried later; the new stop was still claimed.
        self.assertEqual(worker.next_request(), ("urgent", "E_STOP"))
        self.assertEqual(worker.last_error, "http_503")
        self.assertEqual(worker._results.get_nowait(), ("old", True, "start_sent"))


if __name__ == "__main__":
    unittest.main()
