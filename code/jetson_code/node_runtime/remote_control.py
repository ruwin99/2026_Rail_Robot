"""Bounded Firestore command polling, separate from ROS callbacks."""
from __future__ import annotations

import os
from pathlib import Path
import queue
import re
import threading
import time

from .webrtc_signaling import FirestoreSignaling, SignalingError, encode, server_time, timestamp, utc


COMMAND_FIELDS = frozenset({
    "schemaVersion", "robotId", "ownerUid", "action", "state", "createdAt",
    "acceptedAt", "finishedAt", "resultCode",
})
# 수명은 브라우저 시계가 아니라 서버 createdAt 기준이다. 예전 웹의 expiresAt
# 필드는 전환 기간이 끝나 더 이상 받지 않는다(필드가 다르면 형식 오류).
COMMAND_TTL_SECONDS = 10


def command_expiry(data, fields=COMMAND_FIELDS):
    """Return the server-time expiry of an exact request shape."""
    if set(data) != fields:
        raise ValueError("Invalid request shape")
    return timestamp(data["createdAt"]) + COMMAND_TTL_SECONDS
RESULT_CODES = frozenset({
    "start_sent", "start_rejected", "start_unavailable", "start_timeout",
    "start_outcome_unknown", "mission_not_ready", "remote_stop_requested",
    "estop_ack", "estop_rejected", "estop_unavailable", "estop_unconfirmed",
    "expired", "invalid_command", "unauthorized", "clock_not_synced",
})


def robot_allows(robot, sender_uid, owner_uid):
    """The sender rechecks authorization even after Firestore rules did."""
    return (
        isinstance(robot, dict)
        and robot.get("enabled") is True
        and robot.get("remoteControlEnabled") is True
        and robot.get("senderId") == sender_uid
        and isinstance(robot.get("operatorUids"), list)
        and owner_uid in robot["operatorUids"]
    )


def valid_command(document, robot_id, now, *, accepted_stop=False):
    """Accept only a short-lived, exact command shape for this robot."""
    try:
        data = document["data"]
        created, expires = timestamp(data["createdAt"]), command_expiry(data)
        if not (
            type(data["schemaVersion"]) is int and data["schemaVersion"] == 1
            and data["robotId"] == robot_id
            and isinstance(data["ownerUid"], str) and 0 < len(data["ownerUid"]) <= 128
            and data["action"] in ("START", "E_STOP")
            and data["state"] in ("requested", "accepted")
            and data["finishedAt"] is None and data["resultCode"] is None
            and created <= now + 2
        ):
            return False
        if data["state"] == "requested":
            return data["acceptedAt"] is None and now < expires
        # A previously accepted stop may be retried after a process crash.
        return (accepted_stop and data["action"] == "E_STOP"
                and data["acceptedAt"] is not None
                and created <= timestamp(data["acceptedAt"]) <= expires
                and now <= expires + 45)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _expired(data, now):
    try:
        return command_expiry(data) <= now
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


class RemoteCommandStore:
    """Synchronous REST owner; call only from RemoteControlWorker's thread."""

    def __init__(self, config, signaling=None):
        self.robot_id = config["robotId"]
        self.sender_uid = config["senderUid"]
        self.signaling = signaling or FirestoreSignaling(config)
        self.root = f"robots/{self.robot_id}"
        self.prefix = self.signaling.base.split("/v1/", 1)[1] + "/"

    def robot(self):
        return self.signaling.get(self.root)["data"]

    def now(self):
        # 요청의 createdAt은 서버 시각이다. RTC가 없는 Jetson 시계 대신 서버 기준으로 수명을 판정한다.
        return server_time(self.signaling)

    def commands(self):
        # Query only the 60-second recovery horizon; old command history must
        # not cost 100 document reads on every short poll.
        result = self.signaling.request(
            "POST", self.root + ":runQuery", {"structuredQuery": {
                "from": [{"collectionId": "commands"}],
                "where": {"fieldFilter": {
                    "field": {"fieldPath": "createdAt"},
                    "op": "GREATER_THAN_OR_EQUAL",
                    "value": {"timestampValue": utc(self.now() - 60)},
                }},
                "orderBy": [{"field": {"fieldPath": "createdAt"}, "direction": "DESCENDING"}],
                "limit": 100,
            }},
        )
        return [self.signaling.document(row["document"]) for row in result if "document" in row]

    def stop_commands(self):
        """Page every recent stop separately so a START burst cannot hide it."""
        threshold = utc(self.now() - 60)
        cursor = None
        while True:
            query = {
                "from": [{"collectionId": "commands"}],
                "where": {"compositeFilter": {"op": "AND", "filters": [
                    {"fieldFilter": {"field": {"fieldPath": "action"},
                                     "op": "EQUAL", "value": {"stringValue": "E_STOP"}}},
                    {"fieldFilter": {"field": {"fieldPath": "createdAt"},
                                     "op": "GREATER_THAN_OR_EQUAL",
                                     "value": {"timestampValue": threshold}}},
                ]}},
                "orderBy": [
                    {"field": {"fieldPath": "createdAt"}, "direction": "DESCENDING"},
                    {"field": {"fieldPath": "__name__"}, "direction": "DESCENDING"},
                ],
                "limit": 100,
            }
            if cursor is not None:
                query["startAt"] = {"values": cursor, "before": False}
            result = self.signaling.request("POST", self.root + ":runQuery",
                                            {"structuredQuery": query})
            rows = [row["document"] for row in result if "document" in row]
            for row in rows:
                yield self.signaling.document(row)
            if len(rows) < 100:
                return
            last = rows[-1]
            cursor = [last["fields"]["createdAt"], {"referenceValue": last["name"]}]

    def _transition(self, document, state, result_code=None):
        if state not in ("accepted", "succeeded", "failed"):
            raise ValueError("Invalid command state")
        if result_code is not None and result_code not in RESULT_CODES:
            raise ValueError("Invalid command result code")
        values = {"state": encode(state)}
        paths = ["state"]
        if result_code is not None:
            values["resultCode"] = encode(result_code)
            paths.append("resultCode")
        server_time = "acceptedAt" if state == "accepted" else "finishedAt"
        write = {
            "update": {"name": self.prefix + document["path"], "fields": values},
            "updateMask": {"fieldPaths": paths},
            "currentDocument": {"updateTime": document["updateTime"]},
            "updateTransforms": [{"fieldPath": server_time, "setToServerValue": "REQUEST_TIME"}],
        }
        result = self.signaling.request("POST", ":commit", {"writes": [write]})
        next_doc = dict(document)
        next_doc["updateTime"] = result["writeResults"][0]["updateTime"]
        next_doc["data"] = dict(document["data"], state=state, resultCode=result_code)
        return next_doc

    def claim(self, document):
        return self._transition(document, "accepted")

    def finish(self, document, success, result_code):
        return self._transition(document, "succeeded" if success else "failed", result_code)


class RemoteControlWorker:
    """One network thread. ROS node communicates through two bounded queues."""

    def __init__(self, mission_ready, config_path=None, store_factory=RemoteCommandStore):
        path = config_path or os.environ.get("RAIL_ROBOT_WEBRTC_CONFIG") or "~/.config/rail_robot/webrtc.json"
        config = FirestoreSignaling.load(path)
        self._store = store_factory(config)
        # A second local consumer could mark the first one's accepted START as
        # unknown during its audio wait. Keep one consumer per robot per Jetson.
        import fcntl
        lock_dir = Path.home() / ".cache" / "rail_robot"
        lock_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(lock_dir / (self._store.robot_id + "-remote-control.lock"), "a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise
        self._ready = mission_ready
        self._requests = queue.Queue(maxsize=32)
        self._results = queue.Queue(maxsize=32)
        self._inflight = {}
        self._stop = threading.Event()
        self.last_error = None
        self._thread = threading.Thread(target=self._run, name="dashboard-control", daemon=True)
        self._thread.start()

    def next_request(self):
        try:
            return self._requests.get_nowait()
        except queue.Empty:
            return None

    def complete(self, command_id, success, code):
        if code not in RESULT_CODES:
            raise ValueError("Invalid command result code")
        self._results.put_nowait((command_id, success, code))

    def _finish_results(self):
        while True:
            try:
                command_id, success, code = self._results.get_nowait()
            except queue.Empty:
                return
            document = self._inflight.get(command_id)
            if document is None:
                continue
            try:
                self._store.finish(document, success, code)
            except SignalingError:
                # Retry the status update, never the ROS action.
                self._results.put_nowait((command_id, success, code))
                raise
            self._inflight.pop(command_id, None)

    def _poll(self):
        robot = None

        def process(documents):
            nonlocal robot
            for document in documents:
                if self._stop.is_set():
                    return False
                data = document["data"]
                if data.get("state") not in ("requested", "accepted"):
                    continue
                if robot is None:
                    robot = self._store.robot()
                    if self._stop.is_set():
                        return False
                # 조회 응답이 갱신한 서버 시각으로 문서마다 판정한다.
                self._process_command(document, robot, self._store.now())
            return True

        # The dedicated STOP query is paged and processed before the bounded
        # recent-command query; over 100 STARTs cannot crowd out an E_STOP.
        if not process(self._store.stop_commands()):
            return
        documents = self._store.commands()
        documents.sort(key=lambda item: item["data"].get("action") != "E_STOP")
        process(documents)

    def _process_command(self, document, robot, now):
        data = document["data"]
        state = data.get("state")
        command_id = document["path"].rsplit("/", 1)[-1]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", command_id) or command_id in self._inflight:
            return
        authorized = robot_allows(robot, self._store.sender_uid, data.get("ownerUid"))
        valid = valid_command(document, self._store.robot_id, now, accepted_stop=state == "accepted")
        if state == "accepted" and data.get("action") == "START":
            # A crash may have happened after START was sent: never replay it.
            self._store.finish(document, False, "start_outcome_unknown")
            return
        if not authorized or not valid:
            if state == "requested":
                code = ("unauthorized" if not authorized
                        else "expired" if _expired(data, now) else "invalid_command")
                self._store.finish(document, False, code)
            elif state == "accepted" and data.get("action") == "E_STOP" and not valid:
                self._store.finish(document, False, "estop_unconfirmed")
            return
        if data["action"] == "START" and not self._ready():
            self._store.finish(document, False, "mission_not_ready")
            return
        if self._requests.full():
            return
        if state == "requested":
            document = self._store.claim(document)
        self._inflight[command_id] = document
        self._requests.put_nowait((command_id, data["action"]))

    def _run(self):
        while not self._stop.is_set():
            # 명령 조회(E_STOP 우선)를 결과 기록보다 먼저, 서로 다른 예외 경계에서
            # 실행한다. 결과 기록이 계속 실패해도 새 E_STOP 조회는 막히지 않는다.
            errors = []
            for step in (self._poll, self._finish_results):
                try:
                    step()
                except Exception as exc:
                    # URLs, documents and credentials must not enter logs.
                    errors.append(str(exc) if isinstance(exc, SignalingError) else type(exc).__name__)
            self.last_error = errors[0] if errors else None
            self._stop.wait(0.5)
        # Shutdown may follow an E_STOP runtime fault. Persist the outcome
        # already reported by ROS before releasing the worker where possible.
        try:
            self._finish_results()
        except Exception:
            pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=11.0)
        if not self._thread.is_alive():
            self._lock_file.close()
