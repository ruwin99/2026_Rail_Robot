"""Opt-in, always-on Firestore receiver for launching the ROS mission process.

This process owns no UART or ROS publisher. It can only start a fixed systemd
unit; the existing, separate mission START request is still required for motion.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import time

from .remote_control import command_expiry
from .webrtc_signaling import FirestoreSignaling, SignalingError, encode, server_time, timestamp, utc


# 원격 명령과 같은 계약: 서버 createdAt + 10 s가 수명이다(remote_control.command_expiry).
REQUEST_FIELDS = frozenset({
    "schemaVersion", "robotId", "ownerUid", "action", "state", "createdAt",
    "acceptedAt", "finishedAt", "resultCode",
})
# 이 유닛 상태에서는 새 실행을 허용한다. failed는 시작 직전에 reset-failed로 지운다.
LAUNCHABLE_STATES = ("inactive", "failed")
# 요청 수명(최대 16 s)과 수락 후 복구를 덮는 조회 범위. 오래된 이력은 읽지 않는다.
REQUEST_QUERY_WINDOW_SECONDS = 60
RESULT_CODES = frozenset({
    "launch_started", "already_running", "unit_unavailable", "launch_failed",
    "launch_outcome_unknown", "expired", "invalid_request", "unauthorized",
})
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def launch_allowed(robot, sender_uid, owner_uid):
    return (isinstance(robot, dict) and robot.get("enabled") is True
            and robot.get("remoteLaunchEnabled") is True
            and robot.get("senderId") == sender_uid
            and isinstance(robot.get("operatorUids"), list)
            and owner_uid in robot["operatorUids"])


def valid_launch_request(document, robot_id, now):
    try:
        data = document["data"]
        created = timestamp(data["createdAt"])
        # 정확한 필드 집합만 통과한다. 그 밖의 필드 조합은 ValueError다.
        expires = command_expiry(data, REQUEST_FIELDS)
        return (type(data["schemaVersion"]) is int and data["schemaVersion"] == 1
                and data["robotId"] == robot_id
                and isinstance(data["ownerUid"], str)
                and 0 < len(data["ownerUid"]) <= 128
                and data["action"] == "LAUNCH_MISSION"
                and data["state"] == "requested"
                and data["acceptedAt"] is None and data["finishedAt"] is None
                and data["resultCode"] is None
                and created <= now + 5 and now < expires)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


class LaunchRequestStore:
    def __init__(self, config, signaling=None):
        self.robot_id = config["robotId"]
        self.sender_uid = config["senderUid"]
        self.signaling = signaling or FirestoreSignaling(config)
        self.root = f"robots/{self.robot_id}"
        self.prefix = self.signaling.base.split("/v1/", 1)[1] + "/"

    def robot(self):
        return self.signaling.get(self.root)["data"]

    def now(self):
        # 요청 수명은 서버 createdAt 기준이다. Jetson 시계 대신 서버 기준 현재 시각을 쓴다.
        return server_time(self.signaling)

    def requests(self):
        # One-field filter/order avoids a composite index; the client checks action/TTL.
        # Only the recent window is read so old history does not cost reads every poll.
        result = self.signaling.request("POST", self.root + ":runQuery", {
            "structuredQuery": {
                "from": [{"collectionId": "launchRequests"}],
                "where": {"fieldFilter": {
                    "field": {"fieldPath": "createdAt"},
                    "op": "GREATER_THAN_OR_EQUAL",
                    "value": {"timestampValue": utc(self.now() - REQUEST_QUERY_WINDOW_SECONDS)},
                }},
                "orderBy": [{"field": {"fieldPath": "createdAt"}, "direction": "DESCENDING"}],
                "limit": 100,
            },
        })
        return [self.signaling.document(row["document"]) for row in result if "document" in row]

    def _transition(self, document, state, code=None):
        if state not in ("accepted", "succeeded", "failed"):
            raise ValueError("Invalid launch state")
        if code is not None and code not in RESULT_CODES:
            raise ValueError("Invalid launch result")
        values = {"state": encode(state)}
        paths = ["state"]
        if code is not None:
            values["resultCode"] = encode(code)
            paths.append("resultCode")
        stamp = "acceptedAt" if state == "accepted" else "finishedAt"
        write = {
            "update": {"name": self.prefix + document["path"], "fields": values},
            "updateMask": {"fieldPaths": paths},
            "currentDocument": {"updateTime": document["updateTime"]},
            "updateTransforms": [{"fieldPath": stamp, "setToServerValue": "REQUEST_TIME"}],
        }
        result = self.signaling.request("POST", ":commit", {"writes": [write]})
        next_doc = dict(document)
        next_doc["updateTime"] = result["writeResults"][0]["updateTime"]
        next_doc["data"] = dict(document["data"], state=state, resultCode=code)
        return next_doc

    def claim(self, document):
        return self._transition(document, "accepted")

    def finish(self, document, success, code):
        return self._transition(document, "succeeded" if success else "failed", code)

    def status(self, ready, mission_running):
        fields = {
            "schemaVersion": encode(1), "senderId": encode(self.sender_uid),
            "ready": encode(bool(ready)), "missionRunning": encode(bool(mission_running)),
        }
        write = {
            "update": {"name": self.prefix + self.root + "/launcher/status", "fields": fields},
            "updateTransforms": [{"fieldPath": "updatedAt", "setToServerValue": "REQUEST_TIME"}],
        }
        self.signaling.request("POST", ":commit", {"writes": [write]})


class MissionService:
    """The sole launch target is a fixed systemd user unit, never input text."""

    UNIT = "rail-mission.service"
    START_ARGS = ("systemctl", "--user", "start", UNIT)
    RESET_FAILED_ARGS = ("systemctl", "--user", "reset-failed", UNIT)
    SHOW_ARGS = ("systemctl", "--user", "show", UNIT,
                 "--property=LoadState", "--property=ActiveState")

    @staticmethod
    def inspect():
        result = subprocess.run(MissionService.SHOW_ARGS, capture_output=True,
                                text=True, timeout=5, check=False)
        if result.returncode != 0:
            return False, "unknown"
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        return values.get("LoadState") == "loaded", values.get("ActiveState", "unknown")

    @staticmethod
    def start():
        # 이전 실행이 비정상 종료해 failed로 남았으면 그 기록만 지운다. 결과와 무관하게
        # 아래 start 결과와 이후 inspect로 성공 여부를 판정한다.
        subprocess.run(MissionService.RESET_FAILED_ARGS, capture_output=True,
                       text=True, timeout=5, check=False)
        result = subprocess.run(MissionService.START_ARGS, capture_output=True,
                                text=True, timeout=10, check=False)
        if result.returncode != 0:
            raise RuntimeError("mission_unit_start_failed")

    @staticmethod
    def other_mission_running(proc_root=Path("/proc")):
        """Catch a manually started or partly orphaned robot launch stack."""
        core_nodes = (b"camera_node", b"vision_node", b"control_node",
                      b"stm_bridge_node", b"inspection_node", b"mission_node")
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                argv = (entry / "cmdline").read_bytes()[:4096].split(b"\0")
            except (OSError, PermissionError):
                continue
            if (any(arg.endswith(b"/rail_robot_nodes/" + node)
                    for arg in argv for node in core_nodes)
                    or (b"robot.launch.py" in argv and b"rail_robot_nodes" in argv)):
                return True
        return False


class MissionLauncher:
    def __init__(self, store, service):
        self.store = store
        self.service = service

    def publish_status(self):
        robot = self.store.robot()
        loaded, state = self.service.inspect()
        permitted = (robot.get("enabled") is True
                     and robot.get("remoteLaunchEnabled") is True
                     and robot.get("senderId") == self.store.sender_uid)
        running = (state in ("active", "activating", "reloading")
                   or self.service.other_mission_running())
        self.store.status(bool(permitted and loaded and state in LAUNCHABLE_STATES and not running), running)

    def poll(self):
        robot = self.store.robot()
        documents = self.store.requests()
        now = self.store.now()
        for document in documents:
            data = document["data"]
            state = data.get("state")
            if state not in ("requested", "accepted"):
                continue
            request_id = document["path"].rsplit("/", 1)[-1]
            if not SAFE_ID.fullmatch(request_id):
                continue
            if state == "accepted":
                # An earlier receiver may have started the service. Never retry.
                self.store.finish(document, False, "launch_outcome_unknown")
                continue
            authorized = launch_allowed(robot, self.store.sender_uid, data.get("ownerUid"))
            valid = valid_launch_request(document, self.store.robot_id, now)
            if not authorized or not valid:
                code = ("unauthorized" if not authorized else
                        "expired" if _expired(data, now) else "invalid_request")
                self.store.finish(document, False, code)
                continue
            document = self.store.claim(document)
            loaded, unit_state = self.service.inspect()
            if (unit_state in ("active", "activating", "reloading")
                    or self.service.other_mission_running()):
                self.store.finish(document, False, "already_running")
                continue
            if not loaded or unit_state not in LAUNCHABLE_STATES:
                self.store.finish(document, False, "unit_unavailable")
                continue
            try:
                self.service.start()
            except Exception:
                # A timed-out systemctl may still have started it; never retry.
                self.store.finish(document, False, "launch_outcome_unknown")
                continue
            loaded, unit_state = self.service.inspect()
            self.store.finish(document, unit_state in ("active", "activating"),
                              "launch_started" if unit_state in ("active", "activating")
                              else "launch_failed")


def _expired(data, now):
    try:
        return command_expiry(data, REQUEST_FIELDS) <= now
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def main():
    import fcntl
    path = os.environ.get("RAIL_ROBOT_WEBRTC_CONFIG") or "~/.config/rail_robot/webrtc.json"
    config = FirestoreSignaling.load(path)
    store = LaunchRequestStore(config)
    lock_dir = Path.home() / ".cache" / "rail_robot"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_dir / (store.robot_id + "-mission-launcher.lock"), "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher = MissionLauncher(store, MissionService())
        next_status = 0.0
        while True:
            try:
                if time.monotonic() >= next_status:
                    launcher.publish_status()
                    next_status = time.monotonic() + 5.0
                launcher.poll()
            except SignalingError:
                # No credentials or Firestore response content enters logs.
                pass
            except Exception:
                # A single malformed document cannot terminate the receiver.
                pass
            time.sleep(0.5)


if __name__ == "__main__":
    main()
