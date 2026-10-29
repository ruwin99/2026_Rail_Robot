"""Exercise production launch request REST writes against a demo emulator only.

The caller seeds an authorized request and supplies an unsigned emulator-only
sender token. No systemctl, ROS, UART, real Jetson, or production Firebase use.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))

from node_runtime.mission_launcher import LaunchRequestStore, MissionLauncher  # noqa: E402
from node_runtime.webrtc_signaling import FirestoreSignaling  # noqa: E402


class FakeService:
    def __init__(self):
        self.state = "inactive"
        self.starts = 0

    def inspect(self):
        return True, self.state

    def other_mission_running(self):
        return False

    def start(self):
        self.starts += 1
        self.state = "active"


def main():
    project_id = os.environ["RAIL_VIRTUAL_PROJECT_ID"]
    robot_id = os.environ["RAIL_VIRTUAL_ROBOT_ID"]
    sender_uid = os.environ["RAIL_VIRTUAL_SENDER_UID"]
    host = os.environ["FIRESTORE_EMULATOR_HOST"]
    if (project_id != "demo-rail-launch" or robot_id != "robot"
            or sender_uid != "sender" or host != "127.0.0.1:8087"):
        raise ValueError("Only the fixed local demo emulator is permitted")
    config = {"projectId": project_id, "robotId": robot_id,
              "senderUid": sender_uid, "senderId": sender_uid,
              "apiKey": "emulator-only", "refreshToken": "emulator-only"}
    signal = FirestoreSignaling(config)
    signal.base = (f"http://{host}/v1/projects/{project_id}"
                   "/databases/(default)/documents")
    signal._token = os.environ["RAIL_VIRTUAL_SENDER_TOKEN"]
    signal._token_until = time.monotonic() + 60
    store = LaunchRequestStore(config, signal)
    service = FakeService()
    launcher = MissionLauncher(store, service)
    launcher.publish_status()
    launcher.poll()
    launcher.publish_status()
    document = signal.get("robots/robot/launchRequests/request_one")
    status = signal.get("robots/robot/launcher/status")
    if not (service.starts == 1 and document["data"]["state"] == "succeeded"
            and document["data"]["resultCode"] == "launch_started"
            and status["data"]["ready"] is False
            and status["data"]["missionRunning"] is True):
        raise AssertionError("Virtual launch contract failed")
    print("mission_launcher_store_and_fake_service_ok")


if __name__ == "__main__":
    main()
