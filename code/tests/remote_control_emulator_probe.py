"""Exercise the production Firestore command store against a local demo emulator.

Invoked by rail_dashboard/tests/remote-control-virtual.test.mjs. This probe
never instantiates a ROS node, opens UART, or contacts production Firebase.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))

from node_runtime.remote_control import RemoteCommandStore, robot_allows, valid_command  # noqa: E402
from node_runtime.webrtc_signaling import FirestoreSignaling  # noqa: E402


def main():
    project_id = os.environ["RAIL_VIRTUAL_PROJECT_ID"]
    robot_id = os.environ["RAIL_VIRTUAL_ROBOT_ID"]
    command_id = os.environ["RAIL_VIRTUAL_COMMAND_ID"]
    sender_uid = os.environ["RAIL_VIRTUAL_SENDER_UID"]
    host = os.environ["FIRESTORE_EMULATOR_HOST"]
    if (project_id != "demo-rail-remote-virtual" or robot_id != "virtual-robot"
            or command_id != "store_stop" or sender_uid != "virtual-sender"
            or host != "127.0.0.1:8087"):
        raise ValueError("Only the fixed local demo emulator is permitted")

    config = {"projectId": project_id, "robotId": robot_id,
              "senderUid": sender_uid, "senderId": sender_uid,
              "apiKey": "emulator-only", "refreshToken": "emulator-only"}
    signaling = FirestoreSignaling(config)
    signaling.base = (f"http://{host}/v1/projects/{project_id}"
                      "/databases/(default)/documents")
    signaling._token = os.environ["RAIL_VIRTUAL_SENDER_TOKEN"]
    signaling._token_until = time.monotonic() + 60
    store = RemoteCommandStore(config, signaling)

    robot = store.robot()
    stop_documents = list(store.stop_commands())
    command = next(doc for doc in stop_documents
                   if doc["path"] == f"robots/{robot_id}/commands/{command_id}")
    start = next(doc for doc in store.commands()
                 if doc["path"] == f"robots/{robot_id}/commands/store_start")
    if not robot_allows(robot, sender_uid, command["data"]["ownerUid"]):
        raise AssertionError("Virtual robot did not authorize the operator")
    if not valid_command(command, robot_id, time.time()) or not valid_command(start, robot_id, time.time()):
        raise AssertionError("Virtual command was not valid")

    claimed_start = store.claim(start)
    store.finish(claimed_start, True, "start_sent")
    claimed = store.claim(command)
    store.finish(claimed, True, "estop_ack")
    print("store_claim_and_finish_ok")


if __name__ == "__main__":
    main()
