import math
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
from node_runtime.live_dashboard import (LiveDashboardClient, LatestTelemetryWorker,
                                         TelemetrySnapshot, encode_value)
from node_runtime.webrtc_signaling import SignalingError, decode, utc

CONFIG = dict(projectId="project", robotId="robot", senderId="sender", senderUid="sender",
              apiKey="test-placeholder", refreshToken="test-placeholder", storageBucket="project.test")


class LiveDashboardTests(unittest.TestCase):
    def test_values_round_trip_and_nonfinite_rejected(self):
        values = {"camera": {"fps": 12.2, "ms": None}, "topics": ["/camera"], "count": 3}
        self.assertEqual(decode(encode_value(values)), values)
        with self.assertRaises(ValueError):
            encode_value(math.nan)

    def test_identity_and_server_timestamp_cannot_be_overridden(self):
        client = LiveDashboardClient(CONFIG)
        client.signaling.request = Mock(return_value={})
        client.publish_capture({"captureId": "cap_1", "senderId": "other", "updatedAt": "fake"})
        method, path, body = client.signaling.request.call_args.args
        self.assertEqual((method, path), ("POST", ":commit"))
        write = body["writes"][0]
        self.assertTrue(write["update"]["name"].endswith("robots/robot/captures/cap_1"))
        self.assertEqual(write["update"]["fields"]["senderId"], {"stringValue": "sender"})
        self.assertNotIn("updatedAt", write["update"]["fields"])
        self.assertEqual(write["updateTransforms"], [{"fieldPath": "updatedAt", "setToServerValue": "REQUEST_TIME"}])

    def test_upload_rejects_cross_robot_and_traversal_before_network(self):
        client = LiveDashboardClient(CONFIG)
        client.signaling._auth = Mock()
        for key in ("robots/other/captures/a.jpg", "robots/robot/captures/../a.jpg",
                    "robots/robot/captures/a%2fb.jpg", "robots/robot/captures//a.jpg"):
            with self.assertRaises(ValueError):
                client.upload_image("unused.jpg", key)
        client.signaling._auth.assert_not_called()

    def test_capture_requires_valid_bucket_but_telemetry_does_not(self):
        with patch.dict(os.environ, {"FIREBASE_STORAGE_BUCKET": ""}):
            for bucket in (None, "", "gs://project.test", "project.test/path", 123):
                with self.subTest(bucket=bucket):
                    client = LiveDashboardClient(dict(CONFIG, storageBucket=bucket))
                    client.signaling.request = Mock(return_value={})
                    client.publish_telemetry({"topics": [], "mission": None, "cameras": {}})
                    client.signaling.request.assert_called_once()
                    with self.assertRaises(ValueError):
                        client.require_capture_storage()
        with patch.dict(os.environ, {"FIREBASE_STORAGE_BUCKET": "project.firebasestorage.app"}):
            client = LiveDashboardClient(dict(CONFIG, storageBucket=""))
            self.assertEqual(client.require_capture_storage(), "project.firebasestorage.app")
        self.assertEqual(LiveDashboardClient(CONFIG).require_capture_storage(), "project.test")

    def test_upload_uses_sender_token_and_returns_private_object_key(self):
        client = LiveDashboardClient(CONFIG)
        client.signaling._auth = Mock()
        client.signaling._token = "test-id-token"
        client.signaling._raw = Mock(return_value={"downloadTokens": "must-not-be-returned"})
        key = "robots/robot/captures/cap_1/raw.jpg"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jpg"
            path.write_bytes(b"\xff\xd8\xff\xd9")
            self.assertEqual(client.upload_image(path, key), key)
            request = client.signaling._raw.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer test-id-token")
            self.assertEqual(request.get_header("Content-type"), "image/jpeg")
            self.assertEqual(request.data, path.read_bytes())
            client.signaling._token_until = 100
            client.signaling._raw.side_effect = SignalingError("http_401")
            with self.assertRaises(SignalingError):
                client.upload_image(path, key)
            self.assertEqual(client.signaling._token_until, 0)

    def test_upload_rejects_non_jpeg_without_authentication(self):
        client = LiveDashboardClient(CONFIG)
        client.signaling._auth = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jpg"
            path.write_bytes(b"not a JPEG")
            with self.assertRaises(ValueError):
                client.upload_image(path, "robots/robot/captures/cap_1/raw.jpg")
        client.signaling._auth.assert_not_called()

    def test_stale_model_does_not_refresh_with_camera_fps(self):
        snapshot = TelemetrySnapshot(5)
        snapshot.metric("side", "rustMs", 22.2, 100)
        snapshot.metric("side", "fps", 15, 107)
        camera = snapshot.payload([], now=107)["cameras"]["side"]
        self.assertIsNone(camera["rustMs"])
        self.assertEqual(camera["fps"], 15)
        self.assertEqual(camera["observedAt"], utc(107))
        snapshot.metric("side", "fps", 1, 90)
        self.assertEqual(snapshot.payload([], now=107)["cameras"]["side"]["fps"], 15)

    def test_mission_absent_until_observed_and_uses_actual_counts(self):
        snapshot = TelemetrySnapshot(5)
        self.assertIsNone(snapshot.payload([])["mission"])
        snapshot.mission_status("RESCAN", 5, 2, False, 100)
        mission = snapshot.payload([], now=107)["mission"]
        self.assertEqual((mission["completed"], mission["total"]), (7, 10))
        self.assertEqual(mission["observedAt"], utc(100))
        self.assertTrue(mission["remainingTasks"])
        snapshot.mission_status("COMPLETE", 5, 5, False, 101)
        self.assertEqual(snapshot.payload([])["mission"]["remainingTasks"], [])

    def test_failed_and_out_of_order_mission_preserve_observation_time(self):
        snapshot = TelemetrySnapshot(5)
        snapshot.mission_status("RESCAN", 5, 2, True, 100)
        snapshot.mission_status("COMPLETE", 5, 5, False, 99)
        mission = snapshot.payload([], now=200)["mission"]
        self.assertEqual((mission["phase"], mission["completed"]), ("FAILED", 7))
        self.assertEqual(mission["observedAt"], utc(100))

    def test_graph_discovery_only_active_bounded_metadata(self):
        topics = [(f"/topic{i}", 1, ["Image"] * 20) for i in range(150)] + [("/idle", 0, ["String"])]
        payload = TelemetrySnapshot(5).payload(topics)
        self.assertEqual(len(payload["topics"]), 128)
        self.assertTrue(all(len(t["types"]) == 8 and t["publishers"] > 0 for t in payload["topics"]))

    def test_health_requires_fresh_direct_observation(self):
        snapshot = TelemetrySnapshot(5)
        unknown = {item["id"]: item for item in snapshot.payload([], now=100)["nodeHealth"]}
        self.assertEqual(len(unknown), 11)
        self.assertTrue(all(item["status"] == "unknown" for item in unknown.values()))
        snapshot.health_signal("camera_side", "ok", 100)
        snapshot.health_signal("stm_bridge", "fault", 100)
        current = {item["id"]: item for item in snapshot.payload([], now=102)["nodeHealth"]}
        self.assertEqual(current["camera_side"]["status"], "ok")
        self.assertEqual(current["stm_bridge"]["status"], "fault")
        self.assertEqual(current["cleaner_front"]["status"], "unknown")
        self.assertEqual(current["pump_side"]["observedAt"], None)
        old = {item["id"]: item for item in snapshot.payload([], now=107)["nodeHealth"]}
        self.assertEqual(old["camera_side"]["status"], "stale")
        self.assertEqual(old["stm_bridge"]["status"], "stale")
        self.assertEqual(old["camera_side"]["observedAt"], utc(100))

    def test_health_rejects_unapproved_or_out_of_order_observations(self):
        snapshot = TelemetrySnapshot(5)
        snapshot.health_signal("encoder_left", "observed", 100)
        snapshot.health_signal("encoder_left", "fault", 99)
        snapshot.health_signal("cleaner_front", "ok", 101)
        snapshot.health_signal("arbitrary_node", "ok", 101)
        snapshot.health_signal("encoder_right", "ok", math.nan)
        states = {item["id"]: item for item in snapshot.payload([], now=101)["nodeHealth"]}
        self.assertEqual(states["encoder_left"]["status"], "observed")
        self.assertEqual(states["encoder_right"]["status"], "unknown")
        self.assertEqual(states["cleaner_front"]["status"], "unknown")
        self.assertNotIn("arbitrary_node", states)

    def test_worker_overwrites_pending_snapshot_without_network_in_submit(self):
        factory = Mock()
        # Delay worker entirely to prove submit does not authenticate or publish.
        with patch.object(threading.Thread, "start"):
            worker = LatestTelemetryWorker(factory)
        worker.submit({"sequence": 1})
        second = {"sequence": 2}
        worker.submit(second)
        second["sequence"] = 3
        self.assertEqual(worker._pending, {"sequence": 2})
        factory.assert_not_called()

    def test_failed_write_retries_latest_snapshot_and_sanitizes_error(self):
        client = Mock()
        with patch.object(threading.Thread, "start"):
            worker = LatestTelemetryWorker(lambda: client)
        worker._stop = Mock()
        worker._stop.is_set.return_value = False
        worker._stop.wait.side_effect = [False, False, True]
        worker.submit({"sequence": 1})

        def interrupted_send(payload):
            if payload["sequence"] == 1:
                worker.submit({"sequence": 2})
                raise RuntimeError("secret-refresh-token-must-not-be-logged")
            self.assertEqual(worker.last_error, "RuntimeError")

        client.publish_telemetry.side_effect = interrupted_send
        worker._run()
        self.assertEqual([c.args[0] for c in client.publish_telemetry.call_args_list],
                         [{"sequence": 1}, {"sequence": 2}])
        self.assertIsNone(worker.last_error)

    def test_ros_observer_hands_off_without_network_and_preserves_source_clocks(self):
        # Minimal ROS doubles keep this contract runnable without ROS/CUDA/OpenCV.
        class Node:
            def __init__(self, *args, **kwargs):
                self.subscriptions = {}
                self.now = 107.0

            def create_subscription(self, message_type, topic, callback, qos):
                self.subscriptions[topic] = callback

            def create_timer(self, period, callback):
                self.period = period

            def get_clock(self):
                return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self.now * 1e9)))

            def get_topic_names_and_types(self):
                return [("/camera/side", ["Image"]), ("/idle", ["String"])]

            def count_publishers(self, topic):
                return int(topic != "/idle")

            def destroy_node(self):
                pass

        replacements = {}
        for name, values in {
            "rclpy.node": dict(Node=Node),
            "std_msgs.msg": dict(Float64=object, String=object),
            "rail_robot_interfaces.msg": dict(EncoderState=object, MissionStatus=object, ModelTiming=object),
            "node_runtime.ros.common": dict(qos_sensor=object(), qos_events=object(), qos_state=object(),
                                           stamp_seconds=lambda stamp: stamp.sec + stamp.nanosec / 1e9),
            "node_runtime.settings": dict(AUTOMATIC_RAIL_SECTION_TARGET=5),
        }.items():
            replacements[name] = ModuleType(name)
            replacements[name].__dict__.update(values)
        path = Path(__file__).resolve().parents[1] / "jetson_code/node_runtime/ros/dashboard_telemetry_node.py"
        spec = importlib.util.spec_from_file_location("node_runtime.ros._telemetry_contract_test", path)
        module = importlib.util.module_from_spec(spec)
        worker = Mock(last_error=None)
        with patch.dict(sys.modules, replacements):
            spec.loader.exec_module(module)
        with patch.object(module, "LatestTelemetryWorker", return_value=worker):
            node = module.DashboardTelemetryNode()
        header = SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=0))
        node.subscriptions["debug/vision/timing"](SimpleNamespace(
            camera_role="top", model="crack", processing_ms=20, header=header))
        node.subscriptions["debug/camera/front/fps"](SimpleNamespace(data=15))
        node.subscriptions["mission/state"](SimpleNamespace(
            mode="RESCAN", initial_count=5, rescan_count=2, failed=False, error="", header=header))
        node.subscriptions["runtime/stm_bridge_health"](SimpleNamespace(data="ready"))
        node.subscriptions["debug/encoder/state"](SimpleNamespace(feedback_valid=True, error=""))
        worker.submit.assert_not_called()
        node._publish()
        payload = worker.submit.call_args.args[0]
        self.assertEqual(node.period, 2.0)
        self.assertEqual(payload["cameras"]["front"]["fps"], 15)
        self.assertIsNone(payload["cameras"]["front"]["crackMs"])
        self.assertEqual(payload["cameras"]["front"]["observedAt"], utc(107))
        self.assertEqual(payload["mission"]["observedAt"], utc(100))
        self.assertEqual(payload["mission"]["completed"], 7)
        self.assertEqual([topic["name"] for topic in payload["topics"]], ["/camera/side"])
        health = {item["id"]: item["status"] for item in payload["nodeHealth"]}
        self.assertEqual(health["stm_bridge"], "ok")
        self.assertEqual(health["encoder_left"], "observed")
        self.assertEqual(health["pump_side"], "unknown")
        node.prepare_shutdown()
        node._publish()
        node.destroy_node()
        worker.close.assert_called_once()
        worker.submit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
