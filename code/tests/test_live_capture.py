"""Durability and stage ordering without Firebase, ROS, or Jetson hardware."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
from node_runtime.live_capture import LiveCaptureOutbox


class Client:
    robot_id = "test-robot"
    sender_id = "test-sender"

    def __init__(self):
        self.uploads = []
        self.events = []
        self.fail = False

    def upload_image(self, path, key):
        if self.fail:
            raise ConnectionError("do-not-log-secret")
        if not path.exists():
            raise FileNotFoundError(path)
        self.uploads.append(key)
        return key

    def publish_capture(self, payload):
        self.events.append(dict(payload))


class CaptureOutboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.jpg"
        self.raw.write_bytes(b"durable-raw")
        self.task = SimpleNamespace(raw_capture_path=self.raw, camera_role="top", phase="initial",
                                    phase_sequence=1, captured_at=datetime(2026, 9, 6, tzinfo=timezone.utc))
        self.client = Client()
        self.errors = []
        self.outbox = LiveCaptureOutbox(self.client, "run-one", self.root / "outbox", report_error=self.errors.append)

    def test_capture_publishes_before_analysis_and_retains_source_identity(self):
        self.outbox.captured(self.task)
        self.assertEqual(self.client.events, [])
        self.outbox.flush_once()
        raw = self.client.events[-1]
        self.assertEqual(raw["status"], "captured")
        self.assertEqual(raw["cameraRole"], "front")
        self.assertEqual(raw["capturedAt"], "2026-09-06T00:00:00+00:00")
        self.assertEqual(raw["processedObjectKeys"], [])
        self.outbox.processed(self.task, np.zeros((8, 16, 3), np.uint8))
        self.outbox.flush_once()
        done = self.client.events[-1]
        self.assertEqual(done["status"], "processed")
        self.assertEqual(done["captureId"], raw["captureId"])
        self.assertEqual(done["capturedAt"], raw["capturedAt"])
        self.assertIsNotNone(done["analyzedAt"])
        self.assertEqual(len(self.client.uploads), 2)
        self.assertEqual(list(self.outbox.directory.glob("*.json")), [])

    def test_offline_results_survive_restart_and_replay_both_stages_in_order(self):
        self.client.fail = True
        self.outbox.captured(self.task)
        self.outbox.processed(self.task, np.zeros((8, 16, 3), np.uint8))
        self.outbox.flush_once()
        self.assertEqual(len(list(self.outbox.directory.glob("*.json"))), 1)
        self.assertNotIn("do-not-log-secret", str(self.errors))
        restarted = LiveCaptureOutbox(self.client, "different-run", self.root / "outbox")
        self.client.fail = False
        restarted.flush_once()
        self.assertEqual([event["status"] for event in self.client.events], ["captured", "processed"])
        self.assertTrue(all(event["runId"] == "run-one" for event in self.client.events))

    def test_disk_capacity_failure_never_deletes_previous_pending_record(self):
        self.outbox.capacity = 1
        self.outbox.captured(self.task)
        other = SimpleNamespace(**vars(self.task))
        other.raw_capture_path = self.root / "second.jpg"
        with self.assertRaisesRegex(RuntimeError, "full"):
            self.outbox.captured(other)
        self.assertEqual(len(list(self.outbox.directory.glob("*.json"))), 1)
        self.assertTrue(self.raw.exists())

    def test_slow_network_does_not_hold_lock_needed_for_analysis_notification(self):
        entered, release = threading.Event(), threading.Event()
        original = self.client.upload_image
        def upload(path, key):
            entered.set()
            release.wait(5)
            return original(path, key)
        self.client.upload_image = upload
        self.outbox.captured(self.task)
        self.outbox.start()
        try:
            self.assertTrue(entered.wait(2))
            done = threading.Event()
            def analyze():
                self.outbox.processed(self.task, np.zeros((8, 16, 3), np.uint8))
                done.set()
            analysis = threading.Thread(target=analyze)
            analysis.start()
            self.assertTrue(done.wait(1), "Network upload blocked capture analysis")
            analysis.join(1)
        finally:
            release.set()
            self.outbox.close()

    def test_analysis_worker_notifies_per_task_without_a_mission_finalize(self):
        from node_runtime import inspection
        from node_runtime.data import CaptureAnalysisTask

        task = CaptureAnalysisTask(raw_capture_path=self.raw, camera_role="side", phase="initial",
                                   phase_sequence=1, trigger="CAMERA_CAPTURE", captured_at=self.task.captured_at)
        observer = Mock()
        image = np.zeros((8, 16, 3), np.uint8)
        worker = inspection.CaptureAnalysisWorker(Mock(), Mock(), self.root, live_capture_observer=observer)
        with patch.object(inspection.cv2, "imread", return_value=image), \
                patch.object(inspection, "capture_and_analyze", return_value=image):
            worker.start()
            worker.submit(task)
            self.assertTrue(worker.wait_until_idle(2))
            self.assertIsNone(worker.shutdown())
        self.assertEqual([call[0] for call in observer.mock_calls], ["captured", "processed"])
        self.assertIs(observer.processed.call_args.args[0], task)
        self.assertIs(observer.processed.call_args.args[1], image)

    def test_optional_outbox_failures_do_not_fault_capture_analysis(self):
        from node_runtime import inspection
        from node_runtime.data import CaptureAnalysisTask

        task = CaptureAnalysisTask(raw_capture_path=self.raw, camera_role="side", phase="initial",
                                   phase_sequence=1, trigger="CAMERA_CAPTURE", captured_at=self.task.captured_at)
        observer = Mock()
        observer.captured.side_effect = OSError("private-error-detail")
        observer.processed.side_effect = OSError("private-error-detail")
        image = np.zeros((8, 16, 3), np.uint8)
        worker = inspection.CaptureAnalysisWorker(Mock(), Mock(), self.root, live_capture_observer=observer)
        with patch.object(inspection.cv2, "imread", return_value=image), \
                patch.object(inspection, "capture_and_analyze", return_value=image) as analyze, \
                patch("sys.stderr") as errors:
            worker.start()
            worker.submit(task)
            self.assertTrue(worker.wait_until_idle(2))
            self.assertIsNone(worker.shutdown())
            analyze.assert_called_once()
            self.assertNotIn("private-error-detail", str(errors.mock_calls))
        self.assertIsNone(worker.failure_message())


if __name__ == "__main__":
    unittest.main()
