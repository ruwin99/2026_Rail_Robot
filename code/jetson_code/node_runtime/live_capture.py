"""Durable capture notifications; all network I/O stays on one retry worker.

The outbox references immutable raw/processed JPEGs in the normal capture tree.
Do not remove that tree while pending outbox records remain. A record is removed
only after both image objects and their Firestore notification are acknowledged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid


class LiveCaptureOutbox:
    def __init__(self, client, run_id, directory, *, report_error=print,
                 capacity=1000, retry_seconds=5.0):
        self.client = client
        self.run_id = run_id
        self.directory = Path(directory).expanduser() / client.robot_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.report_error = report_error
        self.capacity = capacity
        self.retry_seconds = retry_seconds
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="live-capture-upload", daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._stop.set()
        self._wake.set()
        self._thread.join(1.0)

    def _identity(self, task):
        return "cap_" + hashlib.sha256((self.run_id + str(task.raw_capture_path.resolve())).encode()).hexdigest()[:32]

    @staticmethod
    def _iso(value):
        # Legacy non-ROS callers can provide naive local datetimes.
        return value.astimezone(timezone.utc).isoformat()

    def _base(self, task):
        capture_id = self._identity(task)
        prefix = f"robots/{self.client.robot_id}/captures/{capture_id}"
        return {
            "captureId": capture_id, "runId": self.run_id,
            "cameraRole": "front" if task.camera_role == "top" else "side",
            "phase": task.phase, "zone": task.phase_sequence,
            "capturedAt": self._iso(task.captured_at), "analyzedAt": None,
            "status": "captured", "rawObjectKey": prefix + "/raw.jpg",
            "processedObjectKeys": [],
        }

    def _write(self, path, record):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)

    def captured(self, task):
        """Persist a small local notification, never wait for the network."""
        payload = self._base(task)
        path = self.directory / (payload["captureId"] + ".json")
        with self._lock:
            if path.exists():
                return
            if sum(1 for _ in self.directory.glob("*.json")) >= self.capacity:
                raise RuntimeError("Live capture outbox is full; raw capture remains on disk.")
            self._write(path, {"payload": payload, "rawPath": str(task.raw_capture_path.resolve()),
                               "processedPath": None, "rawPublished": False,
                               "revision": uuid.uuid4().hex})
        self._wake.set()

    def processed(self, task, image):
        import cv2

        # Keep this separate from report artifacts; retry uses the exact image
        # that belonged to this task, never a newer live camera frame.
        self.captured(task)
        capture_id = self._identity(task)
        processed_path = task.raw_capture_path.parent / (task.raw_capture_path.stem + "_live_processed.jpg")
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Could not encode live processed capture.")
        processed_path.write_bytes(encoded.tobytes())
        path = self.directory / (capture_id + ".json")
        with self._lock:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record["payload"]
            payload.update(status="processed", analyzedAt=self._iso(datetime.now(timezone.utc)),
                           processedObjectKeys=[{"label": "annotated", "objectKey":
                               f"robots/{self.client.robot_id}/captures/{capture_id}/processed.jpg"}])
            record.update(processedPath=str(processed_path.resolve()), revision=uuid.uuid4().hex)
            self._write(path, record)
        self._wake.set()

    def flush_once(self):
        """One bounded pass. Failures retain the record for the next retry."""
        for path in sorted(self.directory.glob("*.json"))[:self.capacity]:
            if self._stop.is_set():
                return
            try:
                with self._lock:
                    record = json.loads(path.read_text(encoding="utf-8"))
                payload = record["payload"]
                if not record["rawPublished"]:
                    self.client.upload_image(Path(record["rawPath"]), payload["rawObjectKey"])
                    raw = dict(payload, status="captured", analyzedAt=None, processedObjectKeys=[])
                    self.client.publish_capture(raw)
                    with self._lock:
                        latest = json.loads(path.read_text(encoding="utf-8"))
                        latest["rawPublished"] = True
                        self._write(path, latest)
                    record = latest
                    payload = record["payload"]
                if record["processedPath"] is None:
                    continue
                self.client.upload_image(Path(record["processedPath"]), payload["processedObjectKeys"][0]["objectKey"])
                self.client.publish_capture(payload)
                with self._lock:
                    latest = json.loads(path.read_text(encoding="utf-8"))
                    if latest["revision"] == record["revision"]:
                        path.unlink()
            except Exception as exc:
                # Do not print HTTP bodies or config; those can contain tokens.
                self.report_error(f"Live capture upload pending ({type(exc).__name__}); retained {path.name}")

    def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            self.flush_once()
            self._wake.wait(self.retry_seconds)
