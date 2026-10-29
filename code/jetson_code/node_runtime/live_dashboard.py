"""Optional bounded dashboard I/O. Never call network methods in ROS callbacks."""
from __future__ import annotations

from copy import deepcopy
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import quote, urlencode
from urllib.request import Request

from .webrtc_signaling import FirestoreSignaling, SignalingError, utc


def encode_value(value):
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float) and math.isfinite(value):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [encode_value(item) for item in value]}}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {key: encode_value(item) for key, item in value.items()}}}
    raise ValueError("Invalid dashboard value")


class LiveDashboardClient:
    """One worker owns each client; credentials are read only from private config."""

    def __init__(self, config):
        self.signaling = FirestoreSignaling(config)
        self.robot_id = config["robotId"]
        self.sender_id = config["senderUid"]
        self.bucket = config.get("storageBucket") or os.environ.get("FIREBASE_STORAGE_BUCKET", "")

    @classmethod
    def from_environment(cls):
        path = os.environ.get("RAIL_ROBOT_WEBRTC_CONFIG") or "~/.config/rail_robot/webrtc.json"
        return cls(FirestoreSignaling.load(path))

    def _publish(self, path, payload):
        values = dict(payload, schemaVersion=1, senderId=self.sender_id)
        values.pop("updatedAt", None)
        write = {"update": {
            "name": self.signaling.base.split("/v1/", 1)[1] + "/" + path,
            "fields": {key: encode_value(value) for key, value in values.items()}},
            "updateTransforms": [{"fieldPath": "updatedAt", "setToServerValue": "REQUEST_TIME"}]}
        return self.signaling.request("POST", ":commit", {"writes": [write]})

    def publish_telemetry(self, payload):
        return self._publish(f"robots/{self.robot_id}/telemetry/current", payload)

    def publish_capture(self, payload):
        capture_id = payload.get("captureId", "")
        if not isinstance(capture_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", capture_id):
            raise ValueError("Invalid capture ID")
        return self._publish(f"robots/{self.robot_id}/captures/{capture_id}", payload)

    def require_capture_storage(self):
        """Validate before starting a capture outbox; telemetry needs no bucket."""
        if (not isinstance(self.bucket, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{1,221}", self.bucket)):
            raise ValueError("storageBucket is required in private WebRTC config")
        return self.bucket

    def upload_image(self, path, object_key):
        prefix = f"robots/{self.robot_id}/captures/"
        if (not isinstance(object_key, str) or not object_key.startswith(prefix)
                or len(object_key) > 512 or not re.fullmatch(r"[A-Za-z0-9_./-]+", object_key)
                or any(part in ("", ".", "..") for part in object_key.split("/"))):
            raise ValueError("Image object must belong to this robot's capture prefix")
        self.require_capture_storage()
        file = Path(path)
        if not 0 < file.stat().st_size <= 10 * 1024 * 1024:
            raise ValueError("Image exceeds upload limit")
        with file.open("rb") as stream:
            content = stream.read(10 * 1024 * 1024 + 1)
        if len(content) > 10 * 1024 * 1024 or not content.startswith(b"\xff\xd8\xff"):
            raise ValueError("Only bounded JPEG captures are supported")
        self.signaling._auth()
        url = ("https://firebasestorage.googleapis.com/v0/b/" + quote(self.bucket, safe="")
               + "/o?" + urlencode({"uploadType": "media", "name": object_key}))
        try:
            self.signaling._raw(Request(url, method="POST", data=content, headers={
                "Authorization": "Bearer " + self.signaling._token, "Content-Type": "image/jpeg"}))
        except SignalingError as exc:
            if str(exc) == "http_401":
                self.signaling._token_until = 0
            raise
        return object_key


class LatestTelemetryWorker:
    """One pending snapshot, no growing queue; failed snapshots are retried."""

    def __init__(self, client_factory=LiveDashboardClient.from_environment):
        self._factory = client_factory
        self._pending = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.last_error = None
        self._thread = threading.Thread(target=self._run, name="dashboard-telemetry", daemon=True)
        self._thread.start()

    def submit(self, payload):
        with self._lock:
            if not self._stop.is_set():
                self._pending = deepcopy(payload)

    def _run(self):
        client = None
        while not self._stop.wait(2.0):
            with self._lock:
                payload, self._pending = self._pending, None
            if payload is None:
                continue
            try:
                client = client or self._factory()
                client.publish_telemetry(payload)
                self.last_error = None
            except Exception as exc:
                # Do not log request URLs, credentials or config contents.
                self.last_error = str(exc) if isinstance(exc, SignalingError) else type(exc).__name__
                with self._lock:
                    if self._pending is None:
                        self._pending = payload

    def close(self):
        self._stop.set()
        self._thread.join(timeout=11.0)


class TelemetrySnapshot:
    """Pure state adapter; timestamps remain observation times, not upload times."""

    HEALTH_IDS = (
        "camera_side", "camera_front", "vision", "mission", "stm_bridge",
        "encoder_left", "encoder_right", "cleaner_front", "cleaner_side",
        "pump_front", "pump_side",
    )
    # The mission UART protocol does not expose actuator-operation feedback.
    # Command requests and ROS publisher discovery must never imply device health.
    OBSERVABLE_HEALTH_IDS = frozenset(HEALTH_IDS[:7])
    HEALTH_STALE_SECONDS = {
        "camera_side": 5.0, "camera_front": 5.0, "vision": 5.0,
        "mission": 3.0, "stm_bridge": 3.0,
        "encoder_left": 3.0, "encoder_right": 3.0,
    }

    def __init__(self, section_target):
        self.section_target = section_target
        self.mission = None
        self._mission_stamp = 0.0
        self.metrics = {role: {} for role in ("side", "front")}
        self.health = {}

    def health_signal(self, node_id, status, observed_at):
        if (node_id not in self.OBSERVABLE_HEALTH_IDS
                or status not in ("ok", "observed", "fault", "unknown")
                or not isinstance(observed_at, (int, float))
                or not math.isfinite(observed_at) or observed_at <= 0):
            return
        previous = self.health.get(node_id)
        if previous is None or observed_at >= previous[1]:
            self.health[node_id] = (status, observed_at)

    def metric(self, role, name, value, observed_at):
        if role not in self.metrics or name not in ("fps", "rustMs", "crackMs", "obstacleMs"):
            return
        if math.isfinite(value) and value >= 0 and math.isfinite(observed_at) and observed_at > 0:
            old = self.metrics[role].get(name)
            if old is None or observed_at >= old[1]:
                self.metrics[role][name] = (float(value), observed_at)

    def mission_status(self, mode, initial, rescan, failed, observed_at):
        if not math.isfinite(observed_at) or observed_at <= 0:
            return
        if observed_at < self._mission_stamp:
            return
        self._mission_stamp = observed_at
        phases = ["CAPTURE_SCAN", "RETURN", "REALTIME", "RESCAN_RETURN", "RESCAN", "RESCAN_DONE", "COMPLETE"]
        labels = ["초기 구간 검사", "시작점 복귀", "청소 및 윤활", "재검사 시작점 복귀", "구간 재검사", "재검사 종료 확인", "임무 종료 확인"]
        remaining = labels[phases.index(mode):] if mode in phases and mode != "COMPLETE" else []
        if failed:
            remaining = ["오류 확인 및 임무 상태 점검"]
        self.mission = {"phase": "FAILED" if failed else mode,
                        "completed": int(initial) + int(rescan), "total": 2 * self.section_target,
                        "remainingTasks": remaining, "observedAt": utc(observed_at)}

    def payload(self, topics, now=None):
        now = time.time() if now is None else now
        cameras = {}
        for role, metrics in self.metrics.items():
            # Individual old timings must not look fresh when only FPS is alive.
            camera = {name: metrics[name][0] if name in metrics and 0 <= now - metrics[name][1] <= 5 else None
                      for name in ("fps", "rustMs", "crackMs", "obstacleMs")}
            camera["observedAt"] = utc(max(stamp for _, stamp in metrics.values())) if metrics else None
            cameras[role] = camera
        bounded_topics = [{"name": name[:256], "publishers": int(count), "types": [t[:256] for t in types[:8]]}
                          for name, count, types in sorted(topics) if count > 0][:128]
        node_health = []
        for node_id in self.HEALTH_IDS:
            observation = self.health.get(node_id)
            if observation is None:
                status, observed_at = "unknown", None
            else:
                status, observed_at = observation
                if now - observed_at > self.HEALTH_STALE_SECONDS[node_id] or now < observed_at:
                    status = "stale"
            detail = {
                "ok": "최근 직접 상태 신호 수신",
                "observed": "공통 MCU 피드백 수신, 개별 회전 미검증",
                "fault": "오류 또는 무효 신호 수신",
                "stale": "상태 신호 수신 지연",
                "unknown": ("작동 확인 피드백 없음" if node_id not in self.OBSERVABLE_HEALTH_IDS
                            else "상태 신호 미확인"),
            }[status]
            node_health.append({"id": node_id, "status": status,
                                "observedAt": utc(observed_at) if observed_at is not None else None,
                                "detail": detail})
        return {"topics": bounded_topics, "mission": deepcopy(self.mission),
                "cameras": cameras, "nodeHealth": node_health}
