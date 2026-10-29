"""Authenticated, bounded Firestore signaling; no media or Storage operations."""
from __future__ import annotations

import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class SignalingError(RuntimeError):
    pass


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp required")
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def utc(value=None):
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat().replace("+00:00", "Z")


def response_server_time(result):
    """Firestore server time carried by a REST response (commitTime/readTime), or None."""
    for item in (result if isinstance(result, list) else [result]):
        if isinstance(item, dict):
            for key in ("commitTime", "readTime"):
                if isinstance(item.get(key), str):
                    try:
                        return timestamp(item[key])
                    except ValueError:
                        return None
    return None


def server_time(clock=None):
    """Server-based current time from an object with server_now(); the local clock otherwise.

    RTC가 없는 Jetson은 부팅 직후나 NTP가 막힌 현장에서 시스템 시계가 크게 틀릴 수 있다.
    서버가 기록한 시각(createdAt, heartbeat, lease)과 비교할 때는 이 값을 쓴다.
    """
    reader = getattr(clock, "server_now", None)
    value = reader() if callable(reader) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return time.time()


def encode(value):
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {k: encode(v) for k, v in value.items()}}}
    if isinstance(value, str):
        return {"stringValue": value}
    raise ValueError("Unsupported signaling field")


TIME_FIELDS = {"createdAt", "viewerHeartbeatAt", "expiresAt", "endedAt"}


def fields(values):
    return {k: {"timestampValue": v} if k in TIME_FIELDS and v is not None else encode(v)
            for k, v in values.items()}


def decode(value):
    if "mapValue" in value:
        return {k: decode(v) for k, v in value["mapValue"].get("fields", {}).items()}
    if "integerValue" in value:
        return int(value["integerValue"])
    if "arrayValue" in value:
        return [decode(item) for item in value["arrayValue"].get("values", [])]
    return next(iter(value.values()), None)


def session_valid(document, robot_id, sender_id, allowed_uids, now):
    """Local allowlist supplements (never replaces) external Firestore rules."""
    try:
        d = document["data"]
        offer = d["offer"]
        heartbeat, expiry, created = (timestamp(d[k]) for k in
                                      ("viewerHeartbeatAt", "expiresAt", "createdAt"))
        return (type(d["schemaVersion"]) is int and d["schemaVersion"] == 1 and d["robotId"] == robot_id
                and d["senderId"] == sender_id and isinstance(d["ownerUid"], str)
                and (allowed_uids is None or d["ownerUid"] in allowed_uids)
                and d["state"] in ("requested", "negotiating", "connected")
                and 0 <= now - heartbeat < 30 and now < expiry <= created + 3660
                and created <= heartbeat and created < expiry
                and all(k in d for k in ("answer", "endedAt", "errorCode"))
                and d["endedAt"] is None and d["errorCode"] is None
                and offer["type"] == "offer" and isinstance(offer["sdp"], str)
                and 0 < len(offer["sdp"]) <= 100000
                and "m=video " in offer["sdp"] and "a=recvonly" in offer["sdp"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def context_valid(context, config, allowed_uids, now):
    try:
        robot, lock, doc = (context[k] for k in ("robot", "lock", "session"))
        r, lease, d = robot["data"], lock["data"], doc["data"]
        return (r["enabled"] is True and r["senderId"] == config["senderUid"] == config["senderId"]
                and isinstance(r["viewerUids"], list) and d["ownerUid"] in r["viewerUids"]
                and doc["path"] == f'robots/{config["robotId"]}/webrtcSessions/{lease["sessionId"]}'
                and lease["ownerUid"] == d["ownerUid"]
                and now < timestamp(lease["leaseUntil"]) <= now + 40
                and session_valid(doc, config["robotId"], config["senderUid"], allowed_uids, now))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def validate_candidate(value, offer_sdp=None):
    if value.get("endOfCandidates") is True:
        return "", 0
    candidate, index = value.get("candidate"), value.get("sdpMLineIndex")
    mid, fragment = value.get("sdpMid"), value.get("usernameFragment")
    if (not isinstance(candidate, str) or not candidate.startswith("candidate:")
            or len(candidate) > 8192
            or (index is not None and (type(index) is not int or not 0 <= index <= 255))
            or any(item is not None and (not isinstance(item, str) or len(item) > 256)
                   for item in (mid, fragment))):
        raise ValueError("Invalid ICE candidate")
    if index is None:
        sections = re.split(r"(?m)^m=", (offer_sdp or "").replace("\r", ""))[1:]
        matches = [i for i, section in enumerate(sections)
                   if mid is not None and f"a=mid:{mid}" in section.splitlines()]
        if len(matches) != 1:
            raise ValueError("ICE candidate has no matching media section")
        index = matches[0]
    return candidate, index


class FirestoreSignaling:
    """One synchronous I/O owner, called exclusively outside ROS callbacks."""

    def __init__(self, config):
        self.config = config
        for key in ("projectId", "robotId"):
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", config.get(key, "")):
                raise ValueError("Invalid signaling identity")
        if not config.get("apiKey") or not config.get("refreshToken") or not config.get("senderUid"):
            raise ValueError("Dedicated Firebase sender credentials required")
        uid = config["senderUid"]
        if not isinstance(uid, str) or not 0 < len(uid) <= 128 or config.get("senderId") != uid:
            raise ValueError("senderId must equal the authenticated senderUid")
        allowed = config.get("allowedViewerUids")
        if allowed is not None and (not isinstance(allowed, list)
                                   or not all(isinstance(x, str) and x for x in allowed)):
            raise ValueError("allowedViewerUids must be a list when supplied")
        self.allowed_uids = None if allowed is None else frozenset(allowed)
        self.base = f'https://firestore.googleapis.com/v1/projects/{config["projectId"]}/databases/(default)/documents'
        self.collection = f'robots/{config["robotId"]}/webrtcSessions'
        self._token = None
        self._token_until = 0
        # Latest Firestore server timestamp and the local monotonic instant it
        # corresponds to. Monotonic time never jumps with NTP or a reset RTC.
        self._server_epoch = None
        self._server_monotonic = 0.0

    def server_now(self):
        """Firestore server time estimated from the latest response, or None before any."""
        if self._server_epoch is None:
            return None
        return self._server_epoch + (time.monotonic() - self._server_monotonic)

    def now(self):
        """Server-based current time; the Jetson clock only until the first response."""
        return server_time(self)

    @staticmethod
    def load(path):
        file = Path(path).expanduser()
        if file.stat().st_size > 65536:
            raise ValueError("Configuration too large")
        return json.loads(file.read_text(encoding="utf-8"))

    def _raw(self, request):
        try:
            with urlopen(request, timeout=5) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise SignalingError("response_too_large")
            return json.loads(raw) if raw else {}
        except HTTPError as exc:
            # Never propagate response bodies, headers, SDP or credential URLs.
            raise SignalingError(f"http_{exc.code}") from None
        except SignalingError:
            raise
        except Exception:
            raise SignalingError("network_error") from None

    def _auth(self):
        if time.monotonic() < self._token_until:
            return
        body = urlencode({"grant_type": "refresh_token", "refresh_token": self.config["refreshToken"]}).encode()
        response = self._raw(Request("https://securetoken.googleapis.com/v1/token?" +
                                     urlencode({"key": self.config["apiKey"]}), data=body,
                                     headers={"Content-Type": "application/x-www-form-urlencoded"}))
        if response.get("user_id") != self.config["senderUid"] or not response.get("id_token"):
            raise SignalingError("sender_identity_mismatch")
        self._token = response["id_token"]
        lifetime = float(response.get("expires_in", 3600))
        if not math.isfinite(lifetime) or lifetime < 60:
            raise SignalingError("invalid_token_lifetime")
        self._token_until = time.monotonic() + min(lifetime - 60, 3300)

    def request(self, method, path, body=None, query=()):
        if method == "DELETE" or any("delete" in write for write in (body or {}).get("writes", [])):
            raise ValueError("Sender cannot delete Firestore history")
        self._auth()
        url = self.base + (path if path.startswith(":") else "/" + path if path else "")
        if query:
            url += "?" + urlencode(query)
        sent = time.monotonic()
        try:
            result = self._raw(Request(url, method=method,
                                      data=None if body is None else json.dumps(body).encode(),
                                      headers={"Authorization": "Bearer " + self._token,
                                               "Content-Type": "application/json"}))
        except SignalingError as exc:
            if str(exc) == "http_401":
                self._token_until = 0
            raise
        server = response_server_time(result)
        if server is not None:
            # The server stamped the response between send and receive; the
            # midpoint bounds the error by half the round trip.
            self._server_epoch = server
            self._server_monotonic = (sent + time.monotonic()) / 2
        return result

    @staticmethod
    def document(raw):
        return {"path": raw["name"].split("/documents/", 1)[1],
                "updateTime": raw["updateTime"],
                "data": {k: decode(v) for k, v in raw.get("fields", {}).items()}}

    def context(self, session_path=None):
        """Read a coherent robot/lease/session snapshot under a read-only transaction.

        Rules on robot/lock reads verify device_uploaders.enabled internally;
        the sender deliberately does not GET that otherwise-private document.
        """
        root = f'robots/{self.config["robotId"]}'
        prefix = self.base.split("/v1/", 1)[1] + "/"
        transaction = self.request("POST", ":beginTransaction", {"options": {"readOnly": {}}})["transaction"]
        try:
            result = self.request("POST", ":batchGet", {"documents": [prefix + root, prefix + root + "/webrtcLocks/active"],
                                                       "transaction": transaction})
            found = {self.document(x["found"])["path"]: self.document(x["found"])
                     for x in result if "found" in x}
            robot, lock = found.get(root), found.get(root + "/webrtcLocks/active")
            context = {"robot": robot, "lock": lock, "session": None}
            if not robot or not lock:
                return context
            session_id = lock["data"].get("sessionId", "")
            if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
                return context
            path = self.collection + "/" + session_id
            if session_path is not None and session_path != path:
                return context
            result = self.request("POST", ":batchGet", {"documents": [prefix + path], "transaction": transaction})
            context["session"] = next((self.document(x["found"]) for x in result if "found" in x), None)
            return context
        finally:
            try:
                self.request("POST", ":rollback", {"transaction": transaction})
            except SignalingError as exc:
                if str(exc) in ("http_401", "http_403", "sender_identity_mismatch"):
                    raise

    def get(self, path):
        return self.document(self.request("GET", path))

    def patch(self, document, values):
        if not values or set(values) - {"answer", "state", "endedAt", "errorCode"}:
            raise ValueError("Sender may update only sender-owned session fields")
        if values.get("answer") is not None:
            answer = values["answer"]
            if answer.get("type") != "answer" or not isinstance(answer.get("sdp"), str) or not 0 < len(answer["sdp"]) <= 100000:
                raise ValueError("Invalid answer")
        result = self._write(document["path"], values, {"updateTime": document["updateTime"]})
        return {"path": document["path"], "updateTime": result["writeResults"][0]["updateTime"],
                "data": dict(document["data"], **{k: result.get("commitTime") if k == "endedAt" and v is not None else v
                                                 for k, v in values.items()})}

    def _write(self, path, values, precondition):
        return self.request("POST", ":commit", {"writes": [self._write_entry(path, values, precondition)]})

    def _write_entry(self, path, values, precondition):
        server_times = [key for key in ("createdAt", "endedAt") if values.get(key) is not None]
        plain = {key: value for key, value in values.items() if key not in server_times}
        write = {"update": {"name": self.base.split("/v1/", 1)[1] + "/" + path, "fields": fields(plain)},
                 "currentDocument": precondition}
        if "updateTime" in precondition:
            write["updateMask"] = {"fieldPaths": list(plain)}
        if server_times:
            write["updateTransforms"] = [{"fieldPath": key, "setToServerValue": "REQUEST_TIME"} for key in server_times]
        return write

    def candidates(self, path):
        result = self.request("GET", path + "/viewerCandidates", query=[("pageSize", 129), ("orderBy", "createdAt asc")])
        docs = result.get("documents", [])
        if len(docs) > 128 or result.get("nextPageToken"):
            raise SignalingError("candidate_limit")
        return [self.document(d) for d in docs]

    def send_candidate(self, path, index, value):
        self._write(path + f"/senderCandidates/{index:03d}", value, {"exists": False})

    def send_candidates(self, path, start_index, values):
        if not 1 <= len(values) <= 4 or start_index < 0 or start_index + len(values) > 128:
            raise ValueError("Invalid candidate batch")
        writes = [self._write_entry(path + f"/senderCandidates/{start_index + offset:03d}",
                                   value, {"exists": False})
                  for offset, value in enumerate(values)]
        return self.request("POST", ":commit", {"writes": writes})
