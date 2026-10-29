"""Single-session coordinator. Network I/O never runs in ROS callbacks."""
import queue
import copy
import threading
import time
import logging

from .webrtc_signaling import context_valid, server_time, timestamp, utc, validate_candidate, SignalingError


def candidate_kind(candidate):
    """ICE 후보의 종류만 남긴다(주소·포트는 기록하지 않음). 예: host/udp, host/udp/mdns, srflx/udp."""
    parts = candidate.split() if isinstance(candidate, str) else []
    if len(parts) < 8:
        return "end" if not candidate else "unknown"
    kind = parts[parts.index("typ") + 1] if "typ" in parts[:-1] else "unknown"
    address = parts[4]
    suffix = "/mdns" if address.endswith(".local") else "/v6" if ":" in address else ""
    return f"{kind}/{parts[2].lower()}{suffix}"


def kind_summary(kinds):
    counts = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + 1
    return ",".join(f"{kind}x{count}" for kind, count in sorted(counts.items())) or "none"


class SessionSender:
    def __init__(self, signaling, media_factory):
        self.signaling = signaling
        self.media_factory = media_factory
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._events = queue.PriorityQueue(maxsize=132)
        self._event_serial = 0
        self._event_overflow = False
        self._session = None
        self._context = None
        self._media = None
        self._retired = []
        self._generation = 0
        self._deadline = 0
        self._negotiation_deadline = 0
        self._cleanup = []
        self._seen = set()
        self._outgoing = 0
        # 협상 실패 진단용 후보 종류(주소 없음). 세션마다 초기화한다.
        self._local_kinds = []
        self._remote_kinds = []
        self._cleanup_after = 0
        self.state = "idle"
        self.error = ""
        self._thread = threading.Thread(target=self._run, name="webrtc-signaling", daemon=True)
        self._thread.start()

    def status(self):
        with self._lock:
            return {"state": self.state, "errorCode": self.error,
                    "active": self._media is not None,
                    "teardownPending": len(self._retired),
                    "media": {} if self._media is None else self._media.stats()}

    def push(self, payload, stamp_ns):
        with self._lock:
            if self._media is None or time.monotonic() >= self._deadline:
                return False
            return self._media.push_jpeg(payload, stamp_ns)

    def watchdog(self):
        with self._lock:
            error = ""
            if self._media is not None:
                if self._event_overflow:
                    error = "event_queue_overflow"
                elif time.monotonic() >= self._deadline:
                    error = "session_expired"
                    if self._context is not None:
                        data = self._context["session"]["data"]
                        _, error = min((timestamp(data["expiresAt"]), "session_expired"),
                            (timestamp(data["viewerHeartbeatAt"]) + 30, "heartbeat_timeout"),
                            (timestamp(self._context["lock"]["data"]["leaseUntil"]), "lock_lease_expired"))
                elif self.state != "connected" and time.monotonic() >= self._negotiation_deadline:
                    error = "negotiation_timeout"
                    # answer는 나갔는데 ICE가 연결되지 않은 경우다. 후보 종류로 원인을 가른다:
                    # 원격에 host/mdns만 있고 relay가 없으면 다른 네트워크 시청(TURN 필요)을 의심한다.
                    logging.getLogger(__name__).warning(
                        "WebRTC negotiation_timeout: local=%s remote=%s",
                        kind_summary(self._local_kinds), kind_summary(self._remote_kinds))
        if error:
            self._end(error)

    def _event(self, generation, kind, value):
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            self._event_serial += 1
            serial = self._event_serial
        # Publish SDP before ICE; preserve the original order within each kind.
        priority = 0 if kind == "answer" else 1 if kind == "state" else 2
        try:
            self._events.put_nowait((priority, serial, generation, kind, value))
        except queue.Full:
            # Callback must not block or close its own GLib loop.
            with self._lock:
                self._event_overflow = True
                self._deadline = 0

    def _end(self, error=""):
        with self._lock:
            media, session = self._media, self._session
            self._media = self._session = None
            self._context = None
            self._generation += 1
            self._deadline = 0
            if media is not None:
                self._retired.append(media)
            self.state = "failed" if error else "idle"
            self.error = error
            if session is not None and len(self._cleanup) < 8:
                self._cleanup.append((session["path"], error))
        if media is not None:
            media.close()

    def _start(self, context):
        if not self._valid(context) or context["session"]["data"]["state"] != "requested":
            return
        # context() is already a coherent read-only transaction. Starting media
        # from it avoids another multi-request round trip before answer creation.
        claimed = context["session"]
        with self._lock:
            if self._stop.is_set():
                return
            self._generation += 1
            self._event_overflow = False
            generation = self._generation
            self._session = claimed
            self._context = copy.deepcopy(context)
            self._seen = set()
            self._outgoing = 0
            self._local_kinds, self._remote_kinds = [], []
            self._deadline = self._lease(context, server_time(self.signaling))
            self._negotiation_deadline = time.monotonic() + 20
            self._media = self.media_factory()
            self.state, self.error = "negotiating", ""
            media = self._media
        media.start(claimed["data"]["offer"]["sdp"],
                    lambda sdp: self._event(generation, "answer", sdp),
                    lambda candidate, index: self._event(generation, "ice", (candidate, index)),
                    lambda state: self._event(generation, "state", state))

    def _valid(self, context):
        # 세션·lease·heartbeat 시각은 서버가 기록한다. Jetson 시계 대신 서버 기준 현재 시각과 비교한다.
        return context_valid(context, self.signaling.config,
                             self.signaling.allowed_uids, server_time(self.signaling))

    @staticmethod
    def _identity(context):
        data = context["session"]["data"]
        lock = context["lock"]["data"]
        return (context["robot"]["updateTime"],
                context["session"]["path"], lock["sessionId"], lock["ownerUid"],
                {key: data.get(key) for key in ("schemaVersion", "robotId", "senderId",
                 "ownerUid", "offer", "createdAt", "expiresAt")})

    @staticmethod
    def _lease(context, now=None):
        data = context["session"]["data"]
        expiry = min(timestamp(data["expiresAt"]), timestamp(data["viewerHeartbeatAt"]) + 30,
                     timestamp(context["lock"]["data"]["leaseUntil"]))
        return time.monotonic() + max(0, expiry - (time.time() if now is None else now))

    def _current(self, generation, path):
        with self._lock:
            return (generation == self._generation and self._session is not None
                    and self._session["path"] == path and self._media is not None)

    def _poll_active(self, original, generation):
        doc = original
        # Process the small callback batch first. Firestore rules and update-time
        # preconditions revalidate authorization and lease ownership on every
        # write; a conflict falls back to a coherent client-side refresh.
        network_events = 0
        outgoing_candidates = []
        for _ in range(self._events.maxsize):
            if network_events >= 4 or len(outgoing_candidates) >= 4:
                break
            try:
                _, _, event_generation, kind, value = self._events.get_nowait()
            except queue.Empty:
                break
            if event_generation != generation or not self._current(generation, doc["path"]):
                continue
            # Progress notifications are already exposed in media.stats(). They
            # must not consume the network batch and delay the queued SDP answer
            # behind another slow coherent Firestore context read.
            if kind == "state" and value not in ("connected", "failed", "closed", "disconnected", "error") and not value.startswith("error"):
                continue
            if kind == "answer":
                network_events += 1
                # Publish the answer and requested -> negotiating transition atomically.
                # Firestore rules intentionally validate this as one sender-owned step.
                doc = self._patch_active(doc, generation, {
                    "answer": {"type": "answer", "sdp": value},
                    "state": "negotiating",
                })
            elif kind == "ice":
                if self._outgoing + len(outgoing_candidates) >= 128:
                    self._end("candidate_limit")
                    return
                candidate, index = value
                data = {"candidate": candidate, "sdpMid": None, "sdpMLineIndex": index,
                        "usernameFragment": None, "createdAt": utc()}
                if not candidate:
                    data = {"endOfCandidates": True, "createdAt": utc()}
                outgoing_candidates.append(data)
                self._local_kinds.append(candidate_kind(candidate))
            elif kind == "state":
                if value == "connected":
                    network_events += 1
                    doc = self._patch_active(doc, generation, {"state": "connected"})
                    with self._lock:
                        if self._current(generation, doc["path"]):
                            self.state = "connected"
                elif value in ("failed", "closed", "disconnected", "error") or value.startswith("error"):
                    self._end("media_failed")
                    return
        if not self._current(generation, doc["path"]):
            return
        if outgoing_candidates:
            self._signaling_call("sender_candidates_create", self.signaling.send_candidates,
                                 doc["path"], self._outgoing, outgoing_candidates)
            self._outgoing += len(outgoing_candidates)
        context = self._signaling_call("active_context_read", self.signaling.context, doc["path"])
        if not self._accept_context(context, doc["path"], generation):
            return
        doc = context["session"]
        for candidate in self._signaling_call("viewer_candidates_read", self.signaling.candidates, doc["path"]):
            if candidate["path"] in self._seen:
                continue
            value, index = validate_candidate(candidate["data"], doc["data"]["offer"]["sdp"])
            with self._lock:
                if not self._current(generation, doc["path"]):
                    return
                self._media.add_ice(value, index)
                self._seen.add(candidate["path"])
                self._remote_kinds.append(candidate_kind(value))

    @staticmethod
    def _signaling_call(operation, function, *args):
        try:
            return function(*args)
        except SignalingError as exc:
            # Only fixed operation labels and sanitized codes; never arguments,
            # request bodies, SDP, ICE addresses or authentication information.
            logging.getLogger(__name__).warning("WebRTC %s failed: %s", operation, str(exc))
            raise

    def _accept_context(self, context, path, generation):
        with self._lock:
            if not self._current(generation, path):
                return False
            valid = (self._valid(context) and self._context is not None
                     and self._identity(context) == self._identity(self._context))
            if valid:
                self._session = context["session"]
                self._context = copy.deepcopy(context)
                self._deadline = self._lease(context, server_time(self.signaling))
                return True
        self._end("session_closed_or_invalid")
        return False

    def _patch_active(self, original, generation, values):
        for attempt in range(3):
            doc = original
            if attempt:
                # A heartbeat CAS retry rechecks authorization and lock ownership.
                context = self.signaling.context(original["path"])
                if not self._accept_context(context, original["path"], generation):
                    raise SignalingError("session_closed_or_invalid")
                doc = context["session"]
            try:
                return self.signaling.patch(doc, values)
            except SignalingError as exc:
                # 브라우저 heartbeat와 겹친 updateTime 충돌(400/409/412)은 다시 읽고 재시도하는
                # 정상 경로라 경고하지 않는다. 다른 오류나 마지막 시도 실패만 남긴다.
                if str(exc) not in ("http_409", "http_400", "http_412") or attempt == 2:
                    logging.getLogger(__name__).warning("WebRTC session_patch failed: %s", str(exc))
                    raise
        raise SignalingError("session_conflict")

    def _tidy(self):
        with self._lock:
            item = self._cleanup[0] if self._cleanup else None
        if item is None:
            return
        path, error = item
        try:
            doc = self.signaling.get(path)
            if (doc["data"].get("senderId") == self.signaling.config["senderUid"]
                    and doc["data"].get("state") not in ("closed", "failed")):
                self.signaling.patch(doc, {"state": "failed" if error else "closed",
                                          "endedAt": utc(), "errorCode": error})
        except SignalingError as exc:
            if str(exc) != "http_404":
                raise
        with self._lock:
            if self._cleanup and self._cleanup[0] == item:
                self._cleanup.pop(0)

    def _run(self):
        while not self._stop.is_set():
            try:
                with self._lock:
                    doc, generation = self._session, self._generation
                    self._retired = [media for media in self._retired
                                     if not getattr(media, "is_closed", lambda: True)()]
                    teardown_pending = bool(self._retired)
                if doc is None:
                    if teardown_pending:
                        self._stop.wait(0.1)
                        continue
                    if time.monotonic() >= self._cleanup_after:
                        try:
                            self._tidy()
                        except Exception:
                            self._cleanup_after = time.monotonic() + 30
                    context = self.signaling.context()
                    if self._valid(context):
                        self._start(context)
                else:
                    self._poll_active(doc, generation)
            except Exception as exc:
                # Remote failures fail closed; no credentials or SDP in status.
                self._end(str(exc) if isinstance(exc, SignalingError) else "signaling_or_media_error")
            self._stop.wait(1.0)

    def close(self):
        self._stop.set()
        self._end()
        self._thread.join(timeout=0.2)
        # External trusted cleanup handles abandoned documents after shutdown.
