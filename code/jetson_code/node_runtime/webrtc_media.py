"""Session-scoped GStreamer sender. No ROS, signaling, or credentials on disk.

Callbacks run on GStreamer threads and must enqueue work, never perform network I/O.
One latest JPEG, 128 pending ICE candidates and leaky GStreamer queues bound work.
"""
from collections import deque
from dataclasses import dataclass, field
import re
import threading
import time


@dataclass(frozen=True)
class WebRTCMediaConfig:
    width: int = 640
    height: int = 720
    fps: int = 12
    encoder: str = "auto"
    software_encoder: str = "x264enc"
    bitrate_kbps: int = 2000
    stun_server: str = ""
    turn_servers: tuple = field(default=(), repr=False)

    def __post_init__(self):
        if (self.width, self.height) not in ((640, 720), (1280, 360), (1280, 720)) or type(self.fps) is not int or not 1 <= self.fps <= 15:
            raise ValueError("Expected 640x720, 1280x360 or 1280x720 and 1..15 FPS")
        if self.encoder not in ("auto", "nvv4l2h264enc", "x264enc", "openh264enc"):
            raise ValueError("Unsupported encoder")
        if self.software_encoder not in ("x264enc", "openh264enc", "disabled"):
            raise ValueError("Unsupported software fallback")
        if type(self.bitrate_kbps) is not int or not 100 <= self.bitrate_kbps <= 20000:
            raise ValueError("Invalid bitrate")
        if self.stun_server and not self.stun_server.startswith("stun://"):
            raise ValueError("STUN must use GStreamer stun:// URI")
        if len(self.turn_servers) > 4 or any(not s.startswith(("turn://", "turns://")) for s in self.turn_servers):
            raise ValueError("Invalid TURN configuration")


def select_encoder(config, available):
    if config.encoder != "auto":
        if not available(config.encoder):
            raise RuntimeError("Requested encoder is unavailable")
        return config.encoder
    if available("nvv4l2h264enc") and available("nvvidconv"):
        return "nvv4l2h264enc"
    if config.software_encoder != "disabled" and available(config.software_encoder):
        return config.software_encoder
    raise RuntimeError("No configured H.264 encoder is available")


def h264_payload(offer):
    """Select offered packetization-mode=1 baseline H264; never assume PT 96."""
    if not isinstance(offer, str) or len(offer) > 100000:
        raise ValueError("Invalid offer")
    sections = re.split(r"(?m)^m=", offer.replace("\r", ""))[1:]
    video = [s for s in sections if s.startswith("video ")]
    if len(video) != 1 or len(sections) != 1 or "a=recvonly" not in video[0]:
        raise ValueError("Offer must contain one recvonly video section")
    for pt in re.findall(r"(?mi)^a=rtpmap:(\d+) H264/90000\s*$", video[0]):
        fmtp = re.search(r"(?m)^a=fmtp:" + pt + r" (.+)$", video[0])
        settings = fmtp.group(1).lower() if fmtp else ""
        profile = re.search(r"profile-level-id=([0-9a-f]{6})", settings)
        if "packetization-mode=1" in settings and (profile is None or profile.group(1).startswith("42")):
            return int(pt)
    raise ValueError("Viewer must offer baseline H.264 packetization-mode=1")


def pipeline_description(config, encoder, payload):
    if encoder == "nvv4l2h264enc":
        encode = (f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc "
                  f"bitrate={config.bitrate_kbps * 1000} iframeinterval={config.fps * 2} insert-sps-pps=true")
    elif encoder == "x264enc":
        encode = (f"x264enc tune=zerolatency speed-preset=ultrafast bframes=0 "
                  f"bitrate={config.bitrate_kbps} key-int-max={config.fps * 2} byte-stream=true")
    else:
        encode = f"openh264enc bitrate={config.bitrate_kbps * 1000} gop-size={config.fps * 2} complexity=low"
    return ("webrtcbin name=peer bundle-policy=max-bundle "
            "appsrc name=source is-live=true format=time block=false max-buffers=1 leaky-type=downstream "
            # jpegdec negotiates actual source dimensions, preserving both the
            # vertical camera stream and legacy publishers without stretch.
            f"caps=image/jpeg,framerate={config.fps}/1 ! "
            "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! jpegdec ! "
            "videoconvert ! video/x-raw,format=I420,width=(int){640,1280},height=(int){360,720} ! " + encode +
            " ! h264parse config-interval=-1 ! video/x-h264,profile=constrained-baseline ! "
            f"rtph264pay config-interval=-1 pt={payload} ! application/x-rtp,media=video,encoding-name=H264,payload={payload} ! peer.")


class GStreamerWebRTCSender:
    def __init__(self, config=None):
        self.config = config or WebRTCMediaConfig()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._ice = deque(maxlen=128)
        self._remote_ready = False
        self._pending_promise = None
        self._counts = dict(received=0, replaced=0, pushed=0, rejected=0, ice_rejected=0,
                            jpeg_bytes=0, encoder=None, state="idle")
        self._last_stamp = None

    def start(self, offer_sdp, on_answer, on_ice, on_state):
        payload = h264_payload(offer_sdp)
        with self._lock:
            if self._thread is not None or self._stop.is_set():
                raise RuntimeError("Create a new sender for each session")
            self._on_answer, self._on_ice, self._on_state = on_answer, on_ice, on_state
            self._thread = threading.Thread(target=self._run, args=(offer_sdp, payload), daemon=True,
                                            name="rail-webrtc-media")
            self._thread.start()

    def push_jpeg(self, payload, timestamp_ns):
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                return False
            if not payload or len(payload) > 15 * 1024 * 1024 or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
                self._counts["rejected"] += 1
                return False
            if self._last_stamp is not None and timestamp_ns <= self._last_stamp:
                self._counts["rejected"] += 1
                return False
            self._last_stamp = timestamp_ns
            self._counts["received"] += 1
            self._counts["replaced"] += self._latest is not None
            self._latest = (bytes(payload), timestamp_ns, time.monotonic_ns())
            return True

    def add_ice(self, candidate, mline_index):
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                return False
            if not isinstance(candidate, str) or len(candidate) > 8192 or mline_index != 0 or len(self._ice) == 128:
                self._counts["ice_rejected"] += 1
                return False
            self._ice.append((candidate, mline_index))
            return True

    def stats(self):
        with self._lock:
            return dict(self._counts, pending_frames=int(self._latest is not None), pending_ice=len(self._ice))

    def close(self):
        self._stop.set()
        with self._lock:
            self._latest = None
            self._ice.clear()
            pending = self._pending_promise
        if pending is not None:
            pending.interrupt()
        # Teardown occurs on the media thread; never wait in a ROS callback.

    def is_closed(self):
        return self._thread is None or not self._thread.is_alive()

    def _state(self, state):
        with self._lock:
            self._counts["state"] = state
        self._on_state(state)

    def _run(self, offer, payload):
        pipeline = None
        context = None
        source = None
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstWebRTC", "1.0")
            gi.require_version("GstSdp", "1.0")
            from gi.repository import Gst, GstWebRTC, GstSdp, GLib
            Gst.init(None)
            self._Gst, self._GstWebRTC = Gst, GstWebRTC
            context = GLib.MainContext.new()
            context.push_thread_default()
            encoder = select_encoder(self.config, lambda name: Gst.ElementFactory.find(name) is not None)
            with self._lock:
                self._counts["encoder"] = encoder
            pipeline = Gst.parse_launch(pipeline_description(self.config, encoder, payload))
            peer, appsrc = pipeline.get_by_name("peer"), pipeline.get_by_name("source")
            self._peer = peer
            if self.config.stun_server:
                peer.set_property("stun-server", self.config.stun_server)
            for server in self.config.turn_servers:
                if not peer.emit("add-turn-server", server):
                    raise RuntimeError("TURN configuration rejected")
            peer.connect("on-ice-candidate", lambda _, index, candidate: self._on_ice(candidate, index) if not self._stop.is_set() else None)
            peer.connect("notify::ice-gathering-state", lambda obj, _: self._on_ice("", 0)
                         if not self._stop.is_set() and obj.get_property("ice-gathering-state").value_nick == "complete" else None)
            peer.connect("notify::connection-state", lambda obj, _: self._state(obj.get_property("connection-state").value_nick) if not self._stop.is_set() else None)
            result, sdp = GstSdp.SDPMessage.new()
            if result != GstSdp.SDPResult.OK or GstSdp.sdp_message_parse_buffer(offer.encode(), sdp) != GstSdp.SDPResult.OK:
                raise ValueError("SDP parse failed")
            description = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdp)
            if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Media pipeline could not start")
            self._negotiate(peer, description)
            bus = pipeline.get_bus()
            started = time.monotonic_ns()
            last_push = 0
            while not self._stop.wait(0.005):
                # Drain a bounded number of GLib callbacks per iteration.
                for _ in range(16):
                    if not context.pending():
                        break
                    context.iteration(False)
                message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message is not None:
                    raise RuntimeError("Media pipeline stopped")
                now = time.monotonic_ns()
                with self._lock:
                    candidates = list(self._ice) if self._remote_ready else []
                    if self._remote_ready:
                        self._ice.clear()
                    frame = None
                    if self._remote_ready and now - last_push >= 1_000_000_000 / self.config.fps:
                        frame, self._latest = self._latest, None
                for candidate, index in candidates:
                    peer.emit("add-ice-candidate", index, candidate)
                if frame is None:
                    continue
                jpeg, stamp, received = frame
                if now - received > 800_000_000:
                    continue
                buffer = Gst.Buffer.new_allocate(None, len(jpeg), None)
                buffer.fill(0, jpeg)
                # PTS uses pipeline running time, never a ROS wall-clock epoch.
                buffer.pts = max(0, now - started)
                buffer.dts = buffer.pts
                buffer.duration = int(1_000_000_000 / self.config.fps)
                if appsrc.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
                    raise RuntimeError("Media source rejected frame")
                last_push = now
                with self._lock:
                    self._counts["pushed"] += 1
                    self._counts["jpeg_bytes"] += len(jpeg)
        except Exception:
            # Gst errors may embed TURN credentials: never expose their text.
            self._state("failed")
        finally:
            self._stop.set()
            if pipeline is not None:
                pipeline.set_state(self._Gst.State.NULL)
            self._peer = None
            self._answer_reply = None
            self._answer_description = None
            if context is not None:
                context.pop_thread_default()
            with self._lock:
                self._latest = None
                self._ice.clear()
            if self.stats()["state"] != "failed":
                self._state("closed")

    def _promise_ok(self, promise):
        if self._stop.is_set():
            return False
        if promise.wait() != self._Gst.PromiseResult.REPLIED:
            return False
        reply = promise.get_reply()
        return reply is None or not reply.has_field("error")

    def _wait_promise(self, promise):
        with self._lock:
            if self._stop.is_set():
                promise.interrupt()
                return False
            self._pending_promise = promise
        try:
            return self._promise_ok(promise)
        finally:
            with self._lock:
                if self._pending_promise is promise:
                    self._pending_promise = None

    def _negotiate(self, peer, description):
        """Complete SDP setup on the dedicated media thread.

        Waiting here avoids depending on GLib promise callback scheduling inside
        a long-running ROS process. close() interrupts the current wait.
        """
        self._state("setting_remote")
        promise = self._Gst.Promise.new()
        peer.emit("set-remote-description", description, promise)
        if not self._wait_promise(promise):
            raise RuntimeError("Remote SDP setup failed")
        with self._lock:
            self._remote_ready = True

        self._state("creating_answer")
        promise = self._Gst.Promise.new()
        peer.emit("create-answer", None, promise)
        if not self._wait_promise(promise):
            raise RuntimeError("SDP answer creation failed")
        reply = promise.get_reply()
        answer = reply.get_value("answer")
        answer_text = answer.sdp.as_text()

        self._state("setting_local")
        promise = self._Gst.Promise.new()
        peer.emit("set-local-description", answer, promise)
        if not self._wait_promise(promise):
            raise RuntimeError("Local SDP setup failed")
        self._answer_reply = reply
        self._answer_description = answer
        self._on_answer(answer_text)
        self._state("connecting")

    def _remote_set(self, promise, *_):
        if not self._promise_ok(promise):
            self._state("failed")
            self.close()
            return
        with self._lock:
            self._remote_ready = True
        self._peer.emit("create-answer", None, self._Gst.Promise.new_with_change_func(self._answer_created, None, None))

    def _answer_created(self, promise, *_):
        if not self._promise_ok(promise):
            self._state("failed")
            self.close()
            return
        # Keep the owning GstStructure alive while using its boxed SDP value.
        # Chaining get_reply().get_value() can free the owner early on GI 1.20.
        reply = promise.get_reply()
        answer = reply.get_value("answer")
        answer_text = answer.sdp.as_text()
        self._answer_reply = reply
        self._answer_description = answer
        self._peer.emit("set-local-description", answer, self._Gst.Promise.new_with_change_func(self._local_set, answer_text, None))

    def _local_set(self, promise, answer, *_):
        if self._promise_ok(promise):
            self._on_answer(answer)
        elif not self._stop.is_set():
            self._state("failed")
            self.close()
