"""Explicit operator-only LAN check over authenticated SSH, NOT production signaling.

Receiver requires aiortc; sender requires ROS/GStreamer. No dashboard/Firestore writes.
SDP/ICE travel over the SSH pipe; video travels directly via WebRTC (STUN, no TURN).
"""
import argparse
import asyncio
import json
import os
import re
import sys
import threading
import time
import struct
import zlib
import subprocess

_output_lock = threading.Lock()


def emit(value):
    with _output_lock:
        print(json.dumps(value), flush=True)


def write_latency_marker(frame, stamp_ns):
    """Test-only marker at the actual image bottom, for either GUI layout."""
    import numpy as np

    if frame is None or frame.shape[0] < 16 or frame.shape[1] < 112 * 8:
        raise ValueError("Frame is too small for the latency marker")
    raw = struct.pack(">HQ", 0xCAFE, stamp_ns // 1000)
    raw += struct.pack(">I", zlib.crc32(raw))
    for index, bit in enumerate(np.unpackbits(np.frombuffer(raw, np.uint8))):
        frame[-16:, index*8:(index+1)*8] = 255 if bit else 0


def read_latency_marker(pixels):
    """Return correlated source seconds only when the marker checksum matches."""
    import numpy as np

    if pixels is None or pixels.shape[0] < 16 or pixels.shape[1] < 112 * 8:
        return None
    bits = [np.mean(pixels[-12:-4, i*8+2:i*8+6]) > 128 for i in range(112)]
    raw = np.packbits(bits).tobytes()
    if raw[:2] == b"\xca\xfe" and zlib.crc32(raw[:10]) == struct.unpack(">I", raw[10:])[0]:
        return struct.unpack(">Q", raw[2:10])[0] / 1e6
    return None


def sender(seconds, latency_marker=False, metrics=False):
    import faulthandler
    faulthandler.enable()
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from node_runtime.ros.common import qos_sensor
    from node_runtime.webrtc_media import GStreamerWebRTCSender, WebRTCMediaConfig
    offer = json.loads(sys.stdin.readline())
    media = GStreamerWebRTCSender(WebRTCMediaConfig(stun_server="stun://stun.l.google.com:19302"))
    rclpy.init()
    node = Node("rail_webrtc_lan_check")
    media.start(offer["sdp"], lambda sdp: emit({"type": "answer", "sdp": sdp}),
                lambda candidate, index: emit({"type": "ice", "candidate": candidate, "index": index}),
                lambda state: emit({"type": "state", "state": state}))
    first = [True]
    def receive(msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        age = (node.get_clock().now().nanoseconds - stamp) / 10**9
        if not 0 <= age <= .8:
            return
        if first[0]:
            first[0] = False
            infos = node.get_publishers_info_by_topic("/rail_robot/gui/image/compressed")
            emit({"type": "input", "topic": "/rail_robot/gui/image/compressed", "format": msg.format,
                  "frameId": msg.header.frame_id,
                  "publishers": [{"messageType": i.topic_type, "depth": i.qos_profile.depth,
                                  "reliability": str(i.qos_profile.reliability),
                                  "durability": str(i.qos_profile.durability)} for i in infos]})
        payload = bytes(msg.data)
        if latency_marker:
            # Test-only, checksummed marker. Production sender leaves the GUI intact.
            import cv2
            import numpy as np
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            write_latency_marker(frame, stamp)
            payload = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        media.push_jpeg(payload, stamp)
    node.create_subscription(CompressedImage, "/rail_robot/gui/image/compressed", receive, qos_sensor)
    node.create_timer(5.0, lambda: emit({"type": "progress", "stats": media.stats()}))
    stop = threading.Event()
    def measure_load():
        if not stop.wait(8):
            result = subprocess.run(["timeout", "-k", "2s", "6s", "tegrastats", "--interval", "1000"],
                                    capture_output=True, text=True, timeout=10)
            emit({"type": "tegrastats", "samples": result.stdout.splitlines()})
    if metrics:
        threading.Thread(target=measure_load, daemon=True).start()
    def read_control():
        for line in sys.stdin:
            data = json.loads(line)
            if data.get("type") == "close":
                break
            if data.get("type") == "ice":
                media.add_ice(data["candidate"], data["index"])
            if data.get("type") == "clock":
                emit({"type": "clock", "sent": data["sent"], "remote": time.time()})
        stop.set()
    threading.Thread(target=read_control, daemon=True).start()
    deadline = time.monotonic() + seconds
    try:
        while not stop.is_set() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
    finally:
        emit({"type": "senderStats", "stats": media.stats()})
        media.close()
        node.destroy_node()
        rclpy.shutdown()


async def receiver(args):
    import paramiko
    from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription, RTCRtpSender
    from aiortc.sdp import candidate_from_sdp
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")]))
    transceiver = pc.addTransceiver("video", direction="recvonly")
    transceiver.setCodecPreferences([c for c in RTCRtpSender.getCapabilities("video").codecs
                                    if c.mimeType.lower() == "video/h264"])
    frames, shape, arrival = 0, None, []
    markers, clock_samples = [], []
    @pc.on("track")
    def track_received(track):
        async def consume():
            nonlocal frames, shape
            try:
                while True:
                    frame = await track.recv()
                    frames += 1
                    shape = [frame.width, frame.height]
                    arrival.append(time.monotonic())
                    if args.latency_marker:
                        source = read_latency_marker(frame.to_ndarray(format="gray"))
                        if source is not None:
                            markers.append((time.time(), source))
            except Exception:
                pass
        asyncio.create_task(consume())
    await pc.setLocalDescription(await pc.createOffer())
    offer_sdp = pc.localDescription.sdp
    local_candidates = []
    if args.trickle:
        lines = offer_sdp.replace("\r\n", "\n").splitlines()
        local_candidates = [line.removeprefix("a=") for line in lines if line.startswith("a=candidate:")]
        if args.candidate_host:
            if not re.fullmatch(r"[A-Za-z0-9-]{1,80}\.local", args.candidate_host):
                raise ValueError("candidate host must be a bounded .local name")
            replaced = []
            for candidate in local_candidates:
                parts = candidate.split()
                if len(parts) >= 6:
                    parts[4] = args.candidate_host
                replaced.append(" ".join(parts))
            local_candidates = replaced
        offer_sdp = "\r\n".join(line for line in lines
                                  if not line.startswith("a=candidate:") and line != "a=end-of-candidates") + "\r\n"
    ros_domain_id = os.environ.get("RAIL_ROBOT_ROS_DOMAIN_ID", "76")
    if not ros_domain_id.isdigit() or not 0 <= int(ros_domain_id) <= 232:
        raise ValueError("RAIL_ROBOT_ROS_DOMAIN_ID must be 0..232")
    command = ("source /opt/ros/humble/setup.bash; "
               "source ~/Desktop/rail_robot/code/ros2/install/setup.bash; "
               "export ROS_DOMAIN_ID=" + ros_domain_id + "; "
               "export GST_PLUGIN_PATH=$HOME/.local/share/rail_robot/gst-deps/usr/lib/aarch64-linux-gnu/gstreamer-1.0; "
               "python3 ~/Desktop/rail_robot/code/tools/webrtc_lan_check.py sender --seconds " + str(args.seconds + 10)
               + (" --latency-marker" if args.latency_marker else "")
               + (" --metrics" if args.metrics else "")
               + "; echo SENDER_PROCESS_STATUS=$? >&2")
    ssh_config = paramiko.SSHConfig()
    with open(os.path.expanduser("~/.ssh/config")) as stream:
        ssh_config.parse(stream)
    host = ssh_config.lookup(args.host)
    client = paramiko.SSHClient()
    client.load_system_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    await asyncio.to_thread(client.connect, host.get("hostname", args.host),
                            username=host.get("user"), key_filename=args.key,
                            look_for_keys=False, allow_agent=False, timeout=8)
    stdin, stdout, stderr = client.exec_command(command)
    stdin.write(json.dumps({"sdp": offer_sdp}) + "\n")
    for candidate in local_candidates:
        stdin.write(json.dumps({"type": "ice", "candidate": candidate, "index": 0}) + "\n")
    if args.trickle:
        stdin.write(json.dumps({"type": "ice", "candidate": "", "index": 0}) + "\n")
    stdin.flush()
    async def clock_sync():
        # Exclude remote interpreter/GStreamer startup from the clock RTT bound.
        await asyncio.sleep(5)
        for _ in range(5):
            stdin.write(json.dumps({"type": "clock", "sent": time.time()}) + "\n")
            stdin.flush()
            await asyncio.sleep(.2)
    clock_task = asyncio.create_task(clock_sync())
    async def errors():
        while True:
            line = await asyncio.to_thread(stderr.readline)
            if not line:
                return
            print(line.rstrip(), file=sys.stderr)
    error_task = asyncio.create_task(errors())
    pending = []
    async def incoming():
        while True:
            line = await asyncio.to_thread(stdout.readline)
            if not line:
                emit({"type": "senderExit", "returncode": await asyncio.to_thread(stdout.channel.recv_exit_status)})
                return
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if data["type"] == "answer":
                await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="answer"))
                for candidate in pending:
                    await pc.addIceCandidate(candidate)
                pending.clear()
            elif data["type"] == "ice":
                if not data["candidate"]:
                    continue
                candidate = candidate_from_sdp(data["candidate"].removeprefix("candidate:"))
                candidate.sdpMLineIndex = data["index"]
                if pc.remoteDescription:
                    await pc.addIceCandidate(candidate)
                else:
                    pending.append(candidate)
            elif data["type"] == "clock":
                received = time.time()
                clock_samples.append((received - data["sent"], data["remote"] - (received + data["sent"]) / 2))
            else:
                emit(data)
    task = asyncio.create_task(incoming())
    await asyncio.sleep(args.seconds)
    stats = await pc.getStats()
    transports = [dict(s.__dict__) for s in stats.values() if s.type == "transport"]
    for s in transports:
        s.pop("timestamp", None)
    clock_best = min(clock_samples) if clock_samples else None
    ages = sorted((received + clock_best[1] - source) * 1000 for received, source in markers) if clock_best else []
    emit({"type": "receiverStats", "connection": pc.connectionState, "frames": frames,
          "shape": shape, "activeFps": (frames - 1) / (arrival[-1] - arrival[0]) if frames > 1 else 0,
          "transports": transports,
          "transportMbps": sum(s.get("bytesReceived", 0) for s in transports)*8/args.seconds/1e6,
          "guiToDecodedLatencyMsP95": ages[int((len(ages)-1)*.95)] if ages else None,
          "latencySamples": len(ages), "clockErrorBoundMs": clock_best[0]*500 if clock_best else None,
          "latencyNote": "Test marker includes additional JPEG decode/re-encode; GUI composition to decoded LAN receiver, not camera capture or browser paint." if ages else "Not measured: no correlated marker samples."})
    if not stdout.channel.exit_status_ready():
        stdin.write('{"type":"close"}\n')
        stdin.flush()
    await pc.close()
    try:
        await asyncio.wait_for(task, timeout=10)
        await asyncio.wait_for(error_task, timeout=5)
        await clock_task
    except asyncio.TimeoutError:
        stdout.channel.close()
    finally:
        stdin.close()
        stdout.close()
        stderr.close()
        client.close()
    if frames == 0:
        raise SystemExit("No decoded frames; LAN check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["sender", "receiver"])
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--host", default="bitdol")
    parser.add_argument("--key", default=os.path.expanduser("~/.ssh/codex_jetson_ed25519"))
    parser.add_argument("--latency-marker", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--trickle", action="store_true",
                        help="Send receiver candidates separately, like the Firebase browser flow")
    parser.add_argument("--candidate-host", default="",
                        help="Replace trickled host address for mDNS compatibility testing")
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120:
        parser.error("seconds must be 5..120")
    sender(args.seconds, args.latency_marker, args.metrics) if args.role == "sender" else asyncio.run(receiver(args))
