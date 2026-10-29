# Jetson WebRTC sender

## Scope and contract

The sender follows [signaling contract v1](webrtc-signaling-contract.md). The dashboard, Firebase rules and TURN server configuration are in `../../rail_dashboard`. The optional five-second Firebase snapshot path remains independently available.

The optional `webrtc_sender_node` subscribes only to the final GUI JPEG. It owns no camera, TensorRT model, motor, pump or UART. `webrtc.launch.py` starts only this sender; the mission stack must supply the GUI topic separately.

`webrtc_signaling.py` implements authenticated Firestore REST; `webrtc_sender.py` owns session admission/expiry/candidate limits; `webrtc_media.py` owns GStreamer. Missing configuration leaves sender disabled and sends no video. Valid authorized request starts one encoder; invalid/closed/expired session stops it. No always-on video transmission.

## Observed input and hardware

Current local source emits camera-only 1280×360 JPEGs with detection/ROI overlays.
The sender negotiates actual input dimensions and still accepts legacy 1280×720
publishers. Local HighGUI remains 1280×720; mission/timing data is carried on the
GUI status interface. Update display and sender together. The historical results
below describe the 1280×720 deployment, not measurements of the new source.
See [live capture output](live-capture-output.md) and
[live dashboard configuration](live-dashboard.md) for the opt-in photo/telemetry path.

Observed on bitdol, 2026-09-06:

| Item | Evidence |
|---|---|
| Model | `/proc/device-tree/model`: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| Platform | Ubuntu 22.04, ROS Humble, GStreamer 1.20.3 |
| Topic | `/rail_robot/gui/image/compressed` |
| Message | `sensor_msgs/msg/CompressedImage` |
| Format / frame ID | `bgr8; jpeg compressed bgr8` / `rail_robot_gui` |
| Decoded input | 1280×720, final camera feeds + inference overlays |
| Input delivery | 120 JPEGs/10.013s; 11.984 FPS full window, 11.999 active FPS |
| Input JPEG bandwidth | 7.276 Mbit/s for that scene |
| Input QoS | actual publisher discovery BEST_EFFORT, VOLATILE; source and subscription configured KEEP_LAST depth 1 |
| Depth discovery limitation | RMW endpoint reports depth 0, so depth 1 is verified from publisher QoS construction, not claimed from discovery |
| Encoders | `nvv4l2h264enc` absent; `x264enc`, `openh264enc` available |
| Selection | auto → x264enc on this Jetson; explicit hardware mode fails if unavailable |

Input header stamp is GUI composition time, not physical camera exposure. Original per-camera age/identity remains on `/rail_robot/gui/status`; existing inspection/contamination/status interfaces are untouched.

## Dependencies and launch

System packages (operator can install missing packages):

```bash
sudo apt install python3-gi python3-gst-1.0 gir1.2-gst-plugins-bad-1.0 \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-nice
```

During this test `webrtcbin` existed but `nicesrc` was missing. `gstreamer1.0-nice` was downloaded from the Ubuntu apt repository and extracted under the user's home, without sudo or system changes. Until installed system-wide, this Jetson needs:

```bash
export GST_PLUGIN_PATH="$HOME/.local/share/rail_robot/gst-deps/usr/lib/aarch64-linux-gnu/gstreamer-1.0${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"
gst-inspect-1.0 nicesrc
gst-inspect-1.0 webrtcbin
gst-inspect-1.0 x264enc
```

Build and prepare each terminal:

```bash
cd ~/Desktop/rail_robot/code/ros2
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
# Use the same domain in every participating robot process. Test used 76.
export ROS_DOMAIN_ID=76
```

Terminal A, existing camera/inference/GUI only:

```bash
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="$HOME/Desktop/rail_robot/code/.venv/lib/python3.10/site-packages:$PYTHONPATH"
ros2 launch rail_robot_nodes robot.launch.py mode:=dual_camera display:=false gui_output:=true firebase_snapshot:=false
```

Terminal B, sender only (system Python GI; no model venv needed):

```bash
ros2 launch rail_robot_nodes webrtc.launch.py \
  config_file:=$HOME/.config/rail_robot/webrtc.json
```

The operator/dashboard task provisions the external JSON based on `code/config/webrtc.example.json`; use `chmod 600` and never store a real file inside the repo. `config_file` is just a path, not a credential ROS parameter. Alternatively `RAIL_ROBOT_WEBRTC_CONFIG` selects the file for `ros2 run rail_robot_nodes webrtc_sender_node`. No credentials were provisioned or committed here.

Media settings: `encoder=auto|nvv4l2h264enc|x264enc|openh264enc`, software fallback `x264enc|openh264enc|disabled`, integer bitrateKbps 100..20000, integer fps 1..15 (default12), GStreamer `stun://host:port`, external `turn://`/`turns://` URI list. Example uses STUN and no TURN. URI credentials must be percent-escaped per GStreamer and kept only in the private config. No coturn or paid service is provisioned. Hardware pipeline is implemented but not exercised on this non-NVENC test configuration.

Status:

```bash
ros2 topic echo /rail_robot/webrtc/status --qos-reliability best_effort
```

The status includes state, sanitized errorCode, JPEG admission/drop counts, selected encoder and bounded pending counts. It does not claim JPEG bytes are H.264 wire bandwidth. Authentication/rules errors fail closed. The separate dashboard task must finish robot-specific authorization before production signaling can succeed.

To retain fallback, run Terminal A with `firebase_snapshot:=true` and existing external Firebase snapshot credentials. It remains a separate five-second consumer; it is not automatically toggled by this sender. The dashboard task selects video or the latest snapshot. Actual snapshot network upload was not tested in this WebRTC task.

## Bounded work and lifecycle

ROS callback validates format/stamp/age/size, then hands off immutable compressed bytes; no HTTP, JPEG decode or encoding there. Latest JPEG slot=1, GStreamer appsrc max-buffers=1/block=false, raw pre-encode leaky queue=1. No leaky queue after H264 encoding/RTP fragmentation: dropping part of a compressed frame corrupts decoding. ICE input=128, signaling callbacks=132, one referenced session selected by the active viewer lock, one signaling worker, one active media thread. No frame queue grows with connection speed.

Heartbeat every 10s with a 30s timeout, lock lease 30s, fixed session deadline one hour, negotiation20s; local watchdog 100ms and monotonic lease check gate frame admission independently of network I/O. HTTP has a 5s request timeout. CAS retries revalidate the coherent robot/lock/session context, immutable fields and hard expiry so heartbeat conflicts do not normally terminate a valid stream. Robot configuration changes or revoked uploader/viewer authorization stop the stream. Local generation IDs prevent late callbacks from reviving ended sessions. Terminal-report failures back off without preventing new admission; Jetson never deletes sessions, candidates or locks. Shutdown stops new work immediately; media teardown and an already-running HTTP call may finish asynchronously. Abandoned documents require the separately-owned janitor described in the contract.

`teardownPending` blocks new session admission until the prior native media thread has actually exited, preventing repeated reconnects from accumulating encoders. If a native driver never completes teardown, restart only this optional sender process after diagnosing it; the node does not create more encoders or affect robot control.

## Tests and measurement boundaries

Local unit checks cover signaling authorization/expiry, timestamp/candidate schema, compare-and-set, stale callbacks, latest-frame bounds, encoder selection and SDP lifetime. They do not substitute for real Firestore rules or native media tests.

Jetson native debugging identified and fixed the GI boxed-SDP owner lifetime crash on GStreamer1.20 and RTP-fragment drops from an incorrectly located leaky queue. Jetson package build succeeded. Actual same-LAN peer was Windows Python aiortc, with authenticated SSH used ONLY by the explicit test harness to exchange SDP/ICE. Media uses direct WebRTC/STUN, no TURN. This establishes the media path, not dashboard browser compatibility or Firebase authentication/rules correctness.

Reproduce from an operator PC with aiortc and paramiko installed (SSH key and known_hosts must already be provisioned):

```bash
python code/tools/webrtc_lan_check.py receiver --host bitdol --seconds 30
# Collect five tegrastats samples while the stream is active:
python code/tools/webrtc_lan_check.py receiver --host bitdol --seconds 30 --metrics
# Optional test-only checksummed timestamp strip, additional JPEG decode/re-encode:
python code/tools/webrtc_lan_check.py receiver --host bitdol --seconds 30 --latency-marker
```

The test helper starts no cameras itself, never writes Firebase, and is not production unauthenticated signaling. It requires an operator's SSH key, limits session duration, and exits when the SSH control channel closes. Production node has no equivalent network test endpoint.

First successful 30s media trial: 313 decoded frames, 1280×720, active FPS11.705, receiver transport bytes6,613,571 → 1.764Mbit/s over the full30s (includes WebRTC transport overhead, not raw Ethernet overhead). Input329 / pushed321; latest-frame replacements8. This is scene-dependent variable bitrate, with encoder target2Mbit/s. Final measurements follow below.

Initial connected-stream tegrastats sample (4 seconds): total RAM2104..2122MB, six-core mean CPU approximately46%, GPU samples0..99%, temperature approximately52..53°C. This is total machine load including camera/inference/GUI and x264, not isolated sender overhead or long-term thermal validation. Samples from a different boot/desktop state must not be subtracted from earlier ROS migration measurements.

Capture-to-browser latency is not measured here because browser/dashboard implementation is out of scope. Optional marker mode measures GUI composition→decoded LAN receiver, with an SSH-clock round-trip uncertainty and extra test-only JPEG processing. Never label RTCP RTT as end-to-end video latency.

### Final measurements and validation

2026-09-06 final normal-stream trial (`--seconds 30 --metrics`, no marker): 317 decoded 1280×720 frames, active11.663 FPS, receiver transport6,645,381 bytes / 1.772Mbit/s over30s. Sender received329, pushed319, latest replacements9; all remaining queues held at most one JPEG. Repeated fresh sessions connected successfully and explicit close ended the media process cleanly. This is manual fresh-session reconnection, not a tested dashboard automatic reconnect flow.

Five synchronized tegrastats samples during that stream: total RAM2252..2316MB, six-core CPU mean53.6% (individual cores39..79%), GR3D samples99/99/26/0/65%, temperature52..53.3°C. Camera/inference/GUI and small verification processes share this machine; these are total short-window readings, not isolated encoding cost or sustained thermal/stability certification. The H.264 encoder was x264enc on CPU; GPU readings include existing inference.

Separate marker trial: 304 decoded frames, active11.640 FPS, 1.697Mbit/s; 304 checksum-validated timestamp samples yielded GUI composition→decoded receiver p95 **226.22ms**, with best measured clock round-trip half-width **4.02ms**. This includes extra test-only JPEG decode/re-encode and receiver marker decoding. It excludes physical capture→GUI composition and browser presentation. An earlier startup-biased clock sample (±407ms) was discarded; clock sync now waits for remote startup.

Local: 23 WebRTC unit tests and 15 existing launch tests passed; Python99 files compiled. Jetson: the same23 WebRTC tests passed with its Python3.10, native GStreamer WebRTC path connected and decoded video over LAN, optional ROS launch published disabled/config_required and shut down cleanly without credentials. Actual Firestore login/session/ICE rules and trusted cleanup remain untested because that configuration is owned by the other task. Browser playback, remote-network connectivity, TURN relay and long-duration congestion/thermal tests remain separate follow-up verification. No TURN secrets, Firebase credentials, dashboard edits or Firebase deployments were made.


### Dashboard contract alignment verification (2026-09-06)

The sender now follows [the deployed dashboard contract](../../../rail_dashboard/WEBRTC_SIGNALING_CONTRACT.md): lock-based admission, remote robot authorization, fixed one-hour expiry and server-timestamp terminal/candidate writes. `senderId` must equal `senderUid`, the dedicated Firebase Authentication UID. Remote `robots/{robotId}.viewerUids` is mandatory; local `allowedViewerUids` is an optional additional restriction. Existing operators using a label such as `jetson-main` for senderId must correct their private config and trusted robot registration; no credentials are edited by this task.

Coherent read-only transactions inspect robot, active lock and its referenced session. A configuration revision, lock takeover/expiry, heartbeat expiry, terminal state or authentication/permission failure ends sending. Revocation is recognized at the next successful poll or failed authorized read; the independent local deadline remains authoritative during network loss. The device uploader's enabled flag is enforced by rules on those reads, not by an unauthorized direct read of its private document. No session/candidate/lock deletion is issued by Jetson; trusted recursive cleanup remains externally owned.

Local alignment verification: all 29 WebRTC tests passed (20 signaling/session and 9 media tests), 99 Python files compiled, and `git diff --check` passed. Signaling tests use synthetic documents and mocked network/media. Coverage includes remote viewer authorization, missing/changed/expired locks, hard expiry immutability, configuration revision, 403 failure, coherent reads, server timestamp transforms, create-only candidate writes, no DELETE, bounded CAS retry, watchdog race and stalled teardown. Outgoing events are drained at most four per polling cycle, with the callback queue still bounded at 132.

Jetson verification of this revision: updated sender/signaling/media modules, tests and example config were copied to bitdol; all 29 WebRTC tests passed with Python 3.10. `colcon build --packages-select rail_robot_nodes --symlink-install` passed. With ROS_DOMAIN_ID=77 and no credential config, `webrtc.launch.py` started the sender and an intentional six-second SIGINT timeout ended the node cleanly. This launch check did not initiate a media session or authenticate to Firebase.

These checks do not establish production Firestore permission or browser interoperability. The earlier 23-test and LAN performance results above belong to the pre-alignment implementation; they are not a fresh end-to-end validation of this changed signaling contract. No production identities, rules, dashboard files or snapshot code were changed. Actual authenticated dashboard playback remains to be tested after provisioning by the separate dashboard task.
