# Camera-only video and incremental capture output

These are local source changes. They have not been installed or measured on Jetson.

## Video and inspection photos are separate

The existing GUI JPEG/WebRTC stream is already live during a mission. Previously,
inspection raw images, analyzed images, XLSX, and dashboard records were generated
locally for each capture, but the ROS cloud upload node received a finalized
manifest only after all initial/reinspection captures completed.

The new optional capture output publishes one raw image notification after the
raw capture is saved, followed by the corresponding annotated image after that
capture's analysis and local reporting finish. It does not wait for mission
completion. Normal finalized reports and their upload pipeline are preserved.

## Enable on a future Jetson deployment

After deploying and rebuilding the ROS workspace, use the launch parameter
`dashboard_live:=true` with the normal mission launcher. The feature is off by
default. An inspection-producing mission mode is required; a camera-only test
without capture requests does not produce a captured-photo history.

The private configuration is read from `RAIL_ROBOT_WEBRTC_CONFIG`, or
`~/.config/rail_robot/webrtc.json`. In addition to the existing project, robot, and
sender authentication configuration, set the actual Firebase `storageBucket`
(or `FIREBASE_STORAGE_BUCKET`). Do not commit credentials. Deploy matching
Firestore and Storage rules before attempting uploads.

Missing or malformed configuration disables live capture output with a warning;
local inspection continues. Correct the configuration and restart the inspection
process to enable uploads. Network upload failures retry in the background.

## Delivery and recovery

- Firestore: `robots/{robotId}/captures/cap_{32-character-hash}`.
- Storage: `robots/{robotId}/captures/{captureId}/raw.jpg` and `processed.jpg`.
- Each event includes run ID, camera role, phase, zone, original capture time,
  analysis completion time, status, and object keys. `top` is displayed as
  `front`, matching the GUI. The run ID matches the finalized dashboard export.
- The persistent outbox is
  `~/.local/share/rail_robot/live-capture-outbox/{robotId}`. Up to 1,000 pending
  capture records are retained. Each record references the saved raw JPEG and
  its `_live_processed.jpg` neighbor in the capture directory.
- Network operations run on a dedicated thread. Retry passes occur every five
  seconds (or earlier when new captures arrive). A slow upload does not hold the
  capture/analysis lock. Delivery time still depends on network and backlog.
- Restarting an inspection process with live output enabled resumes pending
  records, including earlier runs. Do not delete the referenced capture tree or
  outbox before delivery. Raw-only records remain when analysis never completes.
- Disk-full/capacity errors are logged; originals and the existing finalized
  report flow remain available. Live upload failures do not issue robot faults.

## Camera-only composition

Remote JPEG output is now 1280 x 360: SIDE and FRONT camera panes are each
uniformly reduced from 1280 x 720 to 640 x 360. ROI guides and matched detection
annotations remain. Mission text and timing tables are sent as structured GUI
status data instead of being burned into the remote video. The local HighGUI
window retains its existing 1280 x 720 diagnostic layout.

WebRTC negotiates the JPEG's real dimensions without stretching, accepting both
the new 1280 x 360 and legacy 1280 x 720 publishers. Snapshot and probe consumers
accept both. Update sender and display code together. Native GStreamer encoder
negotiation, Firebase delivery latency, and camera performance still require
Jetson testing; desktop tests do not establish their measured performance.

## Local verification (2026-09-07)

46 tests passed without skips across capture outbox, ROS inspection, GUI output,
snapshot/probe consumers, WebRTC media, and the LAN-check marker tests. These use
ROS/GStreamer doubles and real local OpenCV JPEG encoding/decoding. The operator
LAN-check marker now uses the actual frame bottom and is tested with both 360-
and 720-pixel heights. Missing/invalid capture bucket configuration is checked
before starting the outbox; local inspection continues when it is disabled.
Outbox write failures are also checked not to fault analysis.

No Jetson deployment, shell/service edit, Firebase write, or actual LAN stream
test was performed in this verification.
