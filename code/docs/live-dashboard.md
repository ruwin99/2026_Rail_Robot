# Live dashboard observations and captures

This opt-in bridge publishes ROS observations while the robot runs. It does not
send actuator commands and its failure does not stop/restart the robot nodes.
All Firebase I/O runs in background workers, never in ROS subscription callbacks.

## Configuration and launch

Rebuild the ROS workspace after updating the local source on the Jetson. This
change does not deploy anything or modify the Jetson's existing services.

The private `~/.config/rail_robot/webrtc.json` supplies `projectId`, `robotId`,
`senderId`, `senderUid`, `apiKey`, and `refreshToken`. `senderId` must match the
authenticated dedicated sender UID. Add the actual Firebase `storageBucket` for
capture images (or export `FIREBASE_STORAGE_BUCKET`). Do not commit this file.
An alternate file may be selected through `RAIL_ROBOT_WEBRTC_CONFIG`.

### Shell environment for the operator

Keep the existing ROS/CUDA/GStreamer setup. No new Python package is required
for this REST bridge. Add the following to `~/.bashrc` only when applying the
deployment yourself, then run `source ~/.bashrc` in each launch terminal:

```bash
export RAIL_ROBOT_WEBRTC_CONFIG="$HOME/.config/rail_robot/webrtc.json"
# Only needed when storageBucket is empty/missing in the private JSON:
export FIREBASE_STORAGE_BUCKET="YOUR_ACTUAL_FIREBASE_STORAGE_BUCKET"
```

Use the bucket name only (no `gs://` prefix or URL); replace the placeholder
before enabling uploads. A nonempty JSON `storageBucket` takes precedence over
the environment variable. `config/webrtc.example.json` leaves it empty so the
environment fallback can be used. The legacy report/snapshot setting
`RAIL_ROBOT_FIREBASE_STORAGE_BUCKET` does not configure this new capture bridge.
Existing credentials remain in the private JSON; do not copy them into shell
commands, ROS parameters, or repository files.

`dashboard_live:=true` is a launch argument, not an environment variable.
It remains opt-in on each launch. Use the explicit commands below if an existing
`rail_dual` or `rail_mission` shell function does not forward `"$@"`.

### Camera metrics and full mission launch

```bash
cd ~/Desktop/rail_robot/code
# Apply the same ROS/CUDA setup used for the existing camera launcher first.
./scripts/run.sh mode:=dual_camera display:=false gui_output:=true dashboard_live:=true
```

For inspection photo output, stop the dual-camera launch first, then use:

```bash
./scripts/run.sh mode:=mission display:=false gui_output:=true dashboard_live:=true
# Separate prepared terminal, when ready to start the mission:
ros2 service call /mission/start std_srvs/srv/Trigger '{}'
```

Keep the existing WebRTC sender launch/service running separately. It reads the
same private configuration through `RAIL_ROBOT_WEBRTC_CONFIG` when no explicit
`config_file` is passed. `firebase_snapshot:=true` can still enable the existing
five-second snapshot consumer; its existing authentication setup is unchanged.

Dual-camera mode sends topic discovery and metrics, but has no mission owner or
inspection capture service. Mission observations and analyzed inspection captures
require the existing mission launch with `dashboard_live:=true`. This flag never
automatically issues START or enables actuators. The existing WebRTC service still
provides video; telemetry does not replace WebRTC.

## Telemetry contract

`robots/{robotId}/telemetry/current` is replaced approximately every two seconds.
Only one pending latest snapshot is held; connection failures cannot grow a queue.
The dedicated sender writes this document; robot-authorized viewers may read it.

- `schemaVersion: 1`, `senderId`, `updatedAt`: Firestore server timestamp.
- `topics`: at most 128 `{name, publishers, types}` entries with publishers > 0;
  names/types are capped at 256 characters and types at 8. Discovery indicates
  registered publishers, not proof of message flow. No arbitrary topic payloads
  are sent.
- `mission`: null until `mission/state` is observed; otherwise `{phase, completed,
  total, remainingTasks, observedAt}`. `completed` is actual initial_count plus
  rescan_count. `total` is twice the enforced automatic rail section target.
  This measures accepted raw capture section counts, before analysis completes,
  not overall elapsed mission progress. The dashboard labels it 촬영 구간 진행률.
  Remaining tasks describe current/subsequent protocol phases; they are not
  individually acknowledged work items. `failed` is shown as phase `FAILED`.
- `cameras.side` and `.front`: `{fps, rustMs, crackMs, obstacleMs, observedAt}`.
  Missing or >5-second-old individual metrics are null. `top` maps to `front`.
  Timing observations keep ModelTiming completion timestamps. FPS Float64 has
  no timestamp and therefore uses callback receipt time. `observedAt` is the latest
  metric observation; uploads do not refresh source observations. Display stale
  mission and whole-document timestamps separately in the dashboard.

## Capture transport contract

`LiveDashboardClient.upload_image(path, object_key)` accepts JPEG files up to
10 MiB only in `robots/{robotId}/captures/...`, using the same authenticated sender.
It returns the Storage object key, not a public download token. Dashboard viewers
must retrieve images with authorized Firebase Storage requests.
`publish_capture(payload)` writes `robots/{robotId}/captures/{captureId}`, adding
`schemaVersion`, authenticated `senderId`, and server `updatedAt`. The capture
worker supplies per-capture provenance and image keys, retrying independently of
the finalized report uploader. Rules for both new paths must be deployed before
enabling the bridge. No client deletes sessions, locks, or cloud capture history.

## Verification

2026-09-07 local verification: 13 dashboard contract/worker/ROS-adapter tests and
20 launch/routing tests passed. Checks include callback-only worker handoff,
source observation timestamps, stale/null metrics, private image keys and token
refresh, bucket validation without disabling telemetry, and dual-camera launch
without mission/control/UART owners. These are local tests, not Firebase or Jetson
measurements. Existing signaling and executor lifecycle regression tests also pass.

Local tests cover identity/path restrictions, server timestamps, nullable/stale
metrics, actual mission counts, bounded discovery, latest-only handoff and optional
launch isolation. Hardware acceptance remains pending: enable the bridge on the
bench, observe real ROS metrics and newly analyzed capture documents before mission
completion, disconnect network and confirm ROS callbacks/control stay responsive,
then restore network and confirm retries recover without exposing credentials.
