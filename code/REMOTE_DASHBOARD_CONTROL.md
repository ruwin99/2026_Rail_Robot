# Dashboard remote mission control

The dashboard submits `START` and `E_STOP` requests to
`robots/{robotId}/commands/{commandId}`. The optional Jetson
`remote_control_node` polls Firestore, checks the command and robot configuration
again, and calls the existing ROS 2 `/mission/start` or `/stm/command` service.
The dashboard subscribes to the command document for the result. Video and
telemetry use their existing independent paths.

`E_STOP` here is a software request over the internet. It is **not** a
hardwired emergency stop, and `estop_ack` means the STM acknowledged the
command, not that physical motion has been independently measured to stop.
Maintain a physical stop and local supervision. A lost network request cannot
stop the robot.

## Contract and enablement

- The viewer must be an email-verified administrator in the dashboard rules,
  listed in `robots/{robotId}.viewerUids` and `operatorUids`.
- The robot document must have `enabled: true`, `remoteControlEnabled: true`,
  the existing `senderId`, and the matching `operatorUids` list. Provision it
  with a trusted Firebase administrator, not from the dashboard.
- The Jetson must authenticate as the matching `senderId` and have an enabled
  `device_uploaders/{senderUid}` document. It reuses the existing private
  `RAIL_ROBOT_WEBRTC_CONFIG` file. Never put that file or credentials in Git.
- Deploy the matching `rail_dashboard/firestore.rules` and
  `rail_dashboard/firestore.indexes.json`. The composite `commands` index is
  needed for the priority STOP query; wait until Firebase reports it ready.
  Requested commands expire after at most 15 seconds; the client uses 10 seconds. Only the Jetson
  sender can acknowledge a command. Closed command records cannot be edited
  or deleted by the dashboard or sender.
- Build/install the ROS 2 package after updating the Jetson code, then start
  mission mode with `dashboard_control:=true` and `uart_protocol:=v2`. The
  control flag defaults to `false` and does not follow `dashboard_live` or
  WebRTC automatically. Run `./scripts/run.sh mode:=mission
  dashboard_live:=true dashboard_control:=true uart_protocol:=v2
  serial_port:=/dev/ttyACM0` from `~/Desktop/rail_robot/code`, using the
  actual STM serial path. Verify the `rail_mission` shell function with
  `type rail_mission` before relying on it to forward arguments.

Enable `remoteControlEnabled` and the launch flag only after a supervised
testbed check of the physical stop, STM RR2 firmware, serial link, and local
start/stop behavior. The dashboard `START` button additionally requires fresh
initial `CAPTURE_SCAN` telemetry; the Jetson independently checks mission
state and readiness. `start_sent` records a successful `/mission/start` reply
and UART START send, not confirmed robot movement. A previously accepted START
is never replayed after a Jetson restart. An accepted E_STOP may be retried.

For troubleshooting, inspect the latest command's `state`, `resultCode`,
`createdAt`, `expiresAt`, `acceptedAt`, and `finishedAt`; then inspect the
Jetson `rail_remote_control` log, `/mission/state`, `/runtime/fault`, and STM
UART status. A command stuck in `requested` points to authorization, the
optional receiver, or connectivity. `accepted` means the receiver claimed it,
not that the STM acted. `failed` and `succeeded` are application results; check
the actual robot and physical stop separately. To disable remote control,
set `remoteControlEnabled: false` with a trusted administrator and restart
mission mode without `dashboard_control:=true`.
