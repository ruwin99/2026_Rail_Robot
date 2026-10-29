# Dashboard `rail_mission` process launch

This is an optional Jetson process launcher. The dashboard's **`rail_mission 실행`**
button starts ROS nodes only. It does **not** call `/mission/start`, write `START`
to UART, or permit motion. The existing in-mission start request remains a
separate operator action after the normal preparation and safety checks.
The supplied mission unit sets `dashboard_control:=true` (with
`uart_protocol:=v2`), so after a dashboard launch the same dashboard can send the
separate remote START/E_STOP commands. Those commands still require the robot
document's `remoteControlEnabled: true`, the operator in `operatorUids`, and a
fresh initial mission state; a LAUNCH itself never sends START.

## Firestore contract

The authenticated operator creates `robots/{robotId}/launchRequests/{requestId}`:

```text
schemaVersion: 1, robotId, ownerUid, action: "LAUNCH_MISSION",
state: "requested", createdAt (server timestamp),
acceptedAt: null, finishedAt: null, resultCode: null
```

The request lives for 10 seconds from the server `createdAt`; no browser clock
is used. During the transition the rules and the receiver still accept the old
shape with an `expiresAt` at most 15 seconds after creation.

The dedicated sender UID alone changes `requested → accepted → succeeded/failed`
and writes `acceptedAt`, `finishedAt`, and `resultCode` with server timestamps.
An accepted request left by a crash is finalized as `launch_outcome_unknown`;
it is never replayed. The sender independently checks `robots/{robotId}` for
`enabled`, `remoteLaunchEnabled`, matching `senderId`, and
the operator in `operatorUids`. The command is fixed to `LAUNCH_MISSION`; no
shell text, unit name, launch flags, or executable path comes from Firestore.

The launcher writes `robots/{robotId}/launcher/status` every 5 seconds with
exactly `{schemaVersion:1, senderId, ready, missionRunning, updatedAt}`.
`updatedAt` is a Firestore server timestamp. The dashboard should enable the
button only when this status is under 15 seconds old, `ready=true`, and
`missionRunning=false`. `launch_started` means the fixed ROS launch process
became active; it does not mean every ROS node is healthy. Use mission telemetry
to determine whether `/mission/start` is actually available.

Deploy matching Firestore rules and add `remoteLaunchEnabled: true` to the
robot document **before** installing the service. Existing WebRTC credentials
in `~/.config/rail_robot/webrtc.json` are reused; no key belongs in Git.

## Install on the Jetson

After the updated repository and ROS build are present on the Jetson, inspect
the absolute paths, Python environment and arguments in the units. They assume
`~/Desktop/rail_robot/code`, ROS Humble, CUDA 12.6 and the project's Python 3.10
venv. The mission and WebRTC units no longer read the interactive `~/.bashrc`:
`scripts/systemd/rail-ros-env.sh` sets the ROS, CUDA, venv and GStreamer paths
explicitly, and both units read the same `~/.config/rail_robot/runtime.env`
(copied from `runtime.env.example`) so their `ROS_DOMAIN_ID` cannot differ.
Engines come from `deployment.py`. The mission unit invokes the fixed launch
arguments with `dashboard_control:=true upload:=true uart_protocol:=v2`. `upload:=true` sends the
finalized report and detection previews to the dashboard after DONE (needs
`~/.config/rail_robot/firebase.env`). The installed unit is a **copy**: after a `git pull` that changes
`rail-mission.service`, copy it again and run `systemctl --user daemon-reload`.
After a mission completes normally the stack exits by itself (after the completion
announcement, 20 s, and the report upload result), so the unit returns to inactive and the dashboard
can launch it again without an STM reset. A failed mission does not exit by itself.

```bash
mkdir -p ~/.config/systemd/user ~/.config/rail_robot
cp -n ~/Desktop/rail_robot/code/scripts/systemd/runtime.env.example ~/.config/rail_robot/runtime.env
cp ~/Desktop/rail_robot/code/scripts/systemd/rail-mission.service ~/.config/systemd/user/
cp ~/Desktop/rail_robot/code/scripts/systemd/rail-mission-launcher.service ~/.config/systemd/user/
cp ~/Desktop/rail_robot/code/scripts/systemd/rail-webrtc.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now rail-mission-launcher.service rail-webrtc.service
systemctl --user status rail-mission-launcher.service rail-webrtc.service --no-pager
systemctl --user show rail-mission.service -p LoadState -p ActiveState
```

### After every `git pull`

The ROS install links each Python file individually (`--symlink-install`), so a
pull that adds a new module under `jetson_code/node_runtime/` fails at start with
`ModuleNotFoundError` until the package is rebuilt. The module list comes from
CMake `file(GLOB ...)`, which is evaluated only when CMake configures, so a plain
rebuild keeps the old list and silently skips new files (seen 2026-10-03 with
`cuda_crack_preprocess.py`). Rebuild with `--cmake-force-configure` after each
pull, then restart the mission if it is running:

```bash
cd ~/Desktop/rail_robot/code/ros2
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select rail_robot_nodes --symlink-install --cmake-force-configure
```

### Logs and clock (one-time, needs sudo)

Keep logs across reboots so a field fault can be inspected after a power cycle:

```bash
sudo mkdir -p /etc/systemd/journald.conf.d /var/log/journal
sudo cp ~/Desktop/rail_robot/code/scripts/systemd/journald-rail.conf /etc/systemd/journald.conf.d/rail.conf
sudo systemctl restart systemd-journald
```

Keep the GPU at 918 MHz or above so per-frame inference is not slowed by the
default governor (realtime crack TensorRT 6.8 ms → 2.6 ms, about +0.7 W at the
module input, measured 2026-10-01 in 25W mode). This is a **system** unit and
runs after `nvpmodel.service`:

```bash
sudo cp ~/Desktop/rail_robot/code/scripts/systemd/rail-gpu-min-freq.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now rail-gpu-min-freq.service
cat /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/min_freq
```

The last line should print `918000000`. To return to the default floor:
`sudo systemctl disable --now rail-gpu-min-freq.service`.

Without an RTC backup battery the Jetson boots with the last shutdown time
and jumps when NTP first answers. The software already copes: Firestore
comparisons use the server time from each response, and `/mission/start`
refuses to send START while the kernel clock is unsynchronized
(`require_clock_sync`, default true). Give the Jetson a reachable time source
so START is possible: internet NTP, an NTP server on the site LAN (a laptop or
router), or HTTPS-based time when UDP 123 is blocked. Prefer `chrony` with
`makestep 1 3` so the clock only steps right after boot and slews afterwards.

`systemctl --user enable` starts the launcher when the **user systemd manager**
runs. It does not by itself guarantee launch before the user logs in. If
operation before login is explicitly required, the operator must separately
decide whether to enable user lingering after reviewing that boot behavior;
this installation does not enable lingering automatically.

Do **not** enable `rail-mission.service` for boot; only the receiver starts on
boot. The receiver uses `systemctl --user start rail-mission.service`, with no
shell and no user-supplied arguments. If `rail_mission` was started manually,
stop it normally before requesting another launch. The receiver also checks
for a running mission node outside the unit and rejects a duplicate request.

A `failed` unit (an earlier abnormal exit) does not block the next launch: the
receiver runs `systemctl --user reset-failed rail-mission.service` and then
starts the unit. Inspect the earlier failure with `journalctl --user -u
rail-mission.service -n 100 --no-pager`. A latched STM FAULT (E_STOP, or a lost
link while prepared/moving) still needs an STM reset; a relaunch then ends with
`STM RR2 MISSION READY timed out`. Check the launcher with `journalctl --user -u
rail-mission-launcher.service -n 100 --no-pager`. If the launcher is down or
Firestore is unreachable, its heartbeat expires and the button becomes
disabled. Existing physical emergency-stop and MCU interlocks remain necessary.
