# Remote mission control virtual test

This test uses the local Firestore emulator and a `demo-` Firebase project.
It never connects to the production Firestore project or a physical robot.
`virtual-robot` is seeded with `enabled: true`, `remoteControlEnabled: true`,
`senderId: virtual-sender`, `viewerUids` and `operatorUids` containing
`virtual-operator`, plus an enabled `device_uploaders/virtual-sender` record.
Authentication is simulated by `@firebase/rules-unit-testing`; no real
Firebase credentials are needed.

On Windows, run the emulator in one PowerShell terminal from `rail_dashboard`:

```powershell
firebase emulators:start --only firestore --project demo-rail-remote-virtual --config firebase.emulator.json
```

Run the Firestore scenarios in a second terminal. Install Java 21 and the Firebase CLI first. The test uses Python from PATH;
set `RAIL_VIRTUAL_PYTHON` to select another Python 3 executable:

```powershell
node --test tests/remote-control-virtual.test.mjs tests/remote-commands-rules.test.mjs
```

The test exercises authorized START and E_STOP through Firestore rules and
mock services, operator denial, opt-in disabled, mission not ready, and expired
commands. It also runs the production Jetson `RemoteCommandStore` against the
local emulator REST API to query, claim, and finish START and STOP commands. The token
is an unsigned emulator-only test token, passed to Python through an environment
variable. This Store probe checks transport and rules, while the Jetson-side
test checks STOP priority and START blocking. It calls the actual command worker and
ROS node logic with in-memory Firestore, mission, ROS service, and STM fakes:

```powershell
cd ..
python -m unittest discover -s code/tests -p 'test_remote_control*.py' -v
```

Use an installed Python 3 interpreter for the last command. Stop the emulator
with Ctrl+C after testing. The mock `start_sent` and `estop_ack` outcomes mean
only that the simulated service returned success; they do not prove physical
movement or stopping. No test starts `rail_mission`, opens UART, or enables
`robots/rail-robot-01` in production.
