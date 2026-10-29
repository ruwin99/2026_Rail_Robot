"""The operator shutdown helper must source ROS and publish one cleanup event."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


CODE = Path(__file__).resolve().parents[1]


class RosShutdownScriptTests(unittest.TestCase):
    def test_shutdown_publishes_the_reliable_runtime_event(self):
        bash = Path(r"C:\\Program Files\\Git\\bin\\bash.exe")
        bash = str(bash) if bash.is_file() else shutil.which("bash")
        if not bash:
            self.skipTest("bash unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts, binary = root / "scripts", root / "bin"
            scripts.mkdir()
            binary.mkdir()
            helper = scripts / "ros_shutdown.sh"
            helper.write_bytes((CODE / "scripts" / "ros_shutdown.sh").read_bytes())
            helper.chmod(0o755)
            ros2 = binary / "ros2"
            ros2.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${ROUTE_LOG}\"\n", encoding="utf-8")
            ros2.chmod(0o755)
            log = root / "route.txt"
            env = os.environ.copy()
            env.update(PATH=str(binary) + os.pathsep + env["PATH"], ROUTE_LOG=log.as_posix(),
                       RAIL_ROBOT_ROS_SETUP=str(root / "not-installed.bash"))
            result = subprocess.run([bash, str(helper)], env=env, capture_output=True,
                                    text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(log.read_text().splitlines(), [
                "topic", "pub", "--once", "--wait-for-subscriptions", "1", "/runtime/fault",
                "std_msgs/msg/String", "{data: 'operator shutdown'}",
            ])


if __name__ == "__main__":
    unittest.main()
