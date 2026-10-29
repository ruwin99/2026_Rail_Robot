"""Source-tree convenience entry; installed nodes use entrypoints.py directly."""
import subprocess
import sys


def main(args=None):
    arguments = list(sys.argv[1:] if args is None else args)
    if arguments == ["--training"]:
        arguments = ["mode:=training"]
    if any(argument.startswith("--") and argument not in ("--help", "--show-args", "--debug")
           for argument in arguments):
        print("Use ROS launch arguments such as mode:=vision display:=false. "
              "Retained specialist tests require run.py --legacy.", file=sys.stderr)
        return 2
    try:
        return subprocess.call(["ros2", "launch", "rail_robot_nodes", "robot.launch.py", *arguments])
    except FileNotFoundError:
        print("ROS 2 is unavailable. Source ROS and code/ros2/install/setup.bash first.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
