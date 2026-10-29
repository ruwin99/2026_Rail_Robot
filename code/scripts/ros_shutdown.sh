#!/usr/bin/env bash
set -euo pipefail

# Ask every active rail-robot runtime node to run its normal cleanup path.
# Topics are graph endpoints, so they disappear when their owning nodes exit.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_BASE_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
ROS_LOCAL_SETUP="${RAIL_ROBOT_ROS_SETUP:-${PROJECT_ROOT}/ros2/install/setup.bash}"

if ! command -v ros2 >/dev/null 2>&1 && [[ -f "${ROS_BASE_SETUP}" ]]; then
    set +u
    source "${ROS_BASE_SETUP}"
    set -u
fi
if [[ -f "${ROS_LOCAL_SETUP}" ]]; then
    set +u
    source "${ROS_LOCAL_SETUP}"
    set -u
fi
if ! command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 is unavailable. Source ROS and the local overlay first." >&2
    exit 1
fi

# runtime/fault has reliable event QoS in every robot runtime wrapper. Waiting
# lets current subscribers discover this publisher before the one-shot message.
ros2 topic pub --once --wait-for-subscriptions 1 /runtime/fault std_msgs/msg/String \
    "{data: 'operator shutdown'}"
echo "Shutdown signal sent. The launch terminal should finish after node cleanup."
