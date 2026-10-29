#!/usr/bin/env bash
# systemd 유닛용 ROS 실행 환경. 대화형 ~/.bashrc에 의존하지 않고 필요한 값만 명시한다.
# ROS_DOMAIN_ID 같은 운용 값은 두 유닛이 함께 읽는 ~/.config/rail_robot/runtime.env 하나에서 온다.
# 사용: rail-ros-env.sh <명령> [인자...]   (예: ./scripts/run.sh mode:=mission ...)
set -eo pipefail

CODE="${RAIL_ROBOT_HOME:-$HOME/Desktop/rail_robot/code}"
: "${ROS_DOMAIN_ID:?ROS_DOMAIN_ID must be set in ~/.config/rail_robot/runtime.env}"

export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-0}"

# ROS setup 스크립트는 정의되지 않은 변수를 참조하므로 set -u 없이 불러온다.
source /opt/ros/humble/setup.bash
if [[ -f "$CODE/ros2/install/setup.bash" ]]; then
  source "$CODE/ros2/install/setup.bash"
fi

# 프로젝트 venv의 numpy·OpenCV 등을 ROS Python 경로보다 먼저 찾는다(.bashrc와 같은 순서).
export PYTHONPATH="$CODE/.venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export GST_PLUGIN_PATH="$HOME/.local/share/rail_robot/gst-deps/usr/lib/aarch64-linux-gnu/gstreamer-1.0${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}"

cd "$CODE"
exec "$@"
