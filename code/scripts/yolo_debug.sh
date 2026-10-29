#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RAIL_ROBOT_VENV:-${PROJECT_ROOT}/.venv}"
PYTHON="${VENV}/bin/python"

TOP_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0"
OBSTACLE_ENGINE="/home/bitdol/models/obstacle_yolo26n_foreign_matter_v2/plans/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-int-h736-fp16-notf32-trt10.3.plan"
OBSTACLE_ENGINE_SHA256="cb896747dc9d076a30aedea30517499a3a6d2d0be80a1804c596b8b746a6412c"
OBSTACLE_CONFIDENCE_THRESHOLD="0.424"

if [[ "$#" != "0" ]]; then
    echo "Usage: ./scripts/yolo_debug.sh" >&2
    exit 2
fi
if [[ -n "${NVIDIA_TF32_OVERRIDE:-}" && "${NVIDIA_TF32_OVERRIDE}" != "0" ]]; then
    echo "NVIDIA_TF32_OVERRIDE must be 0 for the pinned obstacle plan." >&2
    exit 2
fi
export NVIDIA_TF32_OVERRIDE=0

bash "${PROJECT_ROOT}/scripts/requirement.sh"
exec "${PYTHON}" "${PROJECT_ROOT}/jetson_code/run.py" --legacy \
    --yolo-debug-benchmark \
    --top-camera-device "${TOP_CAMERA_DEVICE}" \
    --obstacle-engine "${OBSTACLE_ENGINE}" \
    --obstacle-engine-sha256 "${OBSTACLE_ENGINE_SHA256}" \
    --obstacle-confidence-threshold "${OBSTACLE_CONFIDENCE_THRESHOLD}" \
    --headless
