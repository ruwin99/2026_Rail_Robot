#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# 경로는 현재 터미널 위치가 아니라 이 스크립트 위치로 계산한다.
# ROS 패키지를 못 찾으면 아래 setup 경로와 ros2/install 빌드 결과를 먼저 확인한다.
# [검토 시작점] 기본 실행은 ROS 2 launch로 넘긴다. 실행할 노드 조합과 기본 모드는
# ros2/rail_robot_nodes/launch/robot.launch.py에서 결정하며, 이 셸이 FSM을 돌리지는 않는다.
# --legacy를 첫 인자로 준 경우에만 아래쪽의 기존 run.py 실행 경로를 사용한다.
if [[ "${1:-}" != "--legacy" ]]; then
    # [1. 실행 인자 검사] --training 하나만 쓰는 기존 호출을 mode:=training으로 바꾼다.
    # 다른 ROS 설정을 함께 주려면 처음부터 mode:=training 형식으로 입력해야 한다.
    if [[ "${1:-}" == "--training" ]]; then
        if [[ "$#" != "1" ]]; then
            echo "Use mode:=training with ROS launch arguments, or --training alone." >&2
            exit 2
        fi
        set -- mode:=training
    fi
    for argument in "$@"; do
        # 예전 run.py 전용 플래그를 ROS launch로 넘기지 않도록 차단한다.
        # 여기서 exit 2이면 노드/장치 오류가 아니라 호출 형식 오류다. 출력된 안내와 인자를 확인한다.
        case "${argument}" in
            --capture-test|--front-capture-test|--realtime-test|--realtime-test-uart|--no-uart|--no-crack|--headless)
                echo "Use ROS launch arguments (for example mode:=vision display:=false), or --legacy for the retained specialist tests." >&2
                exit 2
                ;;
        esac
    done
    ROS_BASE_SETUP="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
    ROS_LOCAL_SETUP="${RAIL_ROBOT_ROS_SETUP:-${PROJECT_ROOT}/ros2/install/setup.bash}"
    # ros2 명령과 로컬 패키지를 찾도록 환경을 불러온다. 설치/빌드 자체를 수행하지는 않는다.
    # [2. ROS 환경] 기본 ROS는 ros2 명령이 없을 때만, 프로젝트 overlay는 파일이 있으면 불러온다.
    # source 중 미정의 변수를 허용하려고 잠시 set +u를 사용하고 이후 엄격 모드로 복구한다.
    # 이 분기에서 venv의 TensorRT/CUDA 설치나 PYTHONPATH 보충까지 하지는 않는다.
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
        # ROS 자체가 없으면 여기서 종료한다. ros2는 있지만 rail_robot_nodes를 못 찾는 오류는
        # 다음 exec에서 발생하므로 로컬 패키지 빌드와 ROS_LOCAL_SETUP 적용 여부를 구별해서 본다.
        echo "ROS 2 is unavailable. Source your ROS installation and build code/ros2 first; see code/ros2/README.md." >&2
        exit 1
    fi
    # mode:=dual_camera, display:=false 등의 인자를 그대로 전달한다.
    # exec로 셸을 launch 프로세스로 바꿔 Ctrl+C가 실행 프로세스에 전달되도록 한다.
    # [3. 실행권 전달] 인자가 없으면 launch 기본 mode=mission을 사용한다.
    # 실제 주행 시작은 mission/start 요청과 STM 승인 단계이므로 이 줄에서 START를 쓰지는 않는다.
    exec ros2 launch rail_robot_nodes robot.launch.py "$@"
fi
shift

# 여기부터는 --legacy 전용이다. 일반 ROS 실행의 모델 설정은 node_runtime/ros/deployment.py 참조.
# [기존 실행 경로] shift로 첫 --legacy만 제거했으며 남은 인자는 아래에서 검증한다.
# ROS 모드를 디버깅할 때 아래 엔진 변수만 바꿔도 위 exec 경로의 노드 설정은 바뀌지 않는다.
LOCAL_VENV="${PROJECT_ROOT}/.venv"
VENV="${RAIL_ROBOT_VENV:-${LOCAL_VENV}}"
export RAIL_ROBOT_VENV="${VENV}"
PYTHON="${VENV}/bin/python"

TRAINING=0
# [기존 원본 수집] --legacy --training은 OpenCV만 확인한 후 training_capture.py로 바로 간다.
# 아래 임무용 모델 설정이나 requirement.sh까지 내려가지 않으므로 원본 수집 실패를 별도로 추적한다.
for argument in "$@"; do
    if [[ "${argument}" == "--training" ]]; then
        TRAINING=1
    fi
done
if [[ "${TRAINING}" == "1" ]]; then
    if [[ "$#" != "1" || "$1" != "--training" ]]; then
        echo "--training must be used alone." >&2
        exit 2
    fi
    if [[ ! -x "${PYTHON}" ]]; then
        echo "Python virtual environment is unavailable: ${PYTHON}" >&2
        exit 1
    fi
    if ! "${PYTHON}" -c 'import cv2' 2>/dev/null; then
        echo "OpenCV is unavailable in ${VENV}." >&2
        exit 1
    fi
    exec "${PYTHON}" "${PROJECT_ROOT}/jetson_code/training_capture.py"
fi

for argument in "$@"; do
    if [[ "${argument}" == "--no-crack" ]]; then
        echo "--no-crack has been removed from run.sh." >&2
        exit 2
    fi
done

APPROVED_REALTIME_RUST_ENGINE_PATH="/home/bitdol/models/plans_new/realtime-rust-mnv2-os16-finetuned-w1280-h240-fp16-super25w-7d61dbcf.plan"
APPROVED_REALTIME_RUST_ENGINE_SHA256="e5093573ce43ec5f4c37b5e4de5321bd9af9dfacbcbb3cdc28f51bdebe0997cc"
APPROVED_CAPTURE_TEST_RUST_ENGINE_PATH="/home/bitdol/models/capture_r101_hardneg_v6_20260821/plans/capture-rust-r101-os8-hardneg-v6-w1280-h720-fp32-notf32.plan"
APPROVED_CAPTURE_TEST_RUST_ENGINE_SHA256="7c442abf7049bcc56d093e0dc7dd2a347caaee74dc8e20006b63ee032b6f22d4"
APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_PATH="/home/bitdol/models/hrsegnet_b32_steelcrack_aug400_20260924/plans/hrsegnet-b32-steelcrack-aug400-realtime-w1280-h128-fp16.plan"
APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256="964306b153b8ae9580851bb9e93157c621fbb397c57e8e5681ea3c4e71f72ce6"
APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_PATH="/home/bitdol/models/hrsegnet_b32_20260816/plans/hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan"
APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256="01c2dc7e4467e8888eda7eea68670ba6598ad4f7b664e57d420bda76e293111a"
APPROVED_SIDE_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
APPROVED_TOP_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0"
APPROVED_OBSTACLE_ENGINE_PATH="/home/bitdol/models/obstacle_yolo26n_foreign_matter_v2/plans/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-int-h736-fp16-notf32-trt10.3.plan"
APPROVED_OBSTACLE_ENGINE_SHA256="cb896747dc9d076a30aedea30517499a3a6d2d0be80a1804c596b8b746a6412c"
APPROVED_OBSTACLE_CONFIDENCE_THRESHOLD="0.424"

STUDENT_ENGINE_PATH="${RAIL_ROBOT_REALTIME_RUST_ENGINE:-${RAIL_ROBOT_STUDENT_ENGINE:-${RAIL_ROBOT_ENGINE:-${APPROVED_REALTIME_RUST_ENGINE_PATH}}}}"
OPTIMIZED_STUDENT_ENGINE_PATH="${RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE:-}"
TEACHER_ENGINE_PATH="${RAIL_ROBOT_CAPTURE_RUST_ENGINE:-${RAIL_ROBOT_TEACHER_ENGINE:-${HOME}/models/plans_new/corrosion-capture-r101-os8-w1280-h720-fp32.plan}}"
CAPTURE_HRSEGNET_CRACK_ENGINE_PATH="${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}"
REALTIME_CRACK_ENGINE_PATH="${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-${HOME}/models/plans_new/realtime-crack-bgcrack-w1280-h128-fp32.plan}"
REALTIME_MULTITASK_ENGINE_PATH="${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE:-}"
REALTIME_HRSEGNET_CRACK_ENGINE_PATH="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE:-}"
STUDENT_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256:-${RAIL_ROBOT_STUDENT_ENGINE_SHA256:-${RAIL_ROBOT_ENGINE_SHA256:-${APPROVED_REALTIME_RUST_ENGINE_SHA256}}}}"
OPTIMIZED_STUDENT_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE_SHA256:-}"
TEACHER_ENGINE_SHA256="${RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256:-${RAIL_ROBOT_TEACHER_ENGINE_SHA256:-}}"
CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256="${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}"
REALTIME_CRACK_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}"
REALTIME_MULTITASK_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256:-}"
REALTIME_HRSEGNET_CRACK_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256:-}"
CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD="0.55"
REALTIME_CRACK_THRESHOLD="${RAIL_ROBOT_REALTIME_CRACK_THRESHOLD:-0.5}"
CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS="20"
REALTIME_CRACK_MIN_COMPONENT_PIXELS="${RAIL_ROBOT_REALTIME_CRACK_MIN_COMPONENT_PIXELS:-20}"
REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD:-0.55}"
REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS:-20}"

if [[ -n "${RAIL_ROBOT_CRACK_ENGINE:-}" || -n "${RAIL_ROBOT_CRACK_ENGINE_SHA256:-}" ]]; then
    echo "RAIL_ROBOT_CRACK_ENGINE is no longer supported; capture and realtime HrSegNet baselines are pinned separately." >&2
    exit 2
fi
if [[ -n "${RAIL_ROBOT_CRACK_THRESHOLD:-}" ]]; then
    echo "RAIL_ROBOT_CRACK_THRESHOLD is no longer supported; use the pinned role-specific HrSegNet baselines." >&2
    exit 2
fi
if [[ -n "${RAIL_ROBOT_CRACK_MIN_COMPONENT_PIXELS:-}" ]]; then
    echo "RAIL_ROBOT_CRACK_MIN_COMPONENT_PIXELS is no longer supported; use the pinned role-specific HrSegNet baselines." >&2
    exit 2
fi

if [[ -n "${RAIL_ROBOT_CAPTURE_CRACK_ENGINE:-}" ||
    -n "${RAIL_ROBOT_CAPTURE_CRACK_ENGINE_SHA256:-}" ||
    -n "${RAIL_ROBOT_CAPTURE_CRACK_THRESHOLD:-}" ||
    -n "${RAIL_ROBOT_CAPTURE_CRACK_MIN_COMPONENT_PIXELS:-}" ]]; then
    echo "Ignoring legacy capture BGCrack settings; capture now uses the pinned original HrSegNet-B32 full-frame baseline." >&2
fi
if [[ ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE}" != "${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}" ) ||
    ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}" != "${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}" ) ||
    ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}" != "0.55" ) ||
    ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}" != "20" ) ]]; then
    echo "Ignoring unapproved capture HrSegNet overrides; run.sh uses the pinned plan, SHA-256, threshold, and component baseline." >&2
fi

CAPTURE_TEST=0
FRONT_CAPTURE_TEST=0
REALTIME_TEST=0
HRSEGNET_CLI_CONFIGURED=0
REALTIME_MODEL_CLI_CONFIGURED=0
CAPTURE_MODEL_CLI_CONFIGURED=0
TEACHER_CLI_CONFIGURED=0
NORMAL_CAMERA_OR_OBSTACLE_CLI_CONFIGURED=0
for argument in "$@"; do
    if [[ "${argument}" == "--capture-test" ]]; then
        CAPTURE_TEST=1
    elif [[ "${argument}" == "--front-capture-test" ]]; then
        FRONT_CAPTURE_TEST=1
    elif [[ "${argument}" == "--realtime-test" ]]; then
        REALTIME_TEST=1
    elif [[ "${argument}" == --realtime-hrsegnet-crack-* ]]; then
        HRSEGNET_CLI_CONFIGURED=1
        REALTIME_MODEL_CLI_CONFIGURED=1
    elif [[ "${argument}" == --capture-crack-* ||
        "${argument}" == --capture-hrsegnet-crack-* ]]; then
        CAPTURE_MODEL_CLI_CONFIGURED=1
    elif [[ "${argument}" == --teacher-engine* ]]; then
        TEACHER_CLI_CONFIGURED=1
    elif [[ "${argument}" == --student-engine* ||
        "${argument}" == --engine* ||
        "${argument}" == --optimized-student-engine* ||
        "${argument}" == --realtime-crack-* ||
        "${argument}" == --realtime-multitask-* ]]; then
        REALTIME_MODEL_CLI_CONFIGURED=1
    elif [[ "${argument}" == --camera-index ||
        "${argument}" == --camera-index=* ||
        "${argument}" == --capture-camera-device ||
        "${argument}" == --capture-camera-device=* ||
        "${argument}" == --side-camera-device ||
        "${argument}" == --side-camera-device=* ||
        "${argument}" == --top-camera-device ||
        "${argument}" == --top-camera-device=* ||
        "${argument}" == --obstacle-engine ||
        "${argument}" == --obstacle-engine=* ||
        "${argument}" == --obstacle-engine-sha256 ||
        "${argument}" == --obstacle-engine-sha256=* ||
        "${argument}" == --obstacle-confidence-threshold ||
        "${argument}" == --obstacle-confidence-threshold=* ]]; then
        NORMAL_CAMERA_OR_OBSTACLE_CLI_CONFIGURED=1
    fi
done

if [[ "${CAPTURE_TEST}" == "0" && "${FRONT_CAPTURE_TEST}" == "0" &&
    "${REALTIME_TEST}" == "0" &&
    "${NORMAL_CAMERA_OR_OBSTACLE_CLI_CONFIGURED}" == "1" ]]; then
    echo "Normal run.sh missions use pinned SIDE/TOP cameras and the pinned obstacle model; camera and obstacle CLI overrides are disabled." >&2
    exit 2
fi

if [[ "${CAPTURE_MODEL_CLI_CONFIGURED}" == "1" ]]; then
    echo "Capture model CLI overrides are disabled in run.sh; the pinned original HrSegNet-B32 full-frame baseline is mandatory." >&2
    exit 2
fi
if [[ "${FRONT_CAPTURE_TEST}" == "1" ]]; then
    if [[ "$#" != "1" || "$1" != "--front-capture-test" ]]; then
        echo "--front-capture-test must be used alone; its camera and models are pinned." >&2
        exit 2
    fi
    if [[ "${CAPTURE_TEST}" == "1" || "${REALTIME_TEST}" == "1" ]]; then
        echo "--front-capture-test cannot be combined with other test modes." >&2
        exit 2
    fi
    if [[ "${NORMAL_CAMERA_OR_OBSTACLE_CLI_CONFIGURED}" == "1" ||
        "${TEACHER_CLI_CONFIGURED}" == "1" ||
        "${REALTIME_MODEL_CLI_CONFIGURED}" == "1" ]]; then
        echo "FRONT capture camera and model overrides are disabled; the approved TOP camera, obstacle, and capture HrSegNet baselines are pinned." >&2
        exit 2
    fi
    if [[ -n "${NVIDIA_TF32_OVERRIDE:-}" && "${NVIDIA_TF32_OVERRIDE}" != "0" ]]; then
        echo "NVIDIA_TF32_OVERRIDE must be 0 because the pinned plans were built with TF32 disabled." >&2
        exit 2
    fi
    export NVIDIA_TF32_OVERRIDE=0
    bash "${PROJECT_ROOT}/scripts/requirement.sh"
    exec "${PYTHON}" "${PROJECT_ROOT}/jetson_code/run.py" --legacy \
        --front-capture-test \
        --capture-camera-device "${APPROVED_TOP_CAMERA_DEVICE}" \
        --capture-hrsegnet-crack-engine "${CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}" \
        --capture-hrsegnet-crack-engine-sha256 "${CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}" \
        --capture-hrsegnet-crack-probability-threshold "${CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}" \
        --capture-hrsegnet-crack-min-component-pixels "${CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}" \
        --obstacle-engine "${APPROVED_OBSTACLE_ENGINE_PATH}" \
        --obstacle-engine-sha256 "${APPROVED_OBSTACLE_ENGINE_SHA256}" \
        --obstacle-confidence-threshold "${APPROVED_OBSTACLE_CONFIDENCE_THRESHOLD}"
fi
if [[ "${CAPTURE_TEST}" == "1" && "${TEACHER_CLI_CONFIGURED}" == "1" ]]; then
    echo "Teacher engine CLI overrides are disabled for --capture-test." >&2
    exit 2
fi
if [[ "${CAPTURE_TEST}" == "1" ]]; then
    if [[ ( -n "${RAIL_ROBOT_CAPTURE_RUST_ENGINE:-}" &&
        "${RAIL_ROBOT_CAPTURE_RUST_ENGINE}" != "${APPROVED_CAPTURE_TEST_RUST_ENGINE_PATH}" ) ||
        ( -n "${RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256:-}" &&
        "${RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256}" != "${APPROVED_CAPTURE_TEST_RUST_ENGINE_SHA256}" ) ||
        -n "${RAIL_ROBOT_TEACHER_ENGINE:-}" ||
        -n "${RAIL_ROBOT_TEACHER_ENGINE_SHA256:-}" ]]; then
        echo "Ignoring capture rust environment overrides; --capture-test uses the pinned hard-negative R101 plan." >&2
    fi
    TEACHER_ENGINE_PATH="${APPROVED_CAPTURE_TEST_RUST_ENGINE_PATH}"
    TEACHER_ENGINE_SHA256="${APPROVED_CAPTURE_TEST_RUST_ENGINE_SHA256}"
    echo "Capture test rust engine (pinned): ${TEACHER_ENGINE_PATH}" >&2
    echo "Capture test rust SHA-256: ${TEACHER_ENGINE_SHA256}" >&2
fi

if [[ "${CAPTURE_TEST}" == "0" ]]; then
    if [[ "${REALTIME_MODEL_CLI_CONFIGURED}" == "1" ]]; then
        echo "Realtime model CLI overrides are disabled in run.sh; the pinned Rust + HrSegNet baseline is mandatory." >&2
        exit 2
    fi
    if [[ -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-}" ||
        -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}" ||
        -n "${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE:-}" ||
        -n "${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256:-}" ]]; then
        echo "Ignoring legacy BGCrack/multitask realtime environment settings; the approved HrSegNet baseline is pinned." >&2
    fi
    if [[ ( -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE:-}" &&
        "${RAIL_ROBOT_REALTIME_RUST_ENGINE}" != "${APPROVED_REALTIME_RUST_ENGINE_PATH}" ) ||
        ( -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256:-}" &&
        "${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256}" != "${APPROVED_REALTIME_RUST_ENGINE_SHA256}" ) ||
        ( -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE:-}" &&
        "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE}" != "${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" ) ||
        ( -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256:-}" &&
        "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}" != "${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}" ) ]]; then
        echo "Ignoring unapproved realtime Rust/HrSegNet environment overrides; run.sh uses the pinned baseline paths and SHA-256 values." >&2
    fi
    unset RAIL_ROBOT_REALTIME_CRACK_ENGINE
    unset RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256
    unset RAIL_ROBOT_REALTIME_MULTITASK_ENGINE
    unset RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256
    unset RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE
    unset RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE_SHA256
    REALTIME_CRACK_ENGINE_PATH=""
    REALTIME_CRACK_ENGINE_SHA256=""
    REALTIME_MULTITASK_ENGINE_PATH=""
    REALTIME_MULTITASK_ENGINE_SHA256=""
    OPTIMIZED_STUDENT_ENGINE_PATH=""
    OPTIMIZED_STUDENT_ENGINE_SHA256=""
    STUDENT_ENGINE_PATH="${APPROVED_REALTIME_RUST_ENGINE_PATH}"
    STUDENT_ENGINE_SHA256="${APPROVED_REALTIME_RUST_ENGINE_SHA256}"
    REALTIME_HRSEGNET_CRACK_ENGINE_PATH="${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_PATH}"
    REALTIME_HRSEGNET_CRACK_ENGINE_SHA256="${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}"
    REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD="0.55"
    REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS="20"
fi

HRSEGNET_CONFIGURED=0
if [[ "${CAPTURE_TEST}" == "0" && (
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE:-}" ||
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256:-}" ||
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD:-}" ||
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS:-}"
) ]]; then
    HRSEGNET_CONFIGURED=1
fi
if [[ "${CAPTURE_TEST}" == "0" ]]; then
    HRSEGNET_CONFIGURED=1
fi
HRSEGNET_SELECTED=0
if [[ "${HRSEGNET_CONFIGURED}" == "1" || "${HRSEGNET_CLI_CONFIGURED}" == "1" ]]; then
    HRSEGNET_SELECTED=1
fi
if [[ -z "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" && "${HRSEGNET_CONFIGURED}" == "1" ]]; then
    echo "HrSegNet threshold, component, and SHA settings require RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE." >&2
    exit 2
fi
if [[ "${HRSEGNET_SELECTED}" == "1" && -n "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" && -z "${REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}" ]]; then
    echo "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE requires RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256." >&2
    exit 2
fi
if [[ "${HRSEGNET_SELECTED}" == "1" && (
    -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-}" ||
    -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}" ||
    -n "${REALTIME_MULTITASK_ENGINE_PATH}" ||
    -n "${REALTIME_MULTITASK_ENGINE_SHA256}"
) ]]; then
    echo "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE cannot be combined with BGCrack or multitask realtime crack settings." >&2
    exit 2
fi

SEPARATE_RUST_PATH_CONFIGURED=0
SEPARATE_RUST_SHA_CONFIGURED=0
if [[ "${CAPTURE_TEST}" == "0" ]]; then
    RAW_RUST_PATH_CONFIGURED=0
    RAW_RUST_SHA_CONFIGURED=0
    if [[ -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE:-}" ||
        -n "${RAIL_ROBOT_STUDENT_ENGINE:-}" ||
        -n "${RAIL_ROBOT_ENGINE:-}" ]]; then
        RAW_RUST_PATH_CONFIGURED=1
    fi
    if [[ -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256:-}" ||
        -n "${RAIL_ROBOT_STUDENT_ENGINE_SHA256:-}" ||
        -n "${RAIL_ROBOT_ENGINE_SHA256:-}" ]]; then
        RAW_RUST_SHA_CONFIGURED=1
    fi
    if [[ ( "${RAW_RUST_PATH_CONFIGURED}" == "1" ||
        "${RAW_RUST_SHA_CONFIGURED}" == "1" ) && (
        -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ||
        -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}"
    ) ]]; then
        echo "Raw realtime rust engine settings cannot be combined with RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE settings." >&2
        exit 2
    fi
    if [[ -z "${OPTIMIZED_STUDENT_ENGINE_PATH}" &&
        -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}" ]]; then
        echo "RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE_SHA256 requires RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE." >&2
        exit 2
    fi
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" && -z "${REALTIME_MULTITASK_ENGINE_SHA256}" ]]; then
        echo "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE requires RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256." >&2
        exit 2
    fi
    if [[ -z "${REALTIME_MULTITASK_ENGINE_PATH}" && -n "${REALTIME_MULTITASK_ENGINE_SHA256}" ]]; then
        echo "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256 requires RAIL_ROBOT_REALTIME_MULTITASK_ENGINE." >&2
        exit 2
    fi
    if [[ -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ||
        "${RAW_RUST_PATH_CONFIGURED}" == "1" ]]; then
        SEPARATE_RUST_PATH_CONFIGURED=1
    fi
    if [[ -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}" ||
        "${RAW_RUST_SHA_CONFIGURED}" == "1" ]]; then
        SEPARATE_RUST_SHA_CONFIGURED=1
    fi
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" &&
        "${SEPARATE_RUST_PATH_CONFIGURED}" != "${SEPARATE_RUST_SHA_CONFIGURED}" ]]; then
        echo "Hybrid realtime mode requires both the separate realtime rust engine and its SHA-256." >&2
        exit 2
    fi
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" && (
        -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-}" ||
        -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}"
    ) ]]; then
        echo "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE cannot be combined with a separate realtime crack engine; multitask supplies the crack decision." >&2
        exit 2
    fi
fi
if [[ -n "${NVIDIA_TF32_OVERRIDE:-}" && "${NVIDIA_TF32_OVERRIDE}" != "0" ]]; then
    echo "NVIDIA_TF32_OVERRIDE must be 0 because the crack plans were built with TF32 disabled; this setting applies to every active plan in this process." >&2
    exit 2
fi
export NVIDIA_TF32_OVERRIDE=0

ENGINE_ARGUMENTS=(--teacher-engine "${TEACHER_ENGINE_PATH}")
if [[ -n "${TEACHER_ENGINE_SHA256}" ]]; then
    ENGINE_ARGUMENTS+=(--teacher-engine-sha256 "${TEACHER_ENGINE_SHA256}")
fi

NORMAL_DUAL_CAMERA_ARGUMENTS=()
if [[ "${CAPTURE_TEST}" == "0" && "${REALTIME_TEST}" == "0" ]]; then
    NORMAL_DUAL_CAMERA_ARGUMENTS=(
        --side-camera-device "${APPROVED_SIDE_CAMERA_DEVICE}"
        --top-camera-device "${APPROVED_TOP_CAMERA_DEVICE}"
        --obstacle-engine "${APPROVED_OBSTACLE_ENGINE_PATH}"
        --obstacle-engine-sha256 "${APPROVED_OBSTACLE_ENGINE_SHA256}"
        --obstacle-confidence-threshold "${APPROVED_OBSTACLE_CONFIDENCE_THRESHOLD}"
    )
    echo "Normal mission dual-camera baseline: SIDE=${APPROVED_SIDE_CAMERA_DEVICE}, TOP=${APPROVED_TOP_CAMERA_DEVICE}." >&2
fi

if [[ "${CAPTURE_TEST}" == "1" ]]; then
    ENGINE_ARGUMENTS+=(
        --capture-hrsegnet-crack-engine "${CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}"
        --capture-hrsegnet-crack-engine-sha256 "${CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}"
        --capture-hrsegnet-crack-probability-threshold "${CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}"
        --capture-hrsegnet-crack-min-component-pixels "${CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}"
    )
else
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" ]]; then
        ENGINE_ARGUMENTS+=(
            --realtime-multitask-engine "${REALTIME_MULTITASK_ENGINE_PATH}"
            --realtime-multitask-engine-sha256 "${REALTIME_MULTITASK_ENGINE_SHA256}"
        )
        if [[ "${SEPARATE_RUST_PATH_CONFIGURED}" == "1" ]]; then
            if [[ -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ]]; then
                ENGINE_ARGUMENTS+=(
                    --optimized-student-engine "${OPTIMIZED_STUDENT_ENGINE_PATH}"
                    --optimized-student-engine-sha256 "${OPTIMIZED_STUDENT_ENGINE_SHA256}"
                )
            else
                ENGINE_ARGUMENTS+=(
                    --student-engine "${STUDENT_ENGINE_PATH}"
                    --student-engine-sha256 "${STUDENT_ENGINE_SHA256}"
                )
            fi
        fi
    else
        if [[ -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ]]; then
            ENGINE_ARGUMENTS+=(
                --optimized-student-engine "${OPTIMIZED_STUDENT_ENGINE_PATH}"
            )
            if [[ -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}" ]]; then
                ENGINE_ARGUMENTS+=(
                    --optimized-student-engine-sha256 "${OPTIMIZED_STUDENT_ENGINE_SHA256}"
                )
            fi
        else
            ENGINE_ARGUMENTS+=(--student-engine "${STUDENT_ENGINE_PATH}")
            if [[ -n "${STUDENT_ENGINE_SHA256}" ]]; then
                ENGINE_ARGUMENTS+=(--student-engine-sha256 "${STUDENT_ENGINE_SHA256}")
            fi
        fi
    fi

    if [[ "${REALTIME_TEST}" == "0" ]]; then
        ENGINE_ARGUMENTS+=(
            --capture-hrsegnet-crack-engine "${CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}"
            --capture-hrsegnet-crack-engine-sha256 "${CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}"
            --capture-hrsegnet-crack-probability-threshold "${CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}"
            --capture-hrsegnet-crack-min-component-pixels "${CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}"
        )
    fi
    if [[ "${HRSEGNET_SELECTED}" == "1" ]]; then
        if [[ -n "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" ]]; then
            ENGINE_ARGUMENTS+=(
                --realtime-hrsegnet-crack-engine "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}"
                --realtime-hrsegnet-crack-engine-sha256 "${REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}"
                --realtime-hrsegnet-crack-probability-threshold "${REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}"
                --realtime-hrsegnet-crack-min-component-pixels "${REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}"
            )
            echo "Realtime crack baseline: operator-selected pinned HrSegNet threshold=${REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}, min_component_pixels=${REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}; field accuracy is not certified." >&2
        else
            echo "WARNING: UNLOCKED exploratory HrSegNet CLI settings; TEST ONLY." >&2
        fi
    else
        ENGINE_ARGUMENTS+=(
            --realtime-crack-threshold "${REALTIME_CRACK_THRESHOLD}"
            --realtime-crack-min-component-pixels "${REALTIME_CRACK_MIN_COMPONENT_PIXELS}"
        )
        if [[ -z "${REALTIME_MULTITASK_ENGINE_PATH}" ]]; then
            ENGINE_ARGUMENTS+=(--realtime-crack-engine "${REALTIME_CRACK_ENGINE_PATH}")
        fi
        if [[ -z "${REALTIME_MULTITASK_ENGINE_PATH}" && -n "${REALTIME_CRACK_ENGINE_SHA256}" ]]; then
            ENGINE_ARGUMENTS+=(--realtime-crack-engine-sha256 "${REALTIME_CRACK_ENGINE_SHA256}")
        fi
    fi
fi

bash "${PROJECT_ROOT}/scripts/requirement.sh"

exec "${PYTHON}" "${PROJECT_ROOT}/jetson_code/run.py" --legacy \
    "${ENGINE_ARGUMENTS[@]}" \
    "$@" \
    "${NORMAL_DUAL_CAMERA_ARGUMENTS[@]}"
