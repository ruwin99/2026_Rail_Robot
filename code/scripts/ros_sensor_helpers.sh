#!/usr/bin/env bash
# .bashrc에서 source하여 사용한다. Jetson 설정을 직접 수정하거나 노드를 자동 실행하지 않는다.
export RAIL_ROBOT_HOME="${RAIL_ROBOT_HOME:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
# 한 번짜리 요청은 같은 폴더의 파이썬 스크립트(rail_ros_once.py, rail_alert.py)가 한 프로세스로 보낸다.
# ros2 명령줄은 켜는 데만 약 1초가 든다(Jetson 실측).
_RAIL_SENSOR_HELPERS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ROS의 setup.bash는 set -u(nounset)에서 AMENT_TRACE_SETUP_FILES unbound variable로 실패한다.
# source하는 동안만 nounset을 풀고, 호출자의 원래 설정으로 복구한다.
_rail_sensor_env() {
    local restore_nounset=0 status=0
    [[ $- == *u* ]] && restore_nounset=1
    set +u
    _rail_sensor_env_load || status=$?
    (( restore_nounset )) && set -u
    return "$status"
}

_rail_sensor_env_load() {
    local setup="/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
    local overlay="${RAIL_ROBOT_ROS_SETUP:-$RAIL_ROBOT_HOME/ros2/install/setup.bash}"
    # 같은 셸에서 이미 불러왔으면 다시 source하지 않는다(Jetson에서 호출마다 약 0.7초).
    # 오버레이의 install 폴더가 AMENT_PREFIX_PATH에 있고 ros2가 보이면 같은 환경으로 본다.
    if [[ -n "${ROS_DISTRO:-}" && "${AMENT_PREFIX_PATH:-}" == *"$(dirname -- "$overlay")/"* ]] \
            && command -v ros2 >/dev/null; then
        export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
        return 0
    fi
    [[ -f "$setup" ]] && source "$setup"
    if [[ ! -f "$overlay" ]]; then
        printf 'ROS workspace is not built: %s\n' "$overlay" >&2
        return 1
    fi
    source "$overlay" || return
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
    command -v ros2 >/dev/null || { printf 'ros2 not found\n' >&2; return 1; }
}

_rail_sensor_topic() {
    local namespace="${RAIL_ROBOT_NAMESPACE:-}"
    namespace="${namespace#/}"
    namespace="${namespace%/}"
    printf '/%s%s' "${namespace:+$namespace/}" "$1"
}

# 주행을 시작하지 않고 현재 토픽만 기록한다. Ctrl+C는 기록만 종료한다.
# 한 번 실행한 기록은 JSONL 파일 하나이며, 실제 저장 경로는 노드가 출력한다.
rail_record() {
    if [[ $# -gt 1 || ( $# -eq 1 && -z "$1" ) ]]; then
        printf 'rail_record [output_directory]\n' >&2
        return 2
    fi
    if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
        printf 'rail_record [output_directory] -- Ctrl+C saves one JSONL recording.\n'
        return 0
    fi
    local directory="${1:-${RAIL_RECORD_DIR:-$RAIL_ROBOT_HOME/records}}"
    local namespace="${RAIL_ROBOT_NAMESPACE:-/}"
    namespace="${namespace#/}"
    namespace="${namespace%/}"
    _rail_sensor_env || return
    ros2 run rail_robot_nodes record_node --ros-args \
        -r "__ns:=/$namespace" -p "output_directory:=$directory"
}

rail_sensors() {
    _rail_sensor_env || return
    ros2 launch rail_robot_nodes robot.launch.py mode:=sensors \
        "serial_port:=${RAIL_ROBOT_SERIAL_PORT:-/dev/ttyACM0}" \
        "namespace:=${RAIL_ROBOT_NAMESPACE:-/}" "$@"
}

# 실행 중인 RR2 mission bridge에 우선 정지 명령을 보낸다.
# 응답 success=true는 STM이 출력을 끄고 FAULT를 고정했다는 ACK다.
rail_estop() {
    if [[ $# -ne 0 ]]; then
        printf 'rail_estop takes no arguments\n' >&2
        return 2
    fi
    _rail_sensor_env || return
    # 서비스 발견(최대 10초)과 응답 대기(최대 5초)를 나눈다. 예전 `timeout 3s ros2 service call`은
    # 명령줄 기동과 DDS 발견만으로 3초를 넘겨 요청을 보내기도 전에 끝날 수 있었다.
    python3 "$_RAIL_SENSOR_HELPERS_DIR/rail_ros_once.py" estop "$(_rail_sensor_topic stm/command)"
}

rail_led() {
    if [[ $# -eq 0 || "$1" == *:=* ]]; then
        rail_sensors "$@"
        return
    fi
    local command="$1" payload
    shift
    case "$command" in
        status)
            [[ $# -eq 0 ]] || return 2
            _rail_sensor_env || return
            ros2 topic echo "$(_rail_sensor_topic debug/led/state)" std_msgs/msg/String --field data
            return ;;
        rgb)
            if [[ $# -ne 3 ]]; then printf 'rail_led rgb R G B (0..255)\n' >&2; return 2; fi
            local value
            for value in "$@"; do
                if [[ ! "$value" =~ ^[0-9]{1,3}$ ]] || (( 10#$value > 255 )); then
                    printf 'RGB values must be 0..255\n' >&2; return 2
                fi
            done
            payload="$1,$2,$3" ;;
        auto|off|red|green|blue|white|yellow|cyan|magenta)
            [[ $# -eq 0 ]] || return 2
            payload="$command" ;;
        *) printf 'rail_led [red|green|blue|white|yellow|cyan|magenta|off|auto|rgb R G B|status]\n' >&2; return 2 ;;
    esac
    _rail_sensor_env || return
    # 요청 토픽은 volatile이므로 이전 LED 요청을 재시작한 노드에 재전달하지 않는다.
    python3 "$_RAIL_SENSOR_HELPERS_DIR/rail_ros_once.py" string "$(_rail_sensor_topic debug/led/set)" "$payload"
}

rail_tts() {
    if [[ $# -eq 0 ]]; then
        _rail_sensor_env || return
        local namespace="${RAIL_ROBOT_NAMESPACE:-/}"
        namespace="/${namespace#/}"
        namespace="${namespace%/}"
        [[ -n "$namespace" ]] || namespace="/"
        ros2 run rail_robot_nodes audio_alert_node --ros-args -r "__ns:=$namespace"
        return
    fi
    if [[ $# -ne 1 ]]; then
        printf 'rail_tts [start|done|result|status]\n' >&2
        return 2
    fi
    local cue="" topic=""
    case "$1" in
        start) cue="mission_started" ;;
        done) cue="mission_completed" ;;
        result|status) topic="$1" ;;
        *) printf 'rail_tts [start|done|result|status]\n' >&2; return 2 ;;
    esac
    _rail_sensor_env || return
    if [[ -n "$topic" ]]; then
        ros2 topic echo "$(_rail_sensor_topic "tts/$topic")" std_msgs/msg/String --field data
    else
        python3 "$_RAIL_SENSOR_HELPERS_DIR/rail_ros_once.py" string "$(_rail_sensor_topic tts/debug)" "$cue"
    fi
}

# 장애물 경고음(obstacle_detected.wav) 시험. 다른 터미널에서 rail_tts로 음성 노드를 먼저 실행해 둔다.
# 경고음은 /tts/obstacle_state가 true인 동안 반복 재생되므로 기본은 3초 뒤 자동으로 false를 보낸다.
# true와 false를 한 파이썬 프로세스(rail_alert.py)에서 보낸다. ros2 명령줄을 두 번 켜면 Jetson에서 3.4초,
# 이 방식은 1.0초가 든다. 센서·STM·모터에는 영향이 없다. rail_mission 실행 중에는 쓰지 않는다
# (실제 장애물 판단과 섞인다). Ctrl+C로 끊으면 정지(false)를 보내고 끝난다.

rail_alert() {
    local usage='rail_alert [seconds|on|off|status]  (default: play 3 seconds, then stop)'
    local mode=3
    (( $# )) && mode="$1"   # 빈 인자("")는 기본값이 아니라 사용법 오류로 둔다.
    if [[ $# -gt 1 ]]; then
        printf '%s\n' "$usage" >&2
        return 2
    fi
    case "$mode" in
        on|off|status) ;;
        *)
            if ! [[ "$mode" =~ ^[0-9]+([.][0-9]+)?$ ]] \
                    || ! awk -v s="$mode" 'BEGIN { exit !(s > 0 && s <= 60) }'; then
                printf '%s\n' "$usage" >&2
                return 2
            fi
            ;;
    esac
    _rail_sensor_env || return
    if [[ "$mode" == status ]]; then
        ros2 topic echo "$(_rail_sensor_topic tts/status)" std_msgs/msg/String --field data
    else
        python3 "$_RAIL_SENSOR_HELPERS_DIR/rail_alert.py" "$(_rail_sensor_topic tts/obstacle_state)" "$mode"
    fi
}

rail_ultrasonic() {
    if [[ $# -eq 0 || "$1" == *:=* ]]; then
        rail_sensors "$@"
    elif [[ $# -eq 1 && "$1" == status ]]; then
        _rail_sensor_env || return
        ros2 topic echo "$(_rail_sensor_topic debug/ultrasonic/state)" std_msgs/msg/String --field data
    else
        printf 'rail_ultrasonic [status|serial_port:=/dev/ttyACM0]\n' >&2
        return 2
    fi
}

# 시험 모드 임무: rail_mission과 같고(촬영·복귀·실시간 청소·재검사 복귀·재검사, 단계 사이 1초 정지)
# 모든 단계가 6구간이 아니라 2구간이다. 임무 노드·검사·보고서·대시보드가 같은 값(2)을 쓰고,
# STM에는 PREPARE 2로 알린다. PREPARE 구간 인자를 아는 새 STM 펌웨어가 필요하다(이전 펌웨어는
# PREPARE를 거부하므로 6구간으로 잘못 달리지 않고 시작하지 않는다). 대시보드 기록에는 capture_target 2로 남는다.
# upload:=true: 임무 완료(DONE) 뒤 최종 보고서와 검출 미리보기를 대시보드 "검출 기록·요약"에 올린다
# (~/.config/rail_robot/firebase.env 필요). 업로드는 몇 초 걸리므로 "Dashboard report uploaded"를 본 뒤 종료한다.
# 업로드를 끄려면 rail_mission_test upload:=false 처럼 뒤에 덮어쓴다.
rail_mission_test() (
    export RAIL_ROBOT_SECTION_TARGET=2
    cd "$RAIL_ROBOT_HOME" || exit
    ./scripts/run.sh mode:=mission serial_port:="${RAIL_ROBOT_SERIAL_PORT:-/dev/ttyACM0}" \
        display:=false gui_output:=true dashboard_live:=true upload:=true "$@"
)
