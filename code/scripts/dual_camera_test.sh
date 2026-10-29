#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SIDE_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
TOP_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0"
OBSTACLE_ENGINE="/home/bitdol/models/obstacle_yolo26n_foreign_matter_v2/plans/obstacle-yolo26n-foreign-matter-v2-camera-w1280-h720-int-h736-fp16-notf32-trt10.3.plan"
OBSTACLE_ENGINE_SHA256="cb896747dc9d076a30aedea30517499a3a6d2d0be80a1804c596b8b746a6412c"

FORWARDED_ARGUMENTS=()
HEADLESS_REQUESTED=0
UART_REQUESTED=0
TIMING_REQUESTED=0
DEBUG_REQUESTED=0
RUST_DEBUG_REQUESTED=0
usage() {
    echo "Usage: ./scripts/dual_camera_test.sh [--headless] [--uart] [--timing] [--debug] [--rust-debug]" >&2
}
for argument in "$@"; do
    case "${argument}" in
        --headless)
            if [[ "${HEADLESS_REQUESTED}" == "1" ]]; then
                usage
                exit 2
            fi
            HEADLESS_REQUESTED=1
            FORWARDED_ARGUMENTS+=("--headless")
            ;;
        --uart)
            if [[ "${UART_REQUESTED}" == "1" ]]; then
                usage
                exit 2
            fi
            UART_REQUESTED=1
            FORWARDED_ARGUMENTS+=("--realtime-test-uart")
            ;;
        --timing)
            if [[ "${TIMING_REQUESTED}" == "1" ]]; then
                usage
                exit 2
            fi
            TIMING_REQUESTED=1
            FORWARDED_ARGUMENTS+=("--dual-timing")
            ;;
        --debug)
            if [[ "${DEBUG_REQUESTED}" == "1" ]]; then
                usage
                exit 2
            fi
            DEBUG_REQUESTED=1
            ;;
        --rust-debug|--rust_debug)
            if [[ "${RUST_DEBUG_REQUESTED}" == "1" ]]; then
                usage
                exit 2
            fi
            RUST_DEBUG_REQUESTED=1
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

if [[ "${DEBUG_REQUESTED}" == "1" ]]; then
    if [[ "${UART_REQUESTED}" == "1" || "${TIMING_REQUESTED}" == "1" ]]; then
        echo "--debug cannot be combined with --uart or --timing." >&2
        exit 2
    fi
    FORWARDED_ARGUMENTS+=("--dual-debug-benchmark")
    if [[ "${HEADLESS_REQUESTED}" == "0" ]]; then
        FORWARDED_ARGUMENTS+=("--headless")
    fi
fi

if [[ "${RUST_DEBUG_REQUESTED}" == "1" ]]; then
    if [[ "${UART_REQUESTED}" == "1" || "${TIMING_REQUESTED}" == "1" || "${DEBUG_REQUESTED}" == "1" ]]; then
        echo "--rust-debug cannot be combined with --uart, --timing, or --debug." >&2
        exit 2
    fi
    FORWARDED_ARGUMENTS+=("--dual-rust-debug-benchmark")
    if [[ "${HEADLESS_REQUESTED}" == "0" ]]; then
        FORWARDED_ARGUMENTS+=("--headless")
    fi
fi

exec bash "${PROJECT_ROOT}/scripts/run.sh" --legacy \
    --realtime-test \
    --side-camera-device "${SIDE_CAMERA_DEVICE}" \
    --top-camera-device "${TOP_CAMERA_DEVICE}" \
    --obstacle-engine "${OBSTACLE_ENGINE}" \
    --obstacle-engine-sha256 "${OBSTACLE_ENGINE_SHA256}" \
    --obstacle-confidence-threshold 0.424 \
    "${FORWARDED_ARGUMENTS[@]}"
