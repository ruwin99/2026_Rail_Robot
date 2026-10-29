#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${RAIL_ROBOT_FIREBASE_CONFIG_FILE:-${HOME}/.config/rail_robot/firebase.env}"

for argument in "$@"; do
    case "${argument}" in
        --capture-test|--capture-test=*|--realtime-test|--realtime-test=*|--training|--training=*|--no-uart|--no-uart=*|--headless|--headless=*)
            echo "db_test.sh requires the interactive GUI initial+rescan capture flow; --capture-test, --realtime-test, --training, --no-uart, and --headless are not supported." >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Firebase config not found: ${CONFIG_PATH}" >&2
    echo "Run ./scripts/firebase_setup.sh before db_test.sh." >&2
    exit 1
fi

export RAIL_ROBOT_FIREBASE_CONFIG_FILE="${CONFIG_PATH}"
export RAIL_ROBOT_FIREBASE_UPLOAD=1
export RAIL_ROBOT_FIREBASE_UPLOAD_REQUIRED=1
export RAIL_ROBOT_DB_TEST_MANUAL_CAPTURE=1

echo "DB test: press S for 4 before-cleaning and 4 after-cleaning captures."
echo "After capture 4, clean the test section before pressing S again."
echo "The run finalizes, uploads, and exits automatically after capture 8."
echo "Firebase upload is enabled; credentials remain in the external config file."

exec bash "${PROJECT_ROOT}/scripts/run.sh" --legacy "$@"
