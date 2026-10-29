#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${RAIL_ROBOT_FIREBASE_CONFIG_FILE:-${HOME}/.config/rail_robot/firebase.env}"

if [[ -e "${CONFIG_PATH}" ]]; then
    echo "Refusing to overwrite existing Firebase config: ${CONFIG_PATH}" >&2
    exit 1
fi

read -r -p "Firebase Web API key: " FIREBASE_API_KEY
read -r -p "Dedicated Jetson Firebase Auth UID: " FIREBASE_UID
read -r -p "Dedicated Jetson Firebase Auth email: " FIREBASE_EMAIL
read -r -s -p "Dedicated Jetson Firebase Auth password: " FIREBASE_PASSWORD
printf '\n'

if [[ -z "${FIREBASE_API_KEY}" || -z "${FIREBASE_UID}" ||
    -z "${FIREBASE_EMAIL}" || -z "${FIREBASE_PASSWORD}" ]]; then
    echo "All Firebase device-account fields are required." >&2
    exit 2
fi
if [[ "${FIREBASE_API_KEY}${FIREBASE_UID}${FIREBASE_EMAIL}${FIREBASE_PASSWORD}" == *$'\n'* ]]; then
    echo "Firebase settings cannot contain a newline." >&2
    exit 2
fi

CONFIG_DIRECTORY="$(dirname -- "${CONFIG_PATH}")"
mkdir -p -- "${CONFIG_DIRECTORY}"
umask 077
TEMPORARY_CONFIG="$(mktemp "${CONFIG_DIRECTORY}/firebase.env.tmp.XXXXXX")"
trap 'rm -f -- "${TEMPORARY_CONFIG}"' EXIT

{
    printf 'RAIL_ROBOT_FIREBASE_UPLOAD=1\n'
    printf 'RAIL_ROBOT_FIREBASE_API_KEY=%s\n' "${FIREBASE_API_KEY}"
    printf 'RAIL_ROBOT_FIREBASE_UID=%s\n' "${FIREBASE_UID}"
    printf 'RAIL_ROBOT_FIREBASE_EMAIL=%s\n' "${FIREBASE_EMAIL}"
    printf 'RAIL_ROBOT_FIREBASE_PASSWORD=%s\n' "${FIREBASE_PASSWORD}"
    printf 'RAIL_ROBOT_FIREBASE_PROJECT_ID=eswcontest-rail-robot\n'
    printf 'RAIL_ROBOT_FIREBASE_STORAGE_BUCKET=eswcontest-rail-robot.firebasestorage.app\n'
    printf 'RAIL_ROBOT_FIREBASE_TIMEOUT_SECONDS=30\n'
} > "${TEMPORARY_CONFIG}"

chmod 600 -- "${TEMPORARY_CONFIG}"
mv -- "${TEMPORARY_CONFIG}" "${CONFIG_PATH}"
trap - EXIT
echo "Firebase uploader config saved with mode 600: ${CONFIG_PATH}"
