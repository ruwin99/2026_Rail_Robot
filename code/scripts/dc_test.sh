#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_VENV="${PROJECT_ROOT}/.venv"
PARENT_VENV="$(dirname -- "${PROJECT_ROOT}")/.venv"

if [[ -n "${RAIL_ROBOT_VENV:-}" ]]; then
    VENV="${RAIL_ROBOT_VENV}"
elif [[ -x "${LOCAL_VENV}/bin/python" ]]; then
    VENV="${LOCAL_VENV}"
else
    VENV="${PARENT_VENV}"
fi

PYTHON="${VENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Python virtual environment not found: ${PYTHON}" >&2
    echo "Set RAIL_ROBOT_VENV or create the project .venv first." >&2
    exit 1
fi
if ! "${PYTHON}" -c 'import serial' 2>/dev/null; then
    echo "pyserial is unavailable in ${VENV}." >&2
    exit 1
fi

exec "${PYTHON}" "${PROJECT_ROOT}/jetson_code/dc_test.py" "$@"
