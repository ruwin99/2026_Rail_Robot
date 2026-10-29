#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" != "0" ]]; then
    echo "Usage: ./scripts/front_capture_test.sh" >&2
    exit 2
fi

exec bash "${PROJECT_ROOT}/scripts/run.sh" --legacy --front-capture-test
