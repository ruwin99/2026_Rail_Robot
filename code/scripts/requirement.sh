#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${RAIL_ROBOT_VENV:-${PROJECT_ROOT}/.venv}"
PYTHON="${VENV}/bin/python"
REQUIRED_PACKAGES=(
    alsa-utils
    python3
    python3-venv
    python3-pip
    python3-numpy
    python3-opencv
    python3-serial
    python3-libnvinfer
    libnvinfer-bin
)
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'install ok installed'; then
        MISSING_PACKAGES+=("${package}")
    fi
done

if (( ${#MISSING_PACKAGES[@]} > 0 )); then
    echo "Installing required packages: ${MISSING_PACKAGES[*]}"
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Creating Python virtual environment: ${VENV}"
    python3 -m venv --system-site-packages "${VENV}"
fi

VENV_CONFIG="${VENV}/pyvenv.cfg"
if [[ -f "${VENV_CONFIG}" ]] && grep -q '^include-system-site-packages = false' "${VENV_CONFIG}"; then
    sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' "${VENV_CONFIG}"
fi

if ! "${PYTHON}" -c 'import importlib.metadata as metadata; from cuda import cuda, cudart, nvrtc; raise SystemExit(metadata.version("cuda-python") != "12.5.0")' 2>/dev/null; then
    echo "Installing NVIDIA CUDA Python 12.5.0..."
    if ! "${PYTHON}" -m pip install --disable-pip-version-check --only-binary=:all: 'cuda-python==12.5.0'; then
        echo "CUDA Python installation failed; TensorRT inference cannot run." >&2
        exit 1
    fi
fi

if ! "${PYTHON}" -c 'import cv2, numpy, serial, tensorrt; from cuda import cuda, cudart, nvrtc' 2>/dev/null; then
    echo "TensorRT, CUDA Python, OpenCV, NumPy, or pyserial is unavailable in ${VENV}." >&2
    echo "Verify that JetPack and the system Python packages match this Jetson." >&2
    exit 1
fi

if ! "${PYTHON}" -c 'import tensorrt as trt; raise SystemExit(not str(trt.__version__).startswith("10.3."))' 2>/dev/null; then
    echo "This engine requires the TensorRT 10.3.x Python runtime." >&2
    echo "Rebuild the plan before using a different TensorRT release." >&2
    exit 1
fi

if ! "${PYTHON}" -c 'import openpyxl' 2>/dev/null; then
    echo "Installing XLSX report support..."
    if ! "${PYTHON}" -m pip install --disable-pip-version-check 'openpyxl==3.1.5'; then
        echo "openpyxl installation failed; inspection reports cannot be written." >&2
        exit 1
    fi
fi

echo "Requirements are ready (TensorRT DeepLabV3+ runtime enabled)."
