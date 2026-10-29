"""Compile each real STM32 application translation unit with a host HAL double.

No source extraction, production math copies, board connection, or flashing.
The Cube main and interrupt units are covered by the separate ARM link tests.
"""
from functools import lru_cache
import atexit
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

CORE = Path(__file__).resolve().parents[2] / "stm_code/code/encoder/Core"
FIXTURES = Path(__file__).with_suffix("")
MODULES = ("robot_app", "drive_sync_pid", "drive_control", "actuators",
           "uart_protocol", "mission", "debug_console", "rgb_pwm", "ultrasonic", "fault_stop")
_directories = []


def _run(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise AssertionError("Command failed: " + subprocess.list2cmdline(command) + "\n" +
                             result.stdout + result.stderr)
    return result.stdout


@lru_cache(maxsize=16)
def build_host(pid_enabled=None, rgb_active_low=None, right_trim_bp=None, right_reverse_trim_bp=None,
               reverse_pid_enabled=None, forward_target_bp=9980, reverse_target_bp=10000,
               soft_start_ms=0):
    # Existing exact PWM/I fixtures use the original 99.8% calibration.
    # 기존 PWM 시험은 출발 직후 기준 PWM을 검사하므로 출발 가속을 끈다(0).
    # None이면 운영 헤더의 DRIVE_SOFT_START_MS를 그대로 쓴다.
    # Pass None to test the user's current production-header value.
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    msvc = False
    if compiler is None and os.name == "nt":
        candidates = sorted(Path("C:/Program Files (x86)/Microsoft Visual Studio").glob(
            "*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if candidates:
            compiler, msvc = str(candidates[-1]), True
    if compiler is None:
        raise unittest.SkipTest("Host C compiler unavailable; hardware is never executed.")
    if not (CORE / "Src").is_dir():
        # Jetson에는 STM 소스(stm_code)를 두지 않는다. 펌웨어 호스트 시험은 소스가 있는 노트북에서만 돈다.
        raise unittest.SkipTest("STM firmware sources unavailable; host firmware tests run where stm_code exists.")
    directory = tempfile.TemporaryDirectory(prefix="rail-stm32-modules-")
    _directories.append(directory)
    path = Path(directory.name)
    executable = path / ("stm32_host.exe" if os.name == "nt" else "stm32_host")
    include_dirs = [FIXTURES, CORE / "Inc"]
    if msvc:
        vc = Path(compiler).parents[3]
        sdk = sorted(Path("C:/Program Files (x86)/Windows Kits/10/Include").iterdir())[-1]
        sdk_lib = sdk.parents[1] / "Lib" / sdk.name
        include_dirs += [vc / "include", sdk / "ucrt", sdk / "shared", sdk / "um"]
        flags = ["/nologo", "/std:c11", "/utf-8", "/W4", "/D_CRT_SECURE_NO_WARNINGS"]
        flags += ["/I" + str(item) for item in include_dirs]
        if pid_enabled is not None:
            flags += ["/DDRIVE_SYNC_PID_ENABLED=" + str(int(pid_enabled))]
    else:
        flags = ["-std=c11", "-Wall", "-Wextra"] + ["-I" + str(item) for item in include_dirs]
        if pid_enabled is not None:
            flags += ["-DDRIVE_SYNC_PID_ENABLED=" + str(int(pid_enabled))]
    if rgb_active_low is not None:
        flags += [("/D" if msvc else "-D") + "RGB_PWM_ACTIVE_LOW=" + str(int(rgb_active_low))]
    if right_trim_bp is not None:
        flags += [("/D" if msvc else "-D") + "RIGHT_MOTOR_FORWARD_TRIM_BP=" + str(right_trim_bp)]
    if right_reverse_trim_bp is not None:
        flags += [("/D" if msvc else "-D") + "RIGHT_MOTOR_REVERSE_TRIM_BP=" + str(right_reverse_trim_bp)]
    if reverse_pid_enabled is not None:
        flags += [("/D" if msvc else "-D") + "DRIVE_SYNC_PID_REVERSE_ENABLED=" + str(int(reverse_pid_enabled))]
    if forward_target_bp is not None:
        flags += [("/D" if msvc else "-D") + "DRIVE_SYNC_PID_FORWARD_RIGHT_TARGET_BP=" + str(forward_target_bp)]
    if reverse_target_bp is not None:
        flags += [("/D" if msvc else "-D") + "DRIVE_SYNC_PID_REVERSE_LEFT_TARGET_BP=" + str(reverse_target_bp)]
    if soft_start_ms is not None:
        flags += [("/D" if msvc else "-D") + "DRIVE_SOFT_START_MS=" + str(soft_start_ms) + "U"]
    sources = [CORE / "Src" / (name + ".c") for name in MODULES]
    sources += sorted(FIXTURES.glob("*.c"))
    objects = []
    for source in sources:
        obj = path / (source.stem + (".obj" if msvc else ".o"))
        command = [compiler, *flags, "/c", str(source), "/Fo:" + str(obj)] if msvc else [
            compiler, *flags, "-c", str(source), "-o", str(obj)]
        _run(command, path)
        objects.append(str(obj))
    if msvc:
        command = [compiler, "/nologo", *objects, "/Fe:" + str(executable), "/link",
                   "/LIBPATH:" + str(vc / "lib/x64"),
                   "/LIBPATH:" + str(sdk_lib / "ucrt/x64"),
                   "/LIBPATH:" + str(sdk_lib / "um/x64")]
    else:
        command = [compiler, *objects, "-o", str(executable)]
    _run(command, path)
    return executable


def run_scenario(name, pid_enabled=None, rgb_active_low=None, right_trim_bp=None, right_reverse_trim_bp=None,
                 reverse_pid_enabled=None, forward_target_bp=9980, reverse_target_bp=10000,
                 soft_start_ms=0):
    executable = build_host(pid_enabled, rgb_active_low, right_trim_bp, right_reverse_trim_bp,
                            reverse_pid_enabled, forward_target_bp, reverse_target_bp, soft_start_ms)
    return _run([str(executable), name], executable.parent)


atexit.register(lambda: [item.cleanup() for item in _directories])
