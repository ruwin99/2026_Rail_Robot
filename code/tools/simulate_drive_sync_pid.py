"""Run the firmware's C controller against an explicitly synthetic motor model.

This does not flash, connect to hardware, or estimate the robot's fall risk.
Python only models the motors and records results; PID arithmetic runs in C.
This is a standalone controller comparison, not the Motor_Run runtime policy:
controlled=True exercises PI in both directions even when the firmware's
DRIVE_SYNC_PID_REVERSE_ENABLED switch disables reverse correction.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import html
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "stm_code/code/encoder/Core/Inc"
CONTROLLER_SOURCE = INCLUDE.parent / "Src/drive_sync_pid.c"


def read_right_motor_trim_bp(header: Path = INCLUDE / 'robot_config.h', *, direction=1) -> int:
    """로컬 펌웨어 설정만 읽는다. 시뮬레이터에 별도 보정값을 두지 않는다."""
    name = 'RIGHT_MOTOR_REVERSE_TRIM_BP' if direction < 0 else 'RIGHT_MOTOR_FORWARD_TRIM_BP'
    match = re.search(r'^\s*#define\s+' + name + r'\s+(\d+)[uU]?[^\S\n]*'
                      r'(?://[^\n]*|/\*[^\n]*\*/[^\S\n]*)?$',
                      header.read_text(encoding='utf-8'), re.MULTILINE)
    if match is None or not 1 <= int(match[1]) <= 20000:
        raise ValueError(name + ' must be a literal integer in 1..20000')
    return int(match[1])


RIGHT_MOTOR_FORWARD_TRIM_BP = read_right_motor_trim_bp()
RIGHT_MOTOR_REVERSE_TRIM_BP = read_right_motor_trim_bp(direction=-1)


def encoder_1_base_pwm(base: int, direction=1) -> int:
    """Match the STM's physical-right ENC1 fixed-point PWM calculation."""
    trim = RIGHT_MOTOR_REVERSE_TRIM_BP if direction < 0 else RIGHT_MOTOR_FORWARD_TRIM_BP
    return (base * trim + 5000) // 10000

WRAPPER = r'''
#include "drive_sync_pid.h"
static DriveSyncPidConfig config;
static DriveSyncPidState state;
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
EXPORT void sim_reset(unsigned c1, unsigned c2, unsigned now) {
  config = DriveSyncPid_DefaultConfig();
  DriveSyncPid_Reset(&state, c1, c2, now);
}
EXPORT void sim_step(unsigned c1, unsigned c2, unsigned now,
                     int direction, int base1, int base2, int *out) {
  DriveSyncPidOutput r = DriveSyncPid_Step(&state, &config, c1, c2, now,
                                         direction, base1, base2);
  out[0]=r.pwm1; out[1]=r.pwm2; out[2]=r.error;
  out[3]=r.correction; out[4]=r.valid; out[5]=r.fault;
}
EXPORT unsigned sim_state_bytes(void) { return sizeof(state); }
EXPORT unsigned sim_config_bytes(void) { return sizeof(config); }
'''


def build_controller(directory: Path):
    """Build a temporary host library; no project build products are overwritten."""
    source = directory / "controller.c"
    source.write_text(WRAPPER, encoding="utf-8")
    library = directory / ("controller.dll" if os.name == "nt" else "controller.so")
    if os.name == "nt":
        choices = sorted(Path("C:/Program Files (x86)/Microsoft Visual Studio").glob(
            "*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe"))
        if not choices:
            raise RuntimeError("MSVC host compiler is required on Windows")
        compiler = choices[-1]
        vc = compiler.parents[3]
        sdk = sorted(Path("C:/Program Files (x86)/Windows Kits/10/Include").iterdir())[-1]
        libs = sdk.parents[1] / "Lib" / sdk.name
        cmd = [str(compiler), "/nologo", "/LD", "/O2", "/std:c11", "/utf-8",
               str(source), str(CONTROLLER_SOURCE), "/I" + str(INCLUDE), "/I" + str(vc / "include"),
               "/I" + str(sdk / "ucrt"), "/I" + str(sdk / "shared"),
               "/I" + str(sdk / "um"), "/Fo:" + str(directory) + os.sep,
               "/link", "/OUT:" + str(library), "/LIBPATH:" + str(vc / "lib/x64"),
               "/LIBPATH:" + str(libs / "ucrt/x64"), "/LIBPATH:" + str(libs / "um/x64")]
    else:
        compiler = shutil.which("cc") or shutil.which("gcc")
        if not compiler:
            raise RuntimeError("A host C compiler is required")
        cmd = [compiler, "-std=c11", "-O2", "-shared", "-fPIC", str(source), str(CONTROLLER_SOURCE),
               "-I" + str(INCLUDE), "-o", str(library)]
    subprocess.run(cmd, cwd=directory, check=True, capture_output=True, text=True, errors='replace')
    dll = ctypes.CDLL(str(library))
    dll.sim_reset.argtypes = [ctypes.c_uint32] * 3
    dll.sim_step.argtypes = [ctypes.c_uint32] * 3 + [ctypes.c_int] * 3 + [ctypes.POINTER(ctypes.c_int)]
    dll.sim_reset.restype = dll.sim_step.restype = None
    return dll


def step(dll, c1, c2, now, direction=1, base=800):
    out = (ctypes.c_int * 6)()
    dll.sim_step(c1, c2, now & 0xFFFFFFFF, direction,
                 encoder_1_base_pwm(base, direction), base, out)
    return list(out)


def run_case(dll, name, controlled=True, direction=1, base=800,
             gap_ms=0, noise=False, stall=False, jitter=False,
             left_gain=0.86, right_loaded_gain=0.78, time_constants=(0.10, 0.14)):
    """1 ms first-order plant; gains/time constants are assumptions, not measurements.

    A common forward mechanical position unit produces +1 TIM3 count and -2
    TIM4 counts (TI1 vs TI12). Raw polarity follows the observed board wiring,
    independently of PID configuration so a wrong configured sign still fails.
    There is no wheel slip, lateral rail contact, gravity or falling model.
    """
    rng = random.Random(17)
    position = [0.0, 0.0]
    velocity = [0.0, 0.0]
    pwm = [encoder_1_base_pwm(base, direction), base]
    dll.sim_reset(0, 0, 0)
    rows, fault_at, fault_code = [], None, 0
    next_control = 10
    for ms in range(1, 8001):
        load = [left_gain, 1.0 if ms < 3000 else right_loaded_gain]
        for i in (0, 1):
            target = direction * max(0, pwm[i] - 100) * 0.60 * load[i]
            if stall and i == 1 and ms >= 3000:
                target = 0
                velocity[i] = 0
            tau = time_constants[i]
            velocity[i] += (target - velocity[i]) * (1 - math.exp(-0.001 / tau))
            position[i] += velocity[i] * 0.001
        blocked = gap_ms and 3000 <= ms < 3000 + gap_ms
        if ms >= next_control and not blocked:
            c1 = math.floor(position[0])
            c2 = -math.floor(position[1] * 2)
            if noise:
                c1 += rng.choice((-1, 0, 0, 0, 1))
                c2 += rng.choice((-1, 0, 0, 0, 1))
            if controlled:
                result = step(dll, c1 % 601, c2 % 601, ms, direction, base)
                if result[4] or result[5]:
                    pwm = result[:2]
                if result[5] and fault_at is None:
                    fault_at, fault_code = ms, result[5]
            next_control = ms + (rng.randint(10, 14) if jitter else 10)
        if ms % 10 == 0:
            rows.append({"time_ms": ms, "v1": velocity[0], "v2": velocity[1],
                         "position1": position[0], "position2": position[1],
                         "pwm1": pwm[0], "pwm2": pwm[1]})
    steady = [r for r in rows if 2000 <= r["time_ms"] < 2900 or 6500 <= r["time_ms"] < 7900]
    mismatch = sum(abs(r["v1"] - r["v2"]) for r in steady) / len(steady)
    mean_speed = sum((abs(r["v1"]) + abs(r["v2"])) / 2 for r in steady) / len(steady)
    metrics = {"name": name, "controller": controlled, "direction": direction,
               "scope": "standalone_controller_not_direction_enable_policy",
               "base_pwm": base, "mean_speed_mismatch_pct": round(100 * mismatch / max(mean_speed, 1), 3) if fault_at is None else None,
               "end_position_difference_counts": round(position[0] - position[1], 3),
               "max_abs_speed_difference": round(max(abs(r["v1"] - r["v2"]) for r in rows), 3),
               "fault_at_ms": fault_at, "fault_code": fault_code,
               "final_pwm": pwm, "gap_ms": gap_ms,
               "plant": {"left_gain": left_gain, "right_loaded_gain": right_loaded_gain,
                         "time_constants_s": time_constants}}
    return metrics, rows


def make_plot(path: Path, cases):
    """Standalone SVG: actual simulated wheel speeds, not a conceptual drawing."""
    width, row_height = 1080, 230
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{len(cases)*row_height+75}" viewBox="0 0 {width} {len(cases)*row_height+75}">',
             '<rect width="100%" height="100%" fill="#101820"/>',
             '<g font-family="Arial, sans-serif" fill="#e7edf4">',
             '<text x="35" y="30" font-size="21">Synthetic motor simulation — actual firmware C controller</text>',
             '<text x="35" y="54" font-size="14">Controller only; reverse enable policy excluded. Blue: wheel 1; amber: wheel 2. Not hardware validation.</text>']
    for idx, (metrics, rows) in enumerate(cases):
        top = 95 + idx * row_height
        parts.append(f'<text x="35" y="{top}" font-size="17">{html.escape(metrics["name"])}</text>')
        ymax = max(1, max(abs(r[k]) for r in rows for k in ("v1", "v2"))) * 1.1
        for k, color in (("v1", "#51b9ff"), ("v2", "#ffbe55")):
            points = ' '.join(f'{70+r["time_ms"]/8000*950:.1f},{top+155-abs(r[k])/ymax*125:.1f}' for r in rows)
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        parts.append(f'<path d="M70 {top+25} V{top+155} H1020" fill="none" stroke="#69808e"/>')
        for sec in (0, 2, 4, 6, 8):
            parts.append(f'<text x="{70+sec/8*950}" y="{top+175}" font-size="12">{sec}s</text>')
        fault = metrics['fault_at_ms']
        parts.append(f'<text x="70" y="{top+199}" font-size="13">steady mismatch={metrics["mean_speed_mismatch_pct"]}% | fault={fault} ms | final PWM={metrics["final_pwm"]}</text>')
    parts.append('</g></svg>')
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    definitions = [("open_loop_800", dict(controlled=False)),
                   ("sync_pid_800", {}), ("open_loop_1600", dict(controlled=False, base=1600)),
                   ("sync_pid_1600", dict(base=1600)), ("reverse", dict(direction=-1)),
                   ("encoder_noise", dict(noise=True)), ("scheduler_jitter", dict(jitter=True)),
                   ("blocked_20ms", dict(gap_ms=20)), ("blocked_60ms", dict(gap_ms=60)),
                   ("blocked_100ms", dict(gap_ms=100)), ("right_motor_stall", dict(stall=True))]
    cases, sweep = [], []
    with tempfile.TemporaryDirectory(prefix='rail-pid-sim-') as temporary:
        dll = build_controller(Path(temporary))
        memory = {"host_state_bytes": dll.sim_state_bytes(), "host_config_bytes": dll.sim_config_bytes()}
        for name, kwargs in definitions:
            metrics, rows = run_case(dll, name, **kwargs)
            cases.append((metrics, rows))
            with (args.output / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
        # Sweep uncertain physical constants rather than showing only one model.
        for left_gain in (0.70, 0.86, 1.10):
            for load_gain in (0.65, 0.85, 1.0):
                for tau in ((0.05, 0.08), (0.10, 0.14), (0.25, 0.30)):
                    plant = dict(left_gain=left_gain, right_loaded_gain=load_gain, time_constants=tau)
                    before, _ = run_case(dll, 'sweep_open', controlled=False, **plant)
                    after, _ = run_case(dll, 'sweep_pid', **plant)
                    sweep.append({"plant": after['plant'], "open_mismatch_pct": before['mean_speed_mismatch_pct'],
                                  "pid_mismatch_pct": after['mean_speed_mismatch_pct'], "fault": after['fault_code']})
        # Windows cannot delete a loaded DLL. Release after all calls finish.
        if os.name == 'nt':
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
            ctypes.windll.kernel32.FreeLibrary(dll._handle)
        del dll
    result = {"warning": "Synthetic plant; not measured motor dynamics or MCU execution time; no fall-risk validation",
              "assumptions": {"step_ms": 1, "nominal_control_ms": 10, "duration_s": 8,
                              "motor_time_constants_s": [0.10, 0.14], "pwm_deadzone": 100,
                              "encoder_ratio": [1, 2], "encoder_raw_signs": [1, -1], "seed": 17},
              **memory, "cases": [m for m, _ in cases], "parameter_sweep": sweep}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    make_plot(args.output / 'comparison.svg', [cases[i] for i in (0, 1, 8, 10)])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
