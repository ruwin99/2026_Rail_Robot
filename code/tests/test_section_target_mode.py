"""시험 모드(rail_mission_test, 단계별 2구간): 환경 변수 하나로 임무·검사·보고서·대시보드·STM 명령이 같은 구간 수를 쓰는지 확인한다.
장치나 ROS 통신에는 접근하지 않는다."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

CODE = Path(__file__).resolve().parents[1]
JETSON_CODE = CODE / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from inspection_report import (  # noqa: E402
    DEFAULT_RAIL_SECTION_TARGET,
    SECTION_TARGET_ENVIRONMENT,
    rail_section_target_from_environment,
)


def run_child(source: str, target: str | None) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment.pop(SECTION_TARGET_ENVIRONMENT, None)
    if target is not None:
        environment[SECTION_TARGET_ENVIRONMENT] = target
    prelude = textwrap.dedent(f'''
        import sys, types
        sys.path.insert(0, {str(JETSON_CODE)!r})
        serial_stub = types.ModuleType("serial"); serial_stub.SerialException = OSError; serial_stub.Serial = object
        sys.modules.setdefault("serial", serial_stub)
    ''')
    return subprocess.run([sys.executable, "-B", "-c", prelude + textwrap.dedent(source)],
                          env=environment, capture_output=True, text=True, timeout=60)


class SectionTargetEnvironmentTests(unittest.TestCase):
    def test_default_and_valid_values(self):
        self.assertEqual(DEFAULT_RAIL_SECTION_TARGET, 6)
        self.assertEqual(rail_section_target_from_environment({}), 6)
        self.assertEqual(rail_section_target_from_environment({SECTION_TARGET_ENVIRONMENT: ""}), 6)
        for value in ("1", "2", "3", "4", "5", "6", " 2 "):
            self.assertEqual(rail_section_target_from_environment({SECTION_TARGET_ENVIRONMENT: value}),
                             int(value))

    def test_invalid_values_fail_closed(self):
        # STM은 기본보다 많은 구간을 받지 않는다. 잘못된 값은 노드 시작 단계에서 막는다.
        for value in ("0", "7", "-1", "2.5", "two", "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rail_section_target_from_environment({SECTION_TARGET_ENVIRONMENT: value})

    def test_invalid_environment_stops_node_imports(self):
        result = run_child("import node_runtime.settings", "7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(SECTION_TARGET_ENVIRONMENT, result.stderr)


class TwoSectionModeTests(unittest.TestCase):
    def test_every_consumer_uses_two_sections(self):
        result = run_child('''
            import inspection_report, dashboard_export
            from node_runtime import settings
            assert inspection_report.RAIL_SECTION_TARGET == 2
            assert dashboard_export.RAIL_SECTION_TARGET == 2
            assert settings.AUTOMATIC_RAIL_SECTION_TARGET == 2
            assert settings.DEFAULT_RAIL_SECTION_TARGET == 6
            assert settings.CAPTURE_ANALYSIS_QUEUE_CAPACITY == 4
            print("OK")
        ''', "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)

    def test_mission_state_runs_two_initial_and_two_rescan_sections(self):
        result = run_child('''
            from node_runtime.mission import MissionStateMachine
            from node_runtime.settings import (COMPLETE_MODE, DONE_TRIGGER, REALTIME_START_TRIGGER,
                RESCAN_DONE_TRIGGER, RESCAN_RETURN_START_TRIGGER, RESCAN_START_TRIGGER, RETURN_START_TRIGGER)
            mission = MissionStateMachine()
            def capture():
                request = mission.begin_capture(); mission.accept_capture(request); mission.acknowledge_capture(request)
                return request
            def transition(trigger):
                step = mission.prepare_transition(trigger); assert step is not None; mission.complete_transition(step)
            mission.mark_start_sent(); assert mission.acknowledge_started()
            assert [capture().phase_sequence for _ in range(2)] == [1, 2]
            try:
                mission.begin_capture(); raise SystemExit("third initial capture was accepted")
            except RuntimeError:
                pass
            for trigger in (RETURN_START_TRIGGER, REALTIME_START_TRIGGER, RESCAN_RETURN_START_TRIGGER,
                            RESCAN_START_TRIGGER):
                transition(trigger)
            assert [capture().phase_sequence for _ in range(2)] == [1, 2]
            transition(RESCAN_DONE_TRIGGER); transition(DONE_TRIGGER)
            assert mission.mode == COMPLETE_MODE
            assert (mission.initial_rail_section_count, mission.rescan_rail_section_count) == (2, 2)
            print("OK")
        ''', "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_return_before_two_initial_sections_is_ignored(self):
        # 기존 규칙: 촬영 목표 전의 RETURN은 전이 후보를 만들지 않는다(None). 목표가 2면 1장 뒤 RETURN은 무시된다.
        result = run_child('''
            from node_runtime.mission import MissionStateMachine
            from node_runtime.settings import CAPTURE_SCAN_MODE, RETURN_START_TRIGGER
            mission = MissionStateMachine(); mission.mark_start_sent(); mission.acknowledge_started()
            request = mission.begin_capture(); mission.accept_capture(request); mission.acknowledge_capture(request)
            assert mission.prepare_transition(RETURN_START_TRIGGER) is None and mission.mode == CAPTURE_SCAN_MODE
            request = mission.begin_capture(); mission.accept_capture(request); mission.acknowledge_capture(request)
            assert mission.prepare_transition(RETURN_START_TRIGGER) is not None
            print("OK")
        ''', "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class Rr2PrepareArgumentTests(unittest.TestCase):
    def _worker(self):
        from node_runtime.serial_worker import SerialWorker
        worker = SerialWorker.__new__(SerialWorker)
        worker.capability = "MISSION"
        return worker

    def test_prepare_accepts_one_section_count_only(self):
        worker = self._worker()
        self.assertEqual(worker._validate_command("PREPARE", ()), "")
        self.assertEqual(worker._validate_command("PREPARE", ("2",)), "")
        self.assertEqual(worker._validate_command("PREPARE", (str(DEFAULT_RAIL_SECTION_TARGET),)), "")
        for args in (("0",), (str(DEFAULT_RAIL_SECTION_TARGET + 1),), ("x",), ("2", "2")):
            with self.subTest(args=args):
                self.assertNotEqual(worker._validate_command("PREPARE", args), "")
        for opcode in ("START", "CAPTURE_OK", "CANCEL_PREPARE"):
            self.assertNotEqual(worker._validate_command(opcode, ("2",)), "")

    def test_bridge_always_sends_section_count(self):
        source = (JETSON_CODE / "node_runtime/ros/stm_bridge_node.py").read_text(encoding="utf-8")
        branch = source[source.index('elif request.command in ("PREPARE", "CANCEL_PREPARE"):'):
                        source.index('elif request.command == "START":')]
        # 기본 구간 수여도 PREPARE에 구간 수를 붙인다. 기본값이 다른 이전 펌웨어는 조용히 다른 구간 수로
        # 달리지 않고 PREPARE를 거부한다.
        v2 = branch[branch.index("else:"):]
        self.assertIn('if request.command == "PREPARE" else ())', v2)
        self.assertNotIn("!= DEFAULT_RAIL_SECTION_TARGET", v2)
        self.assertIn("self._worker.submit(request.command, args)", v2)
        # legacy 문자열 경로는 구간 수를 전할 수 없으므로 시험 모드를 거부한다.
        legacy = branch[branch.index("if self._worker is None:"):branch.index("else:")]
        self.assertIn("raise RuntimeError", legacy)


class RailMissionTestHelperTests(unittest.TestCase):
    def test_helper_runs_mission_with_two_sections_without_leaking_environment(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        executable = str(bash) if bash.is_file() else shutil.which("bash")
        if not executable:
            self.skipTest("bash is unavailable")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            run = root / "scripts/run.sh"
            run.write_text('#!/usr/bin/env bash\nprintf "TARGET=%s\\n" "$RAIL_ROBOT_SECTION_TARGET"\n'
                           'printf "%s\\n" "$@"\n', encoding="utf-8", newline="\n")
            run.chmod(0o755)
            helpers = root / "helpers.sh"
            helpers.write_bytes((CODE / "scripts/ros_sensor_helpers.sh").read_bytes())
            environment = dict(os.environ, RAIL_ROBOT_HOME=root.as_posix())
            environment.pop(SECTION_TARGET_ENVIRONMENT, None)
            source = ('source "$1"; rail_mission_test record:=false; '
                      'printf "AFTER=%s\\n" "${RAIL_ROBOT_SECTION_TARGET:-unset}"')
            result = subprocess.run([executable, "-c", source, "test", helpers.as_posix()],
                                    capture_output=True, text=True, env=environment, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines[0], "TARGET=2")
        # upload:=true로 임무 완료 뒤 검출 기록·요약 데이터를 올리고, 호출 인자는 그 뒤에 와서 덮어쓸 수 있다.
        self.assertEqual(lines[1:8], ["mode:=mission", "serial_port:=/dev/ttyACM0", "display:=false",
                                      "gui_output:=true", "dashboard_live:=true", "upload:=true",
                                      "record:=false"])
        # 하위 셸에서 실행하므로 호출한 터미널에는 시험 모드 값이 남지 않는다(이후 rail_mission은 기본 6구간).
        self.assertEqual(lines[-1], "AFTER=unset")


if __name__ == "__main__":
    unittest.main()
