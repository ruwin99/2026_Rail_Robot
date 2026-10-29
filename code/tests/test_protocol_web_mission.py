"""웹 임무 시작(START)부터 출발점 복귀 후 DONE까지 프로토콜 통합 시험.

RemoteControlNode → /mission/start → MissionNode → StmBridgeNode → SerialWorker → RR2 UART →
컴파일된 STM 펌웨어(host HAL)를 실제 코드 그대로 연결해 프레임과 상태 전이를 검사한다.
도구는 protocol_e2e.py를 본다. 장치·네트워크·카메라·모델은 쓰지 않으며 시험당 약 10~15초가 걸린다.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]


def expected_events(n):
    return (["STARTED"] + ["CAMERA_CAPTURE"] * n
            + ["RETURN_START", "REALTIME_START", "RESCAN_RETURN_START", "RESCAN_START"]
            + ["CAMERA_CAPTURE"] * n + ["RESCAN_DONE", "DONE"])


def child(scenario):
    from stm32_host import build_host
    import protocol_e2e as e2e

    executable = e2e.install(lambda: build_host(soft_start_ms=None))  # 운영 헤더의 출발 가속 설정 그대로
    from node_runtime.settings import AUTOMATIC_RAIL_SECTION_TARGET as n

    class Interrupted(Exception):
        pass

    def verify_common(plant, expected, prepare_args):
        mission, bridge = plant.mission, plant.bridge
        assert plant.events == expected, f"events differ:\n{plant.events}\n{expected}"
        wire = plant.peer.tx_commands()
        operations = [operation for _, operation, _ in wire]
        sequences = [sequence for sequence, _, _ in wire]
        # 명령 순번은 중복(재전송)이 없고 순서대로 늘어난다.
        assert sequences == sorted(set(sequences)), f"CMD sequences have retries or gaps: {sequences}"
        assert operations.count("START") == 1, operations
        prepare = [args for _, operation, args in wire if operation == "PREPARE"]
        assert prepare == [prepare_args], prepare
        assert operations.index("PREPARE") < operations.index("START")
        # 웹이 START를 두 번 눌러도 STM에는 한 번만 간다.
        assert plant.stm_commands.count("START") == 1
        # STM이 응답한 모든 NACK는 청소 ON 거절(SAFETY·STATE)뿐이어야 한다.
        unexpected = [frame for frame in plant.nacks()
                      if not (frame[3] in e2e.ACCESSORY_OPCODES and frame[4] in ("SAFETY", "STATE"))]
        assert not unexpected, unexpected
        assert not mission._failed and not bridge._failed, (mission._error, plant.faults)
        assert plant.faults == [], plant.faults
        return operations

    def verify_completed(plant, n, prepare_args, *, accessories):
        operations = verify_common(plant, expected_events(n), prepare_args)
        mission = plant.mission
        assert mission.state.mode == "COMPLETE"
        assert (mission.state.initial_rail_section_count, mission.state.rescan_rail_section_count) == (n, n)
        commands = plant.stm_commands
        assert commands[0] == "PREPARE" and commands.count("CAPTURE_OK") == 2 * n
        assert "E_STOP" not in commands
        assert commands.count("SAFE_OFF") >= 3, commands   # REALTIME 진입, 재검사 복귀 시작, DONE
        assert plant.capture_requests == ([("initial", i) for i in range(1, n + 1)]
                                          + [("rescan", i) for i in range(1, n + 1)])
        assert plant.drain_requests == [False, True], plant.drain_requests
        timeline = plant.timeline
        last_return_stop = max(index for index, item in enumerate(timeline)
                               if item == f"boundary:-1:{n}:last")
        # 출발점에 도착(마지막 후진 경계)한 뒤에야 DONE, 최종 정리, 완료 음성이 나온다.
        assert timeline.index("RESCAN_DONE") < last_return_stop < timeline.index("DONE"), timeline
        assert timeline.index("DONE") < timeline.index("drain:finalize") < timeline.index("tts:mission_completed")
        assert timeline.index("tts:mission_started") < timeline.index("cmd:START")
        assert plant.tts_requests.count("mission_started") == 1 and plant.tts_requests.count("mission_completed") == 1
        # 펌웨어 상태 번호 변화(PING·명령 응답의 마지막 필드): 출발점 복귀 상태를 거쳐 IDLE로 끝난다.
        states = plant.firmware_states()
        assert e2e.MISSION_FAULT not in states, [e2e.STATE_NAMES[s] for s in states]
        wanted = [e2e.INITIAL_SCAN, e2e.RETURNING, e2e.REALTIME_CLEANING, e2e.RESCAN_RETURN,
                  e2e.RESCAN, e2e.FINAL_RETURN, e2e.IDLE]
        position = 0
        for state in states:
            if position < len(wanted) and state == wanted[position]:
                position += 1
        assert position == len(wanted), [e2e.STATE_NAMES[s] for s in states]
        if accessories:
            # 실시간 청소 동안 청소 ON/OFF가 오갔고, 단계 경계의 ON 거절은 임무를 멈추지 않았다.
            assert any(op in operations for op in ("FRONT_CLEANER_PWM_25", "SIDE_CLEANER_PWM_12_5")), operations
            assert "FRONT_PUMP_ON" in operations and "SIDE_PUMP_ON" in operations
            refused = [frame for frame in plant.nacks() if frame[3] in e2e.ACCESSORY_OPCODES]
            assert refused, "단계 경계의 ON 거절 경로가 실행되지 않았다"
        return states

    async def duplicate_web_start(plant):
        result = await plant.web_command("web-start-2", "START")
        assert result == (False, "start_rejected"), result

    async def nominal(coast, sections, decisions):
        plant = e2e.Plant(executable, coast=coast, sections=sections, decisions=decisions)
        plant.boot()
        try:
            await plant.run_mission(after_start=duplicate_web_start)
            states = verify_completed(plant, sections, (str(sections),), accessories=decisions)
            if os.environ.get("E2E_DEBUG"):
                print("events", plant.events)
                print("states", [e2e.STATE_NAMES[s] for s in states])
                print("wire", [(op, args) for _, op, args in plant.peer.tx_commands()])
                print("nacks", plant.nacks())
                print("timeline", plant.timeline)
        finally:
            plant.close()

    async def obstacle_during_final_return():
        plant = e2e.Plant(executable, coast=15, sections=n)
        plant.boot()

        async def hold(index):
            if index != 2:
                return
            await plant.ultrasonic(300, 3)                      # 근거리 3회 연속
            await plant.wait(lambda: "OBSTACLE_HOLD" in plant.events, "OBSTACLE_HOLD")
            assert plant.mission._obstacle_held
            before = list(plant.events)
            await plant.step(1500)                              # 정지 중에는 구간도 완료도 진행하지 않는다
            assert plant.events == before and "DONE" not in plant.events
            await plant.ultrasonic(2000, 20)                    # 20회 중 원거리 16회 이상
            await plant.wait(lambda: "OBSTACLE_CLEAR" in plant.events, "OBSTACLE_CLEAR")
            assert not plant.mission._obstacle_held
        try:
            await plant.run_mission(final_return_hook=hold)
            expected = expected_events(n)
            index = expected.index("DONE")
            expected = expected[:index] + ["OBSTACLE_HOLD", "OBSTACLE_CLEAR"] + expected[index:]
            # 정지·해제 이벤트는 출발점 복귀 중(RESCAN_DONE 뒤, DONE 앞)에 와야 한다.
            assert plant.events == expected, plant.events
            verify_common(plant, expected, (str(n),))
            assert plant.mission.state.mode == "COMPLETE"
        finally:
            plant.close()

    async def estop_during_final_return():
        plant = e2e.Plant(executable, coast=15, sections=n)
        plant.boot()

        async def stop(index):
            if index != 2:
                return
            result = await plant.web_command("web-stop-1", "E_STOP")
            assert result == (True, "estop_ack"), result
            raise Interrupted

        try:
            try:
                await plant.run_mission(final_return_hook=stop)
            except Interrupted:
                pass
            else:
                raise AssertionError("E_STOP scenario finished the mission")
            assert plant.bridge._failed and any("E_STOP" in item for item in plant.faults), plant.faults
            events = list(plant.events)
            for _ in range(n + 1):                              # 멈춘 로봇에는 경계 신호도 오지 않지만 오더라도 무시된다
                plant.peer.encoder()
                await plant.step(50)
            await plant.step(2500)                              # 1초 정지가 지나도 DONE이 나오면 안 된다
            assert plant.events == events and "DONE" not in plant.events, plant.events
            states = plant.firmware_states()
            assert e2e.MISSION_FAULT in states, [e2e.STATE_NAMES[s] for s in states]
            assert plant.mission._failed
            assert "tts:mission_completed" not in plant.timeline
        finally:
            plant.close()

    ACCESSORY_ON = ("FRONT_CLEANER_PWM_12_5", "FRONT_CLEANER_PWM_25", "FRONT_PUMP_ON",
                    "SIDE_CLEANER_PWM_12_5", "SIDE_CLEANER_PWM_25", "SIDE_PUMP_ON")

    def accessory_on_count(plant):
        return sum(1 for _, operation, _ in plant.peer.tx_commands() if operation in ACCESSORY_ON)

    async def obstacle_during_realtime_cleaning():
        plant = e2e.Plant(executable, coast=15, sections=n, decisions=True)
        plant.boot()
        marks = {}

        async def hold(index):
            if index != n:
                return
            await plant.wait(lambda: accessory_on_count(plant) > 0, "cleaning outputs ON")
            await plant.ultrasonic(300, 3)
            await plant.wait(lambda: "OBSTACLE_HOLD" in plant.events, "OBSTACLE_HOLD")
            marks["held"] = len(plant.peer.tx_commands())
            marks["nacks_held"] = len(plant.nacks())
            await plant.step(1500)                              # 정지 중에도 제어 노드의 청소 결정은 계속 온다
            during = [operation for _, operation, _ in plant.peer.tx_commands()[marks["held"]:]]
            assert not any(operation in ACCESSORY_ON for operation in during), during
            await plant.ultrasonic(2000, 20)
            await plant.wait(lambda: "OBSTACLE_CLEAR" in plant.events, "OBSTACLE_CLEAR")
            marks["cleared_on"] = accessory_on_count(plant)
            # 해제 뒤에는 새 결정으로 청소가 다시 켜진다(정지 전에 쌓인 결정은 재생하지 않는다).
            await plant.wait(lambda: accessory_on_count(plant) > marks["cleared_on"], "cleaning resumes")
        try:
            await plant.run_mission(realtime_hook=hold)
            assert len(plant.nacks()) >= marks["nacks_held"]
            expected = expected_events(n)
            index = expected.index("REALTIME_START") + 1
            expected = expected[:index] + ["OBSTACLE_HOLD", "OBSTACLE_CLEAR"] + expected[index:]
            verify_common(plant, expected, (str(n),))
            assert plant.mission.state.mode == "COMPLETE"
        finally:
            plant.close()

    async def estop_during_realtime_cleaning():
        plant = e2e.Plant(executable, coast=15, sections=n, decisions=True)
        plant.boot()

        async def stop(index):
            if index != n:
                return
            await plant.wait(lambda: accessory_on_count(plant) > 0, "cleaning outputs ON")
            result = await plant.web_command("web-stop-1", "E_STOP")
            assert result == (True, "estop_ack"), result
            raise Interrupted

        try:
            try:
                await plant.run_mission(realtime_hook=stop)
            except Interrupted:
                pass
            sent = len(plant.peer.tx_commands())
            events = list(plant.events)
            await plant.step(2500)
            # E_STOP 뒤에는 청소 결정이 계속 와도 STM으로 ON이 나가지 않고, 임무도 진행하지 않는다.
            after = [operation for _, operation, _ in plant.peer.tx_commands()[sent:]]
            assert not any(operation in ACCESSORY_ON for operation in after), after
            assert plant.events == events and "RESCAN_RETURN_START" not in plant.events
            assert plant.bridge._failed and plant.mission._failed
            assert e2e.MISSION_FAULT in plant.firmware_states()
        finally:
            plant.close()

    async def jetson_silent_during_final_return():
        # Jetson이 통째로 멈춰도(프로세스 종료·전원 문제) 출발점 복귀 중인 로봇은 STM 감시(600 ms)로 멈춘다.
        plant = e2e.Plant(executable, coast=15, sections=n)
        plant.boot()

        async def silence(index):
            if index != 2:
                return
            plant.bridge._worker.suspend_heartbeats()
            plant.frozen = True
            await plant.step(1000)
            for _ in range(n + 1):
                plant.peer.encoder()
                await plant.step(50)
            await plant.step(2500)
            raise Interrupted
        try:
            try:
                await plant.run_mission(final_return_hook=silence)
            except Interrupted:
                pass
            plant.frozen = False
            received = bytes(plant.peer.rx)
            assert b"MISSION_DONE" not in received, "STM kept driving and finished after the Jetson went silent"
            assert plant.events.count("RESCAN_DONE") == 1 and "DONE" not in plant.events
        finally:
            plant.frozen = False
            plant.close()

    async def relaunch_after_completion():
        # 완료한 임무는 스택이 스스로 종료하고, 웹에서 실행 버튼을 다시 눌러 같은 STM에 새 스택이 붙어도 다음 임무를 마친다.
        from std_msgs.msg import String
        plant = e2e.Plant(executable, coast=15, sections=n, decisions=True, exit_hold=1.0, wait_for_upload=True)
        plant.boot()
        try:
            await plant.run_mission()
            verify_completed(plant, n, (str(n),), accessories=True)
            mission = plant.mission
            # 대기 시간이 지나도 보고서 업로드 결과가 오기 전에는 종료하지 않고, 결과가 오면 종료를 요청한다.
            await plant.step(1500)
            assert not mission.shutdown_requested(), "업로드 결과 전에 종료를 요청했다"
            mission.subscriptions["inspection/upload_status"].callback(String(data="uploaded: report.json"))
            assert mission.shutdown_requested()
            # 완료 상태가 FAILED로 남지 않고, 종료해도 STM은 FAULT 없이 IDLE에 있다.
            statuses = plant.mission.publishers["mission/state"].messages
            assert statuses[-1].mode == "COMPLETE" and not statuses[-1].failed
            plant.stop_stack_keep_stm()
            await plant.step(1500)                                    # Jetson이 없는 동안 STM 감시(600 ms)가 세션을 정리한다
            plant.boot()                                              # 같은 STM에 새 UART 세션(HELLO)으로 붙는다
            for _ in range(20):
                plant.peer.echo(2000)
            plant.begin_next_mission()
            await plant.run_mission(start_id="web-start-relaunch")
            assert plant.events == expected_events(n), plant.events
            assert plant.capture_requests == ([("initial", i) for i in range(1, n + 1)]
                                              + [("rescan", i) for i in range(1, n + 1)])
            second = [operation for _, operation, _ in plant.peer.tx_commands()[plant.wire_mark:]]
            assert second.count("PREPARE") == 1 and second.count("START") == 1, second
            assert plant.mission.state.mode == "COMPLETE" and plant.faults == []
            assert not plant.mission._failed and not plant.bridge._failed
            states = plant.firmware_states()
            assert e2e.MISSION_FAULT not in states and states[-1] == e2e.IDLE, [e2e.STATE_NAMES[s] for s in states]
        finally:
            plant.close()

    scenarios = {
        "nominal_default_sections_with_coast_and_cleaning": lambda: nominal(20, n, True),
        "nominal_without_coast": lambda: nominal(0, n, False),
        "obstacle_during_final_return": obstacle_during_final_return,
        "estop_during_final_return": estop_during_final_return,
        "obstacle_during_realtime_cleaning": obstacle_during_realtime_cleaning,
        "estop_during_realtime_cleaning": estop_during_realtime_cleaning,
        "jetson_silent_during_final_return": jetson_silent_during_final_return,
        "relaunch_after_completion": relaunch_after_completion,
    }
    asyncio.run(scenarios[scenario]())
    print("PASS", scenario)


class WebMissionProtocolTests(unittest.TestCase):
    def run_child(self, scenario, sections=None):
        environment = dict(os.environ)
        environment.pop("RAIL_ROBOT_SECTION_TARGET", None)
        if sections is not None:
            environment["RAIL_ROBOT_SECTION_TARGET"] = str(sections)
        result = subprocess.run([sys.executable, __file__, "--child", scenario], cwd=CODE, env=environment,
                                capture_output=True, text=True, timeout=240)
        if result.returncode == 0 and "SKIP " in result.stdout:
            self.skipTest(result.stdout.split("SKIP ", 1)[1].strip())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS " + scenario, result.stdout)

    def test_default_six_section_mission_with_coasting_and_cleaning(self):
        # 기본 6구간: 관성 보정 이벤트와 청소 ON 거절 경합까지 포함한 전체 흐름.
        self.run_child("nominal_default_sections_with_coast_and_cleaning")

    def test_two_section_test_mode_without_coasting(self):
        # rail_mission_test: PREPARE 2가 STM까지 전달되고 모든 단계가 2구간으로 끝난다.
        self.run_child("nominal_without_coast", sections=2)

    def test_obstacle_hold_while_returning_to_start(self):
        self.run_child("obstacle_during_final_return", sections=2)

    def test_web_estop_while_returning_to_start(self):
        self.run_child("estop_during_final_return", sections=2)

    def test_obstacle_hold_during_cleaning_does_not_leak_stale_commands(self):
        self.run_child("obstacle_during_realtime_cleaning", sections=2)

    def test_web_estop_during_cleaning_blocks_further_outputs(self):
        self.run_child("estop_during_realtime_cleaning", sections=2)

    def test_stm_watchdog_stops_the_return_when_jetson_goes_silent(self):
        self.run_child("jetson_silent_during_final_return", sections=2)

    def test_stack_exits_after_completion_and_a_new_launch_runs_the_next_mission(self):
        self.run_child("relaunch_after_completion", sections=2)


class WebLaunchConfigurationTests(unittest.TestCase):
    """웹 '프로그램 실행'이 띄우는 rail-mission.service의 고정 인자."""

    def setUp(self):
        self.unit = (CODE / "scripts/systemd/rail-mission.service").read_text(encoding="utf-8")
        self.command = next(line for line in self.unit.splitlines() if line.startswith("ExecStart="))

    def test_web_launch_enables_remote_control_rr2_and_report_upload(self):
        for argument in ("mode:=mission", "dashboard_live:=true", "dashboard_control:=true",
                         "uart_protocol:=v2", "upload:=true"):
            self.assertIn(argument, self.command)

    def test_web_launch_is_always_the_default_section_mission(self):
        # 시험 모드(2구간)는 rail_mission_test 명령 안에서만 켜진다. 웹 실행 환경에는 들어가면 안 된다.
        environment = (CODE / "scripts/systemd/runtime.env.example").read_text(encoding="utf-8")
        for text in (self.unit, environment, (CODE / "scripts/systemd/rail-ros-env.sh").read_text(encoding="utf-8")):
            self.assertNotIn("RAIL_ROBOT_SECTION_TARGET", text)


if __name__ == "__main__":
    if "--child" in sys.argv:
        try:
            child(sys.argv[sys.argv.index("--child") + 1])
        except unittest.SkipTest as skipped:
            print("SKIP", skipped)
    else:
        unittest.main()
