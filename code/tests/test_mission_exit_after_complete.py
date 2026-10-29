"""임무 완료 뒤 스택 스스로 종료(MissionNode.shutdown_requested) 단위 시험.

가짜 ROS는 자식 프로세스 안에서만 쓴다. 종료 뒤 같은 STM에 새 스택이 다시 붙는 전체 흐름은
test_protocol_web_mission.py의 relaunch_after_completion이 컴파일된 STM 펌웨어와 함께 확인한다.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


def child_checks():
    from unittest.mock import patch
    from test_ros_robotics import install_fake_ros

    install_fake_ros()
    from node_runtime.ros.mission_node import MissionNode
    from std_msgs.msg import String
    from node_runtime.settings import (COMPLETE_MODE, DONE_TRIGGER, REALTIME_START_TRIGGER,
                                       RESCAN_DONE_TRIGGER, RESCAN_RETURN_START_TRIGGER, RESCAN_START_TRIGGER,
                                       RETURN_START_TRIGGER, AUTOMATIC_RAIL_SECTION_TARGET)

    def node(**parameters):
        overrides = [SimpleNamespace(name="audio_enabled", value=False),
                     SimpleNamespace(name="require_clock_sync", value=False)]
        overrides += [SimpleNamespace(name=name, value=value) for name, value in parameters.items()]
        created = MissionNode(parameter_overrides=overrides)
        created._uart.responder = lambda request: SimpleNamespace(success=True, error="")
        created._drain.responder = lambda request: SimpleNamespace(success=True, error="")
        return created

    def complete(created):
        """START부터 DONE까지 상태 기계를 돌리고 실제 완료 처리(_begin_transition)로 COMPLETE에 도달시킨다."""
        state = created.state
        state.mark_start_sent()
        assert state.acknowledge_started()

        def capture():
            request = state.begin_capture()
            state.accept_capture(request)
            state.acknowledge_capture(request)

        for _ in range(AUTOMATIC_RAIL_SECTION_TARGET):
            capture()
        for trigger in (RETURN_START_TRIGGER, REALTIME_START_TRIGGER, RESCAN_RETURN_START_TRIGGER,
                        RESCAN_START_TRIGGER):
            step = state.prepare_transition(trigger)
            state.complete_transition(step)
        for _ in range(AUTOMATIC_RAIL_SECTION_TARGET):
            capture()
        for trigger in (RESCAN_DONE_TRIGGER, DONE_TRIGGER):
            created._begin_transition(state.prepare_transition(trigger))
        assert state.mode == COMPLETE_MODE and created._complete_since is not None

    # 경계 시각은 큰 monotonic 값에서 (a + 20) - a가 20보다 조금 작아질 수 있어 1 ms 뒤를 본다.
    def tick_at(created, offset):
        with patch("time.monotonic", return_value=created._complete_since + offset):
            created._tick()

    class ExitTests(unittest.TestCase):
        def test_exits_only_after_the_hold(self):
            created = node(exit_hold_seconds=20.0)
            self.assertFalse(created.shutdown_requested())
            complete(created)
            with patch("time.monotonic", return_value=created._complete_since + 19.9):
                self.assertFalse(created.shutdown_requested())
            with patch("time.monotonic", return_value=created._complete_since + 20.001):
                self.assertTrue(created.shutdown_requested())

        def test_waits_for_the_upload_result_when_upload_is_enabled(self):
            created = node(exit_hold_seconds=1.0, wait_for_upload=True, upload_wait_seconds=100.0)
            created.subscriptions["inspection/upload_status"].callback(String(data="uploaded: stale"))
            complete(created)
            self.assertIsNone(created._upload_status, "완료 전의 결과는 이번 임무의 것이 아니다")
            with patch("time.monotonic", return_value=created._complete_since + 50.0):
                self.assertFalse(created.shutdown_requested())
            created.subscriptions["inspection/upload_status"].callback(String(data="uploaded: report.json"))
            with patch("time.monotonic", return_value=created._complete_since + 50.0):
                self.assertTrue(created.shutdown_requested())

        def test_upload_failure_and_timeout_do_not_block_the_exit(self):
            failed = node(exit_hold_seconds=1.0, wait_for_upload=True)
            complete(failed)
            failed.subscriptions["inspection/upload_status"].callback(String(data="upload failed: no network"))
            with patch("time.monotonic", return_value=failed._complete_since + 2.0):
                self.assertTrue(failed.shutdown_requested())
            slow = node(exit_hold_seconds=1.0, wait_for_upload=True, upload_wait_seconds=60.0)
            complete(slow)
            with patch("time.monotonic", return_value=slow._complete_since + 59.0):
                self.assertFalse(slow.shutdown_requested())
            with patch("time.monotonic", return_value=slow._complete_since + 60.001):
                self.assertTrue(slow.shutdown_requested())

        def test_never_exits_when_failed_pending_audio_closing_or_disabled(self):
            for name, prepare in (
                ("failed", lambda created: setattr(created, "_failed", True)),
                ("completion audio pending", lambda created: setattr(created, "_completion_request",
                                                                     dict(request_id="a", cue="mission_completed",
                                                                          expires_at=1e12))),
                ("closed", lambda created: setattr(created, "_closed", True)),
            ):
                with self.subTest(name):
                    created = node(exit_hold_seconds=0.0)
                    complete(created)
                    prepare(created)
                    with patch("time.monotonic", return_value=created._complete_since + 1000.0):
                        self.assertFalse(created.shutdown_requested())
            created = node(exit_hold_seconds=0.0, exit_after_complete=False)
            complete(created)
            with patch("time.monotonic", return_value=created._complete_since + 1000.0):
                self.assertFalse(created.shutdown_requested())

        def test_a_mission_that_has_not_completed_never_asks_to_exit(self):
            created = node(exit_hold_seconds=0.0)
            created.state.mark_start_sent()
            self.assertFalse(created.shutdown_requested())

        def test_shutdown_after_completion_is_not_reported_as_a_failure(self):
            created = node()
            complete(created)
            created.prepare_shutdown()
            self.assertFalse(created._failed)
            self.assertEqual(created.publishers["mission/state"].messages[-1].mode, COMPLETE_MODE)
            self.assertFalse(created.publishers["mission/state"].messages[-1].failed)
            running = node()
            running.state.mark_start_sent()
            running.prepare_shutdown()
            self.assertTrue(running._failed and running.publishers["mission/state"].messages[-1].failed)

        def test_parameters_are_validated(self):
            for parameters in ({"exit_hold_seconds": -1.0}, {"exit_hold_seconds": 601.0},
                               {"exit_hold_seconds": float("nan")}, {"upload_wait_seconds": 901.0},
                               {"exit_after_complete": "yes"}, {"wait_for_upload": 1}):
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    node(**parameters)

    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExitTests))
    return 0 if result.wasSuccessful() else 1


class MissionExitAfterCompleteTests(unittest.TestCase):
    def test_mission_node_exit_contract(self):
        result = subprocess.run([sys.executable, __file__, "--child"], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--child" in sys.argv:
        raise SystemExit(child_checks())
    unittest.main()
