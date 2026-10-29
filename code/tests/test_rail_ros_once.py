"""rail_ros_once.py: 도우미의 한 번짜리 요청(LED·음성 cue 발행, E_STOP 호출). 가짜 연결로 ROS에 붙지 않는다."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rail_ros_once.py"
spec = importlib.util.spec_from_file_location("rail_ros_once", SCRIPT)
rail_ros_once = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rail_ros_once)


class FakeTopic:
    def __init__(self, ready_after=0):
        self.ready_after = ready_after
        self.checks = 0
        self.published = []
        self.closed = 0

    def ready(self):
        self.checks += 1
        return self.checks > self.ready_after

    def publish(self, text):
        self.published.append(text)

    def close(self):
        self.closed += 1


class FakeService(FakeTopic):
    def __init__(self, response=(True, ""), ready_after=0):
        super().__init__(ready_after)
        self.response = response
        self.calls = []

    def call(self, command, timeout):
        self.calls.append((command, timeout))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class Run:
    def __init__(self, argv, topic=None, service=None, tick=0.0, interrupt_on_sleep=False):
        self.topic = topic or FakeTopic()
        self.service = service or FakeService()
        self.created = []
        self.now = 0.0
        self.sleeps = []
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

        def topic_link(target):
            self.created.append(("topic", target))
            return self.topic

        def service_link(target):
            self.created.append(("service", target))
            return self.service

        def sleep(seconds):
            self.sleeps.append(seconds)
            self.now += seconds
            if interrupt_on_sleep:
                raise KeyboardInterrupt

        def clock():
            self.now += tick
            return self.now

        with contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            self.code = rail_ros_once.main(argv, topic_link=topic_link, service_link=service_link,
                                           sleep=sleep, clock=clock)


class StringPublishTests(unittest.TestCase):
    def test_publishes_once_after_a_listener_appears(self):
        run = Run(["string", "/bench/debug/led/set", "red"], topic=FakeTopic(ready_after=3))
        self.assertEqual((run.code, run.topic.published, run.topic.closed), (0, ["red"], 1))
        self.assertEqual(run.created, [("topic", "/bench/debug/led/set")])
        self.assertEqual(run.sleeps[-1], rail_ros_once.FLUSH_SECONDS)

    def test_no_listener_fails_with_a_message_instead_of_waiting_forever(self):
        run = Run(["string", "/tts/debug", "mission_started"], topic=FakeTopic(ready_after=10**9), tick=1.0)
        self.assertEqual((run.code, run.topic.published, run.topic.closed), (1, [], 1))
        self.assertIn("/tts/debug", run.stderr.getvalue())

    def test_ctrl_c_while_waiting_exits_130_without_publishing(self):
        run = Run(["string", "/tts/debug", "mission_started"], topic=FakeTopic(ready_after=5),
                  interrupt_on_sleep=True)
        self.assertEqual((run.code, run.topic.published, run.topic.closed), (130, [], 1))


class EmergencyStopTests(unittest.TestCase):
    def test_acknowledged_estop_reports_success(self):
        run = Run(["estop", "/bench/stm/command"])
        self.assertEqual(run.code, 0)
        self.assertEqual(run.service.calls, [("E_STOP", rail_ros_once.RESPONSE_TIMEOUT_SECONDS)])
        self.assertIn("acknowledged", run.stdout.getvalue())
        self.assertEqual(run.service.closed, 1)

    def test_rejected_estop_fails_with_the_bridge_reason(self):
        run = Run(["estop", "/stm/command"], service=FakeService(response=(False, "Bridge is closing.")))
        self.assertEqual(run.code, 1)
        self.assertIn("Bridge is closing.", run.stderr.getvalue())

    def test_sent_but_unanswered_estop_says_it_was_sent(self):
        run = Run(["estop", "/stm/command"], service=FakeService(response=None))
        self.assertEqual(run.code, 1)
        self.assertIn("was sent", run.stderr.getvalue())

    def test_missing_service_says_estop_was_not_sent(self):
        run = Run(["estop", "/stm/command"], service=FakeService(ready_after=10**9), tick=1.0)
        self.assertEqual((run.code, run.service.calls), (1, []))
        self.assertIn("NOT SENT", run.stderr.getvalue())

    def test_slow_discovery_still_sends_estop(self):
        # 예전 `timeout 3s`로는 끝났을 4초 발견 지연에서도 요청을 보낸다.
        run = Run(["estop", "/stm/command"], service=FakeService(ready_after=200), tick=0.0)
        self.assertEqual(run.code, 0)
        self.assertGreaterEqual(sum(run.sleeps), 4.0)
        self.assertEqual(len(run.service.calls), 1)

    def test_ctrl_c_during_the_call_exits_130_and_closes(self):
        run = Run(["estop", "/stm/command"], service=FakeService(response=KeyboardInterrupt()))
        self.assertEqual((run.code, run.service.closed), (130, 1))


class ArgumentTests(unittest.TestCase):
    def test_bad_arguments_never_connect(self):
        long_text = "x" * (rail_ros_once.MAX_TEXT + 1)
        for argv in ([], ["string"], ["string", "/t"], ["string", "t", "red"], ["string", "/t", ""],
                     ["string", "/t", long_text], ["estop"], ["estop", "stm/command"],
                     ["estop", "/stm/command", "extra"], ["call", "/x"]):
            run = Run(argv)
            self.assertEqual(run.code, rail_ros_once.EXIT_USAGE, argv)
            self.assertEqual(run.created, [], argv)

    def test_qos_and_signal_contracts(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for expected in ("ReliabilityPolicy.RELIABLE", "DurabilityPolicy.VOLATILE", "SignalHandlerOptions.NO",
                         "from rail_robot_interfaces.srv import UartCommand"):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
