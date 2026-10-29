"""rail_alert.py: 장애물 경고음 시험 스크립트의 순서·시간 초과·Ctrl+C 처리. 가짜 게시자를 쓰며 ROS에 연결하지 않는다."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rail_alert.py"
spec = importlib.util.spec_from_file_location("rail_alert", SCRIPT)
rail_alert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rail_alert)

TOPIC = "/bench/tts/obstacle_state"


class FakeLink:
    def __init__(self, subscribers=1):
        self.subscribers = subscribers
        self.published = []
        self.closed = 0

    def subscriber_count(self):
        return self.subscribers

    def publish(self, value):
        self.published.append(value)

    def close(self):
        self.closed += 1


class Run:
    """main을 가짜 연결·가짜 시간으로 실행하고 결과를 모아 둔다."""

    def __init__(self, argv, link=None, on_sleep=None, tick=0.0):
        self.link = link or FakeLink()
        self.sleeps = []
        self.now = 0.0
        self.connected = []
        self.stderr = io.StringIO()

        def connect(topic):
            self.connected.append(topic)
            return self.link

        def sleep(seconds):
            self.sleeps.append(seconds)
            self.now += seconds
            if on_sleep:
                on_sleep(self, seconds)

        def clock():
            self.now += tick
            return self.now

        with contextlib.redirect_stderr(self.stderr):
            self.code = rail_alert.main(argv, connect=connect, sleep=sleep, clock=clock)


class RailAlertTests(unittest.TestCase):
    def test_on_sends_true_once_and_leaves_it_running(self):
        run = Run([TOPIC, "on"])
        self.assertEqual((run.code, run.link.published, run.link.closed), (0, [True], 1))
        self.assertIn("rail_alert off", run.stderr.getvalue())

    def test_off_sends_false_once(self):
        run = Run([TOPIC, "off"])
        self.assertEqual((run.code, run.link.published, run.link.closed), (0, [False], 1))

    def test_seconds_sends_true_waits_that_long_then_false(self):
        run = Run([TOPIC, "3"])
        self.assertEqual((run.code, run.link.published, run.link.closed), (0, [True, False], 1))
        self.assertIn(3.0, run.sleeps)
        # true를 보낸 뒤에 기다리고, false는 기다린 뒤에 보낸다.
        self.assertEqual(run.sleeps.index(3.0), run.sleeps.index(rail_alert.FLUSH_SECONDS) + 1)

    def test_nothing_is_sent_before_a_listener_exists(self):
        link = FakeLink(subscribers=0)

        def appear(run, seconds):
            if len(run.sleeps) == 3:
                link.subscribers = 1

        run = Run([TOPIC, "on"], link=link, on_sleep=appear)
        self.assertEqual((run.code, link.published), (0, [True]))
        self.assertGreaterEqual(len(run.sleeps), 3)

    def test_missing_audio_node_fails_with_a_hint_instead_of_waiting_forever(self):
        run = Run([TOPIC, "3"], link=FakeLink(subscribers=0), tick=1.0)
        self.assertEqual(run.code, rail_alert.EXIT_FAILED)
        self.assertEqual(run.link.published, [])
        self.assertEqual(run.link.closed, 1)
        self.assertIn("rail_tts", run.stderr.getvalue())

    def test_ctrl_c_while_playing_sends_stop_and_exits_130(self):
        def interrupt(run, seconds):
            if seconds == 3.0:
                raise KeyboardInterrupt

        run = Run([TOPIC, "3"], on_sleep=interrupt)
        self.assertEqual(run.code, rail_alert.EXIT_INTERRUPTED)
        self.assertEqual(run.link.published, [True, False])
        self.assertEqual(run.link.closed, 1)

    def test_ctrl_c_after_the_audio_node_disappeared_does_not_wait_to_send_stop(self):
        link = FakeLink()

        def interrupt(run, seconds):
            if seconds == 3.0:
                link.subscribers = 0
                raise KeyboardInterrupt

        run = Run([TOPIC, "3"], link=link, on_sleep=interrupt)
        self.assertEqual((run.code, link.published), (rail_alert.EXIT_INTERRUPTED, [True]))

    def test_ctrl_c_exits_130_even_if_the_link_is_already_closed(self):
        class ClosedLink(FakeLink):
            def subscriber_count(self):
                if self.published:
                    raise RuntimeError("publisher's context is invalid")
                return 1

            def close(self):
                raise RuntimeError("already shut down")

        def interrupt(run, seconds):
            if seconds == 3.0:
                raise KeyboardInterrupt

        run = Run([TOPIC, "3"], link=ClosedLink(), on_sleep=interrupt)
        self.assertEqual(run.code, rail_alert.EXIT_INTERRUPTED)
        self.assertIn("Stop message was not sent", run.stderr.getvalue())

    def test_rclpy_signal_handlers_are_disabled_so_ctrl_c_can_send_stop(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("SignalHandlerOptions.NO", source)

    def test_ctrl_c_before_true_was_sent_sends_nothing(self):
        def interrupt(run, seconds):
            raise KeyboardInterrupt

        run = Run([TOPIC, "3"], link=FakeLink(subscribers=0), on_sleep=interrupt)
        self.assertEqual((run.code, run.link.published, run.link.closed),
                         (rail_alert.EXIT_INTERRUPTED, [], 1))

    def test_bad_arguments_never_connect(self):
        for argv in ([], [TOPIC], [TOPIC, "typo"], [TOPIC, "0"], [TOPIC, "-1"], [TOPIC, "61"],
                     [TOPIC, "1e3"], [TOPIC, "nan"], [TOPIC, "inf"], [TOPIC, ".5"], [TOPIC, ""],
                     ["tts/obstacle_state", "on"], [TOPIC, "on", "extra"]):
            run = Run(argv)
            self.assertEqual(run.code, rail_alert.EXIT_USAGE, argv)
            self.assertEqual(run.connected, [], argv)

    def test_seconds_limits(self):
        self.assertEqual(rail_alert.parse_mode("0.1"), 0.1)
        self.assertEqual(rail_alert.parse_mode("60"), 60.0)
        self.assertIsNone(rail_alert.parse_mode("60.01"))
        self.assertEqual(rail_alert.parse_mode("on"), "on")

    def test_publisher_uses_the_audio_node_state_qos(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for expected in ("ReliabilityPolicy.RELIABLE", "DurabilityPolicy.TRANSIENT_LOCAL", "depth=1",
                         "from std_msgs.msg import Bool"):
            self.assertIn(expected, source)


if __name__ == "__main__":
    unittest.main()
