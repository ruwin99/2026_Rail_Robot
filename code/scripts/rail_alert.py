#!/usr/bin/env python3
"""장애물 경고음 시험: 한 프로세스에서 /tts/obstacle_state에 true를 보내고, 지정한 시간 뒤 false를 보낸다.

사용: rail_alert.py <토픽 절대경로> <on|off|초>   (rail_alert 도우미가 호출한다)

ros2 명령줄은 한 번 켜는 데 약 1초가 들어 true와 false를 따로 보내면 느리다
(Jetson 실측: ros2 topic pub 두 번 3.4초, 이 스크립트 1.0초). 센서·STM·모터에는 아무것도 보내지 않는다.
"""
from __future__ import annotations

import re
import signal
import sys
import time

EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
MAX_SECONDS = 60.0
DISCOVERY_TIMEOUT_SECONDS = 10.0
FLUSH_SECONDS = 0.15  # reliable 전송이 끝나기 전에 프로세스가 내려가지 않게 하는 짧은 대기

USAGE = "usage: rail_alert.py <topic> <on|off|seconds>   (seconds: more than 0, at most 60)"


class RosLink:
    """rclpy 게시자 하나. 시험에서는 같은 메서드를 가진 대역으로 바꾼다."""

    def __init__(self, topic: str) -> None:
        import rclpy
        from rclpy.signals import SignalHandlerOptions
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool

        self._rclpy = rclpy
        self._message = Bool
        # rclpy 기본 신호 처리기는 Ctrl+C·SIGTERM에서 컨텍스트를 먼저 닫아 버려 정지(false)를 보낼 수 없다.
        # 끄고 파이썬의 KeyboardInterrupt로만 처리한다.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node("rail_alert_cli")
        # 음성 노드의 qos_state(reliable, transient_local, depth 1)와 같은 계약이다.
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._publisher = self._node.create_publisher(Bool, topic, qos)

    def subscriber_count(self) -> int:
        return self._publisher.get_subscription_count()

    def publish(self, value: bool) -> None:
        self._publisher.publish(self._message(data=value))

    def close(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()


def parse_mode(text: str):
    """'on'/'off'는 그대로, 숫자는 0 초과 60 이하의 초로 돌려주고, 그 밖은 None."""
    if text in ("on", "off"):
        return text
    # 도우미의 숫자 검사와 같은 규칙이다. nan, inf, 부호, 지수 표기(1e3), ".5"는 거부한다.
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", text) is None:
        return None
    seconds = float(text)
    return seconds if 0.0 < seconds <= MAX_SECONDS else None


def main(argv, *, connect=RosLink, sleep=time.sleep, clock=time.monotonic) -> int:
    if len(argv) != 2 or not argv[0].startswith("/"):
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    topic, mode = argv[0], parse_mode(argv[1])
    if mode is None:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    link = connect(topic)
    sent = False
    try:
        deadline = clock() + DISCOVERY_TIMEOUT_SECONDS
        while link.subscriber_count() < 1:
            if clock() >= deadline:
                print(f"No audio node is listening on {topic}. Run rail_tts in another terminal first.",
                      file=sys.stderr)
                return EXIT_FAILED
            sleep(0.02)
        if mode == "off":
            link.publish(False)
            sleep(FLUSH_SECONDS)
            return 0
        link.publish(True)
        sent = True
        sleep(FLUSH_SECONDS)
        if mode == "on":
            print("Obstacle alert repeats until: rail_alert off", file=sys.stderr)
            return 0
        sleep(mode)
        link.publish(False)
        sent = False
        sleep(FLUSH_SECONDS)
        return 0
    except KeyboardInterrupt:
        # true를 보낸 뒤 끊긴 경우에만 정지를 보낸다. 구독자가 사라졌으면 보낼 필요도 기다릴 이유도 없다.
        try:
            if sent and link.subscriber_count() > 0:
                link.publish(False)
                sleep(FLUSH_SECONDS)
        except Exception as error:  # 통신이 이미 닫혔어도 종료 코드는 130으로 끝낸다.
            print(f"Stop message was not sent: {error}", file=sys.stderr)
        return EXIT_INTERRUPTED
    finally:
        try:
            link.close()
        except Exception:
            pass


def _terminate(_signal, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main(sys.argv[1:]))
