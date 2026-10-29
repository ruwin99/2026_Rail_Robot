#!/usr/bin/env python3
"""도우미 명령(rail_led, rail_tts start/done, rail_estop)의 한 번짜리 ROS 요청을 한 파이썬 프로세스로 보낸다.

사용:
  rail_ros_once.py string <토픽 절대경로> <문자열>   std_msgs/String을 한 번 발행
  rail_ros_once.py estop <서비스 절대경로>           UartCommand(E_STOP) 호출

ros2 명령줄은 켜는 데만 약 1초가 든다(Jetson 실측). 또 rail_estop의 `timeout 3s ros2 service call`은
기동과 DDS 발견(가끔 2~4초)만으로 3초를 넘겨 요청을 보내기도 전에 실패할 수 있었다. 여기서는 서비스를
찾는 시간과 응답을 기다리는 시간을 나누고, 실패하면 어느 단계였는지 출력한다.
"""
from __future__ import annotations

import signal
import sys
import time

EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130
DISCOVERY_TIMEOUT_SECONDS = 10.0
RESPONSE_TIMEOUT_SECONDS = 5.0  # stm_bridge가 STM ACK를 최대 0.35초 기다린 뒤 응답한다.
FLUSH_SECONDS = 0.15  # reliable 전송이 끝나기 전에 프로세스가 내려가지 않게 하는 짧은 대기
MAX_TEXT = 256

USAGE = ("usage: rail_ros_once.py string <topic> <text>\n"
         "       rail_ros_once.py estop <service>")


def _init_rclpy():
    import rclpy
    from rclpy.signals import SignalHandlerOptions

    # rclpy 기본 신호 처리기는 Ctrl+C에서 컨텍스트를 먼저 닫는다. 파이썬 KeyboardInterrupt로만 처리한다.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    return rclpy, rclpy.create_node("rail_ros_once")


def _shutdown(rclpy, node) -> None:
    try:
        node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


class TopicLink:
    """String 게시자 하나. 받는 노드(tts/debug, debug/led/set)와 같이 reliable·volatile이다."""

    def __init__(self, topic: str) -> None:
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        self._rclpy, self._node = _init_rclpy()
        self._message = String
        # volatile이라 재시작한 노드에 지난 요청이 다시 전달되지 않는다.
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self._publisher = self._node.create_publisher(String, topic, qos)

    def ready(self) -> bool:
        return self._publisher.get_subscription_count() > 0

    def publish(self, text: str) -> None:
        self._publisher.publish(self._message(data=text))

    def close(self) -> None:
        _shutdown(self._rclpy, self._node)


class ServiceLink:
    """stm/command 서비스 클라이언트."""

    def __init__(self, service: str) -> None:
        from rail_robot_interfaces.srv import UartCommand

        self._rclpy, self._node = _init_rclpy()
        self._request_type = UartCommand.Request
        self._client = self._node.create_client(UartCommand, service)

    def ready(self) -> bool:
        return self._client.service_is_ready()

    def call(self, command: str, timeout: float):
        """응답이 오면 (success, error), 시간 안에 오지 않으면 None."""
        request = self._request_type()
        request.command = command
        future = self._client.call_async(request)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout)
        if not future.done():
            future.cancel()
            return None
        response = future.result()
        return bool(response.success), str(response.error)

    def close(self) -> None:
        _shutdown(self._rclpy, self._node)


def _wait_ready(link, sleep, clock) -> bool:
    deadline = clock() + DISCOVERY_TIMEOUT_SECONDS
    while not link.ready():
        if clock() >= deadline:
            return False
        sleep(0.02)
    return True


def _publish_string(link, target: str, text: str, sleep, clock) -> int:
    if not _wait_ready(link, sleep, clock):
        print(f"No node is listening on {target}. Is the matching node running?", file=sys.stderr)
        return EXIT_FAILED
    link.publish(text)
    sleep(FLUSH_SECONDS)
    return 0


def _call_estop(link, target: str, sleep, clock) -> int:
    if not _wait_ready(link, sleep, clock):
        print(f"E_STOP NOT SENT: service {target} was not found within {DISCOVERY_TIMEOUT_SECONDS:.0f} s. "
              "Is rail_mission (stm_bridge) running?", file=sys.stderr)
        return EXIT_FAILED
    result = link.call("E_STOP", RESPONSE_TIMEOUT_SECONDS)
    if result is None:
        print(f"E_STOP was sent, but no response within {RESPONSE_TIMEOUT_SECONDS:.0f} s. "
              "Check the robot and the STM bridge.", file=sys.stderr)
        return EXIT_FAILED
    success, error = result
    if not success:
        print(f"E_STOP rejected: {error or 'no reason given'}", file=sys.stderr)
        return EXIT_FAILED
    print("E_STOP acknowledged by STM")
    return 0


def main(argv, *, topic_link=TopicLink, service_link=ServiceLink, sleep=time.sleep, clock=time.monotonic) -> int:
    if len(argv) == 3 and argv[0] == "string":
        target, text = argv[1], argv[2]
        if not target.startswith("/") or not text or len(text) > MAX_TEXT:
            print(USAGE, file=sys.stderr)
            return EXIT_USAGE
        factory, run = topic_link, lambda link: _publish_string(link, target, text, sleep, clock)
    elif len(argv) == 2 and argv[0] == "estop":
        target = argv[1]
        if not target.startswith("/"):
            print(USAGE, file=sys.stderr)
            return EXIT_USAGE
        factory, run = service_link, lambda link: _call_estop(link, target, sleep, clock)
    else:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    link = factory(target)
    try:
        return run(link)
    except KeyboardInterrupt:
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
