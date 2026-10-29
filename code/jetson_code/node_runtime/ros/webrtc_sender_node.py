"""Optional ROS consumer: final JPEG -> session-gated WebRTC only."""
import json
import os
from pathlib import Path

from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from .common import qos_sensor, stamp_seconds
from ..webrtc_media import GStreamerWebRTCSender, WebRTCMediaConfig
from ..webrtc_sender import SessionSender
from ..webrtc_signaling import FirestoreSignaling


# [대시보드 영상 입구] display_node가 합성한 JPEG를 구독하는 독립 소비자다.
# 세션/SDP·ICE 교환은 webrtc_sender.py와 webrtc_signaling.py, 영상 전송은 webrtc_media.py 참조.
class WebRTCSenderNode(Node):
    def __init__(self, **kwargs):
        super().__init__("rail_webrtc_sender", **kwargs)
        # [1. 시작 설정] GUI 구독 토픽과 인증/시그널링 설정 파일 경로를 ROS 파라미터로 읽는다.
        # config_file의 기본값만 환경변수에서 가져온다. 실제 인증정보는 외부 파일에 두며
        # 이 노드 실행만으로 카메라·display·대시보드를 함께 켜지는 않는다.
        descriptor = ParameterDescriptor(read_only=True)
        def param(name, default):
            return self.declare_parameter(name, default, descriptor=descriptor).value
        topic = param("gui_output_topic", "/rail_robot/gui/image/compressed")
        path = param("config_file", os.environ.get("RAIL_ROBOT_WEBRTC_CONFIG", ""))
        self._closed = False
        self._sender = None
        self._lock_file = None
        self._received = self._rejected = 0
        self._last_stamp_ns = -1
        self._status = self.create_publisher(String, "/rail_robot/webrtc/status", qos_sensor)
        if path:
            # [2. 송신 준비] 파일 읽기/검증 뒤 시그널링 객체를 만든다. 경로를 지정했는데 잘못된
            # 파일이면 여기서 초기화 예외가 난다. 경로 자체가 비어 있는 disabled 상태와 구별한다.
            config = FirestoreSignaling.load(path)
            signaling = FirestoreSignaling(config)
            # 같은 로봇의 sender를 이 기기에서 중복 실행하지 않도록 파일 잠금을 유지한다.
            # 여기서 BlockingIOError가 나면 먼저 실행 중인 sender가 있는지 확인한다.
            # 잠금은 파일의 존재가 아니라 열린 파일 핸들에 걸린다. 파일을 삭제해 중복 제한을
            # 우회하지 말고 기존 프로세스를 확인한다. 정상 종료 시 prepare_shutdown이 핸들을 닫는다.
            import fcntl
            lock_dir = Path.home() / ".cache" / "rail_robot"
            lock_dir.mkdir(parents=True, exist_ok=True)
            self._lock_file = open(lock_dir / (config["robotId"] + "-webrtc.lock"), "a")
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            media = config.get("media", {})
            # [3. 미디어 설정 전달] encoder=auto의 실제 선택은 webrtc_media.select_encoder가 맡는다.
            # bitrate/fps는 송신 설정값이다. 이 값만 보고 실제 대역폭/FPS가 달성됐다고 판단하지 않는다.
            # STUN/TURN 주소를 전달할 뿐 서버를 설치하거나 구성하지는 않는다.
            media_config = WebRTCMediaConfig(
                encoder=media.get("encoder", "auto"),
                software_encoder=media.get("softwareEncoder", "x264enc"),
                bitrate_kbps=media.get("bitrateKbps", 2000), fps=media.get("fps", 12),
                stun_server=media.get("stunServer", ""),
                turn_servers=tuple(media.get("turnServers", [])))
            # SessionSender가 별도 시그널링 작업자를 시작한다. 유효한 뷰어 세션을 받아야
            # lambda로 미디어 객체를 만들어 협상하므로 노드 실행과 영상 송신 시작은 다른 단계다.
            self._sender = SessionSender(signaling, lambda: GStreamerWebRTCSender(media_config))
        self.create_subscription(CompressedImage, topic, self._receive, qos_sensor)
        # config가 없으면 disabled 상태만 보고한다. 카메라/추론 노드를 이 노드에서 시작하지 않는다.
        self.create_timer(0.1, self._watchdog)
        self.create_timer(1.0, self._publish_status)

    def _receive(self, message):
        # [4. ROS 입력] 설정 없는 상태에서는 수신 카운터도 증가시키지 않는다.
        # received는 브라우저 재생 프레임 수가 아닌 이 콜백의 입력 수이며, rejected도 여기에 포함된다.
        if self._closed or self._sender is None:
            return
        self._received += 1
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        age = self.get_clock().now().nanoseconds / 1e9 - stamp_seconds(message.header.stamp)
        # 합성 시각 기준으로 오래되거나 중복/역순인 프레임, 잘못된 형식과 과대 메시지는 버린다.
        # rejected가 늘면 format/frame_id → age(0~0.8초) → 크기(4MiB 이하) → 단조 증가 stamp를 본다.
        # age는 ROS 시계 간 차이이므로 다른 기기에서 발행한다면 시계 기준도 확인한다.
        if (message.format != "bgr8; jpeg compressed bgr8" or not message.header.frame_id
                or not 0 <= age <= 0.8 or not 0 < len(message.data) <= 4 * 1024 * 1024
                or stamp_ns <= self._last_stamp_ns):
            self._rejected += 1
            return
        self._last_stamp_ns = stamp_ns
        # 콜백은 압축 바이트를 넘기는 데 그친다. push는 활성 세션/만료를 검사하고,
        # 미디어 계층의 제한된 버퍼로 넘긴다. 디코딩·인코딩·네트워크 처리는 별도 실행 흐름이다.
        self._sender.push(bytes(message.data), stamp_ns)

    def _watchdog(self):
        # [5. 세션 감시] 0.1초 타이머가 세션/협상 만료를 확인한다. ROS 입력이 계속 들어와도
        # 유효 세션이 없으면 송신은 유지되지 않는다. 상세 만료 판정은 SessionSender.watchdog 참조.
        if self._sender and not self._closed:
            self._sender.watchdog()

    def _publish_status(self):
        # [6. 관찰 출력] /rail_robot/webrtc/status의 state/errorCode로 설정·협상·전송 문제를 나눈다.
        # received=0이면 GUI 토픽/QoS/설정을, rejected 증가면 위 검증을, 협상 오류면
        # webrtc_sender.py와 webrtc_signaling.py를 본다. active만으로 브라우저 재생 완료를 단정하지 않는다.
        state = {"state": "disabled", "errorCode": "config_required"} if self._sender is None else self._sender.status()
        state.update({"received": self._received, "rejected": self._rejected})
        self._status.publish(String(data=json.dumps(state, allow_nan=False)))

    def prepare_shutdown(self):
        # [7. 종료] 먼저 _closed로 새 콜백 작업을 막고 세션 작업자를 닫은 뒤 중복 실행 잠금을 해제한다.
        # 공통 실행기와 destroy_node 양쪽에서 호출될 수 있으므로 _closed로 중복 정리를 막는다.
        if self._closed:
            return
        self._closed = True
        if self._sender:
            self._sender.close()
        if self._lock_file:
            self._lock_file.close()

    def destroy_node(self):
        self.prepare_shutdown()
        return super().destroy_node()
