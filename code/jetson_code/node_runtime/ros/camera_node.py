"""ROS camera owner; import does not open devices or initialize ROS."""
from __future__ import annotations

import time
import threading
from uuid import uuid4

from rclpy.node import Node
from rail_robot_interfaces.msg import CameraFrame as CameraFrameMessage
from rail_robot_interfaces.srv import CapturePair
from std_msgs.msg import Float64, String

from ..camera import (
    open_latest_frame_camera,
    read_stopped_capture_pair,
    validate_distinct_camera_devices,
)
from ..settings import FRAME_RATE, REALTIME_INPUT_MAX_AGE_SECONDS
from ..metrics import FrameRateMeter
from .deployment import read_parameters
from .common import qos_events, qos_sensor
from .images import frame_to_message


# 카메라 장치를 여는 진입점: 원본은 camera/{side,top} 토픽으로 보내고,
# 정지 구간의 두 장 촬영은 camera/capture_pair 서비스로 제공한다.
# USB 읽기 스레드·정지 후 대기는 node_runtime/camera.py에서 확인한다.
# ROS 이미지 변환·시각 계산은 ros/images.py, FPS 집계는 node_runtime/metrics.py로 이어진다.
class CameraNode(Node):
    """Own the selected camera readers and publish each sequence at most once."""

    def __init__(
        self,
        side_device: str | None = None,
        top_device: str | None = None,
        *,
        camera_roles: str | None = None,
        camera_factory=open_latest_frame_camera,
        node_name: str = "rail_camera",
        **node_kwargs,
    ) -> None:
        super().__init__(node_name, **node_kwargs)
        self._resource_lock = threading.RLock()
        self._cameras = {}
        self._closed = False
        self._stopping = False
        self._failure = None
        self._last_read_times = {}
        self._stream_id = str(uuid4())
        self._last_sequences = {"side": 0, "top": 0}
        self._faults = self.create_publisher(String, "runtime/fault", qos_events)
        try:
            # 실행 시 정한 camera_roles로 실제 열 장치를 제한한다. 인자로 장치가 없으면
            # ROS 파라미터를 읽고, both에서는 두 이름이 동일 장치를 가리키는지도 검사한다.
            # 장치 열기 실패는 아래 예외 경로에서 이미 열린 장치까지 닫은 후 상위로 전달된다.
            if camera_roles is None:
                from rcl_interfaces.msg import ParameterDescriptor
                camera_roles = (
                    self.get_parameter("camera_roles").value
                    if self.has_parameter("camera_roles")
                    else self.declare_parameter(
                        "camera_roles", "both", descriptor=ParameterDescriptor(read_only=True)
                    ).value
                )
            if camera_roles not in ("both", "side", "front"):
                raise ValueError("camera_roles must be both, side, or front.")
            # 터미널의 front 선택은 내부 토픽과 모델 코드에서 top으로 부른다.
            # 따라서 전방 영상은 camera/top, 전방 FPS는 debug/camera/front/fps를 확인한다.
            self._selected_roles = {
                "both": ("side", "top"), "side": ("side",), "front": ("top",)
            }[camera_roles]
            devices = {"side": side_device, "top": top_device}
            for role in self._selected_roles:
                if devices[role] is None:
                    parameter = "side_camera_device" if role == "side" else "top_camera_device"
                    devices[role] = read_parameters(self, (parameter,))[parameter]
            if camera_roles == "both":
                validate_distinct_camera_devices(devices["side"], devices["top"])
            for role in self._selected_roles:
                self._cameras[role] = camera_factory(devices[role])
            self._frame_publishers = {
                role: self.create_publisher(
                    CameraFrameMessage, f"camera/{role}", qos_sensor
                )
                for role in self._selected_roles
            }
            # 발행 타이머의 FRAME_RATE는 확인 주기이고 실제 카메라 FPS는 아래 계측값이다.
            # 같은 sequence를 다시 읽어도 새 촬영으로 세지 않으므로 FPS는 타이머 횟수와 다르다.
            self._fps_meters = {role: FrameRateMeter() for role in self._selected_roles}
            self._fps_publishers = {
                role: self.create_publisher(
                    Float64, f"debug/camera/{'front' if role == 'top' else role}/fps", qos_sensor
                )
                for role in self._selected_roles
            }
            self._capture_service = self.create_service(
                CapturePair, "camera/capture_pair", self._capture_pair
            )
            self._timer = self.create_timer(1.0 / FRAME_RATE, self._publish_latest)
        except Exception:
            self.destroy_node()
            raise

    def _message(self, role, frame):
        # 촬영 시각은 읽기 완료 시각에서 변환한다. 재시작마다 바뀌는 stream_id와
        # sequence를 함께 전달하므로 이전 실행의 분석 결과와 새 원본을 구별할 수 있다.
        message = frame_to_message(
            frame,
            role,
            ros_now_seconds=self.get_clock().now().nanoseconds / 1e9,
            monotonic_now=time.monotonic(),
        )
        message.stream_id = self._stream_id
        self._last_read_times[role] = frame.read_completed_at
        self._fps_meters[role].record(
            self._stream_id, frame.sequence, frame.read_completed_at, time.monotonic()
        )
        return message

    def _publish(self, role, message) -> None:
        self._frame_publishers[role].publish(message)
        self._last_sequences[role] = message.sequence

    def _fault(self, detail: str) -> None:
        # runtime/fault에는 실패한 역할과 상세 원인을 남기고 _failure를 유지한다.
        # 이후 원본이 잠시 다시 들어와도 readiness는 실패 상태를 자동 해제하지 않는다.
        self._failure = detail
        message = String()
        message.data = f"camera: {detail}"
        if not self._closed:
            self._faults.publish(message)
        self.get_logger().error(message.data)

    def _publish_latest(self) -> None:
        with self._resource_lock:
            if self._closed or self._stopping:
                return
            self._publish_latest_locked()

    def _publish_latest_locked(self) -> None:
        # 타이머에서 새 사진을 기다리지 않는다. 이미 읽힌 최신 프레임만 발행하여
        # 카메라가 느려져도 이 콜백에 대기 프레임이 쌓이지 않게 한다.
        for role, camera in self._cameras.items():
            try:
                frame = camera.read_latest(
                    after_sequence=self._last_sequences[role], timeout=0.0
                )
                if frame.sequence > self._last_sequences[role]:
                    self._publish(role, self._message(role, frame))
            except TimeoutError:
                # timeout=0에서 새 sequence가 없다는 뜻이며 장치 고장으로 확정하지 않는다.
                # 영상이 멈추면 _last_sequences 진행 여부와 FPS, readiness의 최신성을 함께 본다.
                pass
            except Exception as exc:
                self._fault(f"{role}: {exc}")
            fps = Float64()
            fps.data = float(self._fps_meters[role].fps(time.monotonic()) or 0.0)
            self._fps_publishers[role].publish(fps)

    def _capture_pair(self, request, response):
        """Settle once and return two new originals; never acknowledge the STM."""
        # 쌍 촬영과 일반 발행은 같은 자원 잠금을 사용한다. 정지 촬영 중에는 이 노드의
        # 일반 발행이 기다릴 수 있으며, 단일 카메라 모드에서는 쌍 촬영 요청을 거절한다.
        with self._resource_lock:
            if self._closed or self._stopping:
                response.success, response.error = False, "Camera node is closed."
                return response
            if len(self._selected_roles) != 2:
                response.success = False
                response.error = "Paired capture requires camera_roles=both."
                return response
            return self._capture_pair_locked(request, response)

    def _capture_pair_locked(self, request, response):
        try:
            # 정지 후 새 SIDE/TOP 원본을 확보하는 단계다. 이 응답 뒤의 저장·분석 수락은
            # inspection 노드가, STM에 보낼 촬영 확인은 mission 노드가 담당한다.
            # 실패 시 response.error의 request_id로 mission의 촬영 요청과 로그를 연결한다.
            side, top = read_stopped_capture_pair(
                self._cameras["side"], self._cameras["top"]
            )
            side_message = self._message("side", side)
            top_message = self._message("top", top)
            self._publish("side", side_message)
            self._publish("top", top_message)
            response.side = side_message
            response.top = top_message
            response.success = True
            response.error = ""
        except Exception as exc:
            response.success = False
            response.error = f"capture {request.request_id}: {exc}"
            self._fault(response.error)
        return response

    def readiness(self):
        with self._resource_lock:
            if self._closed or self._stopping or self._failure:
                return False, self._failure or "Camera node is closed."
            now = time.monotonic()
            # 장치가 열렸다는 사실만으로 준비 완료로 보지 않고 원본의 최신성도 확인한다.
            # _last_read_times에 역할이 없거나 현재 시각과의 차이가 허용 범위를 벗어나면
            # 시작 준비가 끝나지 않은 상태이므로 카메라 토픽부터 추적한다.
            if not all(
                role in self._last_read_times
                and 0 <= now - self._last_read_times[role] <= REALTIME_INPUT_MAX_AGE_SECONDS
                for role in self._selected_roles
            ):
                return False, "Waiting for fresh frames from the selected cameras."
            return True, "Selected cameras provide fresh originals."

    def prepare_shutdown(self):
        self._stopping = True

    def destroy_node(self):
        with self._resource_lock:
            return self._destroy_locked()

    def _destroy_locked(self):
        # _stopping으로 새 작업을 막은 뒤 장치별 읽기 자원을 반환하는 종료 경로다.
        # 종료 지연은 camera.close()와 하위 LatestFrameCamera 읽기 스레드를 확인한다.
        if not self._closed:
            self._closed = True
            for role, camera in self._cameras.items():
                try:
                    camera.close()
                except Exception as exc:
                    self.get_logger().error(f"Camera {role} shutdown failed: {exc}")
        return super().destroy_node()
