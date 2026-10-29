"""Installed executable dispatch and a shared ROS executor lifecycle.

Executable wrappers call main(); ROS arguments pass unchanged to rclpy. Imports
of cameras, TensorRT engines and UART happen only for the selected executable.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import signal
import sys
import threading
import time
import traceback


# [공통 실행기] 설치된 *_node 실행 파일 이름을 실제 Python 클래스로 연결한다.
# launch의 노드 선택 → create_node의 생성 → run_node의 콜백 처리/종료 순서로 읽으면 된다.
# 특정 기능의 예외 원인은 NODE_CLASSES가 가리키는 클래스와 traceback의 가장 안쪽 호출을 확인한다.
# 이 파일은 공통 수명 관리만 담당한다. QoS는 common.py, 장치 해제는 각 노드의 종료 훅에 있다.
NODE_CLASSES = {
    "record_node": ("record_node", "RecordNode"),
    "sensor_debug_node": ("sensor_debug_node", "SensorDebugNode"),
    "audio_alert_node": ("audio_alert_node", "AudioAlertNode"),
    "dashboard_telemetry_node": ("dashboard_telemetry_node", "DashboardTelemetryNode"),
    "remote_control_node": ("remote_control_node", "RemoteControlNode"),
    "monitor_node": ("monitor_node", "MonitorNode"),
    "webrtc_sender_node": ("webrtc_sender_node", "WebRTCSenderNode"),
    "camera_node": ("camera_node", "CameraNode"),
    "vision_node": ("vision_node", "VisionNode"),
    "control_node": ("control_node", "ControlNode"),
    "stm_bridge_node": ("stm_bridge_node", "StmBridgeNode"),
    "mission_node": ("mission_node", "MissionNode"),
    "inspection_node": ("inspection_node", "InspectionNode"),
    "display_node": ("display_node", "DisplayNode"),
    "upload_node": ("upload_node", "UploadNode"),
    "training_capture_node": ("training_capture_node", "TrainingCaptureNode"),
    "capture_debug_node": ("capture_debug_node", "CaptureDebugNode"),
    "actuator_debug_node": ("actuator_debug_node", "ActuatorDebugNode"),
    "firebase_snapshot_node": ("firebase_snapshot_node", "FirebaseSnapshotNode"),
    "gui_probe_node": ("gui_probe_node", "GuiProbeNode"),
    "frame_profiler_node": ("frame_profiler_node", "FrameProfilerNode"),
}
FAULT_OBSERVER_NODES = frozenset(("monitor_node", "record_node", "webrtc_sender_node"))
READINESS_ROLES = {
    "camera_node": "camera", "vision_node": "vision",
    "control_node": "control", "inspection_node": "inspection",
}


def create_node(executable):
    """Instantiate only the requested standalone node, using its ROS parameters."""
    # [클래스 생성] 실행 파일의 이름이 사전에 없으면 매핑 오류이고, import에서 실패하면
    # 설치 경로/의존 모듈 문제다. 생성자에서 실패하면 해당 장치·모델·파라미터 초기화를 추적한다.
    # 모두 launch의 'process has died'로 보일 수 있으므로 traceback의 실제 실패 지점을 구별한다.
    try:
        module_name, class_name = NODE_CLASSES[executable]
    except KeyError as exc:
        raise ValueError(f"Unknown rail robot executable: {executable}") from exc
    # 선택한 노드만 import한다. 예를 들어 모니터 실행만으로 카메라나 모델을 초기화하지 않는다.
    module = import_module(f"{__package__}.{module_name}")
    return getattr(module, class_name)()


def _terminal_failure(node):
    # [종료할 장애 판정] 노드마다 _failure 문자열 또는 _failed 플래그를 사용하므로 통일해서 읽는다.
    # 단순히 결과가 아직 없다는 이유로 종료하지는 않는다. 이 값이 설정된 최초 위치가 원인이다.
    failure = getattr(node, "_failure", None)
    if failure:
        return str(failure)
    if getattr(node, "_failed", False):
        return getattr(node, "_error", "") or "Node entered a faulted state."
    return None


def _add_readiness_service(node, executable):
    role = READINESS_ROLES.get(executable)
    if role is None:
        return None
    from std_srvs.srv import Trigger

    def ready(request, response):
        # [준비 응답] 닫힘/장애를 먼저 배제한 뒤 노드별 readiness()로 새 원본이나 모델 준비를 검사한다.
        # 노드가 목록에는 있지만 START가 거절되면 해당 <role>/ready의 success와 message를 확인한다.
        # 이 서비스가 true여도 STM의 STARTED 응답까지 받은 것은 아니다.
        try:
            if getattr(node, "_closed", False):
                response.success, response.message = False, "Node is closed."
            elif _terminal_failure(node):
                response.success, response.message = False, _terminal_failure(node)
            elif hasattr(node, "readiness"):
                response.success, response.message = node.readiness()
            else:
                response.success, response.message = True, "Node is initialized."
        except Exception as exc:
            response.success, response.message = False, str(exc)
        return response

    # /mission/start의 준비 확인에 사용된다. 노드 존재 여부와 장치/모델 준비 완료는 다르다.
    return node.create_service(Trigger, f"{role}/ready", ready)


def run_node(executable, ros_args=None):
    """Stop work, quiesce callbacks, then close resources while ROS is alive."""
    if executable not in NODE_CLASSES:
        print(f"Unknown rail robot executable: {executable}", file=sys.stderr)
        return 2
    import rclpy
    from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
    from rclpy.exceptions import InvalidHandle
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import String
    from .common import qos_events

    node = executor = None
    initialized = False
    exit_code = 0
    stopping = threading.Event()
    received_fault = []
    previous_handlers = {}

    def request_stop(signum, frame):
        # [중단 요청] 신호 핸들러는 플래그만 세운다. 여기서 바로 카메라나 UART를 닫으면
        # 실행 중인 콜백과 충돌할 수 있으므로 실제 정리는 아래 finally가 맡는다.
        stopping.set()

    def faulted(message):
        if message.data:
            received_fault.append(message.data)
            stopping.set()

    try:
        # Ctrl+C에서 ROS 통신부터 끊으면 종료 정리 중 OFF 전송/서비스 취소가 불가능해진다.
        # 먼저 정리 절차를 실행하고 마지막에 ROS context를 닫도록 신호를 직접 처리한다.
        rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
        initialized = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
        node = create_node(executable)
        executor = MultiThreadedExecutor(num_threads=2)
        # 콜백 실행 기회가 두 개라는 뜻이다. 실제 동시 실행 범위는 각 노드의 callback group이 제한한다.
        executor.add_node(node)
        _add_readiness_service(node, executable)
        if executable not in FAULT_OBSERVER_NODES:
            # 공통 fault를 받으면 종료한다. monitor/record는 장애 이후도 관찰·기록하도록 제외한다.
            # webrtc_sender는 압축 영상 토픽만 구독하는 원격 보기용이라 장치를 쥐지 않는다. 따로 띄운
            # 송출이 rail_drive 오류나 임무 fault마다 함께 꺼지지 않도록 제외한다.
            # 음성도 이번 작업에서는 세 가지 안내만 맡으므로 fault/운영자 종료 때 재생을 정리한다.
            node.create_subscription(String, "runtime/fault", faulted, qos_events)
        while rclpy.ok() and not stopping.is_set():
            # [정상 실행 루프] 노드 자체 장애/종료 요청을 확인하며 executor가 준비된 콜백을 처리한다.
            # timeout_sec=0.1은 spin의 대기 상한이다. 카메라 FPS나 모델 추론 주기를 뜻하지 않는다.
            # 각 기능의 작업 속도는 노드 타이머, callback group, 별도 작업자에서 결정한다.
            failure = _terminal_failure(node)
            if failure:
                received_fault.append(failure)
                break
            requested = getattr(node, "shutdown_requested", None)
            if getattr(node, "_closed", False) or (callable(requested) and requested()):
                break
            executor.spin_once(timeout_sec=0.1)
        if received_fault:
            exit_code = 1
            print(received_fault[0], file=sys.stderr)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        # A context explicitly closed elsewhere is already unavailable, but
        # still attempt hardware cleanup below.
        pass
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        # [종료 1: 새 작업 차단] 타이머를 취소하고 노드별 prepare_shutdown을 호출한다.
        # 이 시점에는 ROS가 살아 있어 취소된 서비스 응답과 필요한 종료 메시지를 처리할 수 있다.
        if node is not None:
            try:
                # Prevent new timer work before waiting on in-flight callbacks.
                for timer in tuple(node.timers):
                    timer.cancel()
                prepare = getattr(node, "prepare_shutdown", None)
                if callable(prepare):
                    prepare()
            except Exception:
                exit_code = 1
                traceback.print_exc()
        callbacks_stopped = executor is None
        if executor is not None:
            # [종료 2: 진행 중 콜백 정리] 서비스 취소로 깨어난 await 콜백에 정리 기회를 준 뒤
            # executor 종료를 최대 5초 기다린다. 이 시간이 넘으면 자원을 쓰는 콜백이 남았을 수 있다.
            # 이 로그가 나오면 해당 노드의 장치 read/작업자 종료/서비스 취소 경로를 확인한다.
            deadline = time.monotonic() + 0.25
            while rclpy.ok() and time.monotonic() < deadline:
                try:
                    executor.spin_once(timeout_sec=0.01)
                except (InvalidHandle, ExternalShutdownException):
                    pass
                except Exception:
                    exit_code = 1
                    traceback.print_exc()
                    break
            try:
                callbacks_stopped = executor.shutdown(timeout_sec=5.0) is not False
                if not callbacks_stopped:
                    exit_code = 1
                    print("ROS executor did not finish pending callbacks within 5 seconds.", file=sys.stderr)
            except Exception:
                exit_code = 1
                traceback.print_exc()
        if node is not None and callbacks_stopped:
            # 콜백이 자원을 쓰는 동안 destroy_node를 호출하지 않도록 종료 완료를 확인한다.
            try:
                node.destroy_node()
            except Exception:
                exit_code = 1
                traceback.print_exc()
        elif node is not None:
            print("Resources remain open because a callback is still using them.", file=sys.stderr)
        if initialized:
            # [종료 3: ROS 종료] 정상 경로는 콜백 정지 → destroy_node의 장치 해제 → ROS 종료다.
            # 콜백 정지가 실패한 경로에서는 위에서 자원을 그대로 두고 오류로 끝낸다.
            try:
                rclpy.try_shutdown()
            except Exception:
                exit_code = 1
                traceback.print_exc()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return exit_code


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    executable = Path(argv[0]).stem if argv else ""
    raise SystemExit(run_node(executable, ros_args=argv))


if __name__ == "__main__":
    main()
