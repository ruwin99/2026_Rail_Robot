"""Select a working node group without writing another debug shell script."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# [노드 구성] 모드별 필수 역할 목록이다. 카메라·GUI와 선택 기능은 configured_nodes에서 추가한다.
# mission도 전체 노드를 무조건 켜지는 않으며, 실제 임무 START 처리는 mission_node가 담당한다.
# 디버깅 순서: MODES → roles 조합 → parameters/forced → Node → 종료 연동.
# 실행 파일 연결이 안 되면 상위 CMakeLists.txt와 node_runtime/ros/entrypoints.py를 확인한다.
# 모델 파일/임계값은 이 파일이 아닌 node_runtime/ros/deployment.py의 노드 기본값에서 정한다.
MODES = {
    "monitor": ("monitor",),
    "mission": ("vision", "control", "stm_bridge", "inspection", "mission"),
    "vision": ("vision",),
    "camera": (),
    "training": ("training_capture",),
    "front_camera": ("capture_debug",),
    "side_camera": ("capture_debug",),
    "foreign_camera": ("capture_debug",),
    "dual_camera": ("vision",),
    "motor_encoder": ("actuator_debug",),
    "dc_motor": ("actuator_debug",),
    "water_pump": ("actuator_debug",),
    "actuators": ("actuator_debug",),
    "led": ("sensor_debug",),
    "ultrasonic": ("sensor_debug",),
    "sensors": ("sensor_debug",),
}
NODE_NAMES = {"inspection": "inspection", "upload": "report_upload", "capture_debug": "capture_debug"}
FRAME_TRACE_TOPIC = "/rail_robot/debug/frame_trace"


def _boolean(value, name):
    if value.lower() not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value.lower() == "true"


def configured_nodes(context):
    # [1. 실행 인자 해석] LaunchConfiguration은 선언만으로 실제 문자열을 갖지 않는다.
    # OpaqueFunction에서 실행 context가 주어진 뒤 값을 읽어 모드와 true/false를 검사한다.
    # 노드가 하나도 뜨기 전에 ValueError가 나면 이 단계의 mode/옵션 조합부터 확인한다.
    def value(name):
        return LaunchConfiguration(name).perform(context)

    mode = value("mode")
    if mode not in MODES:
        raise ValueError("Unknown mode: " + mode)
    uart_protocol = value("uart_protocol")
    if uart_protocol not in ("", "v2", "legacy"):
        raise ValueError("uart_protocol must be v2 or legacy")
    profiling = _boolean(value("profiling"), "profiling")
    if mode == "monitor":
        # 관찰만 하는 모드: 이 launch로 카메라나 액추에이터를 새로 실행하지 않는다.
        if profiling:
            raise ValueError("profiling requires a composed GUI output mode")
        return [Node(package="rail_robot_nodes", executable="monitor_node",
                     name="rail_monitor", output="screen", emulate_tty=True)]
    camera = _boolean(value("camera"), "camera")
    display = _boolean(value("display"), "display")
    upload = _boolean(value("upload"), "upload")
    gui_output = _boolean(value("gui_output"), "gui_output")
    firebase_snapshot = _boolean(value("firebase_snapshot"), "firebase_snapshot")
    dashboard_live = _boolean(value("dashboard_live"), "dashboard_live")
    dashboard_control = _boolean(value("dashboard_control"), "dashboard_control")
    audio_enabled = _boolean(value("audio_enabled"), "audio_enabled")
    frame_trace_topic = value("frame_trace_topic") or FRAME_TRACE_TOPIC
    if profiling and (not frame_trace_topic.startswith("/") or any(c.isspace() for c in frame_trace_topic)):
        raise ValueError("frame_trace_topic must be an absolute ROS topic name without whitespace")
    if upload and mode != "mission":
        raise ValueError("upload requires mode:=mission and finalized inspection reports")
    if dashboard_control and (mode != "mission" or uart_protocol != "v2"):
        raise ValueError("dashboard_control requires mode:=mission and uart_protocol:=v2")
    actuator_mode = any(role in MODES[mode] for role in ("actuator_debug", "sensor_debug"))
    capture_mode = "capture_debug" in MODES[mode]
    if profiling and (actuator_mode or capture_mode or not gui_output):
        raise ValueError("profiling requires a composed GUI output mode")
    # [2. 필요한 프로세스 조합] 액추에이터 디버깅은 UART를 actuator_debug가 소유하므로
    # mission/control/stm_bridge를 추가하지 않는다. camera:=false는 카메라 소비자를 끄는 옵션이
    # 아니라 외부 원본 토픽/촬영 서비스를 사용하겠다는 뜻이므로, 외부 공급이 없으면 준비가 안 된다.
    roles = (["camera"] if camera and not actuator_mode else []) + list(MODES[mode])
    if mode == "mission" and audio_enabled:
        roles.append("audio_alert")
    # 로컬 화면을 끈 display:=false에서도 gui_output:=true이면 합성 영상 발행은 유지한다.
    if (display or gui_output) and not actuator_mode and not capture_mode:
        roles.append("display")
    if firebase_snapshot:
        # 5초 스냅샷은 합성 GUI 토픽의 소비자다. 수동 capture_debug 모드는 별도 GUI를 사용하므로
        # 여기의 display 출력과 연결되지 않는다. 구독할 합성 출력이 없는 조합은 즉시 거절한다.
        if actuator_mode or capture_mode or not gui_output:
            raise ValueError("firebase_snapshot requires a composed GUI output mode")
        roles.append("firebase_snapshot")
    if upload:
        roles.append("upload")
    if dashboard_live:
        # FPS/임무 상태 전송 노드를 추가하고, 아래 inspection 파라미터로 사진 업로드도 선택한다.
        # WebRTC sender는 이 목록에 없다. 영상 송신은 별도 webrtc.launch.py로 실행한다.
        roles.append("dashboard_telemetry")
    if dashboard_control:
        # Explicit local opt-in: Firebase control is never started by telemetry or video.
        roles.append("remote_control")
    if profiling:
        # 프로파일러는 GUI trace의 선택 소비자이며 제어·UART 필수 역할에 속하지 않는다.
        roles.append("frame_profiler")
    if not roles:
        raise ValueError("The selected configuration has no nodes")

    config = value("params_file")
    if config:
        config = str(Path(config).expanduser().resolve(strict=True))
        if not Path(config).is_file():
            raise ValueError("params_file must be a YAML file")
    overrides = {
        # [3. 파라미터 전달 범위] 비어 있지 않은 launch 인자만 해당 역할에 넘긴다.
        # serial_port를 바꿨는데 반영되지 않으면 실행 모드의 UART 소유 역할과 이 매핑을 확인한다.
        "camera": ("side_camera_device", "top_camera_device"),
        "stm_bridge": ("serial_port", "uart_protocol"),
        "inspection": ("output_directory",),
        "training_capture": ("training_camera_role", "training_output_directory"),
        "actuator_debug": ("serial_port", "uart_protocol"),
        "sensor_debug": ("serial_port", "uart_protocol"),
        "capture_debug": ("capture_output_directory",),
        "display": ("gui_output_topic", "gui_status_topic", "gui_frame_id"),
        "firebase_snapshot": ("gui_output_topic",),
        "audio_alert": ("audio_directory", "audio_device"),
    }
    actions = []
    for role in roles:
        parameters = [config] if config else []
        configured = {name: value(name) for name in overrides.get(role, ()) if value(name)}
        if configured:
            parameters.append(configured)
        # 설정 적용 순서는 YAML → 명시한 장치/출력 인자 → 모드가 강제하는 값이다.
        # 마지막 값이 우선하므로 YAML이 단일 카메라 모드나 디버깅 장치 범위를 바꾸지 못한다.
        forced = {}
        if role == "camera":
            camera_role = mode.removesuffix("_camera") if capture_mode else "both"
            forced["camera_roles"] = "front" if camera_role == "foreign" else camera_role
        elif role == "mission":
            forced["audio_enabled"] = audio_enabled
            # 완료 뒤 스택을 스스로 종료할 때 보고서 업로드가 끝나기를 기다릴지(upload:=true)
            forced["wait_for_upload"] = upload
        elif role == "capture_debug":
            forced.update(capture_role=mode.removesuffix("_camera"), show_gui=display)
        elif role == "actuator_debug":
            forced["debug_capability"] = mode
        elif role == "display":
            forced.update(local_gui=display, gui_output=gui_output, profile_trace=profiling)
            rate = value("gui_publish_rate_hz")
            if rate:
                rate = float(rate)
                if not 10.0 <= rate <= 15.0:
                    raise ValueError("gui_publish_rate_hz must be between 10 and 15")
                forced["gui_publish_rate_hz"] = rate
        elif role == "inspection":
            forced["dashboard_live"] = dashboard_live
        elif role == "frame_profiler":
            output_directory = value("profiling_output_directory")
            if output_directory:
                forced["output_directory"] = output_directory
        if forced:
            parameters.append(forced)
        remappings = []
        if profiling and role in ("display", "frame_profiler"):
            # GUI producer와 profiler subscriber는 같은 선택 trace topic을 사용한다.
            remappings.append((FRAME_TRACE_TOPIC, frame_trace_topic))
        node = Node(
            package="rail_robot_nodes", executable=role + "_node",
            name=NODE_NAMES.get(role, "rail_" + role), namespace=value("namespace") or "/", parameters=parameters,
            remappings=remappings,
            output="screen", emulate_tty=True,
            sigterm_timeout="8", sigkill_timeout="8",
        )
        # 필수 노드 하나가 종료되면 UART 소유 노드를 포함한 이 launch 묶음도 종료한다.
        # GUI·업로드·텔레메트리는 선택 기능이므로 해당 프로세스 종료만으로 전체를 끄지는 않는다.
        # [4. 연쇄 종료 추적] 여러 노드가 동시에 종료되면 마지막 SIGKILL 로그보다 최초 종료 노드의
        # 예외와 Shutdown reason을 먼저 본다. 이 핸들러는 해당 launch 묶음에만 적용된다.
        # 선택 노드도 별도로 runtime/fault를 발행하면 공통 실행기의 장애 전파가 적용될 수 있다.
        if role not in ("display", "upload", "firebase_snapshot", "dashboard_telemetry", "audio_alert", "frame_profiler"):
            actions.append(RegisterEventHandler(OnProcessExit(
                target_action=node,
                on_exit=[EmitEvent(event=Shutdown(reason=f"{role} node exited"))],
            )))
        actions.append(node)
    return actions


def generate_launch_description():
    # [인자 목록 확인] 터미널에서 이 launch에 --show-args를 붙이면 선언된 기본값을 볼 수 있다.
    # 여기의 GUI 주기는 목표 발행률이며 카메라/추론 속도를 보장하지 않는다.
    # 아래 빈 문자열 기본값은 노드 기본값을 사용하기 위한 것으로, 빈 장치 경로를 강제하지 않는다.
    arguments = [
        DeclareLaunchArgument("mode", default_value="mission", choices=list(MODES)),
        DeclareLaunchArgument("camera", default_value="true", description="Start the local camera owner; false uses external camera topics/services"),
        DeclareLaunchArgument("display", default_value="true", description="Enable the optional GUI"),
        DeclareLaunchArgument("gui_output", default_value="true", description="Publish composed JPEGs in mission/dual/vision/camera/training modes even when display is false"),
        DeclareLaunchArgument("firebase_snapshot", default_value="false", description="Enable the optional five-second Firebase GUI fallback consumer"),
        DeclareLaunchArgument("dashboard_live", default_value="false", description="Send live ROS metrics and analyzed inspection captures to the authorized dashboard"),
        DeclareLaunchArgument("dashboard_control", default_value="false", description="Opt in to authenticated remote START and software E_STOP commands; mission RR2 only"),
        DeclareLaunchArgument("upload", default_value="false", description="Upload finalized mission reports"),
        DeclareLaunchArgument("audio_enabled", default_value="true", description="Mission WAV alerts; START waits for successful start WAV playback"),
        DeclareLaunchArgument("profiling", default_value="false", description="Record GUI frame provenance and system timing/resource metrics"),
        DeclareLaunchArgument("profiling_output_directory", default_value="", description="Optional JSONL/summary directory; otherwise use the profiler node default"),
        DeclareLaunchArgument("frame_trace_topic", default_value=FRAME_TRACE_TOPIC, description="FrameTrace topic shared by DisplayNode and the profiler"),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("params_file", default_value="", description="Optional ROS parameter YAML; node defaults use the checked deployment baseline"),
    ]
    arguments.append(DeclareLaunchArgument(
        "uart_protocol", default_value="v2",
        description="RR2 UART protocol for the selected STM owner; legacy is an explicit rollback"))
    for name in ("side_camera_device", "top_camera_device", "serial_port", "output_directory",
                 "training_camera_role", "training_output_directory", "capture_output_directory",
                 "gui_output_topic", "gui_status_topic", "gui_frame_id", "gui_publish_rate_hz",
                 "audio_directory", "audio_device"):
        arguments.append(DeclareLaunchArgument(name, default_value=""))
    return LaunchDescription(arguments + [OpaqueFunction(function=configured_nodes)])
