cmake_minimum_required(VERSION 3.8)
project(rail_robot_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CameraFrame.msg" "msg/DetectionSample.msg" "msg/ActuatorDecision.msg"
  "msg/MissionStatus.msg" "msg/StmEvents.msg"
  "msg/EncoderCommand.msg" "msg/EncoderState.msg"
  "msg/CaptureDebugResult.msg" "msg/ModelTiming.msg" "msg/FrameTrace.msg"
  "srv/CapturePair.srv" "srv/CaptureInspection.srv"
  "srv/DrainInspection.srv" "srv/UartCommand.srv"
  DEPENDENCIES std_msgs sensor_msgs builtin_interfaces
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
