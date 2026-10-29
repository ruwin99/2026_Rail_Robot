"""Only the optional sender; no camera, inference, control or UART processes."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=""),
        DeclareLaunchArgument("gui_output_topic", default_value="/rail_robot/gui/image/compressed"),
        Node(package="rail_robot_nodes", executable="webrtc_sender_node", output="screen",
             parameters=[{"config_file": LaunchConfiguration("config_file"),
                          "gui_output_topic": LaunchConfiguration("gui_output_topic")}]),
    ])
