"""Только мониторинг габарита: облако приходит от реального лидара.

    ros2 launch rail_guard_bringup detector.launch.py \
        config:=/путь/metro.yaml points:=/os_cloud_node/points
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("rail_guard_bringup")
    default_config = os.path.join(share, "config", "metro.yaml")
    args = [
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("points", default_value="/lidar/points"),
        DeclareLaunchArgument("speed", default_value="/train/speed"),
        DeclareLaunchArgument("debug_clouds", default_value="false"),
    ]
    detector = Node(
        package="rail_guard", executable="obstacle_detector", name="obstacle_detector",
        output="screen",
        parameters=[LaunchConfiguration("config"),
                    {"config_file": LaunchConfiguration("config"),
                     "publish_debug_clouds": LaunchConfiguration("debug_clouds")}],
        remappings=[("lidar/points", LaunchConfiguration("points")),
                    ("train/speed", LaunchConfiguration("speed"))],
    )
    return LaunchDescription(args + [detector])
