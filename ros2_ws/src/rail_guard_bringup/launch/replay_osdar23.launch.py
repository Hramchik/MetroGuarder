"""Воспроизведение записи OSDaR23 + мониторинг габарита + RViz.

    ros2 launch rail_guard_bringup replay_osdar23.launch.py \
        sequence:=/путь/к/7_approach_underground_station_7.1
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("rail_guard_bringup")
    default_config = os.path.join(share, "config", "mainline.yaml")
    default_rviz = os.path.join(share, "rviz", "rail_guard.rviz")

    args = [
        DeclareLaunchArgument("sequence", description="каталог распакованной последовательности"),
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("rate", default_value="10.0"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("debug_clouds", default_value="true"),
        DeclareLaunchArgument("route_valid_range", default_value="0.0",
                              description="дальность достоверности внешней оси пути, м"),
        DeclareLaunchArgument("route_curvature", default_value="0.0",
                              description="кривизна маршрута 1/R, + влево"),
    ]

    player = Node(
        package="rail_guard", executable="dataset_player", name="dataset_player",
        output="screen",
        parameters=[{
            "sequence_path": LaunchConfiguration("sequence"),
            "rate": LaunchConfiguration("rate"),
            "loop": LaunchConfiguration("loop"),
        }],
    )
    detector = Node(
        package="rail_guard", executable="obstacle_detector", name="obstacle_detector",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {
                "config_file": LaunchConfiguration("config"),
                "publish_debug_clouds": LaunchConfiguration("debug_clouds"),
                "route_valid_range": LaunchConfiguration("route_valid_range"),
                "route_curvature": LaunchConfiguration("route_curvature"),
            },
        ],
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", default_rviz],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    return LaunchDescription(args + [player, detector, rviz])
