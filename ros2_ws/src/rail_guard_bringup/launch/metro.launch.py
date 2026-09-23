"""Полная демонстрация на записи метро: детектор, вывод результата, при
желании — проигрывание бэга и RViz.

Это точка входа по умолчанию в контейнере: она собирает всю цепочку из ТЗ —
bag → облако → алгоритм → обнаружение → дальность до него — за один вызов.

    ros2 launch rail_guard_bringup metro.launch.py bag:=/bags/doubleT_platform

Имя топика облака указывать не обязательно: детектор подписывается на
`lidar/points`, а если там ничего нет — находит облако сам. Явно задать можно
параметром `points`.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("rail_guard_bringup")
    default_config = os.path.join(share, "config", "metro.yaml")
    default_rviz = os.path.join(share, "rviz", "rail_guard.rviz")

    args = [
        DeclareLaunchArgument("config", default_value=default_config,
                              description="профиль параметров (YAML)"),
        DeclareLaunchArgument("points", default_value="",
                              description="топик облака; пусто — определить самостоятельно"),
        DeclareLaunchArgument("bag", default_value="",
                              description="каталог rosbag2; пусто — ждать облако извне"),
        DeclareLaunchArgument("rate", default_value="1.0",
                              description="темп проигрывания записи"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("offset", default_value="0.0",
                              description="с какой секунды записи начинать"),
        DeclareLaunchArgument("monitor", default_value="true",
                              description="построчный вывод результата в терминал"),
        DeclareLaunchArgument("rviz", default_value="false",
                              description="запустить RViz (нужен проброшенный дисплей)"),
        DeclareLaunchArgument("debug_clouds", default_value="false",
                              description="публиковать отладочные облака"),
        DeclareLaunchArgument("route_valid_range", default_value="0.0",
                              description="дальность достоверности оси из путевой карты, м"),
        DeclareLaunchArgument("route_curvature", default_value="0.0",
                              description="кривизна участка 1/R, знак + влево"),
    ]

    detector = Node(
        package="rail_guard", executable="obstacle_detector", name="obstacle_detector",
        output="screen",
        parameters=[LaunchConfiguration("config"),
                    {"config_file": LaunchConfiguration("config"),
                     "points_topic": LaunchConfiguration("points"),
                     "publish_debug_clouds": LaunchConfiguration("debug_clouds"),
                     "route_valid_range": LaunchConfiguration("route_valid_range"),
                     "route_curvature": LaunchConfiguration("route_curvature")}],
    )
    monitor = Node(
        package="rail_guard", executable="result_monitor", name="rail_guard_monitor",
        output="screen", emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("monitor")),
    )
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", default_rviz], output="log",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    # Запись проигрывается только если её указали. Пауза перед стартом нужна,
    # чтобы детектор успел поднять подписки: облако идёт best-effort, и
    # потерянные первые кадры уже не вернуть. Пауза после — чтобы тракт успел
    # догнать последние кадры до остановки.
    #
    # По концу записи вся сборка гасится: запуск в контейнере должен сам
    # возвращать управление, а монитор — напечатать сводку по проезду.
    play = ExecuteProcess(
        cmd=["bash", "-lc",
             ["sleep 4; ros2 bag play '", LaunchConfiguration("bag"),
              "' --read-ahead-queue-size 8 --rate ", LaunchConfiguration("rate"),
              " --start-offset ", LaunchConfiguration("offset"),
              " $([ '", LaunchConfiguration("loop"), "' = 'true' ] && echo --loop); sleep 3"]],
        output="screen",
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("bag"), "' != ''"])),
        on_exit=[EmitEvent(event=Shutdown(reason="запись закончилась"))],
    )
    return LaunchDescription(args + [detector, monitor, rviz, play])
