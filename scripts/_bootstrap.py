"""Путь к пакету rail_guard для запуска скриптов без сборки ROS-воркспейса."""
import os
import sys

PKG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ros2_ws", "src", "rail_guard")
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "osdar23", "sequences")
