"""Общая подготовка окружения для офлайн-скриптов.

Делает две вещи: добавляет пакет `rail_guard` в `sys.path` (чтобы скрипты
работали без сборки ROS-воркспейса) и, если на хосте нет numpy/scipy,
перезапускает сам скрипт внутри distrobox-контейнера с ROS. На Arch-хосте
научного стека нет — он стоит только в контейнере, и без этого перезапуска
каждый скрипт падал бы на `import numpy`.
"""
import os
import shlex
import shutil
import subprocess
import sys

PKG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ros2_ws", "src", "rail_guard")
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "osdar23", "sequences")

CONTAINER = os.environ.get("RAIL_GUARD_CONTAINER", "ros-gazebo")


def _reexec_in_container() -> None:
    """Перезапускает текущий скрипт в контейнере, где есть numpy и scipy."""
    if os.environ.get("RAIL_GUARD_REEXEC"):
        raise SystemExit(
            f"numpy/scipy недоступны и внутри контейнера «{CONTAINER}». "
            "Установите их или укажите другой контейнер через RAIL_GUARD_CONTAINER.")
    if shutil.which("distrobox") is None:
        raise SystemExit(
            "Для работы нужны numpy и scipy. На этом хосте их нет, а distrobox "
            "не найден — запустите скрипт в окружении с ROS 2 и научным стеком.")

    script = os.path.abspath(sys.argv[0])
    command = " ".join(shlex.quote(part) for part in
                       ["env", "RAIL_GUARD_REEXEC=1", "python3", script, *sys.argv[1:]])
    # Внутри контейнера ROS 2 надо ещё подключить: `bash -lc` читает профиль
    # пользователя, а setup.bash в нём обычно не прописан — и скрипт падает на
    # `import rosbag2_py`, хотя всё установлено. Дистрибутив ищется на месте:
    # humble, jazzy или какой там окажется. Собранный воркспейс подключается
    # следом, если он есть, — от него нужны сообщения rail_guard_msgs.
    workspace = os.path.join(os.path.dirname(PKG_ROOT), "..", "install", "setup.bash")
    inner = (
        'for s in /opt/ros/*/setup.bash; do [ -f "$s" ] && . "$s" && break; done; '
        f'[ -f {shlex.quote(os.path.normpath(workspace))} ] && '
        f'. {shlex.quote(os.path.normpath(workspace))}; '
        + command)
    print(f"[rail-guard] numpy на хосте нет — перезапускаю в distrobox «{CONTAINER}»",
          file=sys.stderr)
    raise SystemExit(subprocess.call(["distrobox", "enter", CONTAINER, "--",
                                      "bash", "-lc", inner]))


try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
except ImportError:
    _reexec_in_container()
