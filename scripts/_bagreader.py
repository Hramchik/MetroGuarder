"""Чтение облаков точек из записи ROS 2 без поднятия ROS-графа.

Вынесено отдельно, потому что нужно двум скриптам — офлайн-прогону и сборке
видео, — и потому что это единственное место в проекте, зависящее от
`rosbag2_py`. Сам алгоритм (`rail_guard.lib`) от ROS не зависит вовсе.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional, Tuple

import numpy as np

DTYPE_MAP = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
             5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def cloud_dtype(msg) -> np.dtype:
    """Структурный dtype под поля PointCloud2, включая выравнивающие дыры.

    Читаем сырой буфер напрямую: `sensor_msgs_py.point_cloud2.read_points`
    на трёхсоттысячном облаке уходит в десятки миллисекунд на копиях.
    """
    fields, offset = [], 0
    for field in sorted(msg.fields, key=lambda f: f.offset):
        if field.offset > offset:
            fields.append((f"__pad{offset}", np.uint8, (field.offset - offset,)))
        fields.append((field.name, DTYPE_MAP[field.datatype]))
        offset = field.offset + np.dtype(DTYPE_MAP[field.datatype]).itemsize
    if msg.point_step > offset:
        fields.append((f"__pad{offset}", np.uint8, (msg.point_step - offset,)))
    return np.dtype(fields)


def storage_id(bag: str) -> str:
    """Тип хранилища записи: sqlite3 или mcap.

    Спрашивается у самой записи, а не задаётся жёстко: `ros2 bag record` в
    Humble пишет sqlite3, но с Iron по умолчанию mcap, и чужая запись вполне
    может оказаться любой из двух.
    """
    meta = os.path.join(bag, "metadata.yaml")
    if os.path.isfile(meta):
        try:
            import yaml
            with open(meta, "r", encoding="utf-8") as handle:
                info = yaml.safe_load(handle) or {}
            found = info.get("rosbag2_bagfile_information", {}).get("storage_identifier")
            if found:
                return str(found)
        except Exception:               # битая метаинформация — не повод падать
            pass
    return "mcap" if bag.endswith(".mcap") else "sqlite3"


def read_frames(bag: str, topic: str = "",
                rotate_to_rep103: bool = False) -> Iterator[Tuple[np.ndarray,
                                                                  Optional[np.ndarray], float]]:
    """Кадры записи: (xyz, интенсивность, метка времени).

    Облако отдаётся в СК датчика — приводить его к рабочей СК должен сам
    пайплайн (`lib/frames.py`), иначе прогон проверял бы не то, что поедет на
    поезде. `rotate_to_rep103` нужен только служебным рисовалкам, которым
    требуются уже развёрнутые точки: поворот +90° вокруг Z, как в записях
    метро.

    Пустой `topic` — взять первое облако, найденное в записи.
    """
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
    except ImportError as exc:                                   # pragma: no cover
        raise SystemExit(
            f"Нужно окружение с ROS 2 (rosbag2_py): {exc}.\n"
            "Запустите скрипт после `source /opt/ros/humble/setup.bash` или в контейнере.")

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage_id(bag)),
                rosbag2_py.ConverterOptions("", ""))
    chosen = topic
    if not chosen:
        clouds = [t.name for t in reader.get_all_topics_and_types()
                  if t.type == "sensor_msgs/msg/PointCloud2"]
        if not clouds:
            raise SystemExit(f"В записи {bag} нет топиков sensor_msgs/msg/PointCloud2")
        chosen = clouds[0]

    while reader.has_next():
        name, data, _stamp = reader.read_next()
        if name != chosen:
            continue
        msg = deserialize_message(data, PointCloud2)
        raw = np.frombuffer(memoryview(msg.data), dtype=cloud_dtype(msg))
        if rotate_to_rep103:
            xyz = np.stack([-raw["y"], raw["x"], raw["z"]], axis=1).astype(np.float32)
        else:
            xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
        # Имя поля интенсивности у разных драйверов разное — берём те же
        # варианты, что и нода (nodes/ros_utils.py), иначе офлайн-прогон и
        # рабочая система считали бы по-разному.
        inten = None
        for name_variant in ("intensity", "i", "reflectivity"):
            if name_variant in raw.dtype.names:
                inten = raw[name_variant].astype(np.float32)
                break
        finite = np.isfinite(xyz).all(axis=1)
        yield (xyz[finite], None if inten is None else inten[finite],
               msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
