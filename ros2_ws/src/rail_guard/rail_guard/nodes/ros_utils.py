"""Мосты между numpy и сообщениями ROS 2.

Держим их в одном месте: конверсия облака — самая горячая точка ноды,
и любое лишнее копирование здесь сразу видно в бюджете 100 мс на кадр.
"""
from __future__ import annotations

import array
from typing import Iterable, Optional, Tuple

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray


def pointcloud2_to_xyzi(msg: PointCloud2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """PointCloud2 → (xyz float32 (N,3), intensity float32 (N,) или None).

    Читаем сырой буфер напрямую: sensor_msgs_py.point_cloud2.read_points
    на 150 тысячах точек уходит в десятки миллисекунд на копиях.
    """
    names = [f.name for f in msg.fields]
    dtype_map = {
        PointField.INT8: np.int8, PointField.UINT8: np.uint8,
        PointField.INT16: np.int16, PointField.UINT16: np.uint16,
        PointField.INT32: np.int32, PointField.UINT32: np.uint32,
        PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
    }
    fields = []
    offset = 0
    for field in sorted(msg.fields, key=lambda f: f.offset):
        if field.offset > offset:                       # выравнивающие дыры
            fields.append((f"__pad{offset}", np.uint8, (field.offset - offset,)))
        fields.append((field.name, dtype_map[field.datatype]))
        offset = field.offset + np.dtype(dtype_map[field.datatype]).itemsize
    if msg.point_step > offset:
        fields.append((f"__pad{offset}", np.uint8, (msg.point_step - offset,)))

    # msg.data — array.array, поддерживающий буферный протокол: frombuffer
    # читает его без копии, а bytes() скопировал бы мегабайты на каждом кадре.
    array = np.frombuffer(memoryview(msg.data), dtype=np.dtype(fields))
    xyz = np.stack([array["x"], array["y"], array["z"]], axis=1).astype(np.float32)
    intensity = None
    for key in ("intensity", "i", "reflectivity"):
        if key in names:
            intensity = array[key].astype(np.float32)
            break
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        xyz = xyz[finite]
        intensity = None if intensity is None else intensity[finite]
    return xyz, intensity


def xyzi_to_pointcloud2(xyz: np.ndarray, intensity: Optional[np.ndarray],
                        header: Header) -> PointCloud2:
    """numpy → PointCloud2 (x, y, z, intensity, float32)."""
    n = int(xyz.shape[0])
    data = np.zeros(n, dtype=[("x", np.float32), ("y", np.float32),
                              ("z", np.float32), ("intensity", np.float32)])
    if n:
        data["x"] = xyz[:, 0]
        data["y"] = xyz[:, 1]
        data["z"] = xyz[:, 2]
        if intensity is not None:
            data["intensity"] = intensity
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_dense = True
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * n
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    # Присваиваем именно array.array: rclpy для bytes/списка в отладочном
    # режиме проверяет каждый элемент по отдельности, и на трёх мегабайтах
    # это стоит ~280 мс — больше, чем весь остальной тракт обработки.
    msg.data = array.array("B", data.tobytes())
    return msg


def make_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def line_marker(header: Header, namespace: str, marker_id: int, points: Iterable[Point],
                color: ColorRGBA, width: float = 0.08,
                lifetime_sec: float = 0.5) -> Marker:
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = width
    marker.color = color
    marker.points = list(points)
    marker.pose.orientation.w = 1.0
    marker.lifetime = Duration(sec=int(lifetime_sec),
                               nanosec=int((lifetime_sec % 1.0) * 1e9))
    return marker


def box_marker(header: Header, namespace: str, marker_id: int, center: np.ndarray,
               size: np.ndarray, color: ColorRGBA, lifetime_sec: float = 0.5) -> Marker:
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
    marker.pose.orientation.w = 1.0
    marker.scale = Vector3(x=float(max(size[0], 0.1)), y=float(max(size[1], 0.1)),
                           z=float(max(size[2], 0.1)))
    marker.color = color
    marker.lifetime = Duration(sec=int(lifetime_sec),
                               nanosec=int((lifetime_sec % 1.0) * 1e9))
    return marker


def text_marker(header: Header, namespace: str, marker_id: int, position: np.ndarray,
                text: str, color: ColorRGBA, height: float = 0.8,
                lifetime_sec: float = 0.5) -> Marker:
    marker = Marker()
    marker.header = header
    marker.ns = namespace
    marker.id = marker_id
    marker.type = Marker.TEXT_VIEW_FACING
    marker.action = Marker.ADD
    marker.pose.position = Point(x=float(position[0]), y=float(position[1]),
                                 z=float(position[2]))
    marker.pose.orientation.w = 1.0
    marker.scale.z = height
    marker.color = color
    marker.text = text
    marker.lifetime = Duration(sec=int(lifetime_sec),
                               nanosec=int((lifetime_sec % 1.0) * 1e9))
    return marker


def clear_marker_array() -> MarkerArray:
    marker = Marker()
    marker.action = Marker.DELETEALL
    return MarkerArray(markers=[marker])
