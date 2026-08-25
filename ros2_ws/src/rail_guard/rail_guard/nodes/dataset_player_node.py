#!/usr/bin/env python3
"""Нода воспроизведения записей лидара (OSDaR23 или каталог .pcd).

Отдаёт облака в темпе съёмки, а вместе с ними — скорость поезда из
бортовой навигации и эталонную разметку. Это позволяет гонять весь тракт
на реальных данных так же, как он поедет на поезде: та же топика, та же
частота, та же система координат.
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Header
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import MarkerArray

from rail_guard.lib.dataset import Osdar23Sequence
from rail_guard.nodes.ros_utils import (box_marker, clear_marker_array, make_color,
                                        xyzi_to_pointcloud2)


class DatasetPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("dataset_player")
        self.declare_parameter("sequence_path", "")
        self.declare_parameter("frame_id", "lidar")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("publish_ground_truth", True)
        self.declare_parameter("sensors", [])          # sensor_index; пусто — все лидары
        self.declare_parameter("preload", True)        # держать кадры в памяти

        path = self.get_parameter("sequence_path").value
        if not path or not os.path.isdir(path):
            raise RuntimeError(f"Параметр sequence_path не указывает на каталог: {path!r}")
        self.frame_id = self.get_parameter("frame_id").value
        self.loop = bool(self.get_parameter("loop").value)
        self.publish_gt = bool(self.get_parameter("publish_ground_truth").value)
        sensors = list(self.get_parameter("sensors").value or [])
        self.sensors = tuple(int(s) for s in sensors) if sensors else None

        self.sequence = Osdar23Sequence(path)
        if len(self.sequence) == 0:
            raise RuntimeError(f"В последовательности нет кадров: {path}")
        self.index = int(self.get_parameter("start_frame").value)
        self._cache: dict = {}
        if bool(self.get_parameter("preload").value):
            import time as _time
            started = _time.perf_counter()
            for i in range(len(self.sequence)):
                self._cache[i] = self.sequence.load(i, sensors=self.sensors)
            self.get_logger().info(
                f"Кадры загружены в память за {_time.perf_counter() - started:.1f} с "
                f"({len(self._cache)} шт.) — чтение с диска не мешает темпу выдачи")

        qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST)
        self.cloud_pub = self.create_publisher(PointCloud2, "lidar/points", qos)
        self.speed_pub = self.create_publisher(Float32, "train/speed", 10)
        self.gt_pub = self.create_publisher(
            MarkerArray, "rail_guard/ground_truth",
            QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                       history=QoSHistoryPolicy.KEEP_LAST))

        # Система координат датасета совпадает с REP-103 (X вперёд, Y влево,
        # Z вверх), начало — на уровне головок рельсов, поэтому base_link
        # и lidar связаны единичным преобразованием.
        self.static_tf = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.get_parameter("base_frame").value
        transform.child_frame_id = self.frame_id
        transform.transform.rotation.w = 1.0
        self.static_tf.sendTransform(transform)

        rate = float(self.get_parameter("rate").value)
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self.publish_next)
        self.get_logger().info(
            f"Последовательность {self.sequence.name}: {len(self.sequence)} кадров, "
            f"частота {rate:.1f} Гц, топик {self.cloud_pub.topic_name}")

    def publish_next(self) -> None:
        if self.index >= len(self.sequence):
            if not self.loop:
                self.get_logger().info("Последовательность закончилась")
                self.timer.cancel()
                return
            self.index = 0

        frame = self.sequence.frames[self.index]
        xyz, intensity, _sensor = self._cache.get(self.index) \
            or self.sequence.load(self.index, sensors=self.sensors)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        self.cloud_pub.publish(xyzi_to_pointcloud2(xyz, intensity, header))

        if not np.isnan(frame.speed):
            self.speed_pub.publish(Float32(data=float(frame.speed)))
        if self.publish_gt and frame.gt_objects:
            self.gt_pub.publish(self._ground_truth_markers(frame, header))

        self.index += 1

    def _ground_truth_markers(self, frame, header) -> MarkerArray:
        markers = clear_marker_array()
        colour = make_color(1.0, 0.6, 0.0, 0.25)
        for i, obj in enumerate(frame.gt_objects):
            markers.markers.append(box_marker(header, "ground_truth", i, obj.center,
                                              obj.size, colour, lifetime_sec=0.3))
        return markers


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
