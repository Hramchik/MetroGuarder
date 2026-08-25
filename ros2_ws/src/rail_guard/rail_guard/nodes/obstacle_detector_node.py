#!/usr/bin/env python3
"""Нода мониторинга габарита: облако точек → препятствия и команда поезду.

Весь тракт (фильтрация, полотно, ось пути, коридор габарита, кластеризация,
сопровождение, решение) выполняется в одной ноде осознанно: разбиение по
нодам добавило бы две сериализации облака на 150 тысяч точек в каждом кадре,
а это половина бюджета реального времени. Промежуточные результаты при этом
доступны снаружи — как отладочные облака и маркеры.
"""
from __future__ import annotations

import os
from dataclasses import fields as dataclass_fields
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import MarkerArray

from rail_guard_msgs.msg import GaugeStatus, Obstacle, ObstacleArray, TrackCorridor
from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.decision import (ACTION_ATTENTION, ACTION_CLEAR, ACTION_EMERGENCY_BRAKE,
                                     ACTION_SERVICE_BRAKE)
from rail_guard.lib.pipeline import FrameResult, Pipeline
from rail_guard.nodes.ros_utils import (box_marker, clear_marker_array, line_marker,
                                        make_color, pointcloud2_to_xyzi, text_marker,
                                        xyzi_to_pointcloud2)

ACTION_COLORS = {
    ACTION_CLEAR: (0.2, 0.9, 0.2),
    ACTION_ATTENTION: (1.0, 0.85, 0.1),
    ACTION_SERVICE_BRAKE: (1.0, 0.5, 0.0),
    ACTION_EMERGENCY_BRAKE: (1.0, 0.1, 0.1),
}


class ObstacleDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_detector")
        self.declare_parameter("config_file", "")
        self.declare_parameter("publish_debug_clouds", True)
        self.declare_parameter("speed_timeout", 2.0)
        self.declare_parameter("route_curvature", 0.0)     # 1/R, знак: + влево
        self.declare_parameter("route_heading", 0.0)
        self.declare_parameter("route_valid_range", 0.0)   # 0 — внешней оси нет

        config_path = self.get_parameter("config_file").value
        cfg = load_yaml(config_path) if config_path and os.path.isfile(config_path) \
            else PipelineConfig()
        self._declare_pipeline_params(cfg)
        self.cfg = self._read_pipeline_params(cfg)
        self.pipeline = Pipeline(self.cfg)

        valid_range = float(self.get_parameter("route_valid_range").value)
        if valid_range > 0.0:
            curvature = float(self.get_parameter("route_curvature").value)
            self.pipeline.set_route_prior(
                lin=float(self.get_parameter("route_heading").value),
                quad=0.5 * curvature, valid_range=valid_range)
            self.get_logger().info(
                f"Ось пути задана извне: R={1.0 / curvature:.0f} м, "
                f"достоверна до {valid_range:.0f} м" if abs(curvature) > 1e-9
                else f"Ось пути задана извне: прямая, достоверна до {valid_range:.0f} м")

        self.publish_debug = bool(self.get_parameter("publish_debug_clouds").value)
        self.cfg.debug = self.publish_debug
        self.speed: Optional[float] = None
        self.speed_stamp = 0.0
        self._frames = 0
        self._slow_frames = 0

        sensor_qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(PointCloud2, "lidar/points", self.on_cloud, sensor_qos)
        self.create_subscription(Float32, "train/speed", self.on_speed, 10)

        self.obstacle_pub = self.create_publisher(ObstacleArray, "rail_guard/obstacles", 10)
        self.status_pub = self.create_publisher(GaugeStatus, "rail_guard/gauge_status", 10)
        self.corridor_pub = self.create_publisher(TrackCorridor, "rail_guard/track_corridor", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "rail_guard/markers", 10)
        self.filtered_pub = self.create_publisher(PointCloud2, "rail_guard/points_filtered", sensor_qos)
        self.candidate_pub = self.create_publisher(PointCloud2, "rail_guard/points_in_gauge", sensor_qos)
        self.get_logger().info("Мониторинг габарита запущен")

    # ------------------------------------------------------------------ параметры
    def _declare_pipeline_params(self, cfg: PipelineConfig) -> None:
        """Объявляет параметры вида gauge.half_width для всех числовых полей."""
        for section in dataclass_fields(cfg):
            value = getattr(cfg, section.name)
            if not hasattr(value, "__dataclass_fields__"):
                continue
            for field in dataclass_fields(value):
                default = getattr(value, field.name)
                if isinstance(default, (int, float, bool, str)):
                    self.declare_parameter(f"{section.name}.{field.name}", default)

    def _read_pipeline_params(self, cfg: PipelineConfig) -> PipelineConfig:
        for section in dataclass_fields(cfg):
            value = getattr(cfg, section.name)
            if not hasattr(value, "__dataclass_fields__"):
                continue
            for field in dataclass_fields(value):
                default = getattr(value, field.name)
                if not isinstance(default, (int, float, bool, str)):
                    continue
                param = self.get_parameter(f"{section.name}.{field.name}").value
                if param is not None:
                    setattr(value, field.name, type(default)(param))
        return cfg

    # ------------------------------------------------------------------ приём данных
    def on_speed(self, msg: Float32) -> None:
        self.speed = float(msg.data)
        self.speed_stamp = self.get_clock().now().nanoseconds * 1e-9

    def _current_speed(self) -> Optional[float]:
        if self.speed is None:
            return None
        age = self.get_clock().now().nanoseconds * 1e-9 - self.speed_stamp
        if age > float(self.get_parameter("speed_timeout").value):
            return None                     # протухшая скорость опаснее отсутствующей
        return self.speed

    def on_cloud(self, msg: PointCloud2) -> None:
        xyz, intensity = pointcloud2_to_xyzi(msg)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        result = self.pipeline.process(xyz, intensity, timestamp=stamp,
                                       ego_speed=self._current_speed())

        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        self.obstacle_pub.publish(self._obstacle_array(header, result))
        self.status_pub.publish(self._gauge_status(header, result))
        self.corridor_pub.publish(self._corridor_msg(header, result))
        self.marker_pub.publish(self._markers(header, result))
        if self.publish_debug:
            if result.filtered_xyz is not None:
                self.filtered_pub.publish(xyzi_to_pointcloud2(result.filtered_xyz, None, header))
            if result.candidate_xyz is not None:
                self.candidate_pub.publish(xyzi_to_pointcloud2(result.candidate_xyz, None, header))

        self._frames += 1
        period_ms = 1000.0 / 10.0
        if result.timings["total_ms"] > period_ms:
            self._slow_frames += 1
            if self._slow_frames % 10 == 1:
                self.get_logger().warn(
                    f"Кадр обработан за {result.timings['total_ms']:.0f} мс — "
                    f"дольше периода лидара ({self._slow_frames} из {self._frames})")
        if not result.decision.clear and result.decision.action >= ACTION_SERVICE_BRAKE:
            self.get_logger().warn(f"{result.decision.action_name}: {result.decision.reason}")

    # ------------------------------------------------------------------ публикация
    def _obstacle_array(self, header: Header, result: FrameResult) -> ObstacleArray:
        msg = ObstacleArray()
        msg.header = header
        msg.processing_time_ms = float(result.timings["total_ms"])
        msg.input_point_count = int(result.input_points)
        msg.filtered_point_count = int(result.filtered_points)
        msg.max_valid_range = float(result.max_valid_range)
        for track in result.obstacles:
            det = track.detection
            item = Obstacle()
            item.id = int(track.id)
            item.classification = int(det.classification)
            item.confidence = float(track.confidence)
            item.centroid = Point(x=float(det.centroid[0]), y=float(det.centroid[1]),
                                  z=float(det.centroid[2]))
            item.size = Vector3(x=float(det.size[0]), y=float(det.size[1]), z=float(det.size[2]))
            item.min_point = Point(x=float(det.min_point[0]), y=float(det.min_point[1]),
                                   z=float(det.min_point[2]))
            item.max_point = Point(x=float(det.max_point[0]), y=float(det.max_point[1]),
                                   z=float(det.max_point[2]))
            item.distance = float(det.distance)
            item.lateral_offset = float(det.lateral_offset)
            item.height_above_rail = float(det.height_above_rail)
            item.clearance_margin = float(det.clearance_margin)
            item.point_count = int(det.point_count)
            item.age = int(track.age)
            item.hits = int(track.hits)
            item.closing_speed = float(track.closing_speed())
            item.time_to_contact = float(track.time_to_contact())
            item.in_gauge = bool(det.in_gauge)
            msg.obstacles.append(item)
        return msg

    def _gauge_status(self, header: Header, result: FrameResult) -> GaugeStatus:
        decision = result.decision
        msg = GaugeStatus()
        msg.header = header
        msg.clear = bool(decision.clear)
        msg.action = int(decision.action)
        msg.nearest_distance = float(decision.nearest_distance)
        msg.nearest_obstacle_id = int(decision.nearest_id)
        msg.obstacle_count = int(decision.obstacle_count)
        msg.braking_distance = float(decision.braking_distance)
        msg.detection_range = float(decision.detection_range)
        msg.reason = decision.reason
        return msg

    def _corridor_msg(self, header: Header, result: FrameResult) -> TrackCorridor:
        corridor = result.corridor
        msg = TrackCorridor()
        msg.header = header
        step = max(1, int(round(1.0 / max(self.cfg.gauge.step, 0.01))))
        for i in range(0, corridor.s.size, step):
            z = float(result.ground.ground_z(np.array([[corridor.s[i], corridor.y[i], 0.0]]))[0])
            msg.centerline.append(Point(x=float(corridor.s[i]), y=float(corridor.y[i]), z=z))
            msg.half_width.append(float(corridor.half_width[i]))
            msg.height.append(float(corridor.top))
        radius = result.track.curvature_radius
        msg.curvature_radius = float(radius if np.isfinite(radius) else 1e9)
        msg.estimated_from_rails = bool(result.track.from_rails)
        msg.valid_range = float(result.track.fit_range)
        return msg

    def _markers(self, header: Header, result: FrameResult) -> MarkerArray:
        markers = clear_marker_array()
        corridor = result.corridor
        ground_z = result.ground.ground_z(
            np.stack([corridor.s, corridor.y, np.zeros_like(corridor.s)], axis=1))
        confirmed = corridor.s <= max(result.track.fit_range, 1.0)
        for idx, (sign, name) in enumerate(((-1.0, "right"), (1.0, "left"))):
            points = [Point(x=float(s), y=float(y + sign * hw), z=float(z + corridor.bottom))
                      for s, y, hw, z in zip(corridor.s, corridor.y, corridor.half_width, ground_z)]
            markers.markers.append(line_marker(header, "gauge", idx, points,
                                               make_color(1.0, 0.2, 0.2, 0.9), 0.10))
        axis = [Point(x=float(s), y=float(y), z=float(z))
                for s, y, z in zip(corridor.s[confirmed], corridor.y[confirmed],
                                   ground_z[confirmed])]
        if len(axis) > 1:
            markers.markers.append(line_marker(header, "track_axis", 0, axis,
                                               make_color(0.2, 0.6, 1.0, 0.9), 0.12))
        for i, track in enumerate(result.obstacles):
            det = track.detection
            colour = make_color(*ACTION_COLORS[ACTION_EMERGENCY_BRAKE], 0.55) if det.in_gauge \
                else make_color(0.6, 0.6, 0.6, 0.35)
            center = 0.5 * (det.min_point + det.max_point)
            markers.markers.append(box_marker(header, "obstacles", i, center, det.size, colour))
            markers.markers.append(text_marker(
                header, "obstacle_labels", i,
                np.array([center[0], center[1], float(det.max_point[2]) + 0.6]),
                f"#{track.id} {det.class_name} {det.distance:.0f} м p={track.confidence:.2f}",
                make_color(1.0, 1.0, 1.0, 0.9)))
        colour = make_color(*ACTION_COLORS.get(result.decision.action, (1.0, 1.0, 1.0)), 1.0)
        markers.markers.append(text_marker(header, "status", 0, np.array([8.0, 0.0, 5.0]),
                                           f"{result.decision.action_name}: {result.decision.reason}",
                                           colour, height=1.0))
        return markers


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObstacleDetectorNode()
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
