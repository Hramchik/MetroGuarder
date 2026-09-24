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
import time
from collections import deque
from dataclasses import fields as dataclass_fields
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import MarkerArray

from rail_guard_msgs.msg import GaugeStatus, Obstacle, ObstacleArray, TrackCorridor
from rail_guard.lib.config import PipelineConfig, load_yaml, parse_param
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
        # Топик облака. Пусто — подписаться на «lidar/points» (его принято
        # ремапить), а если там никто не публикует, найти облако самому:
        # на контрольной записи имя топика заранее неизвестно.
        self.declare_parameter("points_topic", "")
        self.declare_parameter("autodiscover_points", True)
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
        self._degenerate_frames = 0
        # Период кадров лидара измеряется по штампам, а не берётся равным
        # 100 мс: на контрольной записи частота может быть другой, а «успеваем
        # или нет» имеет смысл только относительно настоящего периода.
        self._last_stamp: Optional[float] = None
        self._periods: deque = deque(maxlen=20)

        # Облако лидара — 8 МБ у 64-луча и 24 МБ у 128-луча. DDS режет такое
        # сообщение на фрагменты по 64 КБ, и при BEST_EFFORT потеря одного
        # фрагмента отбрасывает весь кадр: до детектора доходило 43-83 %
        # кадров. RELIABLE заставляет издателя дослать потерянные фрагменты,
        # и кадр либо приходит целым, либо не приходит вовсе — пропуск тогда
        # означает честную нехватку процессорного времени, а не порчу в
        # транспорте. Глубина 2: очередь не нужна, нужен свежий кадр.
        # Издатели облаков (rosbag2 и драйверы лидаров) предлагают RELIABLE,
        # так что подписка совместима; для BEST_EFFORT-издателя остаётся
        # запасная подписка (см. _fallback_qos).
        self.sensor_qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.RELIABLE,
                                     history=QoSHistoryPolicy.KEEP_LAST)
        # Публикуем отладочные облака best-effort: их читает RViz, и там
        # свежесть важнее полноты.
        self.debug_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                    history=QoSHistoryPolicy.KEEP_LAST)
        configured = str(self.get_parameter("points_topic").value or "").strip()
        self.points_topic = configured or "lidar/points"
        self._cloud_sub = self.create_subscription(PointCloud2, self.points_topic,
                                                   self.on_cloud, self.sensor_qos)
        self._clouds_seen = 0
        self._discovery_timer = None
        self._fallback_sub = None
        self._fallback_timer = self.create_timer(3.0, self._ensure_compatible_qos)
        if not configured and bool(self.get_parameter("autodiscover_points").value):
            self._discovery_timer = self.create_timer(2.0, self._discover_points_topic)
        self.create_subscription(Float32, "train/speed", self.on_speed, 10)

        self.obstacle_pub = self.create_publisher(ObstacleArray, "rail_guard/obstacles", 10)
        self.status_pub = self.create_publisher(GaugeStatus, "rail_guard/gauge_status", 10)
        self.corridor_pub = self.create_publisher(TrackCorridor, "rail_guard/track_corridor", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "rail_guard/markers", 10)
        self.filtered_pub = self.create_publisher(PointCloud2, "rail_guard/points_filtered",
                                                  self.debug_qos)
        self.candidate_pub = self.create_publisher(PointCloud2, "rail_guard/points_in_gauge",
                                                   self.debug_qos)
        self.speed_pub = self.create_publisher(Float32, "rail_guard/ego_speed", 10)
        self.get_logger().info(
            f"Мониторинг габарита запущен, облако ожидается в «{self.points_topic}»")

    # ------------------------------------------------------------------ поиск облака
    def _discover_points_topic(self) -> None:
        """Подписка на облако, если в ожидаемом топике его нет.

        На контрольной записи имя топика заранее неизвестно: в одних бэгах это
        «/lidar_points», в других «/sensing/lidar/hesai128/pointcloud». Ждать
        ремапа от оператора значит терять проезд, поэтому нода находит
        единственное облако сама и говорит в лог, на что подписалась.
        """
        if self._clouds_seen:
            self._stop_discovery()
            return
        current = self.resolve_topic_name(self.points_topic)
        if self.count_publishers(current):
            return                          # издатель есть, просто ещё молчит
        candidates = [
            name for name, types in self.get_topic_names_and_types()
            if "sensor_msgs/msg/PointCloud2" in types
            and not name.startswith("/rail_guard/") and name != current
        ]
        if not candidates:
            return
        # Осмысленное имя предпочитаем случайному, а короткое — длинному.
        candidates.sort(key=lambda n: (0 if ("lidar" in n or "point" in n) else 1, len(n)))
        chosen = candidates[0]
        self.get_logger().warn(
            f"В «{current}» облака нет; подписываюсь на найденный топик «{chosen}»")
        self.destroy_subscription(self._cloud_sub)
        if self._fallback_sub is not None:
            self.destroy_subscription(self._fallback_sub)
            self._fallback_sub = None
        self.points_topic = chosen
        self._cloud_sub = self.create_subscription(PointCloud2, chosen, self.on_cloud,
                                                   self.sensor_qos)

    def _ensure_compatible_qos(self) -> None:
        """Добавляет BEST_EFFORT-подписку, если издатель облака best-effort.

        RELIABLE-читатель к BEST_EFFORT-издателю не подключается вовсе: QoS
        несовместимы, и система молча не получает ни одного кадра. Драйверы
        некоторых лидаров публикуют именно так, поэтому через три секунды
        после старта проверяем, есть ли издатель, от которого ничего не
        пришло, и подписываемся вторым, совместимым читателем.
        """
        if self._clouds_seen:
            self._stop_fallback_timer()
            return
        topic = self.resolve_topic_name(self.points_topic)
        best_effort = [info for info in self.get_publishers_info_by_topic(topic)
                       if info.qos_profile.reliability == QoSReliabilityPolicy.BEST_EFFORT]
        if not best_effort or self._fallback_sub is not None:
            return
        self.get_logger().warn(
            f"Издатель «{topic}» публикует BEST_EFFORT — добавляю совместимую подписку")
        self._fallback_sub = self.create_subscription(
            PointCloud2, self.points_topic, self.on_cloud,
            QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       history=QoSHistoryPolicy.KEEP_LAST))

    def _stop_fallback_timer(self) -> None:
        if self._fallback_timer is not None:
            self._fallback_timer.cancel()
            self.destroy_timer(self._fallback_timer)
            self._fallback_timer = None

    def _stop_discovery(self) -> None:
        if self._discovery_timer is not None:
            self._discovery_timer.cancel()
            self.destroy_timer(self._discovery_timer)
            self._discovery_timer = None

    # ------------------------------------------------------------------ параметры
    def _declare_pipeline_params(self, cfg: PipelineConfig) -> None:
        """Объявляет параметры вида gauge.half_width для всех полей профиля.

        Поля, которые система определяет сама (их умолчание — `None`),
        объявляются строкой «auto» с динамической типизацией: так их видно
        в `ros2 param list`, и любое из них можно подавить, задав число —
        и числом, и строкой, потому что тип у такого поля заранее неизвестен.
        """
        auto = ParameterDescriptor(
            dynamic_typing=True,
            description="auto — определяется по данным; число подавляет автоматику")
        for section in dataclass_fields(cfg):
            value = getattr(cfg, section.name)
            if not hasattr(value, "__dataclass_fields__"):
                continue
            for field in dataclass_fields(value):
                default = getattr(value, field.name)
                name = f"{section.name}.{field.name}"
                if default is None:
                    self.declare_parameter(name, "auto", auto)
                elif isinstance(default, (int, float, bool, str)):
                    self.declare_parameter(name, default)

    def _read_pipeline_params(self, cfg: PipelineConfig) -> PipelineConfig:
        for section in dataclass_fields(cfg):
            value = getattr(cfg, section.name)
            if not hasattr(value, "__dataclass_fields__"):
                continue
            for field in dataclass_fields(value):
                default = getattr(value, field.name)
                name = f"{section.name}.{field.name}"
                if default is None:
                    setattr(value, field.name,
                            parse_param(self.get_parameter(name).value))
                    continue
                if not isinstance(default, (int, float, bool, str)):
                    continue
                param = self.get_parameter(name).value
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
        # Время кадра меряется от входа в колбэк: разбор сообщения на
        # 128-луче стоит столько же, сколько четверть тракта обработки, и
        # «задержка обработки» без него — цифра, которая не сходится с тем,
        # сколько кадров система реально успевает взять.
        callback_started = time.perf_counter()
        # Интенсивность читается, только если по ней что-то фильтруется:
        # отдельная выборка столбца из буфера с шагом point_step стоит
        # миллисекунды, а по умолчанию отсечка по интенсивности выключена.
        # Потолок плотности берётся из рабочего бюджета тракта, а не из
        # профиля: он вырабатывается по факту времени кадра и меняется на
        # ходу, а в профиле стоит «auto». Прореживать уже при разборе
        # сообщения важно — на 128-луче это экономит четверть периода.
        xyz, intensity = pointcloud2_to_xyzi(
            msg, self.pipeline.density_budget,
            forward=self.pipeline.frames.forward_column,
            near_limit=self.pipeline.eff.preprocess.decimate_below,
            want_intensity=self.pipeline.eff.preprocess.min_intensity_far > 0.0)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        # Сколько кадр шёл до нас: съёмка, транспорт, очередь, разбор. Это
        # часть задержки, с которой команда дойдёт до тормозов, и она входит
        # в тормозной путь наравне со временем обработки.
        now = self.get_clock().now().nanoseconds * 1e-9
        arrival_delay = now - stamp if 0.0 < now - stamp < 5.0 else 0.0
        result = self.pipeline.process(xyz, intensity, timestamp=stamp,
                                       ego_speed=self._current_speed(),
                                       arrival_delay=arrival_delay)
        if not self._clouds_seen:
            self._stop_discovery()
            self._stop_fallback_timer()
        self._clouds_seen += 1
        self._report_input(result, msg)

        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        frame_ms = (time.perf_counter() - callback_started) * 1e3
        # Тракт сам себя мерит без разбора сообщения — сообщаем ему полную
        # стоимость кадра, иначе подстройка дальности решает, что успевает.
        self.pipeline.note_frame_cost(frame_ms)
        self.obstacle_pub.publish(self._obstacle_array(header, result, frame_ms))
        self.status_pub.publish(self._gauge_status(header, result))
        self.corridor_pub.publish(self._corridor_msg(header, result))
        self.marker_pub.publish(self._markers(header, result))
        if self.publish_debug:
            if result.filtered_xyz is not None:
                self.filtered_pub.publish(xyzi_to_pointcloud2(result.filtered_xyz, None, header))
            if result.candidate_xyz is not None:
                self.candidate_pub.publish(xyzi_to_pointcloud2(result.candidate_xyz, None, header))

        self._frames += 1
        result.timings["callback_ms"] = frame_ms
        if self._last_stamp is not None and 1e-3 < stamp - self._last_stamp < 1.0:
            self._periods.append(stamp - self._last_stamp)
        self._last_stamp = stamp
        period_ms = 1000.0 * (sorted(self._periods)[len(self._periods) // 2]
                              if self._periods else 0.1)
        if frame_ms > period_ms:
            self._slow_frames += 1
            if self._slow_frames % 10 == 1:
                self.get_logger().warn(
                    f"Кадр обработан за {frame_ms:.0f} мс — "
                    f"дольше периода лидара ({period_ms:.0f} мс; "
                    f"{self._slow_frames} из {self._frames})")
        if not result.decision.clear and result.decision.action >= ACTION_SERVICE_BRAKE:
            self.get_logger().warn(f"{result.decision.action_name}: {result.decision.reason}")
        self.speed_pub.publish(Float32(data=float(result.ego_speed or 0.0)))

    # ------------------------------------------------------------------ диагностика входа
    def _report_input(self, result: FrameResult, msg: PointCloud2) -> None:
        """Сообщает о том, что определилось или пошло не так на входе.

        Кадр, от которого после фильтрации осталась сотня точек, — это не
        «путь свободен», а облако в чужой СК, закрытый датчик или неверно
        заданная рабочая зона. Молчать об этом опаснее всего: система
        продолжает штатно публиковать «габарит свободен».
        """
        if result.frame_decided:
            self.get_logger().info(
                f"СК датчика: {result.frame_spec} — {result.frame_source}"
                f"; кадр в топике «{self.points_topic}», {result.input_points} точек"
                f", frame_id «{msg.header.frame_id}»")
        if result.degenerate:
            self._degenerate_frames += 1
            if self._degenerate_frames % 20 == 1:
                self.get_logger().error(
                    f"После фильтрации осталось {result.filtered_points} точек из "
                    f"{result.input_points}: проверьте СК датчика "
                    f"(sensor.forward_axis/up_axis, сейчас «{result.frame_spec}») и "
                    f"рабочую зону preprocess. Решение по такому кадру недостоверно "
                    f"({self._degenerate_frames} кадров подряд)")
        else:
            self._degenerate_frames = 0

    # ------------------------------------------------------------------ публикация
    def _obstacle_array(self, header: Header, result: FrameResult,
                        frame_ms: Optional[float] = None) -> ObstacleArray:
        msg = ObstacleArray()
        msg.header = header
        # Полное время кадра, включая разбор сообщения; если его не передали
        # (офлайн-использование), остаётся время самого тракта.
        msg.processing_time_ms = float(frame_ms if frame_ms is not None
                                       else result.timings["total_ms"])
        msg.input_point_count = int(result.input_points)
        msg.filtered_point_count = int(result.filtered_points)
        msg.max_valid_range = float(result.max_valid_range)
        for track in result.obstacles:
            det = track.detection
            item = Obstacle()
            item.id = int(track.id)
            item.classification = int(track.classification)
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
            item.intrusion_depth = float(min(det.intrusion_depth, 1e6))
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
        msg.confident_range = float(decision.confident_range)
        msg.max_safe_speed = float(decision.max_safe_speed)
        msg.speed_limited = bool(decision.speed_limited)
        msg.ego_speed = float(result.ego_speed if result.ego_speed is not None
                              else self.cfg.decision.default_speed)
        msg.speed_source = result.speed_source
        msg.detection_limit = float(result.detection_limit)
        msg.decision_latency = float(result.latency)
        msg.compute_backend = result.backend
        msg.beam_density = float(result.sensor_profile.points_per_sr
                                 if result.sensor_profile else 0.0)
        msg.scene = result.scene.kind if result.scene else "unknown"
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
                f"#{track.id} {track.class_name} {det.distance:.0f} м p={track.confidence:.2f}",
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
