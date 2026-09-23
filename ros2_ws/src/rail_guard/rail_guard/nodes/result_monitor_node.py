#!/usr/bin/env python3
"""Вывод результата работы алгоритма в терминал — построчно по кадрам.

Нужен там, где RViz недоступен или не нужен: в контейнере без проброшенного
дисплея, в логе прогона, в записи демонстрации. Показывает ровно то, что
система сообщает поезду: сколько точек пришло, докуда видит, где ось пути
подтверждена рельсами, что найдено в габарите и какая выдана команда.

По завершении печатает сводку по всей записи: она же служит результатом
эксперимента, и её не нужно собирать отдельными скриптами.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from rail_guard_msgs.msg import GaugeStatus, ObstacleArray

ACTION_TEXT = {0: "свободен", 1: "ВНИМАНИЕ", 2: "СЛУЖЕБНОЕ", 3: "ЭКСТРЕННОЕ"}
CLASS_TEXT = {0: "неизв", 1: "человек", 2: "крупный", 3: "мелкий", 4: "инфра"}
HEADER = (f"{'кадр':>5} {'точек':>8} {'после':>7} {'видит,м':>8} {'путь,м':>7} "
          f"{'торм,м':>7} {'помех':>6} {'ближайшая помеха':>22} {'скорость, м/с':>15} "
          f"{'решение':<11} {'мс':>6}")


class ResultMonitor(Node):
    def __init__(self) -> None:
        super().__init__("rail_guard_monitor")
        self.declare_parameter("quiet", False)     # только сводка, без построчного вывода
        self.quiet = bool(self.get_parameter("quiet").value)

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
                         history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(ObstacleArray, "rail_guard/obstacles", self.on_obstacles, qos)
        self.create_subscription(GaugeStatus, "rail_guard/gauge_status", self.on_status, qos)

        self._pending: Optional[ObstacleArray] = None
        self.frames = 0
        self.actions = {0: 0, 1: 0, 2: 0, 3: 0}
        self.latency: List[float] = []
        self.ranges: List[float] = []
        self.confident: List[float] = []
        self.safe_speeds: List[float] = []
        self.speed_limited = 0
        self.alarm_distances: List[float] = []
        self.track_ids = set()
        # Измеренные свойства установки и участка: предел обнаружения, плотность
        # лучей датчика и распознанная сцена по кадрам.
        self.detection_limits: List[float] = []
        self.beam_density = 0.0
        self.scenes: dict = {}
        # Период кадров считаем по штампам: «уложились в реальное время» —
        # это про период того лидара, который прислал данные, а не про 10 Гц.
        self.stamps: List[float] = []
        if not self.quiet:
            print(HEADER, flush=True)
            print("-" * len(HEADER), flush=True)

    # Пара сообщений одного кадра публикуется подряд: препятствия, затем
    # решение. Строку печатаем по решению, добрав к нему препятствия.
    def on_obstacles(self, msg: ObstacleArray) -> None:
        self._pending = msg

    def on_status(self, msg: GaugeStatus) -> None:
        obstacles = self._pending
        self._pending = None
        self.frames += 1
        self.actions[int(msg.action)] = self.actions.get(int(msg.action), 0) + 1
        self.ranges.append(float(msg.detection_range))
        self.confident.append(float(msg.confident_range))
        self.safe_speeds.append(float(msg.max_safe_speed))
        self.speed_limited += int(bool(msg.speed_limited))
        # Что система измерила о себе: предел обнаружения и тип участка. Это
        # не настройки, а свойства конкретного лидара и конкретной сцены —
        # на объекте по ним сразу видно, на что система способна здесь.
        if msg.detection_limit > 0.0:
            self.detection_limits.append(float(msg.detection_limit))
        if msg.beam_density > 0.0:
            self.beam_density = float(msg.beam_density)
        if msg.scene:
            self.scenes[msg.scene] = self.scenes.get(msg.scene, 0) + 1
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        if not msg.clear and msg.nearest_distance < 1e6:
            self.alarm_distances.append(float(msg.nearest_distance))

        nearest = "—"
        points = after = 0
        latency = 0.0
        if obstacles is not None:
            points = int(obstacles.input_point_count)
            after = int(obstacles.filtered_point_count)
            latency = float(obstacles.processing_time_ms)
            self.latency.append(latency)
            self.track_ids.update(int(o.id) for o in obstacles.obstacles)
            in_gauge = [o for o in obstacles.obstacles if o.in_gauge]
            if in_gauge:
                closest = min(in_gauge, key=lambda o: o.distance)
                nearest = (f"#{closest.id} {CLASS_TEXT.get(int(closest.classification), '?')}"
                           f" {closest.distance:.0f} м")
        if not self.quiet:
            # При превышении допустимой по дальности обзора скорости показываем
            # её рядом с измеренной: «14.7 ≤11» читается сразу.
            speed = (f"{msg.ego_speed:4.1f} ≤{msg.max_safe_speed:<4.0f}" if msg.speed_limited
                     else f"{msg.ego_speed:4.1f} {msg.speed_source[:9]}")
            print(f"{self.frames:5d} {points:8d} {after:7d} {msg.detection_range:8.0f} "
                  f"{msg.confident_range:7.0f} {msg.braking_distance:7.0f} "
                  f"{msg.obstacle_count:6d} {nearest:>22} "
                  f"{speed:>15} {ACTION_TEXT.get(int(msg.action), '?'):<11} {latency:6.1f}",
                  flush=True)

    # ------------------------------------------------------------------ сводка
    def summary(self) -> None:
        if not self.frames:
            print("Кадров не получено: детектор не публиковал решений.", file=sys.stderr)
            return
        import statistics as st
        print("\n" + "=" * 78)
        print(f"Кадров обработано: {self.frames}")
        order = [(0, "путь свободен"), (1, "внимание"), (2, "служебное торможение"),
                 (3, "экстренное торможение")]
        for key, name in order:
            count = self.actions.get(key, 0)
            if count:
                print(f"  {name:<24} {count:5d} ({100.0 * count / self.frames:4.0f} %)")
        # Период лидара: медиана разниц штампов принятых кадров. Кадры, до
        # которых тракт не добрался, увеличивают эту разницу, поэтому берём
        # минимальный правдоподобный интервал — он и есть период съёмки.
        deltas = sorted(b - a for a, b in zip(self.stamps[:-1], self.stamps[1:])
                        if 1e-3 < b - a < 1.0)
        period_ms = 1000.0 * deltas[max(0, int(0.05 * len(deltas)))] if deltas else 100.0
        if self.latency:
            over = sum(1 for v in self.latency if v > period_ms)
            print(f"Задержка обработки, мс: медиана {st.median(self.latency):.1f}, "
                  f"максимум {max(self.latency):.1f}; дольше периода лидара "
                  f"({period_ms:.0f} мс) {over} кадров "
                  f"({100.0 * over / len(self.latency):.0f} %)")
            print(f"Частота обработки: {1000.0 / max(st.median(self.latency), 1e-3):.1f} кадр/с "
                  f"по медианной задержке")
        if self.ranges:
            print(f"Дальность мониторинга, м: медиана {st.median(self.ranges):.0f}, "
                  f"максимум {max(self.ranges):.0f}")
        if self.confident:
            print(f"Путь подтверждён (рельсы, затем обделка), м: медиана "
                  f"{st.median(self.confident):.0f}, максимум {max(self.confident):.0f}")
        if self.safe_speeds:
            # Главная цифра для оценки «успеет ли поезд остановиться»: с какой
            # скоростью можно ехать, чтобы тормозной путь укладывался в зону,
            # где положение пути известно.
            print(f"Допустимая по обзору скорость, м/с: медиана "
                  f"{st.median(self.safe_speeds):.1f}; скорость выше допустимой в "
                  f"{self.speed_limited} кадрах "
                  f"({100.0 * self.speed_limited / self.frames:.0f} %)")
        if self.detection_limits:
            print(f"Предел обнаружения минимальной цели, м: медиана "
                  f"{st.median(self.detection_limits):.0f} "
                  f"(плотность лучей {self.beam_density:.2g} точек/ср — измерена по кадрам)")
        if self.scenes:
            total = sum(self.scenes.values())
            names = {"tunnel": "замкнутое сечение", "open": "открытый участок",
                     "unknown": "не определена"}
            parts = ", ".join(f"{names.get(kind, kind)} {100.0 * count / total:.0f} %"
                              for kind, count in sorted(self.scenes.items(),
                                                        key=lambda kv: -kv[1]))
            print(f"Сцена по кадрам: {parts}")
        if self.alarm_distances:
            print(f"Дальность обнаружения помехи, м: максимум "
                  f"{max(self.alarm_distances):.0f}, медиана "
                  f"{st.median(self.alarm_distances):.0f}")
        print(f"Уникальных треков за запись: {len(self.track_ids)}")
        print("=" * 78, flush=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ResultMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
