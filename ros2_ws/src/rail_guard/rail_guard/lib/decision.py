"""Перевод списка подтверждённых препятствий в команду для системы
принятия решений поезда.

Решение принимается не по «есть объект / нет объекта», а по сравнению
дальности до препятствия с тормозной дистанцией на текущей скорости:
объект в 120 м при 80 км/ч — это экстренное торможение, а тот же объект
при 20 км/ч — только внимание.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import DecisionConfig
from .tracking import ObjectTrack

ACTION_CLEAR = 0
ACTION_ATTENTION = 1
ACTION_SERVICE_BRAKE = 2
ACTION_EMERGENCY_BRAKE = 3

ACTION_NAMES = {
    ACTION_CLEAR: "CLEAR",
    ACTION_ATTENTION: "ATTENTION",
    ACTION_SERVICE_BRAKE: "SERVICE_BRAKE",
    ACTION_EMERGENCY_BRAKE: "EMERGENCY_BRAKE",
}


@dataclass
class GaugeDecision:
    clear: bool
    action: int
    nearest_distance: float
    nearest_id: int
    obstacle_count: int
    braking_distance: float
    detection_range: float
    reason: str

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self.action, "?")


def braking_distance(speed: float, cfg: DecisionConfig, emergency: bool = True) -> float:
    decel = cfg.emergency_decel if emergency else cfg.service_decel
    return speed * cfg.reaction_time + (speed * speed) / (2.0 * max(decel, 0.1))


def decide(tracks: List[ObjectTrack], speed: Optional[float], detection_range: float,
           cfg: DecisionConfig) -> GaugeDecision:
    v = cfg.default_speed if speed is None else max(0.0, float(speed))
    d_emergency = braking_distance(v, cfg, emergency=True)
    d_service = braking_distance(v, cfg, emergency=False)

    in_gauge = [t for t in tracks if t.detection.in_gauge]
    strong = [t for t in in_gauge if t.confidence >= cfg.min_confidence]
    weak = [t for t in in_gauge if t.confidence < cfg.min_confidence]

    if not in_gauge:
        return GaugeDecision(clear=True, action=ACTION_CLEAR, nearest_distance=float("inf"),
                             nearest_id=0, obstacle_count=0, braking_distance=d_emergency,
                             detection_range=detection_range, reason="габарит свободен")

    if not strong:
        # Объект есть, но либо он на дальности, где положение оси пути уже
        # не подтверждено рельсами, либо точек слишком мало. Тормозить по
        # такому нельзя, молчать — тем более: снижаем скорость и ждём.
        nearest = min(weak, key=lambda t: t.detection.distance)
        return GaugeDecision(clear=False, action=ACTION_ATTENTION,
                             nearest_distance=float(nearest.detection.distance),
                             nearest_id=nearest.id, obstacle_count=len(weak),
                             braking_distance=d_emergency, detection_range=detection_range,
                             reason=(f"{nearest.detection.class_name} на "
                                     f"{nearest.detection.distance:.0f} м, уверенность "
                                     f"{nearest.confidence:.2f} ниже порога торможения"))

    nearest = min(strong, key=lambda t: t.detection.distance)
    d = nearest.detection.distance

    if d <= d_emergency:
        action = ACTION_EMERGENCY_BRAKE
        reason = (f"{nearest.detection.class_name} в габарите на {d:.0f} м, "
                  f"тормозной путь {d_emergency:.0f} м")
    elif d <= d_service:
        action = ACTION_SERVICE_BRAKE
        reason = (f"{nearest.detection.class_name} в габарите на {d:.0f} м, "
                  f"служебное торможение (путь {d_service:.0f} м)")
    elif d <= cfg.attention_factor * d_service:
        action = ACTION_ATTENTION
        reason = f"{nearest.detection.class_name} в габарите на {d:.0f} м, снизить скорость"
    else:
        action = ACTION_ATTENTION
        reason = f"{nearest.detection.class_name} на {d:.0f} м, вне зоны торможения"

    return GaugeDecision(clear=False, action=action, nearest_distance=float(d),
                         nearest_id=nearest.id, obstacle_count=len(strong) + len(weak),
                         braking_distance=d_emergency, detection_range=detection_range,
                         reason=reason)
