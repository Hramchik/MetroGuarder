"""Перевод списка подтверждённых препятствий в команду для системы
принятия решений поезда.

Решение принимается не по «есть объект / нет объекта», а по сравнению
дальности до препятствия с тормозной дистанцией на текущей скорости:
объект в 120 м при 80 км/ч — это экстренное торможение, а тот же объект
при 20 км/ч — только внимание.

Второе, что здесь считается, — допустимая скорость по дальности достоверного
обзора. Поезд не должен ехать быстрее, чем видит: если тормозной путь длиннее
зоны, в которой положение пути подтверждено, остановиться по внезапному
препятствию уже нельзя — независимо от того, насколько хорош детектор. Это
ограничение уходит в систему управления вместе с решением.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import GEOMETRIC_MIN_POINTS, DecisionConfig
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
    # Дальность, на которой положение пути подтверждено (рельсы, а за ними
    # обделка тоннеля), и допустимая на ней скорость.
    confident_range: float = 0.0
    max_safe_speed: float = 0.0
    speed_limited: bool = False
    # Задержка, с которой это решение получено: от съёмки кадра до команды.
    # Входит в тормозной путь наравне с временем срабатывания тормозов.
    latency: float = 0.0

    @property
    def action_name(self) -> str:
        return ACTION_NAMES.get(self.action, "?")


def braking_distance(speed: float, cfg: DecisionConfig, emergency: bool = True,
                     extra_delay: float = 0.0) -> float:
    """Тормозной путь с учётом задержки, с которой пришло само решение.

    `extra_delay` — сколько прошло от съёмки кадра до команды: транспорт,
    обработка, публикация. За это время поезд уже проехал, и не учитывать
    его — значит считать тормозной путь короче настоящего. Ровно поэтому
    задержка тракта меряется, а не предполагается.
    """
    decel = cfg.emergency_decel if emergency else cfg.service_decel
    delay = cfg.reaction_time + max(extra_delay, 0.0)
    return speed * delay + (speed * speed) / (2.0 * max(decel, 0.1))


def safe_speed(distance: float, cfg: DecisionConfig, extra_delay: float = 0.0) -> float:
    """Скорость, с которой поезд успевает остановиться в пределах distance.

    Обратная задача к braking_distance: из d = v*t + v^2/(2a) при
    положительном корне v = -a*t + sqrt((a*t)^2 + 2*a*d).
    """
    a = max(cfg.emergency_decel, 0.1)
    at = a * (cfg.reaction_time + max(extra_delay, 0.0))
    return float(max(0.0, -at + (at * at + 2.0 * a * max(distance, 0.0)) ** 0.5))


def _kinematic_split(strong: List[ObjectTrack], speed: float,
                     cfg: DecisionConfig) -> Tuple[List[ObjectTrack], List[ObjectTrack]]:
    """Делит треки на приближающиеся со скоростью поезда и «стоящие» рядом с ним.

    Неподвижный предмет на пути обязан сокращать дальность ровно на пройденный
    поездом путь. Трек, который держит постоянную дальность при движении, —
    это либо артефакт уехавшего коридора, либо возврат от конструкции самого
    носителя: тормозить по такому нельзя.
    """
    if cfg.kinematic_gate <= 0.0 or speed < cfg.kinematic_min_speed:
        return strong, []
    threshold = cfg.kinematic_gate * speed
    consistent, inconsistent = [], []
    for track in strong:
        (consistent if track.closing_speed() >= threshold else inconsistent).append(track)
    return consistent, inconsistent


def decide(tracks: List[ObjectTrack], speed: Optional[float], detection_range: float,
           cfg: DecisionConfig, speed_measured: bool = False,
           confident_range: Optional[float] = None,
           safety_margin: float = 0.0, latency: float = 0.0) -> GaugeDecision:
    """`safety_margin` — боковой запас габарита (раскачка кузова, погрешность
    оси). Объект, зашедший только в него, поезд физически ещё не задевает:
    это повод снизить скорость и разобраться, а не рвать стоп-кран. Торможение
    выдаётся по тому, что вошло в габарит самого кузова."""
    v = cfg.default_speed if speed is None else max(0.0, float(speed))
    d_emergency = braking_distance(v, cfg, emergency=True, extra_delay=latency)
    d_service = braking_distance(v, cfg, emergency=False, extra_delay=latency)
    # Зона, в которой положение пути подтверждено рельсами: за ней ось ведёт
    # обделка тоннеля, и экстренное торможение по такому объекту не выдаётся.
    confident = detection_range if confident_range is None else float(confident_range)
    v_safe = safe_speed(confident, cfg, extra_delay=latency)
    over_speed = bool(cfg.speed_limit_enabled and speed_measured
                      and v > v_safe + cfg.speed_limit_tolerance)
    speed_note = (f"; обзор достоверен на {confident:.0f} м — это не больше "
                  f"{v_safe:.0f} м/с при текущих {v:.0f} м/с") if over_speed else ""

    in_gauge = [t for t in tracks if t.detection.in_gauge]
    # Порог отражений для торможения обычно выведен из измеренной плотности
    # лучей (lib/derive.py); без вывода остаётся геометрический минимум —
    # столько точек нужно, чтобы у кластера вообще были измеримы габариты.
    min_points_brake = cfg.min_points_brake if cfg.min_points_brake is not None \
        else GEOMETRIC_MIN_POINTS
    strong = [t for t in in_gauge
              if t.confidence >= cfg.min_confidence
              and t.detection.point_count >= min_points_brake
              and t.detection.clearance_margin <= -safety_margin]
    weak = [t for t in in_gauge if t not in strong]
    # Кинематическая проверка имеет смысл только при известной скорости:
    # на подставленной «по умолчанию» она отсеяла бы реальные препятствия
    # у стоящего поезда.
    gated: List[ObjectTrack] = []
    if speed_measured:
        strong, gated = _kinematic_split(strong, v, cfg)
        weak = weak + gated

    if not in_gauge:
        # Ограничение скорости по дальности обзора решение не меняет: помех нет,
        # габарит свободен. Это отдельный выход для системы управления
        # (max_safe_speed), а не тревога — иначе «ложным срабатыванием» станет
        # каждый кадр, где поезд едет быстрее, чем видит.
        return GaugeDecision(
            clear=True, action=ACTION_CLEAR,
            nearest_distance=float("inf"), nearest_id=0, obstacle_count=0,
            braking_distance=d_emergency, detection_range=detection_range,
            reason="габарит свободен" + (speed_note.replace("; обзор", ", обзор", 1)
                                         if over_speed else ""),
            confident_range=confident, max_safe_speed=v_safe, speed_limited=over_speed,
            latency=latency)

    if not strong:
        # Объект есть, но либо он на дальности, где положение оси пути уже
        # не подтверждено рельсами, либо точек слишком мало, либо он не
        # приближается со скоростью поезда. Тормозить по такому нельзя,
        # молчать — тем более: снижаем скорость и ждём.
        nearest = min(weak, key=lambda t: t.detection.distance)
        if gated and nearest in gated:
            reason = (f"{nearest.class_name} на "
                      f"{nearest.detection.distance:.0f} м сближается "
                      f"{nearest.closing_speed():.1f} м/с при скорости поезда "
                      f"{v:.1f} м/с — на препятствие не похоже")
        elif nearest.detection.point_count < min_points_brake:
            reason = (f"{nearest.class_name} на "
                      f"{nearest.detection.distance:.0f} м — всего "
                      f"{nearest.detection.point_count} отражений, для команды "
                      f"торможения нужно {min_points_brake}")
        elif nearest.detection.clearance_margin > -safety_margin:
            reason = (f"{nearest.class_name} на "
                      f"{nearest.detection.distance:.0f} м задевает только запас "
                      f"габарита ({-nearest.detection.clearance_margin:.2f} м из "
                      f"{safety_margin:.2f} м) — снижаем скорость")
        else:
            reason = (f"{nearest.class_name} на "
                      f"{nearest.detection.distance:.0f} м, уверенность "
                      f"{nearest.confidence:.2f} ниже порога торможения")
        return GaugeDecision(clear=False, action=ACTION_ATTENTION,
                             nearest_distance=float(nearest.detection.distance),
                             nearest_id=nearest.id, obstacle_count=len(weak),
                             braking_distance=d_emergency, detection_range=detection_range,
                             reason=reason + speed_note, confident_range=confident,
                             max_safe_speed=v_safe, speed_limited=over_speed,
                             latency=latency)

    nearest = min(strong, key=lambda t: t.detection.distance)
    d = nearest.detection.distance

    if d <= d_emergency:
        action = ACTION_EMERGENCY_BRAKE
        reason = (f"{nearest.class_name} в габарите на {d:.0f} м, "
                  f"тормозной путь {d_emergency:.0f} м")
    elif d <= d_service:
        action = ACTION_SERVICE_BRAKE
        reason = (f"{nearest.class_name} в габарите на {d:.0f} м, "
                  f"служебное торможение (путь {d_service:.0f} м)")
    elif d <= cfg.attention_factor * d_service:
        action = ACTION_ATTENTION
        reason = f"{nearest.class_name} в габарите на {d:.0f} м, снизить скорость"
    else:
        action = ACTION_ATTENTION
        reason = f"{nearest.class_name} на {d:.0f} м, вне зоны торможения"

    # Ограничение по зоне: объект за подтверждённой рельсами дальностью
    # тормозится служебно, а не экстренно. Ось там ведёт обделка тоннеля с
    # точностью около метра при ширине тоннеля 4.5 м: объект настоящий, но
    # его принадлежность именно нашему пути не доказана. Войдёт в
    # подтверждённую зону — получит полную реакцию.
    if d > confident and action > cfg.guided_zone_action:
        action = int(cfg.guided_zone_action)
        reason = (f"{nearest.class_name} на {d:.0f} м — за зоной, где путь "
                  f"подтверждён рельсами ({confident:.0f} м): снижение скорости")

    return GaugeDecision(clear=False, action=action, nearest_distance=float(d),
                         nearest_id=nearest.id, obstacle_count=len(strong) + len(weak),
                         braking_distance=d_emergency, detection_range=detection_range,
                         reason=reason + speed_note, confident_range=confident,
                         max_safe_speed=v_safe, speed_limited=over_speed,
                         latency=latency)
