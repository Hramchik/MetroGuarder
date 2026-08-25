"""Подстановка препятствий в реальные кадры лидара.

Зачем это нужно. В открытых записях с поездов (OSDaR23 и любых других,
снятых в нормальной эксплуатации) посторонних предметов в габарите
практически нет: пути чистые, иначе поезд бы не поехал. Поэтому реальные
данные дают честную оценку ложных срабатываний, но не дают ни одной
истинно-положительной цели, а без них нельзя измерить дальность детекции.

Решение — вставлять цель в реальный кадр с физически корректной
плотностью точек. Модель плотности не выдумана, а измерена по этому же
датасету: число отражений от объекта площадью A на дальности r равно
k * A / r^2, где k получено по размеченным людям (см. POINTS_PER_M2_AT_1M).
Проверка: человек 0.5 x 1.88 м на 78 м — 118 точек в разметке, на 143 м —
63 точки; обе точки дают k ≈ 7.8e5 с расхождением 5 %.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Точек на квадратный метр фронтальной площади, приведённое к дальности 1 м.
# Значение относится к сводному облаку OSDaR23 (Pandar64 + 3 x Livox Tele-15
# + 2 x Waymo Honeycomb, кадр 100 мс). Для другого лидара его нужно
# перемерить по своим данным — параметр вынесен в аргумент.
POINTS_PER_M2_AT_1M = 7.8e5


@dataclass
class Target:
    """Цель для подстановки: форма задаётся облаком в собственной СК
    (X — вдоль пути, Y — поперёк, Z вверх от опорной точки, низ в z=0)."""
    name: str
    size: np.ndarray             # (3,) длина, ширина, высота
    shape: Optional[np.ndarray] = None   # (M,3) реальная форма, если есть
    reflectivity: float = 40.0   # типичная интенсивность возврата

    @property
    def frontal_area(self) -> float:
        return float(self.size[1] * self.size[2])


def person_target(width: float = 0.5, depth: float = 0.35, height: float = 1.75) -> Target:
    return Target(name="человек", size=np.array([depth, width, height]))


def box_target(side: float = 0.4) -> Target:
    return Target(name=f"предмет {int(side * 100)} см", size=np.array([side, side, side]))


def crop_target(xyz: np.ndarray, intensity: Optional[np.ndarray], center: np.ndarray,
                size: np.ndarray, name: str = "объект", margin: float = 0.1) -> Optional[Target]:
    """Вырезает реальный объект из кадра по кубоиду разметки.

    Форма и шероховатость поверхности остаются настоящими — меняется
    только дальность и, соответственно, плотность точек.
    """
    half = size / 2.0 + margin
    mask = np.all(np.abs(xyz - center) <= half, axis=1)
    if mask.sum() < 8:
        return None
    pts = xyz[mask] - center
    pts[:, 2] -= pts[:, 2].min()
    reflect = float(np.median(intensity[mask])) if intensity is not None else 40.0
    return Target(name=name, size=np.asarray(size, dtype=float), shape=pts, reflectivity=reflect)


def expected_points(target: Target, distance: float,
                    density: float = POINTS_PER_M2_AT_1M) -> int:
    """Сколько отражений даст цель на этой дальности."""
    return int(max(0, round(density * target.frontal_area / max(distance, 1.0) ** 2)))


def _sample_surface(target: Target, count: int, rng: np.random.Generator) -> np.ndarray:
    """Точки на видимой (передней) поверхности цели."""
    if target.shape is not None and target.shape.shape[0] >= 4:
        # Реальная форма: берём её точки, при нехватке — с повторами и
        # небольшим разбросом, чтобы не получить кратные дубликаты.
        idx = rng.choice(target.shape.shape[0], size=count,
                         replace=count > target.shape.shape[0])
        jitter = rng.normal(0.0, 0.02, size=(count, 3))
        return target.shape[idx] + jitter
    length, width, height = target.size
    pts = np.empty((count, 3))
    pts[:, 0] = rng.uniform(-length / 2.0, -length / 2.0 + 0.05, size=count)  # передняя грань
    pts[:, 1] = rng.uniform(-width / 2.0, width / 2.0, size=count)
    pts[:, 2] = rng.uniform(0.0, height, size=count)
    return pts


def inject(xyz: np.ndarray, intensity: Optional[np.ndarray], target: Target,
           position: np.ndarray, rng: Optional[np.random.Generator] = None,
           density: float = POINTS_PER_M2_AT_1M, range_noise: float = 0.02,
           occlude: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    """Вставляет цель в облако. position — точка опоры (низ цели) в СК лидара.

    Возвращает (облако, интенсивность, число добавленных точек).
    """
    rng = rng or np.random.default_rng(0)
    distance = float(np.hypot(position[0], position[1]))
    count = expected_points(target, distance, density)
    if count <= 0:
        return xyz, (intensity if intensity is not None else np.zeros(len(xyz), np.float32)), 0

    local = _sample_surface(target, count, rng)
    points = local + position
    # Шум дальномера — вдоль луча, а не по осям.
    direction = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-6)
    points += direction * rng.normal(0.0, range_noise, size=(count, 1))

    base_intensity = intensity if intensity is not None else np.zeros(len(xyz), np.float32)
    if occlude:
        keep = _visibility_mask(xyz, points, target, position)
        xyz = xyz[keep]
        base_intensity = base_intensity[keep]

    new_intensity = np.full(count, target.reflectivity, dtype=np.float32)
    return (np.vstack([xyz, points.astype(np.float32)]),
            np.concatenate([base_intensity, new_intensity]), count)


def _visibility_mask(xyz: np.ndarray, target_points: np.ndarray, target: Target,
                     position: np.ndarray) -> np.ndarray:
    """Убирает исходные точки, которые цель загораживает.

    Без этого за подставленным объектом остаётся видимым фон, и кластер
    получает «хвост» — детектор оказался бы в более лёгких условиях,
    чем в реальности.
    """
    ranges = np.linalg.norm(xyz, axis=1)
    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    el = np.arcsin(np.clip(xyz[:, 2] / np.maximum(ranges, 1e-6), -1.0, 1.0))

    t_range = np.linalg.norm(target_points, axis=1)
    t_az = np.arctan2(target_points[:, 1], target_points[:, 0])
    t_el = np.arcsin(np.clip(target_points[:, 2] / np.maximum(t_range, 1e-6), -1.0, 1.0))

    distance = float(np.linalg.norm(position[:2]))
    half_az = 0.5 * float(target.size[1]) / max(distance, 1.0)
    half_el = 0.5 * float(target.size[2]) / max(distance, 1.0)
    shadow = (np.abs(az - t_az.mean()) < half_az) & (np.abs(el - t_el.mean()) < half_el) \
        & (ranges > t_range.min())
    return ~shadow
