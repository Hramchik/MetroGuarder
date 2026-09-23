"""Подстановка препятствий в реальные кадры лидара.

Зачем это нужно. В открытых записях с поездов (OSDaR23 и любых других,
снятых в нормальной эксплуатации) посторонних предметов в габарите
практически нет: пути чистые, иначе поезд бы не поехал. Поэтому реальные
данные дают честную оценку ложных срабатываний, но не дают ни одной
истинно-положительной цели, а без них нельзя измерить дальность детекции.

Решение — вставлять цель в реальный кадр с физически корректной плотностью
точек: число отражений от цели фронтальной площади A на дальности r равно
fill · k · A / r², где k — плотность лучей, а fill — доля лучей, дающих
полезный возврат.

**Плотность берётся из измерения того самого кадра**, в который вставляется
цель (`lib/sensor_model.py`), а не из константы. Прежде здесь стояло число,
снятое с комплекта лидаров OSDaR23, и оно молча делало оценку дальности
неверной на любом другом датчике: на вдвое более редком лидаре подставленная
цель получала вдвое больше точек, чем получила бы в действительности, и
измеренная «дальность обнаружения» оказывалась завышенной.

Проверка модели по разметке OSDaR23: человек 0.5 × 1.88 м даёт 118 точек на
78 м и 63 на 143 м; измеренная по тем же кадрам плотность лучей переднего
сектора вместе с fill ≈ 0.5 воспроизводит оба числа. Перемерить fill на
своих данных можно скриптом `scripts/measure_sensor.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .sensor_model import SensorProfile, measure_beam_density

# Доля лучей, попавших в габаритный прямоугольник цели и давших полезный
# возврат: силуэт занимает не весь прямоугольник, тёмная поверхность и косой
# угол падения съедают ещё часть. Значение по умолчанию совпадает с
# safety.target_fill и проверяется по разметке (scripts/measure_sensor.py).
DEFAULT_FILL = 0.5


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
                    sensor: Optional[SensorProfile] = None,
                    fill: float = DEFAULT_FILL) -> int:
    """Сколько отражений даст цель на этой дальности.

    `sensor` — профиль датчика, измеренный по кадру, в который вставляется
    цель. Без него функция работать не может: число отражений — свойство
    конкретного лидара, а не цели.
    """
    if sensor is None or not sensor.valid:
        raise ValueError("нужен измеренный профиль датчика: подставленная цель "
                         "обязана иметь плотность точек того лидара, в кадр "
                         "которого её вставляют (см. lib/sensor_model.py)")
    return int(max(0, round(sensor.expected_points(target.frontal_area, distance, fill))))


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
           sensor: Optional[SensorProfile] = None, fill: float = DEFAULT_FILL,
           range_noise: float = 0.02,
           occlude: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    """Вставляет цель в облако. position — точка опоры (низ цели) в СК лидара.

    `sensor` — измеренный профиль датчика; если не передан, плотность лучей
    меряется по самому этому кадру. Так подставленная цель всегда получает
    столько точек, сколько дал бы тот лидар, чей кадр перед нами.

    Возвращает (облако, интенсивность, число добавленных точек).
    """
    rng = rng or np.random.default_rng(0)
    distance = float(np.hypot(position[0], position[1]))
    if sensor is None or not sensor.valid:
        sensor = measure_frame_density(xyz, target, position)
    count = expected_points(target, distance, sensor, fill)
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


def measure_frame_density(xyz: np.ndarray, target: Target,
                          position: np.ndarray) -> SensorProfile:
    """Плотность лучей того кадра, в который вставляется цель.

    Сектор берётся вокруг направления на цель и по её угловому размеру с
    запасом: плотность лучей неоднородна по полю зрения (у комплекта из
    нескольких лидаров она различается на порядок), и значение имеет смысл
    только для того направления, куда цель ставится.
    """
    distance = max(float(np.hypot(position[0], position[1])), 1.0)
    # Не уже нескольких градусов: в совсем узком окне точек не хватит на
    # устойчивую оценку, а плотность в пределах такого окна уже однородна.
    az_half = max(np.arctan2(3.0 * float(target.size[1]), distance), np.radians(5.0))
    el_half = max(np.arctan2(3.0 * float(target.size[2]), distance), np.radians(3.0))
    measured = measure_beam_density(xyz, az_half, el_half)
    if measured is None:
        raise ValueError("не удалось измерить плотность лучей по кадру: "
                         "в секторе цели слишком мало точек")
    return measured


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
