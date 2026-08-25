"""Превращение кластеров в объекты и отсев того, что помехой не является.

Ложное срабатывание на беспилотном поезде стоит экстренного торможения,
поэтому каждый кластер проходит три независимые проверки: геометрия
(размер, высота над УГР), плотность (достаточно ли точек для этой
дальности) и принадлежность инфраструктуре (стена тоннеля, платформа,
сигнальный мост, портал). Классификация — уже поверх этого.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .cluster import min_points_for_range
from .config import ClusterConfig, ObjectFilterConfig
from .track import Corridor

CLASS_UNKNOWN = 0
CLASS_PERSON = 1
CLASS_LARGE_OBJECT = 2
CLASS_SMALL_OBJECT = 3
CLASS_INFRASTRUCTURE = 4

CLASS_NAMES = {
    CLASS_UNKNOWN: "unknown",
    CLASS_PERSON: "person",
    CLASS_LARGE_OBJECT: "large_object",
    CLASS_SMALL_OBJECT: "small_object",
    CLASS_INFRASTRUCTURE: "infrastructure",
}


@dataclass
class Detection:
    centroid: np.ndarray
    min_point: np.ndarray
    max_point: np.ndarray
    size: np.ndarray
    distance: float             # до ближайшей точки объекта
    lateral_offset: float
    height_above_rail: float    # низ объекта над УГР
    top_height: float           # верх объекта над УГР
    clearance_margin: float     # <0 — внутри габарита
    point_count: int
    classification: int = CLASS_UNKNOWN
    confidence: float = 0.0
    in_gauge: bool = True
    reject_reason: str = ""
    point_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.classification, "unknown")


def _classify(det: Detection) -> int:
    length, width, height = float(det.size[0]), float(det.size[1]), float(det.size[2])
    footprint = max(length, width)
    thickness = min(length, width)
    if 0.55 <= height <= 2.30 and footprint <= 1.30 and thickness >= 0.12 \
            and det.height_above_rail < 0.9:
        return CLASS_PERSON
    if footprint > 1.30 or height > 2.30:
        return CLASS_LARGE_OBJECT
    return CLASS_SMALL_OBJECT


def _confidence(det: Detection, cluster_cfg: ClusterConfig) -> float:
    """Уверенность одного кадра: сколько точек пришло относительно
    минимально осмысленного числа для этой дальности, с поправкой на то,
    насколько глубоко объект внутри габарита."""
    need = float(min_points_for_range(det.distance, cluster_cfg))
    density = np.clip(det.point_count / max(need * 2.0, 1.0), 0.0, 1.0)
    depth = np.clip(-det.clearance_margin / 0.5, 0.0, 1.0)   # 0.5 м внутрь = максимум
    size_score = np.clip(max(det.size[2], 0.05) / 0.5, 0.2, 1.0)
    return float(np.clip(0.45 * density + 0.35 * depth + 0.20 * size_score, 0.0, 1.0))


def _is_infrastructure(det: Detection, corridor: Corridor, cfg: ObjectFilterConfig) -> Optional[str]:
    """Признаки конструкций, которые всегда «в габарите», но помехой не являются."""
    length, width, height = float(det.size[0]), float(det.size[1]), float(det.size[2])
    if det.height_above_rail > cfg.overhead_clearance:
        return "overhead"          # сигнальный мост, портал, свод тоннеля
    half = float(corridor.half_width_at(np.array([det.distance]))[0])
    touches_edge = abs(abs(det.lateral_offset) - half) < cfg.wall_edge_margin \
        or det.clearance_margin > -cfg.wall_edge_margin
    if length > cfg.wall_length and width < cfg.wall_width and touches_edge:
        return "wall"              # стена тоннеля, край платформы, ограждение
    if length > cfg.max_length:
        return "structure"
    return None


def detections_from_clusters(xyz: np.ndarray, labels: np.ndarray, heights: np.ndarray,
                             corridor: Corridor, cluster_cfg: ClusterConfig,
                             obj_cfg: ObjectFilterConfig,
                             keep_rejected: bool = False) -> List[Detection]:
    """Собирает объекты по меткам кластеров и применяет фильтры."""
    results: List[Detection] = []
    if labels.size == 0:
        return results
    n_labels = int(labels.max()) + 1
    if n_labels <= 0:
        return results

    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    start = np.searchsorted(sorted_labels, np.arange(n_labels), side="left")
    stop = np.searchsorted(sorted_labels, np.arange(n_labels), side="right")

    lateral_all = corridor.lateral(xyz)
    clearance_all = corridor.clearance(xyz)

    for lab in range(n_labels):
        idx = order[start[lab]:stop[lab]]
        if idx.size == 0:
            continue
        pts = xyz[idx]
        h = heights[idx]
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        det = Detection(
            centroid=pts.mean(axis=0),
            min_point=min_pt,
            max_point=max_pt,
            size=max_pt - min_pt,
            distance=float(min_pt[0]),
            lateral_offset=float(np.median(lateral_all[idx])),
            height_above_rail=float(h.min()),
            top_height=float(h.max()),
            clearance_margin=float(clearance_all[idx].min()),
            point_count=int(idx.size),
            point_indices=idx,
        )

        length, width, height = (float(det.size[0]), float(det.size[1]), float(det.size[2]))
        reason = ""
        if height < obj_cfg.min_size_z or det.top_height < obj_cfg.min_top_height:
            reason = "too_flat"                     # рельс, шпала, балласт, лоток
        elif length > obj_cfg.flat_length and height < obj_cfg.flat_height:
            # Длинный вдоль пути и низкий — путевое оборудование: стрелочная
            # тяга, кабельный лоток, контррельс. Предмет такой длины и высоты
            # в габарите не появляется.
            reason = "long_low"
        elif height < obj_cfg.sheet_height and width > obj_cfg.sheet_width \
                and length > obj_cfg.flat_length:
            # Протяжённая горизонтальная поверхность: стрелка, настил, край
            # платформы или «протёкшее» в коридор полотно. Ширина здесь —
            # главный признак: упавшая опора той же длины и высоты узкая.
            reason = "ground_sheet"
        elif float(np.prod(np.maximum(det.size, 0.02))) < obj_cfg.min_volume:
            reason = "too_small"
        elif det.point_count < float(min_points_for_range(det.distance, cluster_cfg)):
            reason = "sparse"                       # плотность не тянет на реальный объект
        elif length * width < obj_cfg.min_footprint and det.point_count < obj_cfg.sliver_points:
            # Вырожденный кластер: столб или провод, видимый с ребра.
            reason = "sliver"
        else:
            infra = _is_infrastructure(det, corridor, obj_cfg)
            if infra:
                reason = infra

        if reason:
            det.reject_reason = reason
            det.classification = CLASS_INFRASTRUCTURE \
                if reason in ("overhead", "wall", "structure", "ground_sheet", "long_low") \
                else CLASS_UNKNOWN
            det.in_gauge = False
            det.confidence = 0.0
            if keep_rejected:
                results.append(det)
            continue

        det.classification = _classify(det)
        det.confidence = _confidence(det, cluster_cfg)
        det.in_gauge = det.clearance_margin < 0.0
        results.append(det)

    results.sort(key=lambda d: d.distance)
    return results
