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
    intrusion_depth: float = float("inf")   # заход внутрь свободного места тоннеля
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
    # Порог захода внутрь свободного места обычно выведен из собственного
    # шума измеренной границы (lib/pipeline.py); без вывода проверка не
    # делается вовсе — отвергать объекты по порогу, взятому ниоткуда, опаснее,
    # чем не отвергать их совсем.
    min_intrusion = cfg.min_intrusion or 0.0
    if min_intrusion > 0.0 and det.intrusion_depth < min_intrusion:
        # Кластер лежит на границе свободного пространства тоннеля: обделка,
        # кабельный лоток, край платформы. Отличить его от предмета по размеру
        # и плотности нельзя — на 60 м стена даёт столько же точек, сколько
        # человек, — а по положению относительно свободного места можно.
        return "tunnel_surface"
    half = float(corridor.half_width_at(np.array([det.distance]))[0])
    touches_edge = abs(abs(det.lateral_offset) - half) < cfg.wall_edge_margin \
        or det.clearance_margin > -cfg.wall_edge_margin
    if length > cfg.wall_length and width < cfg.wall_width and touches_edge:
        return "wall"              # стена тоннеля, край платформы, ограждение
    if length > cfg.max_length:
        return "structure"
    if cfg.full_height_fraction > 0.0 \
            and det.top_height > cfg.full_height_fraction * corridor.top \
            and det.height_above_rail < 0.3:
        # От полотна до свода — обделка, портал, створка гермозатвора.
        return "full_height"
    return None


def _segment_percentile(values: np.ndarray, labels: np.ndarray, n_labels: int,
                        fraction: float, valid: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    """Перцентиль внутри каждого кластера — сортировкой по (метка, значение).

    Прямого группового перцентиля в numpy нет, а цикл по кластерам — это как
    раз та задержка, которую здесь убирают. Сортировка по паре ключей ставит
    точки каждого кластера подряд и по возрастанию, после чего нужный элемент
    берётся по смещению от начала сегмента.
    """
    out = np.full(n_labels, np.nan)
    keep = np.ones(values.shape[0], dtype=bool) if valid is None else valid
    if not keep.any():
        return out
    lab = labels[keep]
    val = values[keep]
    order = np.lexsort((val, lab))
    lab_sorted, val_sorted = lab[order], val[order]
    counts = np.bincount(lab_sorted, minlength=n_labels)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    present = counts > 0
    pos = starts[present] + np.floor(fraction * (counts[present] - 1)).astype(np.int64)
    out[present] = val_sorted[np.clip(pos, 0, val_sorted.size - 1)]
    return out


def _cluster_stats(xyz: np.ndarray, heights: np.ndarray, lateral: np.ndarray,
                   clearance: np.ndarray, intrusion: Optional[np.ndarray],
                   order: np.ndarray, start: np.ndarray, stop: np.ndarray,
                   n_labels: int) -> dict:
    """Габариты, центр и сводные величины всех кластеров разом.

    Всё считается сегментными свёртками по массиву, отсортированному по
    меткам: `reduceat` проходит его один раз, независимо от того, сколько в
    нём кластеров.
    """
    sorted_xyz = xyz[order]
    sorted_h = heights[order]
    sorted_clearance = clearance[order]
    counts = (stop - start).astype(np.float64)
    nonempty = counts > 0
    # reduceat не любит пустых сегментов: для них берём любой допустимый
    # индекс, а результат всё равно отбрасываем по маске.
    safe_start = np.where(nonempty, start, 0)

    def seg(func, data):
        result = func.reduceat(data, safe_start, axis=0)
        return result

    min_point = seg(np.minimum, sorted_xyz)
    max_point = seg(np.maximum, sorted_xyz)
    total = seg(np.add, sorted_xyz)
    centroid = total / np.maximum(counts, 1.0)[:, None]
    height_min = seg(np.minimum, sorted_h)
    height_max = seg(np.maximum, sorted_h)
    clearance_min = seg(np.minimum, sorted_clearance)

    # Точки, не вошедшие ни в один кластер, имеют метку -1 и лежат в начале
    # отсортированного массива: сегменты кластеров начинаются после них.
    clustered_from = int(start[0]) if n_labels > 0 else 0
    in_clusters = order[clustered_from:]
    labels_flat = np.repeat(np.arange(n_labels), (stop - start))
    lateral_median = _segment_percentile(lateral[in_clusters], labels_flat,
                                         n_labels, 0.5)

    if intrusion is None:
        intrusion_q = np.full(n_labels, np.inf)
    else:
        # Верхний квартиль, а не медиана: у предмета, стоящего на полотне,
        # нижние точки лежат на самой поверхности, и медиана занизила бы заход
        # внутрь свободного места до нуля. Точки, про которые модель ничего не
        # знает, в квартиль не берутся вовсе: иначе высокий кластер (стена от
        # полотна до свода) всегда попадал бы верхней частью в неописанные
        # слои и становился неотвергаемым.
        known = np.isfinite(intrusion[in_clusters])
        intrusion_q = _segment_percentile(intrusion[in_clusters], labels_flat,
                                          n_labels, 0.75, valid=known)
        intrusion_q = np.where(np.isfinite(intrusion_q), intrusion_q, np.inf)

    return {"min_point": min_point, "max_point": max_point, "centroid": centroid,
            "height_min": height_min, "height_max": height_max,
            "clearance": clearance_min, "lateral": lateral_median,
            "intrusion": intrusion_q}


def mark_linear_infrastructure(detections: List[Detection], max_object_length: float,
                              lateral_tolerance: float = 0.25,
                              height_tolerance: float = 0.15,
                              min_members: int = 3) -> int:
    """Помечает кластеры, выстроенные в линию вдоль пути, как инфраструктуру.

    Контактный рельс, кабельный лоток, короб и жёлоб идут вдоль пути на одном
    и том же поперечном смещении и одной высоте над УГР. Лидар видит их
    рвано: на низкой горизонтальной поверхности соседние кольца ложатся через
    r²·Δφ/h — на тридцати метрах это больше метра, — и связать их
    кластеризацией нельзя, не склеив заодно всё остальное. Поэтому линия
    опознаётся не по связности точек, а по расположению самих кластеров: три
    и больше объекта, стоящие на одном поперечном смещении и одной высоте,
    растянутые вдоль пути дальше, чем может быть один предмет, — это
    конструкция. Три предмета, случайно выстроившиеся в линию на одинаковой
    высоте с точностью до четверти метра, — событие, которого не бывает.

    Признак не зависит ни от датчика, ни от участка: он опирается только на
    геометрию пути и на максимальную длину предмета из профиля.
    """
    alive = [d for d in detections if not d.reject_reason]
    if len(alive) < min_members:
        return 0
    marked = 0
    used = set()
    for i, anchor in enumerate(alive):
        if id(anchor) in used:
            continue
        line = [anchor]
        for other in alive[i + 1:]:
            if id(other) in used:
                continue
            if abs(other.lateral_offset - anchor.lateral_offset) <= lateral_tolerance \
                    and abs(other.height_above_rail - anchor.height_above_rail) <= height_tolerance:
                line.append(other)
        if len(line) < min_members:
            continue
        span = max(d.distance for d in line) - min(d.distance for d in line)
        if span <= 0.5 * max_object_length:
            # Кластеры стоят вплотную — это может быть один предмет,
            # рассыпавшийся на части, и трогать его нельзя.
            continue
        for member in line:
            member.reject_reason = "track_line"
            member.classification = CLASS_INFRASTRUCTURE
            member.in_gauge = False
            member.confidence = 0.0
            used.add(id(member))
            marked += 1
    return marked


def detections_from_clusters(xyz: np.ndarray, labels: np.ndarray, heights: np.ndarray,
                             corridor: Corridor, cluster_cfg: ClusterConfig,
                             obj_cfg: ObjectFilterConfig,
                             keep_rejected: bool = False,
                             intrusion: Optional[np.ndarray] = None,
                             point_scale: int = 1,
                             scale_beyond: float = 0.0) -> List[Detection]:
    """Собирает объекты по меткам кластеров и применяет фильтры.

    `point_scale` — сколько кадров сведено в это облако. Число точек объекта
    делится на него: пороги плотности рассчитаны на один кадр, и накопление
    не должно выдавать слабую цель за сильную. Сам факт, что точки нескольких
    кадров легли в один кластер, уже работает на обнаружение — кластер
    собирается там, где по одному кадру он бы рассыпался.

    `scale_beyond` — дальность, с которой накопление вообще работает. Ближе
    неё кластер собран из одного кадра, и делить его точки не на что: иначе
    предмет в двадцати метрах, давший полтора десятка отражений, выглядит
    трёхточечным всплеском и отвергается — при том, что виден он прекрасно.
    """
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

    # Сводка по кластерам считается разом для всех, а не в цикле по каждому.
    # Цикл на Python по десяткам кластеров с срезами по индексам стоил больше,
    # чем вся кластеризация, — а это прямая задержка реакции: время от кадра
    # до команды торможения складывается из таких мест.
    stats = _cluster_stats(xyz, heights, lateral_all, clearance_all, intrusion,
                           order, start, stop, n_labels)

    for lab in range(n_labels):
        idx = order[start[lab]:stop[lab]]
        if idx.size == 0:
            continue
        min_pt, max_pt = stats["min_point"][lab], stats["max_point"][lab]
        # Кластер ближе зоны накопления собран из одного кадра — его точки
        # не масштабируются, сколько бы кадров ни свелось в дальней зоне.
        distance = float(min_pt[0])
        scale = point_scale if distance >= scale_beyond else 1
        det = Detection(
            centroid=stats["centroid"][lab],
            min_point=min_pt,
            max_point=max_pt,
            size=max_pt - min_pt,
            distance=distance,
            lateral_offset=float(stats["lateral"][lab]),
            height_above_rail=float(stats["height_min"][lab]),
            top_height=float(stats["height_max"][lab]),
            clearance_margin=float(stats["clearance"][lab]),
            point_count=max(1, int(round(idx.size / max(scale, 1)))),
            point_indices=idx,
        )
        if intrusion is not None:
            det.intrusion_depth = float(stats["intrusion"][lab])

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
        elif width < obj_cfg.thin_max_width and length > obj_cfg.thin_min_length:
            # Тонкая полоса вдоль пути: кабельный лоток, тяга, накладка, лист.
            # Поперечный размер здесь главный признак: он у препятствия не
            # бывает меньше трети метра, а у путевого оборудования — почти
            # всегда меньше.
            reason = "thin_along_track"
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
                if reason in ("overhead", "wall", "structure", "ground_sheet", "long_low",
                              "tunnel_surface", "thin_along_track", "full_height",
                              "track_line") \
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
