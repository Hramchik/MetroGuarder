"""Фильтрация облака: обрезка рабочей зоны, отсечение собственного корпуса,
воксельное прореживание по полосам дальности, удаление одиночных точек.

Порядок этапов выбран из соображений цены: сначала дешёвые булевы маски,
которые убирают 60-80 % точек, и только потом всё, что требует KD-дерева.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import PreprocessConfig


def workspace_mask(xyz: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Точки внутри рабочей зоны и вне габарита собственного поезда."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    keep = (
        (x > cfg.x_min) & (x < cfg.x_max)
        & (np.abs(y) < cfg.y_abs_max)
        & (z > cfg.z_min) & (z < cfg.z_max)
    )
    ex0, ex1, ey0, ey1, ez0, ez1 = cfg.ego_box
    ego = (x > ex0) & (x < ex1) & (y > ey0) & (y < ey1) & (z > ez0) & (z < ez1)
    return keep & ~ego


def intensity_mask(xyz: np.ndarray, intensity: Optional[np.ndarray],
                   cfg: PreprocessConfig) -> np.ndarray:
    """Слабые дальние возвраты — почти всегда пыль, капли или паразитная засветка."""
    if intensity is None or cfg.min_intensity_far <= 0.0:
        return np.ones(xyz.shape[0], dtype=bool)
    far = xyz[:, 0] > cfg.intensity_far_range
    return ~(far & (intensity < cfg.min_intensity_far))


def voxel_downsample(xyz: np.ndarray, leaf: float,
                     extra: Optional[np.ndarray] = None) -> np.ndarray:
    """Индексы представителей вокселей (первая точка в вокселе).

    Возвращает индексы, а не сами точки: вызывающему коду нужно тем же
    срезом проредить интенсивность и прочие поля.
    """
    if leaf <= 0.0 or xyz.shape[0] == 0:
        return np.arange(xyz.shape[0])
    keys = np.floor(xyz / leaf).astype(np.int64)
    keys -= keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _uniq, first = np.unique(flat, return_index=True)
    return np.sort(first)


def banded_downsample(xyz: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Прореживание с шагом, зависящим от дальности.

    Ближнее поле избыточно плотное — его режем крупным вокселем; дальнее
    поле (ради которого и нужна дальность детекции) не трогаем вовсе.
    """
    if xyz.shape[0] == 0:
        return np.arange(0)
    rng = xyz[:, 0]
    result = []
    lower = 0.0
    for upper, leaf in cfg.voxel_bands:
        band = np.where((rng >= lower) & (rng < upper))[0]
        if band.size:
            result.append(band[voxel_downsample(xyz[band], leaf)])
        lower = upper
    if not result:
        return np.arange(xyz.shape[0])
    return np.sort(np.concatenate(result))


def radius_outlier_mask(xyz: np.ndarray, radius: float = 0.6,
                        min_neighbors: int = 3,
                        radius_per_meter: float = 0.006) -> np.ndarray:
    """Удаление одиночных точек (пыль, капли, интерференция лидаров).

    Радиус растёт с дальностью: на 150 м соседние точки того же объекта
    физически расходятся, и фиксированный радиус выкосил бы реальную цель.
    Считается через k-го соседа, а не через подсчёт в шаре — так на порядок
    дешевле и не зависит от плотности.
    """
    n = xyz.shape[0]
    if n <= min_neighbors:
        return np.ones(n, dtype=bool)
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    dist, _idx = tree.query(xyz, k=min_neighbors + 1, workers=-1)
    kth = dist[:, -1]
    allowed = radius + radius_per_meter * np.abs(xyz[:, 0])
    return kth <= allowed


def preprocess(xyz: np.ndarray, intensity: Optional[np.ndarray],
               cfg: PreprocessConfig) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Полный этап фильтрации. Возвращает (точки, интенсивность, индексы в исходном облаке)."""
    idx = np.where(workspace_mask(xyz, cfg) & intensity_mask(xyz, intensity, cfg))[0]
    if idx.size == 0:
        return xyz[:0], (intensity[:0] if intensity is not None else None), idx
    kept = idx[banded_downsample(xyz[idx], cfg)]
    return xyz[kept], (intensity[kept] if intensity is not None else None), kept
