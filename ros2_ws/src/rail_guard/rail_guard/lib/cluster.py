"""Кластеризация с радиусом связности, зависящим от дальности.

На 20 м человек — это сотни точек в 3 см друг от друга, на 150 м — пять
точек, разнесённых на полметра. Единый порог связности либо склеивает
всё вблизи, либо рассыпает дальнюю цель на одиночные точки. Поэтому
координаты нормируются на локальный радиус связности, и дальше работает
обычная евклидова кластеризация с порогом 1.
"""
from __future__ import annotations

import numpy as np

from .config import ClusterConfig


# Значения связности, когда конфиг ещё не разрешён (см. lib/derive.py):
# базовый радиус порядка размера небольшого предмета и расхождение лучей
# типового сканирующего лидара. Нужны только для прямого вызова этапа вне
# тракта — сам тракт всегда работает на выведенных значениях.
_FALLBACK_EPS0 = 0.15
_FALLBACK_EPS_PER_METER = 0.007


def epsilon_for(xyz: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    eps0 = cfg.eps0 if cfg.eps0 is not None else _FALLBACK_EPS0
    per_meter = cfg.eps_per_meter if cfg.eps_per_meter is not None else _FALLBACK_EPS_PER_METER
    return np.clip(eps0 + per_meter * np.abs(xyz[:, 0]), eps0, cfg.eps_max)


def cluster(xyz: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """Возвращает метки кластеров (-1 — точка не вошла ни в один кластер).

    Связные компоненты считает scipy на разреженной матрице смежности:
    питоновский union-find на тех же данных стоил 200+ мс на кадр и
    съедал весь бюджет реального времени.
    """
    n = xyz.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    if n == 1:
        return np.zeros(1, dtype=np.int32) if cfg.min_cluster_points <= 1 else labels

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    eps = epsilon_for(xyz, cfg)
    scaled = xyz / eps[:, None]
    tree = cKDTree(scaled)
    pairs = tree.query_pairs(r=1.0, output_type="ndarray")

    if pairs.size == 0:
        components = np.arange(n)
        n_comp = n
    else:
        data = np.ones(pairs.shape[0], dtype=np.int8)
        adjacency = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        n_comp, components = connected_components(adjacency, directed=False)

    counts = np.bincount(components, minlength=n_comp)
    keep = counts >= cfg.min_cluster_points
    remap = np.full(n_comp, -1, dtype=np.int32)
    remap[keep] = np.arange(int(keep.sum()), dtype=np.int32)
    return remap[components]


def voxel_leaf_at(distance: np.ndarray | float,
                  bands: tuple) -> np.ndarray | float:
    """Ребро вокселя прореживания на данной дальности."""
    d = np.asarray(distance, dtype=np.float64)
    leaf = np.zeros_like(d)
    lower = 0.0
    for upper, size in bands:
        leaf = np.where((d >= lower) & (d < upper), size, leaf)
        lower = upper
    return leaf


def min_points_for_range(distance: np.ndarray | float, cfg: ClusterConfig) -> np.ndarray | float:
    """Сколько точек считать достаточным на данной дальности.

    Две физические границы, и берётся меньшая из них.

    **Лучи.** Число возвратов от объекта падает как 1/r²: требовать одинаковый
    порог на 20 и на 150 м значит либо ослепнуть вдали, либо утонуть в ложных
    срабатываниях вблизи.

    **Прореживание.** В ближней зоне облако режется вокселем, и цель площадью
    A даёт не больше A/L² точек, сколько бы лучей в неё ни попало: на 20 м
    предмет 40 см при вокселе 12 см — это десяток точек, а не две сотни.
    Порог, посчитанный только по лучам, в ближней зоне оказывается заведомо
    недостижимым — и система не видит предмет прямо перед собой.
    """
    d = np.maximum(np.asarray(distance, dtype=np.float64), 1.0)
    if cfg.ref_min_points is None:
        # Порог выводится из измеренной плотности лучей (lib/derive.py). Пока
        # датчик не измерен, требовать больше минимума нельзя: это означало бы
        # порог, взятый ниоткуда, и пропуск целей на неизвестном лидаре.
        return np.full_like(d, cfg.abs_min_points) if np.ndim(d) else cfg.abs_min_points
    scaled = cfg.ref_min_points * (cfg.ref_range / d) ** 2
    if cfg.voxel_bands and cfg.target_area > 0.0:
        leaf = voxel_leaf_at(d, cfg.voxel_bands)
        cap = np.where(leaf > 0.0,
                       cfg.detection_fraction * cfg.target_area / np.maximum(leaf, 1e-6) ** 2,
                       np.inf)
        scaled = np.minimum(scaled, cap)
    return np.maximum(np.ceil(scaled), cfg.abs_min_points)
