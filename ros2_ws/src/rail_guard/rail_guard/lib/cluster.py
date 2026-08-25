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


def epsilon_for(xyz: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    return np.clip(cfg.eps0 + cfg.eps_per_meter * np.abs(xyz[:, 0]), cfg.eps0, cfg.eps_max)


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


def min_points_for_range(distance: np.ndarray | float, cfg: ClusterConfig) -> np.ndarray | float:
    """Сколько точек считать достаточным на данной дальности.

    Число возвратов от объекта падает как 1/r^2 — требовать одинаковый
    порог на 20 и на 150 м значит либо ослепнуть вдали, либо утонуть
    в ложных срабатываниях вблизи.
    """
    d = np.maximum(np.asarray(distance, dtype=np.float64), 1.0)
    scaled = cfg.ref_min_points * (cfg.ref_range / d) ** 2
    return np.maximum(np.ceil(scaled), cfg.abs_min_points)
