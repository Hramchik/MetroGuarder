"""Оценка поверхности пути (полотна) по продольным ячейкам.

Одна плоскость на всё поле зрения не годится: на 150 м уклон в 1 % даёт
1.5 м ошибки по высоте — это ровно тот масштаб, на котором столб или
насыпь превращаются в «препятствие», а лежащий предмет исчезает.
Поэтому полотно описывается кусочно: своя высота и поперечный наклон
в каждой ячейке длиной bin_size, со сглаживанием между ячейками.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import GroundConfig


@dataclass
class GroundModel:
    edges: np.ndarray        # (B+1,) границы ячеек по x
    centers: np.ndarray      # (B,)
    z0: np.ndarray           # (B,) высота полотна на оси пути в центре ячейки
    slope_y: np.ndarray      # (B,) поперечный наклон dz/dy (возвышение наружного рельса)
    valid: np.ndarray        # (B,) bool — в ячейке было достаточно точек
    rail_offset: float       # высота головки рельса над полотном
    y_ref: np.ndarray        # (B,) поперечное положение оси, вокруг которой считалась ячейка

    def ground_z(self, xyz: np.ndarray) -> np.ndarray:
        """Высота полотна под каждой точкой (линейная интерполяция между ячейками).

        Наклон отсчитывается от оси пути, а не от y=0: в кривой ось уходит
        на десяток метров вбок, и абсолютная y дала бы метровую ошибку высоты.
        """
        if self.centers.size == 0:
            return np.zeros(xyz.shape[0], dtype=np.float32)
        z = np.interp(xyz[:, 0], self.centers, self.z0)
        sl = np.interp(xyz[:, 0], self.centers, self.slope_y)
        ref = np.interp(xyz[:, 0], self.centers, self.y_ref)
        return (z + sl * (xyz[:, 1] - ref)).astype(np.float32)

    def rail_z(self, xyz: np.ndarray) -> np.ndarray:
        """Уровень головок рельсов (УГР) под каждой точкой."""
        return self.ground_z(xyz) + self.rail_offset

    def height_above_rail(self, xyz: np.ndarray) -> np.ndarray:
        return xyz[:, 2] - self.rail_z(xyz)

    @property
    def valid_range(self) -> float:
        if not np.any(self.valid):
            return 0.0
        return float(self.edges[np.max(np.where(self.valid)[0]) + 1])


def estimate_ground(xyz: np.ndarray, cfg: GroundConfig,
                    centerline_y: Optional[np.ndarray] = None,
                    x_min: float = 0.0, x_max: float = 250.0) -> GroundModel:
    """Строит кусочную модель полотна методом «следования за поверхностью».

    Перцентиль по широкой полосе не годится: у платформы, в выемке перед
    порталом тоннеля или рядом с водоотводом в полосу попадают точки на
    два метра ниже полотна, и «землёй» объявляется дно канавы. Поэтому
    уровень ведётся от ближней ячейки к дальней с ограничением на уклон:
    полотно физически не может прыгнуть между соседними ячейками.

    centerline_y — поперечное положение оси пути в каждой ячейке: в кривой
    полоса поиска должна ехать за путём, иначе в неё попадает соседний путь.
    """
    edges = np.arange(x_min, x_max + cfg.bin_size, cfg.bin_size, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = centers.size
    z0 = np.zeros(n_bins)
    slope = np.zeros(n_bins)
    valid = np.zeros(n_bins, dtype=bool)

    y_ref_bins = np.zeros(n_bins)
    if centerline_y is not None:
        ref = np.asarray(centerline_y, dtype=float)
        y_ref_bins = ref[:n_bins] if ref.size >= n_bins else np.pad(ref, (0, n_bins - ref.size), mode="edge")

    if xyz.shape[0] == 0:
        return GroundModel(edges=edges, centers=centers, z0=z0, slope_y=slope,
                           valid=valid, rail_offset=cfg.rail_head_offset, y_ref=y_ref_bins)

    lateral_all = xyz[:, 1] - np.interp(xyz[:, 0], centers, y_ref_bins)
    # Полоса поиска обычно выведена из междупутья (lib/derive.py); без вывода
    # берётся ширина, заведомо укладывающаяся в пределы своего пути.
    search_half_width = cfg.search_half_width if cfg.search_half_width is not None else 2.5
    in_band = np.abs(lateral_all) < search_half_width
    bin_idx = np.digitize(xyz[:, 0], edges) - 1
    order = np.argsort(np.where(in_band, bin_idx, n_bins + 1), kind="stable")
    sorted_bins = np.where(in_band, bin_idx, n_bins + 1)[order]
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left")
    stops = np.searchsorted(sorted_bins, np.arange(n_bins), side="right")

    level: Optional[float] = None
    grade = 0.0
    last_x = centers[0] if n_bins else 0.0
    max_step = cfg.max_slope * cfg.bin_size

    for b in range(n_bins):
        sel = order[starts[b]:stops[b]]
        predicted = None if level is None else level + grade * (centers[b] - last_x)
        if sel.size >= cfg.min_bin_points:
            z = xyz[sel, 2]
            if predicted is not None:
                near = np.abs(z - predicted) < cfg.follow_tolerance
                if near.sum() >= cfg.min_bin_points // 2:
                    sel, z = sel[near], z[near]
                else:
                    sel = sel[:0]
        if sel.size >= max(6, cfg.min_bin_points // 2):
            measured = float(np.percentile(xyz[sel, 2], cfg.low_percentile))
            if predicted is not None:
                measured = float(np.clip(measured, predicted - max_step, predicted + max_step))
            # Поперечный наклон (возвышение наружного рельса в кривой)
            lat = lateral_all[sel]
            inliers = np.abs(xyz[sel, 2] - measured) < cfg.plane_tolerance * 2.0
            lat_slope = 0.0
            if inliers.sum() >= 10:
                a_mat = np.stack([np.ones(int(inliers.sum())), lat[inliers]], axis=1)
                coef, *_ = np.linalg.lstsq(a_mat, xyz[sel][inliers, 2], rcond=None)
                measured, lat_slope = float(coef[0]), float(np.clip(coef[1], -0.12, 0.12))
                if predicted is not None:
                    measured = float(np.clip(measured, predicted - max_step, predicted + max_step))
            if level is not None:
                grade = float(np.clip((measured - level) / max(centers[b] - last_x, 1e-3),
                                      -cfg.max_slope, cfg.max_slope))
            level, last_x = measured, centers[b]
            z0[b], slope[b], valid[b] = measured, lat_slope, True
        elif predicted is not None:
            z0[b] = predicted           # ячейка без данных — продолжаем уклон
            slope[b] = slope[b - 1] if b else 0.0
        elif level is not None:
            z0[b] = level

    if np.any(valid):
        first = int(np.argmax(valid))
        z0[:first] = z0[first]
        slope[:first] = slope[first]

    return GroundModel(edges=edges, centers=centers, z0=z0, slope_y=slope,
                       valid=valid, rail_offset=cfg.rail_head_offset, y_ref=y_ref_bins)
