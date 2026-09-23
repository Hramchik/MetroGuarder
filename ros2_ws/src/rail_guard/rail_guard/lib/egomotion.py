"""Оценка скорости носителя по самим облакам точек.

Тормозной путь считается от скорости, и без неё решение вырождается: при
подставленной «скорости по умолчанию» объект в 20 метрах даёт экстренное
торможение даже тогда, когда поезд стоит. В записях метро топика скорости
нет, на контрольных данных его тоже может не быть, поэтому скорость
оценивается из данных как резерв к бортовой одометрии.

Метод: продольная корреляция занятости. Тоннель вдоль пути неоднороден —
стыки, кабельные кронштейны, ниши, пикеты, — и профиль занятости сдвигается
между кадрами ровно на пройденный путь. Ищется сдвиг, максимизирующий
нормированное скалярное произведение профилей: это дешевле любой регистрации
облаков и не требует ни признаков, ни соответствий.

Оценка продольная и только вперёд: для тормозного расчёта нужна именно она.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from .config import EgoMotionConfig


class EgoMotionEstimator:
    """Скорость по сдвигу профиля занятости между соседними кадрами."""

    def __init__(self, cfg: EgoMotionConfig):
        self.cfg = cfg
        self._prev: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None
        self._recent: Deque[float] = deque(maxlen=max(1, cfg.smooth_frames))
        self._rejected = 0
        self.last_score: float = 0.0
        self.last_raw: Optional[float] = None

    def reset(self) -> None:
        self._prev = None
        self._prev_time = None
        self._recent.clear()
        self._rejected = 0

    # ------------------------------------------------------------------ сетка
    def _occupancy(self, xyz: np.ndarray, heights: np.ndarray) -> Optional[np.ndarray]:
        cfg = self.cfg
        if xyz.shape[0] == 0:
            return None
        band = (heights > cfg.height_min) & (heights < cfg.height_max)
        x, y = xyz[:, 0], xyz[:, 1]
        sel = band & (x >= cfg.x_min) & (x < cfg.x_max) & (np.abs(y) < cfg.lateral_abs_max)
        if int(sel.sum()) < cfg.min_points:
            return None
        xs = x[sel]
        ys = y[sel]
        n_long = int(np.ceil((cfg.x_max - cfg.x_min) / cfg.cell_long))
        n_lat = int(np.ceil(2.0 * cfg.lateral_abs_max / cfg.cell_lat))
        rows = ((xs - cfg.x_min) / cfg.cell_long).astype(np.int32)
        cols = ((ys + cfg.lateral_abs_max) / cfg.cell_lat).astype(np.int32)
        np.clip(rows, 0, n_long - 1, out=rows)
        np.clip(cols, 0, n_lat - 1, out=cols)
        grid = np.zeros(n_long * n_lat, dtype=np.float32)
        grid[rows.astype(np.int64) * n_lat + cols] = 1.0
        return grid.reshape(n_long, n_lat)

    # ------------------------------------------------------------------ сдвиг
    def _best_shift(self, prev: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
        """(сдвиг в ячейках с субъячеечным уточнением, качество совпадения)."""
        max_bins = min(int(round(self.cfg.max_shift / self.cfg.cell_long)), prev.shape[0] - 8)
        if max_bins < 1:
            return 0.0, 0.0
        scores = np.empty(max_bins + 1, dtype=np.float64)
        for s in range(max_bins + 1):
            a = prev[s:] if s else prev
            b = cur[: prev.shape[0] - s] if s else cur
            denom = float(np.sqrt(a.sum() * b.sum()))
            scores[s] = 0.0 if denom <= 0.0 else float((a * b).sum()) / denom
        peak = int(np.argmax(scores))
        best = float(scores[peak])
        # Параболическое уточнение по трём точкам — сдвиг редко кратен ячейке.
        shift = float(peak)
        if 0 < peak < max_bins:
            y0, y1, y2 = scores[peak - 1], scores[peak], scores[peak + 1]
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-9:
                shift += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
        return shift, best

    # ------------------------------------------------------------------ шаг
    def update(self, xyz: np.ndarray, heights: np.ndarray,
               timestamp: float) -> Optional[float]:
        """Скорость в м/с или None, если оценить по этому кадру нельзя."""
        if not self.cfg.enabled:
            return None
        grid = self._occupancy(xyz, heights)
        prev, prev_time = self._prev, self._prev_time
        self._prev, self._prev_time = grid, timestamp
        if grid is None or prev is None or prev_time is None:
            return self.smoothed
        dt = timestamp - prev_time
        if not (self.cfg.min_dt <= dt <= self.cfg.max_dt):
            return self.smoothed

        shift, score = self._best_shift(prev, grid)
        self.last_score = score
        if score < self.cfg.min_score:
            return self.smoothed
        speed = shift * self.cfg.cell_long / dt
        if speed > self.cfg.max_speed:
            return self.smoothed
        self.last_raw = float(speed)
        # Гейт по ускорению: скорость поезда не меняется быстрее, чем позволяет
        # его тормозная характеристика. Оценка, требующая невозможного
        # ускорения, — промах корреляции (пустой участок тоннеля, стык, кадр с
        # редкими данными), и в историю она не идёт. Но если промахи идут
        # подряд, значит изменилась сама дорога, и тогда новой оценке верим.
        previous = self.smoothed
        if previous is not None:
            allowed = self.cfg.max_accel * dt + self.cfg.accel_tolerance
            if abs(speed - previous) > allowed and self._rejected < self.cfg.max_rejected:
                self._rejected += 1
                return previous
        self._rejected = 0
        self._recent.append(float(speed))
        return self.smoothed

    @property
    def smoothed(self) -> Optional[float]:
        """Медиана последних оценок: одиночный промах не должен править решением."""
        if not self._recent:
            return None
        return float(np.median(np.asarray(self._recent)))
