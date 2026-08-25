"""Поиск оси пути по рельсам и построение коридора габарита.

Габарит имеет смысл только относительно того пути, по которому поезд
реально поедет. «Прямо вперёд» — неверный ответ: в кривой R=200 м ось
уходит на 10 м вбок уже к 70 метрам, и коридор либо накрывает соседний
путь (ложные тревоги), либо теряет собственный (пропуск препятствия).

Схема: рельсы ищутся ячейками по ходу движения («следование за парой»),
затем по найденным точкам подгоняется дуга y = b*x + a*x^2 с закреплённым
началом (поезд стоит на своём пути, значит y(0)=0). Дальше зоны, где
рельсы разрешаются лидаром, ось продолжается той же дугой, а уверенность
детекции в этой зоне снижается.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import GaugeConfig, TrackConfig
from .ground import GroundModel


@dataclass
class TrackModel:
    lin: float                    # b в y = b*x + a*x^2
    quad: float                   # a
    fit_range: float              # до какой дальности ось подтверждена рельсами
    from_rails: bool
    bin_x: np.ndarray             # центры ячеек с найденной рельсовой парой
    bin_y: np.ndarray
    bin_z: np.ndarray             # высота головок рельсов в этих ячейках
    n_bins: int = 0
    score: float = 0.0            # качество совпадения сечения с шаблоном пути

    def y_at(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.lin * x + self.quad * x * x

    def slope_at(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.lin + 2.0 * self.quad * x

    @property
    def curvature_radius(self) -> float:
        if abs(self.quad) < 1e-7:
            return float("inf")
        return float((1.0 + self.lin ** 2) ** 1.5 / (2.0 * abs(self.quad)))

    def confidence_at(self, x: np.ndarray | float, decay_range: float = 60.0) -> np.ndarray:
        """Насколько можно доверять положению оси на дальности x.

        В пределах подтверждённой рельсами зоны — единица, дальше падает:
        экстраполяция дуги верна, пока кривая не сменилась переходной.
        """
        x = np.asarray(x, dtype=np.float64)
        beyond = np.maximum(0.0, x - self.fit_range)
        return np.clip(1.0 - beyond / max(decay_range, 1.0), 0.15, 1.0)


def _cross_section_grid(x: np.ndarray, lateral: np.ndarray, height: np.ndarray,
                        cfg: TrackConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Поперечный профиль пути: сколько точек «рельсовой» высоты пришлось
    на каждую ячейку (продольная ячейка x поперечная полоса).

    Работать с профилем, а не с отдельными точками, нужно потому, что на
    реальных данных головка рельса поднимается над балластом всего на
    15-20 см и даёт считанные отражения: различим не сам рельс, а форма
    сечения — два бугра на полшага колеи и провал между ними.
    """
    x_edges = np.arange(cfg.near_start, cfg.max_fit_range + cfg.profile_bin, cfg.profile_bin)
    lat_edges = np.arange(-cfg.search_half_width, cfg.search_half_width + cfg.lat_bin, cfg.lat_bin)
    n_rows, n_cols = x_edges.size - 1, lat_edges.size - 1

    row = np.digitize(x, x_edges) - 1
    col = np.digitize(lateral, lat_edges) - 1
    valid = (row >= 0) & (row < n_rows) & (col >= 0) & (col < n_cols)
    flat = row[valid] * n_cols + col[valid]

    total = np.bincount(flat, minlength=n_rows * n_cols).reshape(n_rows, n_cols)
    rail_band = (height[valid] > cfg.rail_height_min) & (height[valid] < cfg.rail_height_max)
    rails = np.bincount(flat[rail_band], minlength=n_rows * n_cols).reshape(n_rows, n_cols)
    # Нормируем строку: на 80 м точек на два порядка меньше, чем на 10 м,
    # иначе ближние ячейки полностью определяют форму дуги.
    norm = rails / np.maximum(total.sum(axis=1, keepdims=True), 1.0)
    return norm, rails, x_edges, lat_edges


def _template_score(profile: np.ndarray, x_centers: np.ndarray, lat_edges: np.ndarray,
                    shifts: np.ndarray, cfg: TrackConfig) -> np.ndarray:
    """Свёртка профиля с шаблоном сечения пути для набора гипотез оси.

    Шаблон: плюс на местах головок рельсов, минус в середине колеи и
    сразу снаружи. Разностная форма важнее абсолютной — она не реагирует
    ни на общий уровень балласта, ни на широкие сооружения вдоль пути.
    """
    n_rows, n_cols = profile.shape
    lat0, step = lat_edges[0], lat_edges[1] - lat_edges[0]
    half = 0.5 * cfg.gauge
    taps = ((-half, 1.0), (half, 1.0), (0.0, -cfg.center_weight),
            (-half - cfg.outer_offset, -cfg.outer_weight),
            (half + cfg.outer_offset, -cfg.outer_weight))

    scores = np.zeros(shifts.shape[0])
    rows = np.arange(n_rows)
    # Ближние ячейки надёжнее дальних: там рельсы разрешаются уверенно.
    # Без этого веса три-четыре точки на 100 м перевешивают сотни на 20 м.
    row_weight = 1.0 / (1.0 + x_centers / cfg.row_weight_range)
    for offset, weight in taps:
        cols = np.rint((shifts + offset - lat0) / step).astype(np.int64)   # (H, n_rows)
        inside = (cols >= 0) & (cols < n_cols)
        cols = np.clip(cols, 0, n_cols - 1)
        contribution = profile[rows[None, :], cols] * inside * row_weight[None, :]
        scores += weight * contribution.sum(axis=1)
    return scores


def estimate_track(xyz: np.ndarray, intensity: Optional[np.ndarray], ground: GroundModel,
                   cfg: TrackConfig, previous: Optional[TrackModel] = None,
                   prior: Optional["TrackModel"] = None) -> TrackModel:
    """Ищет ось пути по форме поперечного сечения и подгоняет дугу.

    prior — необязательная ось из внешнего источника (маршрут метро,
    путевая карта, одометрия). Если она задана, поиск идёт вокруг неё:
    на реальной линии геометрия пути известна заранее, и лидару остаётся
    её подтвердить, а не выводить с нуля.
    """
    reference = prior if prior is not None else previous
    ref_lin = reference.lin if reference is not None else 0.0
    ref_quad = reference.quad if reference is not None else 0.0

    x_all = xyz[:, 0].astype(np.float64)
    lateral = xyz[:, 1] - (ref_lin * x_all + ref_quad * x_all * x_all)
    height = xyz[:, 2] - ground.ground_z(xyz)
    band = (x_all > cfg.near_start) & (x_all < cfg.max_fit_range) \
        & (np.abs(lateral) < cfg.search_half_width) & (height > -0.3) & (height < 1.0)
    idx = np.where(band)[0]

    model: Optional[TrackModel] = None
    if idx.size >= cfg.min_rail_points:
        profile, raw_counts, x_edges, lat_edges = _cross_section_grid(
            x_all[idx], lateral[idx], height[idx], cfg)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

        quad_max = 1.0 / (2.0 * cfg.min_radius)
        lin_grid = np.linspace(-cfg.lin_span, cfg.lin_span, cfg.coarse_lin)
        quad_grid = np.linspace(-quad_max, quad_max, cfg.coarse_quad)
        best_lin, best_quad, best_score = 0.0, 0.0, -np.inf
        for stage in range(2):
            lin_mesh, quad_mesh = np.meshgrid(lin_grid, quad_grid, indexing="ij")
            lin_flat, quad_flat = lin_mesh.ravel(), quad_mesh.ravel()
            # Смещение гипотезы относительно опорной оси в каждой ячейке
            shifts = (lin_flat[:, None] * x_centers[None, :]
                      + quad_flat[:, None] * (x_centers ** 2)[None, :])
            score = _template_score(profile, x_centers, lat_edges, shifts, cfg)
            score -= cfg.prior_weight * (np.abs(quad_flat) / quad_max
                                         + np.abs(lin_flat) / cfg.lin_span)
            # Ограничение непрерывности: на контрольной дальности ось не
            # может уехать от опорной дальше, чем на max_axis_shift. Проверяем
            # именно на средней дистанции, а не на предельной: соседний путь
            # отстоит на 4.5 м уже там, а настоящая кривая на 100+ м имеет
            # право уходить сколь угодно далеко.
            check = int(np.argmin(np.abs(x_centers - cfg.shift_check_range)))
            score[np.abs(shifts[:, check]) > cfg.max_axis_shift] = -np.inf
            best = int(np.argmax(score))
            best_lin, best_quad = float(lin_flat[best]), float(quad_flat[best])
            best_score = float(score[best])
            if stage == 0:
                d_lin = (lin_grid[1] - lin_grid[0]) if lin_grid.size > 1 else 0.0
                d_quad = (quad_grid[1] - quad_grid[0]) if quad_grid.size > 1 else 0.0
                lin_grid = best_lin + np.linspace(-d_lin, d_lin, cfg.fine_steps)
                quad_grid = best_quad + np.linspace(-d_quad, d_quad, cfg.fine_steps)

        # Гипотеза найдена в системе отсчёта опорной оси — переводим в общую.
        lin = ref_lin + best_lin
        quad = ref_quad + best_quad

        # Дальность доверия: до какой ячейки рельсы реально видны.
        # Считаем не совпадение шаблона (оно нормировано и плохо интерпретируется),
        # а число отражений рельсовой высоты, легших на обе нитки в ячейке.
        x_centers_arr = x_centers
        lat0, lat_step = lat_edges[0], lat_edges[1] - lat_edges[0]
        half_gauge = 0.5 * cfg.gauge
        shift_best = best_lin * x_centers_arr + best_quad * x_centers_arr ** 2
        tol_cols = max(1, int(round(cfg.rail_tolerance / lat_step)))
        hits = np.zeros(x_centers_arr.size, dtype=np.int64)
        for sign in (-1.0, 1.0):
            col = np.rint((shift_best + sign * half_gauge - lat0) / lat_step).astype(np.int64)
            for d in range(-tol_cols, tol_cols + 1):
                c = np.clip(col + d, 0, raw_counts.shape[1] - 1)
                inside = (col + d >= 0) & (col + d < raw_counts.shape[1])
                hits += raw_counts[np.arange(x_centers_arr.size), c] * inside
        confirmed = hits >= cfg.min_bin_points
        fit_range = 0.0
        gap = 0
        for i, ok in enumerate(confirmed):
            if ok:
                fit_range = float(x_edges[i + 1])
                gap = 0
            else:
                gap += 1
                if gap > cfg.max_bin_gap:
                    break
        model = TrackModel(lin=lin, quad=quad, fit_range=fit_range,
                           from_rails=fit_range >= cfg.min_fit_range,
                           bin_x=x_centers_arr[confirmed],
                           bin_y=lin * x_centers_arr[confirmed] + quad * x_centers_arr[confirmed] ** 2,
                           bin_z=np.zeros(0), n_bins=int(confirmed.sum()), score=best_score)

    if model is None or not model.from_rails:
        if previous is not None:
            # Кадр без узнаваемого сечения — едем на прошлой оси, зона доверия тает.
            fallback = TrackModel(lin=previous.lin, quad=previous.quad,
                                  fit_range=max(0.0, previous.fit_range - 10.0),
                                  from_rails=False, bin_x=np.zeros(0), bin_y=np.zeros(0),
                                  bin_z=np.zeros(0))
            if model is None:
                return fallback
            model.lin, model.quad = fallback.lin, fallback.quad
            return model
        if model is None:
            return TrackModel(lin=ref_lin, quad=ref_quad, fit_range=0.0, from_rails=False,
                              bin_x=np.zeros(0), bin_y=np.zeros(0), bin_z=np.zeros(0))

    # Ось пути не может дёрнуться за 100 мс — сглаживаем между кадрами.
    if previous is not None and cfg.smoothing < 1.0:
        model.lin = cfg.smoothing * model.lin + (1.0 - cfg.smoothing) * previous.lin
        model.quad = cfg.smoothing * model.quad + (1.0 - cfg.smoothing) * previous.quad
    return model


@dataclass
class Corridor:
    """Габарит поезда, протянутый вдоль оси пути."""
    s: np.ndarray
    y: np.ndarray
    half_width: np.ndarray
    top: float
    bottom: float

    def lateral(self, xyz: np.ndarray) -> np.ndarray:
        return xyz[:, 1] - np.interp(xyz[:, 0], self.s, self.y)

    def half_width_at(self, x: np.ndarray) -> np.ndarray:
        return np.interp(x, self.s, self.half_width)

    def inside(self, xyz: np.ndarray, height_above_rail: np.ndarray) -> np.ndarray:
        lat = np.abs(self.lateral(xyz))
        return (lat < self.half_width_at(xyz[:, 0])) \
            & (height_above_rail > self.bottom) & (height_above_rail < self.top)

    def clearance(self, xyz: np.ndarray) -> np.ndarray:
        """Запас до боковой границы габарита: <0 — точка внутри."""
        return np.abs(self.lateral(xyz)) - self.half_width_at(xyz[:, 0])


def rail_point_mask(xyz: np.ndarray, corridor: "Corridor", heights: np.ndarray,
                    gauge: float, tolerance: float = 0.12,
                    height_band: Tuple[float, float] = (-0.18, 0.10)) -> np.ndarray:
    """Точки самих рельсов внутри коридора.

    Их надо снимать отдельно, а не поднимать нижнюю границу габарита:
    рельсы лежат ровно на уровне УГР, кластеризация склеивает по ним
    предмет с путём на десятки метров, и «препятствие» уезжает по дальности
    к началу рельсовой нитки. Верхнюю границу поднимать нельзя — тогда
    исчезнут как раз низкие предметы между рельсами.
    """
    lateral = np.abs(corridor.lateral(xyz))
    on_rail = np.abs(lateral - 0.5 * gauge) < tolerance
    return on_rail & (heights > height_band[0]) & (heights < height_band[1])


def build_corridor(track: TrackModel, cfg: GaugeConfig) -> Corridor:
    s = np.arange(0.0, cfg.max_range + cfg.step, cfg.step)
    y = track.y_at(s)
    radius = track.curvature_radius
    # Вынос кузова в кривой: середина вагона смещается внутрь кривой,
    # свесы — наружу. Берём худший случай и расширяем габарит симметрично.
    widening = 0.0 if not np.isfinite(radius) else \
        min(0.5, cfg.car_length ** 2 / (8.0 * max(radius, 50.0)))
    half = np.full_like(s, cfg.half_width + cfg.lateral_margin + widening)
    return Corridor(s=s, y=y, half_width=half, top=cfg.height, bottom=cfg.bottom)
