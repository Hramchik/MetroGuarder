"""Поиск оси пути по рельсам и построение коридора габарита.

Габарит имеет смысл только относительно того пути, по которому поезд
реально поедет. «Прямо вперёд» — неверный ответ: в кривой R=200 м ось
уходит на 10 м вбок уже к 70 метрам, и коридор либо накрывает соседний
путь (ложные тревоги), либо теряет собственный (пропуск препятствия).

Схема: рельсы ищутся ячейками по ходу движения («следование за парой»),
затем по найденным точкам подгоняется дуга y = b*x + a*x^2 с закреплённым
началом (поезд стоит на своём пути, значит y(0)=0).

Рельсы, однако, различимы всего на первых 30-40 м. Дальше ось ведёт сама
обделка: тоннель видно на сотню метров, путь лежит в нём с постоянным
смещением, и это смещение калибруется как раз там, где рельсы ещё видны
(estimate_tunnel_center). За обеими зонами ось — уже экстраполяция дуги, и
уверенность детекции там снижается.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import GaugeConfig, TrackConfig
from .ground import GroundModel


def search_half_width(cfg: TrackConfig) -> float:
    """Полоса поиска вокруг опорной оси.

    Обычно выведена из междупутья (lib/derive.py): поиск не должен дотягиваться
    до соседнего пути. Без вывода берётся доля междупутья прямо здесь — так
    прямой вызов этапа даёт то же, что и полный тракт.
    """
    if cfg.search_half_width is not None:
        return float(cfg.search_half_width)
    return float(np.clip(0.9 * cfg.spacing, 1.5, 6.0))


def guide_band(cfg: TrackConfig, max_object_length: float = 8.0) -> float:
    """Продольная полоса, в которой мерится труба тоннеля."""
    if cfg.guide_band is not None:
        return float(cfg.guide_band)
    return float(max_object_length + 2.0)


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
    guide_range: float = 0.0      # до какой дальности ось ведёт обделка тоннеля
    guide_source: str = ""        # чем именно: центр трубы, левая или правая стена
    # Измеренная ось за рельсовой зоной: центры полос тоннеля, приведённые к
    # оси пути. Это не аппроксимация, а measurement — поэтому за пределами
    # рельсов коридор идёт по ней, а не по продолжению дуги.
    guide_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    guide_y: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def y_at(self, x: np.ndarray | float) -> np.ndarray:
        """Положение оси пути на дальности x.

        До конца рельсовой зоны — дуга, подогнанная по рельсам. Дальше, если
        обделка тоннеля измерена, ось идёт по измеренным центрам: тоннель на
        сотне метров виден, а продолжение дуги на той же сотне уже вымысел —
        ошибка `s²/2R` растёт квадратично и на 100 м при R = 400 м составляет
        12 м. Стык сшивается по значению дуги в конце рельсовой зоны, чтобы
        коридор не имел излома.
        """
        x = np.asarray(x, dtype=np.float64)
        arc = self.lin * x + self.quad * x * x
        if self.guide_s.size < 2:
            return arc
        blend = np.clip(self.guide_s[-1], 0.0, None)
        measured = np.interp(x, self.guide_s, self.guide_y)
        return np.where((x > self.fit_range) & (x <= blend), measured, arc)

    def slope_at(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.lin + 2.0 * self.quad * x

    @property
    def curvature_radius(self) -> float:
        if abs(self.quad) < 1e-7:
            return float("inf")
        return float((1.0 + self.lin ** 2) ** 1.5 / (2.0 * abs(self.quad)))

    def confidence_at(self, x: np.ndarray | float, decay_range: float = 60.0,
                      guide_confidence: float = 0.85) -> np.ndarray:
        """Насколько можно доверять положению оси на дальности x.

        В пределах подтверждённой рельсами зоны — единица. Дальше, пока ось
        ведёт обделка тоннеля, уверенность понижена, но постоянна: труба
        известна не хуже рельсов, а вот привязка пути к ней калибрована на
        ближней зоне. За обеими зонами идёт чистая экстраполяция дуги, и
        уверенность падает: экстраполяция верна, пока кривая не сменилась
        переходной.
        """
        x = np.asarray(x, dtype=np.float64)
        known = max(self.fit_range, self.guide_range)
        beyond = np.maximum(0.0, x - known)
        decay = np.clip(1.0 - beyond / max(decay_range, 1.0), 0.15, 1.0)
        if self.guide_range > self.fit_range:
            guided = (x > self.fit_range) & (x <= self.guide_range)
            decay = np.where(guided, min(guide_confidence, 1.0), decay)
        return decay


def _cross_section_grid(x: np.ndarray, lateral: np.ndarray, height: np.ndarray,
                        cfg: TrackConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Поперечный профиль пути: сколько точек «рельсовой» высоты пришлось
    на каждую ячейку (продольная ячейка x поперечная полоса).

    Работать с профилем, а не с отдельными точками, нужно потому, что на
    реальных данных головка рельса поднимается над балластом всего на
    15-20 см и даёт считанные отражения: различим не сам рельс, а форма
    сечения — два бугра на полшага колеи и провал между ними.
    """
    half = search_half_width(cfg)
    x_edges = np.arange(cfg.near_start, cfg.max_fit_range + cfg.profile_bin, cfg.profile_bin)
    lat_edges = np.arange(-half, half + cfg.lat_bin, cfg.lat_bin)
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


def _band_quantile(band_index: np.ndarray, values: np.ndarray, side: np.ndarray,
                   n_bands: int, quantile: float, min_points: int) -> np.ndarray:
    """Квантиль |смещения| по полосам дальности для одной стороны.

    Возвращает NaN там, где точек меньше порога: полоса без стены — это не
    стена на нулевом расстоянии, а отсутствие измерения.
    """
    result = np.full(n_bands, np.nan)
    idx = np.where(side)[0]
    if idx.size == 0:
        return result
    sel_band = band_index[idx]
    sel_value = np.abs(values[idx])
    order = np.lexsort((sel_value, sel_band))
    sorted_values = sel_value[order]
    counts = np.bincount(sel_band[order], minlength=n_bands)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    enough = counts >= min_points
    if not enough.any():
        return result
    pos = starts[enough] + np.floor(0.01 * min(quantile, 100.0 - quantile) * 2.0
                                    * (counts[enough] - 1)).astype(np.int64)
    # Для левой стены берётся верхний квантиль |смещения|, для правой — тот же
    # по модулю: знак возвращается ниже вызывающим кодом.
    pos = starts[enough] + np.floor(
        (quantile if quantile >= 50.0 else 100.0 - quantile) / 100.0
        * (counts[enough] - 1)).astype(np.int64)
    result[enough] = sorted_values[np.clip(pos, 0, sorted_values.size - 1)]
    return result


def estimate_tunnel_center(x: np.ndarray, lateral: np.ndarray, height: np.ndarray,
                           cfg: TrackConfig, x_max: float
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Труба тоннеля по полосам дальности: центр и обе стены.

    Рельсы различимы на первых десятках метров — дальше головка рельса даёт
    считанные отражения. Обделка же видна на сотне метров и больше: это
    сплошная поверхность в упор к лучу. А путь лежит в трубе с постоянным
    смещением от неё — метро строится так, что тоннель и путь идут вместе.
    Значит тоннель может вести ось там, где рельсов уже не видно, и это не
    экстраполяция дуги, а измерение.

    Возвращает (центры полос по x, центр трубы, левая стена, правая стена) в
    той же поперечной системе, в которой пришёл `lateral`. Полосы, где трубы
    не видно (нет одной из стен, слишком широко, ширина не как у соседей),
    отбрасываются: лучше короткая достоверная зона, чем длинная выдуманная.
    """
    empty = np.zeros(0)
    band = guide_band(cfg)
    # Полоса поиска стен: обычно вдвое шире измеренного сечения (lib/derive.py),
    # без вывода — по полосе поиска пути с запасом на то, что стена дальше рельсов.
    max_half = cfg.guide_max_half_width if cfg.guide_max_half_width is not None \
        else 3.0 * search_half_width(cfg)
    n_bands = int(np.ceil(max(x_max, band) / band))
    wall = (height > cfg.guide_height_min) & (height < cfg.guide_height_max) \
        & (np.abs(lateral) < max_half) & (x > 0.0) & (x < n_bands * band)
    if int(wall.sum()) < 2 * cfg.guide_min_side_points:
        return empty, empty, empty, empty
    xb = x[wall]
    lb = lateral[wall]
    index = np.floor(xb / band).astype(np.int64)

    # Квантиль стены по каждой полосе — сортировкой, а не циклом по полосам:
    # полос до полутора десятков, но в каждой десятки тысяч точек, и питоновский
    # цикл с булевой маской по всему облаку стоил 11 мс на кадр.
    # Одной стены достаточно: вести ось можно и по ней, а вторая нужна только
    # для гипотезы «центр трубы». В двухпутном тоннеле дальняя стена
    # экранирована и на 60 м уже не видна — требовать обе значило бы обрывать
    # зону доверия там, где своя стена видна прекрасно.
    left_arr = _band_quantile(index, lb, lb > 0.0, n_bands, cfg.guide_quantile,
                              cfg.guide_min_side_points)
    right_arr = -_band_quantile(index, lb, lb < 0.0, n_bands, cfg.guide_quantile,
                                cfg.guide_min_side_points)
    have = np.isfinite(left_arr) | np.isfinite(right_arr)
    if int(have.sum()) < 2:
        return empty, empty, empty, empty
    xs_arr = (np.arange(n_bands) + 0.5) * band
    xs_arr, left_arr, right_arr = xs_arr[have], left_arr[have], right_arr[have]
    centres = 0.5 * (left_arr + right_arr)
    width = left_arr - right_arr
    # «Та же труба»: там, где видны обе стены, ширина полосы не должна
    # отличаться от ближней части тоннеля — так отбрасываются платформы,
    # камеры съездов и расширения перед гермозатвором. Там, где видна одна
    # стена, ширину мерить не по чему, и полоса проверяется дальше — по
    # гладкости самой стены (см. estimate_track).
    known = np.isfinite(width)
    same_tube = np.ones(xs_arr.size, dtype=bool)
    if known.any():
        reference = width[known]
        reference_width = float(np.median(reference[:max(2, reference.size // 3)]))
        same_tube[known] = np.abs(width[known] - reference_width) <= cfg.guide_max_width_jump
    return xs_arr[same_tube], centres[same_tube], \
        left_arr[same_tube], right_arr[same_tube]


def _guide_axis(xs: np.ndarray, observations: np.ndarray, rail_x: np.ndarray,
                rail_y: np.ndarray, cfg: TrackConfig) -> Tuple[float, float]:
    """Дуга y = b*x + a*x^2 по наблюдениям оси, начало закреплено в y(0)=0.

    Рельсовые ячейки весят больше стен: они дают положение самого пути, а
    стены — только то, куда идёт тоннель.
    """
    x_all = np.concatenate([xs, rail_x])
    y_all = np.concatenate([observations, rail_y])
    w = np.concatenate([np.ones(xs.size), np.full(rail_x.size, cfg.guide_rail_weight)])
    design = np.stack([x_all, x_all ** 2], axis=1) * w[:, None]
    solution, *_ = np.linalg.lstsq(design, y_all * w, rcond=None)
    return float(solution[0]), float(solution[1])


def _carry_forward(previous: TrackModel, travelled: float, cfg: TrackConfig) -> TrackModel:
    """Ось прошлого кадра, сдвинутая на пройденный путь.

    Кадр без узнаваемого сечения пути — не повод забыть, где путь. Зона
    доверия сокращается на то, что поезд проехал (плюс минимум за само
    неподтверждение), а измеренная по тоннелю ось переносится вперёд: то, что
    было измерено на 90 м, после полутора метров хода лежит на 88.5 м.
    """
    decay = max(travelled, cfg.fallback_decay_min)
    carried = TrackModel(lin=previous.lin, quad=previous.quad,
                         fit_range=max(0.0, previous.fit_range - decay),
                         from_rails=False, bin_x=np.zeros(0), bin_y=np.zeros(0),
                         bin_z=np.zeros(0),
                         guide_range=max(0.0, previous.guide_range - decay),
                         guide_source=previous.guide_source)
    if previous.guide_s.size >= 2 and carried.guide_range > 0.0:
        shifted = previous.guide_s - travelled
        keep = (shifted >= 0.0) & (shifted <= carried.guide_range)
        if int(keep.sum()) >= 2:
            # Начало координат уехало вперёд вместе с поездом, поэтому ось
            # отсчитывается от её значения в новой точке стояния.
            origin = float(np.interp(travelled, previous.guide_s, previous.guide_y)) \
                if travelled > 0.0 else 0.0
            carried.guide_s = shifted[keep]
            carried.guide_y = previous.guide_y[keep] - origin
        else:
            carried.guide_range, carried.guide_source = 0.0, ""
    return carried


def estimate_track(xyz: np.ndarray, intensity: Optional[np.ndarray], ground: GroundModel,
                   cfg: TrackConfig, previous: Optional[TrackModel] = None,
                   prior: Optional["TrackModel"] = None,
                   travelled: float = 0.0) -> TrackModel:
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
        & (np.abs(lateral) < search_half_width(cfg)) & (height > -0.3) & (height < 1.0)
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
        # Продление оси обделкой тоннеля. Рельсы кончились — труба продолжается,
        # и путь идёт в ней с тем же смещением, что и в ближней зоне. Смещение
        # калибруется там, где рельсы видны; если оно там непостоянно, значит
        # выбранная опора к пути не привязана (соседний путь, платформа) —
        # тогда тоннель ось не ведёт вовсе.
        guide_range, guide_source = 0.0, ""
        guide_len = guide_band(cfg)      # длина полосы, в которой мерится труба
        guide_axis: Optional[Tuple[np.ndarray, np.ndarray]] = None
        # guide_enabled=None — «решает сцена»; при прямом вызове ведение
        # пробуется и само не находит опоры там, где стен нет.
        if cfg.guide_enabled is not False and fit_range >= cfg.min_fit_range:
            gx, centre, left, right = estimate_tunnel_center(
                x_all, lateral, height, cfg,
                x_max=float(min(x_all.max(), 2.0 * cfg.max_fit_range)) if x_all.size else 0.0)
            inside = gx <= fit_range
            if gx.size >= 2 and int(inside.sum()) >= 2:
                rail_shift = best_lin * gx + best_quad * gx ** 2
                chosen = None
                for name, values in (("центр трубы", centre), ("левая стена", left),
                                     ("правая стена", right)):
                    delta = values[inside] - rail_shift[inside]
                    delta = delta[np.isfinite(delta)]
                    if delta.size < 2:
                        continue
                    spread = float(delta.max() - delta.min())
                    # Длина опоры важна не меньше её стабильности: стена,
                    # видная на 100 м, ведёт ось дальше, чем центр трубы,
                    # видный на 50. Поэтому при сравнимом разбросе выбирается
                    # та опора, которая измерена в большем числе полос.
                    span = int(np.isfinite(values).sum())
                    score = (spread, -span)
                    if spread <= cfg.guide_max_spread and (chosen is None or score < chosen[2]):
                        chosen = (name, float(np.median(delta)), score, values)
                if chosen is not None:
                    guide_source, offset, _score, values = chosen
                    # Ось по трубе: измеренные центры полос, приведённые к пути
                    # калиброванным смещением. Зона доверия — непрерывная
                    # цепочка полос, в которой труба остаётся той же и не
                    # ломается: между соседними полосами ось не может уйти
                    # вбок круче, чем позволяет минимальный радиус кривой.
                    measured = values - offset
                    guide_range = 0.0
                    previous_x = first_x = first_y = None
                    for band_x, band_y in zip(gx, measured):
                        if not np.isfinite(band_y):
                            continue                     # в этой полосе опоры нет
                        # Насколько труба вправе отойти от продолжения дуги.
                        # `measured` — отклонение трубы от рельсовой оси, и
                        # расти оно может ровно настолько, насколько путь
                        # вправе сменить кривизну: переход с прямой на кривую
                        # минимального для линии радиуса уводит путь на Δ²/2R.
                        # Труба, убегающая быстрее, — не наша: соседний
                        # тоннель, камера съезда или стена, принятая за свою.
                        # Прежде здесь стоял предел наклона между полосами, и
                        # его значение приходилось подбирать под тоннель.
                        if first_x is None:
                            first_x, first_y = band_x, band_y
                        else:
                            span = band_x - first_x
                            allowed = cfg.guide_max_slope * span \
                                if cfg.guide_max_slope is not None \
                                else span * span / (2.0 * max(cfg.min_radius, 1.0)) \
                                + cfg.guide_max_near_error
                            if band_x - previous_x > 2.5 * guide_len \
                                    or abs(band_y - first_y) > allowed:
                                break
                        guide_range = band_x + 0.5 * guide_len
                        previous_x = band_x
                    if guide_range <= fit_range:
                        guide_range, guide_source = 0.0, ""
                    kept = (gx <= guide_range) & np.isfinite(measured)
                    guided_lin, guided_quad = _guide_axis(
                        gx[kept], measured[kept], x_centers_arr[confirmed],
                        shift_best[confirmed], cfg) if (cfg.guide_refits_axis and kept.any()) \
                        else (best_lin, best_quad)
                    # Там, где рельсы видны, они и задают ось: труба вправе
                    # уточнить её только на дальности. Если подгонка уводит
                    # ближнюю зону, значит опора выбрана не по нашему пути —
                    # дуга остаётся рельсовой, а продлённая зона отменяется.
                    near = x_centers_arr[confirmed]
                    if near.size:
                        moved = np.abs((guided_lin - best_lin) * near
                                       + (guided_quad - best_quad) * near ** 2).max()
                    else:
                        moved = 0.0
                    if moved <= cfg.guide_max_near_error:
                        best_lin, best_quad = guided_lin, guided_quad
                        lin, quad = ref_lin + best_lin, ref_quad + best_quad
                    elif cfg.guide_refits_axis:
                        guide_range, guide_source = 0.0, ""
                    if guide_range > 0.0 and int(kept.sum()) >= 2:
                        # В общую СК: к смещению относительно опорной оси
                        # добавляем саму опорную. Стык с дугой сшивается по
                        # значению в конце рельсовой зоны — иначе коридор
                        # получил бы излом там, где кончаются рельсы.
                        s = gx[kept]
                        y = ref_lin * s + ref_quad * s ** 2 + measured[kept]
                        edge = max(fit_range, float(s[0]))
                        arc_edge = lin * edge + quad * edge ** 2
                        guide_s = np.concatenate(([edge], s[s > edge]))
                        guide_y = np.concatenate(([arc_edge], y[s > edge]))
                        if guide_s.size >= 2:
                            guide_y = guide_y + (arc_edge - guide_y[0])
                            guide_axis = (guide_s, guide_y)
                        else:
                            guide_range, guide_source = 0.0, ""

        model = TrackModel(lin=lin, quad=quad, fit_range=fit_range,
                           from_rails=fit_range >= cfg.min_fit_range,
                           bin_x=x_centers_arr[confirmed],
                           bin_y=lin * x_centers_arr[confirmed] + quad * x_centers_arr[confirmed] ** 2,
                           bin_z=np.zeros(0), n_bins=int(confirmed.sum()), score=best_score,
                           guide_range=guide_range, guide_source=guide_source,
                           guide_s=guide_axis[0] if guide_axis else np.zeros(0),
                           guide_y=guide_axis[1] if guide_axis else np.zeros(0))

    if model is None or not model.from_rails:
        if previous is not None:
            fallback = _carry_forward(previous, travelled, cfg)
            if model is None:
                return fallback
            model.lin, model.quad = fallback.lin, fallback.quad
            # Найденное в этом кадре короче того, что уже было известно:
            # зону доверия и измеренную ось сохраняем, иначе один кадр с
            # плохо распознанным сечением обнуляет обзор.
            if fallback.fit_range > model.fit_range:
                model.fit_range = fallback.fit_range
            if fallback.guide_range > model.guide_range:
                model.guide_range = fallback.guide_range
                model.guide_source = fallback.guide_source
                model.guide_s, model.guide_y = fallback.guide_s, fallback.guide_y
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

    @property
    def reach(self) -> float:
        """Дальность, до которой коридор вообще построен."""
        return float(self.s[-1]) if self.s.size else 0.0

    def inside(self, xyz: np.ndarray, height_above_rail: np.ndarray,
               lateral: Optional[np.ndarray] = None) -> np.ndarray:
        """Точки внутри габарита. `lateral` передаётся, если уже посчитан:
        интерполяция по оси коридора идёт по всему облаку, и считать её дважды
        в одном кадре — это лишние миллисекунды на каждом кадре."""
        lat = np.abs(self.lateral(xyz) if lateral is None else lateral)
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


def build_corridor(track: TrackModel, cfg: GaugeConfig,
                   max_range: Optional[float] = None) -> Corridor:
    # Коридор не строится дальше, чем ось пути вообще что-то знает о пути:
    # за подтверждённой рельсами зоной идёт экстраполяция дуги, и в тоннеле
    # ошибка оси в метр означает коридор, лежащий на обделке. Внешняя ось из
    # путевой карты поднимает fit_range сама — тогда ограничение не срабатывает.
    reach = cfg.max_range if max_range is None else min(cfg.max_range, max_range)
    if cfg.extrapolation_margin > 0.0:
        known = max(track.fit_range, track.guide_range)
        reach = min(reach, max(known + cfg.extrapolation_margin, 2.0 * cfg.step))
    s = np.arange(0.0, reach + cfg.step, cfg.step)
    y = track.y_at(s)
    radius = track.curvature_radius
    # Вынос кузова в кривой: середина вагона смещается внутрь кривой,
    # свесы — наружу. Берём худший случай и расширяем габарит симметрично.
    widening = 0.0 if not np.isfinite(radius) else \
        min(0.5, cfg.car_length ** 2 / (8.0 * max(radius, 50.0)))
    half = np.full_like(s, cfg.half_width + cfg.lateral_margin + widening)
    return Corridor(s=s, y=y, half_width=half, top=cfg.height, bottom=cfg.bottom)
