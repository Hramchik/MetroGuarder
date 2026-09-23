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
    """Точки внутри рабочей зоны и вне габарита собственного поезда.

    Границы зоны по дальности и ширине обычно выводятся из дальнобойности
    датчика и габарита носителя (`lib/derive.py`). Если сюда пришёл профиль,
    в котором они ещё не выведены, зона не сужается вовсе: лучше обработать
    лишнее, чем молча выбросить дальнюю часть кадра — ровно ту, ради которой
    борются за дальность обнаружения.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    x_max = cfg.x_max if cfg.x_max is not None else np.inf
    y_abs_max = cfg.y_abs_max if cfg.y_abs_max is not None else np.inf
    keep = (
        (x > cfg.x_min) & (x < x_max)
        & (np.abs(y) < y_abs_max)
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


# Потолок на размер таблицы вокселей: 16 млн ячеек — это 64 МБ int32, дороже
# уже не окупается. Второе условие — плотность заполнения: раскладка по таблице
# выгодна, пока ячеек не в разы больше, чем точек (иначе платим за память, а не
# за сортировку). Оба порога проверяются до выделения памяти.
_MAX_VOXEL_CELLS = 16_000_000
_MAX_CELLS_PER_POINT = 32


def voxel_downsample(xyz: np.ndarray, leaf: float) -> np.ndarray:
    """Индексы представителей вокселей (первая точка в вокселе).

    Возвращает индексы, а не сами точки: вызывающему коду нужно тем же
    срезом проредить интенсивность и прочие поля.

    Представитель ищется раскладкой по таблице вокселей, а не сортировкой
    ключей: запись «минимального индекса» в таблицу линейна по числу точек,
    тогда как np.unique сортирует всё облако. На ближней полосе (160 тыс.
    точек в кадре 64-луча) это 10 мс вместо 25 при том же результате —
    четверть бюджета реального времени. Там, где таблица не по размеру
    (дальняя полоса с мелким вокселем: ячеек на два порядка больше, чем
    точек), остаётся сортировка.
    """
    n = xyz.shape[0]
    if leaf <= 0.0 or n == 0:
        return np.arange(n)
    # Решётка привязана к абсолютным координатам, а не к границам облака:
    # иначе её начало смещается от кадра к кадру вместе с крайней точкой, и
    # одна и та же поверхность каждый кадр прореживается по-новому. Для
    # сопровождения треков это чистое дрожание.
    inv = np.float32(1.0 / leaf) if xyz.dtype == np.float32 else 1.0 / leaf
    kx = np.floor(xyz[:, 0] * inv).astype(np.int32)
    ky = np.floor(xyz[:, 1] * inv).astype(np.int32)
    kz = np.floor(xyz[:, 2] * inv).astype(np.int32)
    kx -= kx.min(); ky -= ky.min(); kz -= kz.min()
    ny, nz = int(ky.max()) + 1, int(kz.max()) + 1
    n_cells = (int(kx.max()) + 1) * ny * nz
    table = 0 < n_cells <= min(_MAX_VOXEL_CELLS, _MAX_CELLS_PER_POINT * n)
    if table:
        flat = (kx * ny + ky) * nz + kz
        first = np.full(n_cells, -1, dtype=np.int32)
        # Запись в обратном порядке: для каждой ячейки последней запишется
        # (а значит, останется) точка с наименьшим индексом — то же
        # «первая точка в вокселе», что даёт np.unique.
        first[flat[::-1]] = np.arange(n - 1, -1, -1, dtype=np.int32)
        out = first[first >= 0]
        out.sort()
        return out
    flat = (kx.astype(np.int64) * ny + ky) * nz + kz
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


def limit_input(xyz: np.ndarray, intensity: Optional[np.ndarray],
                cfg: PreprocessConfig,
                axes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                max_points: Optional[int] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Прореживание входного облака до потолка по числу точек.

    Шаг берётся по индексу, а не по пространству: точки в облаке лидара идут
    по лучам, поэтому равномерный шаг прореживает угловое разрешение
    однородно и не выедает целиком ни один сектор. Нужно для 128-луча:
    920 тыс. точек в кадре не обрабатываются за период лидара на стендовом
    процессоре, а половина из них — переизбыточное ближнее поле.
    """
    n = xyz.shape[0]
    configured = cfg.max_input_points or 0
    limit = configured if max_points is None else max_points
    if limit <= 0 or n <= limit:
        return xyz, intensity
    if cfg.decimate_below > 0.0:
        # Дальность берём по продольной оси: если облако ещё не развёрнуто в
        # рабочую СК, её столбец и знак подсказывает нормализатор.
        if axes is not None:
            perm, signs = axes
            along = xyz[:, perm[0]] * signs[0]
        else:
            along = xyz[:, 0]
        far = along >= cfg.decimate_below
        budget = max(limit - int(far.sum()), limit // 4)
        near_count = n - int(far.sum())
        step = max(1, int(np.ceil(near_count / max(budget, 1))))
        keep = far
        keep[::step] = True
        return xyz[keep], (None if intensity is None else intensity[keep])
    step = int(np.ceil(n / limit))
    return xyz[::step], (None if intensity is None else intensity[::step])


def preprocess(xyz: np.ndarray, intensity: Optional[np.ndarray],
               cfg: PreprocessConfig) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Полный этап фильтрации. Возвращает (точки, интенсивность, индексы в исходном облаке)."""
    idx = np.where(workspace_mask(xyz, cfg) & intensity_mask(xyz, intensity, cfg))[0]
    if idx.size == 0:
        return xyz[:0], (intensity[:0] if intensity is not None else None), idx
    kept = idx[banded_downsample(xyz[idx], cfg)]
    return xyz[kept], (intensity[kept] if intensity is not None else None), kept
