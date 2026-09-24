"""Фильтрация облака: обрезка рабочей зоны, отсечение собственного корпуса,
воксельное прореживание по полосам дальности, удаление одиночных точек.

Порядок этапов выбран из соображений цены: сначала дешёвые булевы маски,
которые убирают 60-80 % точек, и только потом всё, что требует KD-дерева.

Этап целиком состоит из поэлементных операций, сортировок и гистограмм,
поэтому он написан один раз и работает на любом бэкенде: модуль массивов
приходит аргументом `xp` (numpy или cupy, см. `lib/backend.py`). Это первый
из трёх этапов, вынесенных на видеокарту, и самый выгодный: он единственный
работает с полным кадром — до девятисот тысяч точек на 128-луче.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import PreprocessConfig


def workspace_mask(xyz, cfg: PreprocessConfig, xp=np):
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
        & (xp.abs(y) < y_abs_max)
        & (z > cfg.z_min) & (z < cfg.z_max)
    )
    ex0, ex1, ey0, ey1, ez0, ez1 = cfg.ego_box
    ego = (x > ex0) & (x < ex1) & (y > ey0) & (y < ey1) & (z > ez0) & (z < ez1)
    return keep & ~ego


def intensity_mask(xyz, intensity, cfg: PreprocessConfig, xp=np):
    """Слабые дальние возвраты — почти всегда пыль, капли или паразитная засветка."""
    if intensity is None or cfg.min_intensity_far <= 0.0:
        return xp.ones(xyz.shape[0], dtype=bool)
    far = xyz[:, 0] > cfg.intensity_far_range
    return ~(far & (intensity < cfg.min_intensity_far))


# Потолок на размер таблицы вокселей: 16 млн ячеек — это 64 МБ int32, дороже
# уже не окупается. Второе условие — плотность заполнения: раскладка по таблице
# выгодна, пока ячеек не в разы больше, чем точек (иначе платим за память, а не
# за сортировку). Оба порога проверяются до выделения памяти.
_MAX_VOXEL_CELLS = 16_000_000
_MAX_CELLS_PER_POINT = 32


def voxel_downsample(xyz, leaf: float, xp=np):
    """Индексы представителей вокселей (первая точка в вокселе).

    Возвращает индексы, а не сами точки: вызывающему коду нужно тем же
    срезом проредить интенсивность и прочие поля.

    На процессоре представитель ищется раскладкой по таблице вокселей: запись
    «минимального индекса» в таблицу линейна по числу точек, тогда как
    np.unique сортирует всё облако. На ближней полосе (160 тыс. точек в кадре
    64-луча) это 10 мс вместо 25 при том же результате.

    На видеокарте та же раскладка непригодна: при записи в одну ячейку из
    разных потоков выигрывает произвольный, и представитель вокселя менялся бы
    от запуска к запуску. Поэтому там всегда сортировка — она даёт ровно тот
    же ответ, что и процессорная ветка, а результат тракта не должен зависеть
    от того, где его посчитали.
    """
    n = xyz.shape[0]
    if leaf <= 0.0 or n == 0:
        return xp.arange(n)
    # Решётка привязана к абсолютным координатам, а не к границам облака:
    # иначе её начало смещается от кадра к кадру вместе с крайней точкой, и
    # одна и та же поверхность каждый кадр прореживается по-новому. Для
    # сопровождения треков это чистое дрожание.
    inv = np.float32(1.0 / leaf) if xyz.dtype == np.float32 else 1.0 / leaf
    kx = xp.floor(xyz[:, 0] * inv).astype(xp.int32)
    ky = xp.floor(xyz[:, 1] * inv).astype(xp.int32)
    kz = xp.floor(xyz[:, 2] * inv).astype(xp.int32)
    kx -= kx.min(); ky -= ky.min(); kz -= kz.min()
    ny, nz = int(ky.max()) + 1, int(kz.max()) + 1
    n_cells = (int(kx.max()) + 1) * ny * nz
    table = xp is np and 0 < n_cells <= min(_MAX_VOXEL_CELLS, _MAX_CELLS_PER_POINT * n)
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
    flat = (kx.astype(xp.int64) * ny + ky) * nz + kz
    _uniq, first = xp.unique(flat, return_index=True)
    return xp.sort(first)


def banded_downsample(xyz, cfg: PreprocessConfig, xp=np):
    """Прореживание с шагом, зависящим от дальности.

    Ближнее поле избыточно плотное — его режем крупным вокселем; дальнее
    поле (ради которого и нужна дальность детекции) не трогаем вовсе.
    """
    if xyz.shape[0] == 0:
        return xp.arange(0)
    rng = xyz[:, 0]
    result = []
    lower = 0.0
    for upper, leaf in cfg.voxel_bands:
        band = xp.where((rng >= lower) & (rng < upper))[0]
        if band.size:
            result.append(band[voxel_downsample(xyz[band], leaf, xp)])
        lower = upper
    if not result:
        return xp.arange(xyz.shape[0])
    return xp.sort(xp.concatenate(result))


def radius_outlier_mask(xyz: np.ndarray, radius: float = 0.6,
                        min_neighbors: int = 3,
                        radius_per_meter: float = 0.006) -> np.ndarray:
    """Удаление одиночных точек (пыль, капли, интерференция лидаров).

    Радиус растёт с дальностью: на 150 м соседние точки того же объекта
    физически расходятся, и фиксированный радиус выкосил бы реальную цель.
    Считается через k-го соседа, а не через подсчёт в шаре — так на порядок
    дешевле и не зависит от плотности.

    Остаётся на процессоре: работает по точкам-кандидатам, а их сотни, и
    KD-дерево на таком объёме быстрее любого переноса на устройство.
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


def limit_input(xyz, intensity, cfg: PreprocessConfig,
                axes: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                max_points: Optional[int] = None, xp=np) -> Tuple:
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
            along = xyz[:, int(perm[0])] * float(signs[0])
        else:
            along = xyz[:, 0]
        far = along >= cfg.decimate_below
        budget = max(limit - int(far.sum()), limit // 4)
        near_count = n - int(far.sum())
        step = max(1, int(np.ceil(near_count / max(budget, 1))))
        keep = far.copy()
        keep[::step] = True
        return xyz[keep], (None if intensity is None else intensity[keep])
    step = int(np.ceil(n / limit))
    return xyz[::step], (None if intensity is None else intensity[::step])


def preprocess(xyz, intensity, cfg: PreprocessConfig, xp=np) -> Tuple:
    """Полный этап фильтрации. Возвращает (точки, интенсивность, индексы в исходном облаке)."""
    idx = xp.where(workspace_mask(xyz, cfg, xp) & intensity_mask(xyz, intensity, cfg, xp))[0]
    if idx.size == 0:
        return xyz[:0], (intensity[:0] if intensity is not None else None), idx
    kept = idx[banded_downsample(xyz[idx], cfg, xp)]
    return xyz[kept], (intensity[kept] if intensity is not None else None), kept
