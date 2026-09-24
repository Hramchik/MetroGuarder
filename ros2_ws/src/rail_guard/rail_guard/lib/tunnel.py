"""Модель «нормального тоннеля»: где в этом кадре свободное пространство.

Главный источник ложных торможений в тоннеле — не шум и не слабые кластеры,
а сама стена. Ось пути подтверждается рельсами на первых десятках метров, а
дальше идёт экстраполяция дуги; ошибка в метр-полтора на дальности, где
тоннель всего в четыре с половиной метра шириной, укладывает коридор габарита
на обделку, и та честно распознаётся как «объект в габарите».

Отличить стену от предмета по размеру нельзя: на 60 метрах от стены приходит
такой же десяток точек, как от человека. Но у стены есть свойство, которого
нет у предмета: она — граница свободного пространства. Поэтому тоннель
описывается так, как он и выглядит из кабины: для каждой полосы дальности и
каждого слоя высоты измеряется, насколько далеко вбок от оси пути уходит
свободное место. Точка у самой этой границы — обделка, кабельный лоток, край
платформы; точка, забравшаяся внутрь на метр и больше, — то, чего здесь быть
не должно.

Модель строится по каждому кадру заново и не помнит предыдущих: расширение
тоннеля перед платформой или гермозатвором тогда не приходится «доучивать»,
а на контрольной записи нечего переобучать. Единственное сглаживание —
продольное: предмет занимает одну-две полосы дальности, и скользящая медиана
по соседним полосам восстанавливает ту стену, которую предмет закрыл.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backend import sliding_median, to_host
from .config import TunnelConfig
from .scene import support_threshold

LEFT, RIGHT = 0, 1


@dataclass
class FreeSpace:
    """Полуширина свободного пространства по полосам дальности и слоям высоты."""

    width: np.ndarray            # (полосы, слои, 2): метры от оси пути до обделки
    x_min: float
    band: float
    h_min: float
    slice_h: float
    filled_cells: int = 0
    # Собственная неровность границы: медианное отклонение измеренной ширины
    # между соседними полосами дальности. Это шум самого измерения — обделка
    # не гладкая, отражений на дальней полосе считанные единицы. Заход
    # кластера внутрь свободного места имеет смысл сравнивать именно с ним:
    # порог, меньший этого разброса, ничего не доказывает.
    boundary_noise: float = 0.0

    @property
    def valid(self) -> bool:
        return self.width.size > 0 and self.filled_cells > 0

    def slack(self, x: np.ndarray, lateral: np.ndarray, height: np.ndarray) -> np.ndarray:
        """Запас до границы свободного пространства, м.

        Положительное значение — точка внутри свободного места (настолько
        глубоко); около нуля — точка лежит на самой обделке. Там, где модель
        ничего не знает (нет данных, точка вне сетки), возвращается `inf`:
        неизвестность не должна работать как признак препятствия.
        """
        n_bands, n_slices, _ = self.width.shape
        out = np.full(x.shape[0], np.inf, dtype=np.float32)
        if not self.valid:
            return out
        band = np.floor((x - self.x_min) / self.band).astype(np.int64)
        sl = np.floor((height - self.h_min) / self.slice_h).astype(np.int64)
        inside = (band >= 0) & (band < n_bands) & (sl >= 0) & (sl < n_slices)
        if not inside.any():
            return out
        side = np.where(lateral[inside] >= 0.0, LEFT, RIGHT)
        w = self.width[band[inside], sl[inside], side]
        out[inside] = (w - np.abs(lateral[inside])).astype(np.float32)
        np.nan_to_num(out, copy=False, nan=np.inf, posinf=np.inf)
        return out


def _cell_boundary(values, cell, n_cells: int,
                   boundary_points: int, min_points: int,
                   outlier_fraction: float, xp=np):
    """Докуда в этой ячейке простирается свободное место.

    Берётся край **основной массы** отражений слоя, а не самая дальняя точка.
    Разница принципиальная и стоила дорого. Слой высоты в сорок сантиметров у
    полотна заполнен полотном и путевым оборудованием; стена тоннеля в этом
    слое даёт считанные отражения, а сквозь ниши, проёмы и кабельные проходы
    прилетают ещё более редкие. Если считать границей самую дальнюю точку,
    граница уезжает на стену — и кабельный лоток в полутора метрах от оси
    оказывается «глубоко внутри свободного места», то есть предметом. На
    записях метро это давало экстренное торможение в трети кадров.

    Поэтому за границей допускается доля точек `outlier_fraction`: редкие
    возвраты сквозь проёмы и шум дальномера. Это инженерный допуск, а не
    подобранное под запись число: он отвечает на вопрос, какую часть
    отражений слоя мы готовы считать не принадлежащей его поверхности.

    Вторая защита — `boundary_points`: ячейка, где отражений меньше, границей
    не объявляется вовсе, чтобы одиночный возврат не становился поверхностью.
    Считается сортировкой со смещениями, без группового цикла: ячеек тысячи,
    а точек — сотни тысяч.
    """
    result = xp.full(n_cells, xp.nan, dtype=xp.float32)
    if values.size == 0:
        return result
    order = xp.lexsort(xp.stack((values, cell)))
    sorted_values = values[order]
    counts = xp.bincount(cell[order], minlength=n_cells)
    starts = xp.concatenate((xp.zeros(1, dtype=counts.dtype), xp.cumsum(counts)[:-1]))
    enough = counts >= max(min_points, boundary_points)
    if not bool(enough.any()):
        return result
    quantile = float(np.clip(1.0 - outlier_fraction, 0.5, 1.0))
    inside = xp.floor(quantile * (counts[enough] - 1)).astype(xp.int64)
    result[enough] = sorted_values[xp.clip(starts[enough] + inside,
                                           0, sorted_values.size - 1)]
    return result


def _nanmedian_where(values, axis: int, xp=np):
    """nanmedian без предупреждений на полностью пустых срезах.

    numpy ругается на срез из одних NaN, а такие срезы здесь нормальны: на
    высоте свода границы может не быть вовсе. Поэтому пустые срезы
    отбрасываются заранее, и медиана считается только там, где есть данные.
    """
    shape = tuple(v for i, v in enumerate(values.shape) if i != axis)
    out = xp.full(shape, xp.nan, dtype=xp.float32)
    has_data = xp.isfinite(values).any(axis=axis)
    if bool(has_data.any()):
        moved = xp.moveaxis(values, axis, -1)
        out[has_data] = xp.nanmedian(moved[has_data], axis=-1)
    return out


def _drop_unsupported_slices(width, band, n_bands: int,
                             cfg: TunnelConfig, min_support: float, xp=np):
    """Убирает слои, где «граница» встречается лишь в единичных полосах.

    Это и есть отличие обделки от предмета: стена, лоток и край платформы
    тянутся вдоль пути и попадают почти в каждую полосу дальности, а предмет
    занимает одну-две. Если слой высоты заполнен лишь местами, значит на этой
    высоте границы нет вовсе — и ограничивать там нечего.
    """
    per_band = xp.bincount(band, minlength=n_bands)
    # «Полоса с данными» — не та, где есть хоть что-то, а та, где точек хватает
    # на описание сечения. Иначе дальние полосы, куда дошло по десятку
    # отражений, считаются полноценными: слой высоты в них пуст, поддержка
    # слоя падает, и граница свободного места перестаёт признаваться границей —
    # в том числе там, где она видна отлично. Порог берётся от плотности самого
    # кадра, а не числом: у 64- и 128-луча она различается в три раза.
    filled = per_band[per_band > 0]
    threshold = max(cfg.min_cell_points,
                    int(cfg.band_density_fraction * float(to_host(xp.median(filled))))
                    if filled.size else 0)
    with_data = per_band >= threshold
    n_eff = int(with_data.sum())
    if n_eff == 0:
        return width
    support = xp.isfinite(width[with_data]).sum(axis=0) / float(n_eff)
    unsupported = support < min_support
    if bool(unsupported.any()):
        width[:, unsupported] = xp.nan
    return width


def _smooth_along_bands(width, radius: int, xp=np):
    """Скользящая медиана по полосам дальности с игнорированием пустых ячеек.

    Предмет закрывает собой обделку в своей полосе и занижает там измеренную
    ширину; медиана по соседним полосам возвращает ту границу, которая есть в
    действительности. Считается сдвигами (см. lib/backend.py): представление
    с произвольными шагами есть не во всех версиях cupy, а сдвиг работает
    одинаково на обоих бэкендах.
    """
    if radius <= 0:
        return width
    smoothed = sliding_median(width, radius, xp)
    # Полосы, где данных нет даже в окне, добираем медианой слоя по всему кадру:
    # тоннель вдоль пути однороден, и одна цифра на слой лучше, чем ничего.
    fallback = _nanmedian_where(width, axis=0, xp=xp)
    empty = xp.isnan(smoothed)
    if bool(empty.any()):
        smoothed = xp.where(empty, xp.broadcast_to(fallback, smoothed.shape), smoothed)
    return smoothed.astype(xp.float32)


def _boundary_noise(width, xp=np) -> float:
    """Собственная неровность границы: разброс ширины между соседними полосами.

    Обделка не гладкая, а на дальней полосе от неё приходят единицы
    отражений, поэтому измеренная граница гуляет сама по себе. Эта величина —
    масштаб, с которым имеет смысл сравнивать заход предмета внутрь
    свободного места: заход меньше собственного шума границы не доказывает
    ничего. Раньше на его месте стояло подобранное число (0.45 м).
    """
    if width.shape[0] < 2:
        return 0.0
    diff = xp.abs(xp.diff(width, axis=0))
    finite = diff[xp.isfinite(diff)]
    if finite.size == 0:
        return 0.0
    return float(to_host(xp.median(finite)))


def estimate_free_space(x, lateral, height,
                        cfg: TunnelConfig, x_max: float,
                        max_object_length: float = 8.0, xp=np) -> Optional[FreeSpace]:
    """Строит модель свободного пространства по облаку одного кадра.

    `lateral` и `height` — поперечное смещение от оси пути и высота над УГР,
    то есть та же система отсчёта, в которой работает габаритный коридор.
    Благодаря этому модель нечувствительна к уходу оси: и стена, и кластер
    смещаются вместе с коридором, а разница между ними сохраняется.

    `max_object_length` — длина, с которой кластер перестаёт быть предметом.
    Из неё выводится, в какой доле полос дальности слой высоты должен быть
    заполнен, чтобы считаться границей: предмет не может тянуться вдоль пути
    дальше собственной длины, а обделка тянется всегда.
    """
    # enabled=None — «решает сцена» (lib/derive.py). При прямом вызове этапа
    # решать нечему, и модель строится: она сама окажется пустой там, где
    # границ свободного места нет.
    if cfg.enabled is False or x.shape[0] == 0:
        return None
    band_len = cfg.band if cfg.band else max(max_object_length / 2.0, 1.0)
    height_max = cfg.height_max if cfg.height_max is not None else 4.0
    n_bands = int(np.ceil((x_max - cfg.x_min) / band_len))
    n_slices = int(np.ceil((height_max - cfg.height_min) / cfg.slice_height))
    if n_bands <= 0 or n_slices <= 0:
        return None

    band = xp.floor((x - cfg.x_min) / band_len).astype(xp.int64)
    sl = xp.floor((height - cfg.height_min) / cfg.slice_height).astype(xp.int64)
    keep = (band >= 0) & (band < n_bands) & (sl >= 0) & (sl < n_slices)
    if int(keep.sum()) < cfg.min_cell_points:
        return None
    band, sl = band[keep], sl[keep]
    lat = lateral[keep]
    side = xp.where(lat >= 0.0, LEFT, RIGHT).astype(xp.int64)
    cell = ((band * n_slices + sl) * 2 + side).astype(xp.int64)

    raw = _cell_boundary(xp.abs(lat).astype(xp.float32), cell, n_bands * n_slices * 2,
                         cfg.boundary_points, cfg.min_cell_points,
                         cfg.outlier_fraction, xp)
    width = raw.reshape(n_bands, n_slices, 2)
    support = cfg.min_band_support if cfg.min_band_support is not None \
        else support_threshold(max_object_length, band_len, n_bands)
    width = _drop_unsupported_slices(width, band, n_bands, cfg, support, xp)
    filled = int(xp.isfinite(width).sum())
    if filled == 0:
        return None
    noise = _boundary_noise(width, xp)
    width = _smooth_along_bands(width, cfg.smooth_bands, xp)
    # Сетка возвращается в память процессора: она мала (полосы x слои x две
    # стороны), а спрашивают её точки-кандидаты, которые считаются там же.
    return FreeSpace(width=to_host(width), x_min=cfg.x_min, band=band_len,
                     h_min=cfg.height_min, slice_h=cfg.slice_height, filled_cells=filled,
                     boundary_noise=noise)
