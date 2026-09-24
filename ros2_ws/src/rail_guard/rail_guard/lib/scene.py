"""Распознавание сцены по кадру: замкнутое сечение или открытый перегон.

От ответа на этот вопрос зависят три механизма тракта:

* модель свободного пространства (`lib/tunnel.py`) — главное средство против
  ложных торможений на обделке; на открытом перегоне границ нет, строить её
  незачем;
* ведение оси пути обделкой (`lib/track.py`) — работает только в трубе;
* насколько далеко за подтверждённой рельсами зоной можно строить коридор:
  в тоннеле шириной четыре с половиной метра ошибка оси в метр укладывает
  коридор на стену, на перегоне тот же метр ничего не задевает.

Раньше это задавалось профилем: `tunnel.enabled: true` для метро,
`false` для магистральной линии. Но профиль пишется заранее и одинаков на
всю запись, а поезд метро выезжает на открытый участок, магистральный поезд
въезжает в тоннель, и обе системы в этот момент работают не по той модели
сцены, в которой едут. Поэтому сцена определяется по каждому кадру.

**Главный признак замкнутости — непрерывность границы.** Абсолютный порог по
ширине («стена ближе восьми метров») пришлось бы подбирать под конкретный
тоннель. Вместо него проверяется то, что отличает обделку от чего угодно
другого: она тянется вдоль пути непрерывно и попадает почти в каждую полосу
дальности с обеих сторон, тогда как предмет, опора или край платформы
занимают одну-две. Порог доли полос выводится из максимальной длины
предмета, который система обязана считать препятствием
(`objects.max_length`): граница обязана присутствовать везде, кроме участка,
который таким предметом может быть закрыт.

**Второй признак — близость.** Насыпь и лесополоса по краям выемки тоже дают
непрерывную границу, но в десятке метров от оси: коридор её не достанет
никогда, и тоннельные механизмы там только тратят время. Поэтому граница
учитывается, лишь пока она ближе междупутья — расстояния, на котором
начинается соседний путь: всё, что дальше, для габарита безразлично.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from .backend import to_host

TUNNEL = "tunnel"
OPEN = "open"
UNKNOWN = "unknown"


@dataclass
class SceneModel:
    """Что видно в этом кадре: труба вокруг пути или открытое пространство."""

    kind: str = UNKNOWN
    # Полуширина свободного пространства по бокам от оси пути, м (nan — не измерено).
    half_width: float = float("nan")
    # Доля полос дальности, в которых боковая граница видна с обеих сторон.
    wall_support: float = 0.0
    # Высота перекрытия над УГР, м (nan — перекрытия нет).
    ceiling_height: float = float("nan")
    # Доля полос дальности, в которых перекрытие видно: свод тянется вдоль
    # пути, а контактная сеть и кроны деревьев — нет.
    ceiling_support: float = 0.0
    # Разброс ширины трубы относительно её медианы: у тоннеля сечение постоянно.
    width_spread: float = float("nan")
    bands_measured: int = 0

    @property
    def enclosed(self) -> bool:
        return self.kind == TUNNEL

    @property
    def has_ceiling(self) -> bool:
        return bool(np.isfinite(self.ceiling_height))

    def describe(self) -> str:
        if self.kind == UNKNOWN:
            return "сцена не определена"
        if self.kind == OPEN:
            return (f"открытый участок (боковая граница в "
                    f"{self.wall_support * 100:.0f} % полос)")
        return (f"замкнутое сечение: полуширина {self.half_width:.1f} м, "
                f"граница в {self.wall_support * 100:.0f} % полос"
                + (f", перекрытие на {self.ceiling_height:.1f} м"
                   if self.has_ceiling else ", без перекрытия"))


def _side_quantile(band, values, n_bands: int,
                   quantile: float, min_points: int, xp=np):
    """Квантиль |смещения| по полосам дальности; NaN там, где точек мало.

    Полоса без данных — это отсутствие измерения, а не стена на нулевом
    расстоянии, и сливать эти два случая нельзя.
    """
    result = xp.full(n_bands, xp.nan)
    if values.size == 0:
        return result
    order = xp.lexsort(xp.stack((values, band)))
    sorted_values = values[order]
    counts = xp.bincount(band[order], minlength=n_bands)
    starts = xp.concatenate((xp.zeros(1, dtype=counts.dtype), xp.cumsum(counts)[:-1]))
    enough = counts >= min_points
    if not bool(enough.any()):
        return result
    pos = starts[enough] + xp.floor(quantile * (counts[enough] - 1)).astype(xp.int64)
    result[enough] = sorted_values[xp.clip(pos, 0, sorted_values.size - 1)]
    return result


def support_threshold(max_object_length: float, band: float, n_bands: int) -> float:
    """Доля полос, начиная с которой протяжённость признаётся конструкцией.

    Предмет длиной `max_object_length` закрывает собой не больше, чем
    `max_object_length / band` полос (плюс одна — предмет может лечь на
    границу полос). Всё, что присутствует в большем числе полос, тянется
    вдоль пути и предметом быть не может. Порог не подбирается, а следует
    из того, какой предмет система обязана считать препятствием.
    """
    if n_bands <= 0 or band <= 0.0:
        return 1.0
    object_bands = max_object_length / band + 1.0
    return float(np.clip(object_bands / n_bands, 0.15, 0.9))


def enclosure_threshold(max_object_length: float, band: float, n_bands: int) -> float:
    """Доля полос с границей, при которой сечение считается замкнутым.

    Требование строже, чем «длиннее предмета»: обделка идёт вдоль пути
    непрерывно и пропадает только на нишах и камерах съездов. От границы
    поэтому требуется присутствовать везде, кроме участка, который способен
    закрыть собой самый длинный допустимый предмет.

    Верхний предел — 0.7, а не единица: на дальних полосах стена разрешается
    хуже, чем на ближних, и требовать её во всех полосах значило бы не
    признавать тоннелем ни один настоящий тоннель.
    """
    if n_bands <= 0 or band <= 0.0:
        return 1.0
    object_bands = max_object_length / band + 1.0
    return float(np.clip(1.0 - object_bands / n_bands, 0.4, 0.7))


def classify_scene(x: np.ndarray, lateral: np.ndarray, height: np.ndarray,
                   gauge_height: float, max_object_length: float = 8.0,
                   band: float = 10.0, max_range: float = 0.0,
                   min_side_points: int = 12, search_half_width: float = 20.0,
                   quantile: float = 0.9,
                   relevant_half_width: float = 0.0,
                   corridor_half_width: float = 0.0, xp=np) -> SceneModel:
    """Определяет тип сцены по одному кадру.

    `lateral` и `height` — поперечное смещение от оси пути и высота над УГР:
    та же система отсчёта, в которой работает габаритный коридор, поэтому
    результат не зависит от того, куда ушла ось в кривой.

    `gauge_height` — высота габарита носителя: перекрытием считается то, что
    находится над ним, а не над произвольно выбранной отметкой.

    `corridor_half_width` — полуширина габаритного коридора. Граница, лежащая
    ближе двух его полуширин, сама по себе означает трубу: свободного места
    по бокам остаётся меньше, чем сам габарит, и никакой другой сцены с такой
    геометрией не бывает. Этот признак заменяет свод там, где свод виден
    плохо — в круглом тоннеле он идёт почти по габариту и в отдельных кадрах
    даёт слишком мало отражений, чтобы считаться непрерывным.

    `relevant_half_width` — расстояние, дальше которого боковая граница уже не
    влияет на габаритный коридор (обычно междупутье). Стена в полутора метрах
    от габарита и лесополоса в двенадцати метрах — это разные вещи: первую
    коридор при малейшей ошибке оси принимает за препятствие, вторую не
    достанет никогда, и тоннельные механизмы там только тратят время.
    """
    if x.size == 0:
        return SceneModel()
    reach = float(max_range if max_range > 0.0 else to_host(x.max()))
    n_bands = int(np.ceil(reach / band))
    if n_bands < 2:
        return SceneModel()

    # Слой стен: выше полотна и путевого оборудования, но ниже перекрытия.
    # Границы берутся от габарита носителя, а не от абсолютных отметок —
    # так признак переносится на любую линию вместе с профилем.
    wall_low, wall_high = 0.4 * gauge_height, 0.9 * gauge_height
    in_wall = (height > wall_low) & (height < wall_high) \
        & (xp.abs(lateral) < search_half_width) & (x > 0.0) & (x < n_bands * band)
    if int(in_wall.sum()) < 2 * min_side_points:
        return SceneModel(kind=OPEN, bands_measured=n_bands)

    idx = xp.floor(x[in_wall] / band).astype(xp.int64)
    lat = lateral[in_wall]
    left = _side_quantile(idx[lat > 0.0], xp.abs(lat[lat > 0.0]), n_bands,
                          quantile, min_side_points, xp)
    right = _side_quantile(idx[lat < 0.0], xp.abs(lat[lat < 0.0]), n_bands,
                           quantile, min_side_points, xp)

    both = xp.isfinite(left) & xp.isfinite(right)
    # Для признака замкнутости достаточно границы с одной стороны: в
    # двухпутном тоннеле дальняя стена экранирована составом и на шестидесяти
    # метрах уже не видна, и требовать обе значило бы не признать тоннелем
    # ровно тот случай, ради которого всё и делается. Отличить такую стену от
    # лесополосы позволяет второй признак — свод над путём.
    either = xp.isfinite(left) | xp.isfinite(right)
    # Полоса «с данными» — не та, где есть хоть что-то, а та, где точек хватает
    # на описание сечения. Дальние полосы, куда дошло по десятку отражений,
    # стену не разрешают, и считать их свидетельством против замкнутости
    # нельзя: иначе круглый тоннель с перекрытием в четырёх метрах над путём
    # оказывается «открытым участком» просто потому, что на ста пятидесяти
    # метрах его стен уже не видно.
    counts = xp.bincount(xp.floor(x[(x > 0.0) & (x < n_bands * band)] / band
                                  ).astype(xp.int64), minlength=n_bands)
    filled = counts[counts > 0]
    floor = 0.25 * float(to_host(xp.median(filled))) if filled.size else 0.0
    populated = counts >= max(floor, float(min_side_points))
    n_populated = max(int(populated.sum()), 1)
    support = float(to_host((either & populated).sum())) / n_populated

    threshold = enclosure_threshold(max_object_length, band, n_populated)
    widths = to_host((left + right)[both])
    half_width = float(np.median(widths) / 2.0) if widths.size else float("nan")
    spread = float(np.median(np.abs(widths - np.median(widths))) / max(np.median(widths), 1e-3)) \
        if widths.size >= 2 else float("nan")

    # Перекрытие: не отдельные точки над путём, а свод, идущий вдоль него.
    # Контактная сеть и кроны деревьев тоже дают точки выше габарита, но лишь
    # в отдельных полосах — именно этим свод от них и отличается.
    ceiling = float("nan")
    ceiling_support = 0.0
    if np.isfinite(half_width):
        above = (height > gauge_height) & (xp.abs(lateral) < half_width + 1.0) \
            & (x > 0.0) & (x < n_bands * band)
        if int(above.sum()) >= min_side_points:
            over = xp.bincount(xp.floor(x[above] / band).astype(xp.int64),
                               minlength=n_bands) >= min_side_points
            ceiling_support = float(to_host((over & populated).sum())) / n_populated
            if ceiling_support > 0.0:
                ceiling = float(to_host(xp.percentile(height[above], 10)))

    # Замкнутое сечение — это непрерывная боковая граница плюс одно из двух:
    # либо свод над путём, либо сама теснота. Одной границы мало — её даёт и
    # выемка, и лесополоса, и платформа; но граница, подошедшая ближе двух
    # полуширин габарита, — это уже обделка, чем бы она ни была.
    tight = bool(corridor_half_width > 0.0 and np.isfinite(half_width)
                 and half_width <= 2.0 * corridor_half_width)
    enclosed = (support >= threshold and np.isfinite(half_width)
                and (ceiling_support >= threshold or tight))
    if enclosed and relevant_half_width > 0.0 and half_width > relevant_half_width:
        # Границы есть, но они слишком далеко, чтобы коридор их задел: это
        # выемка или просека, а не сечение, в габаритах которого идёт путь.
        enclosed = False
    return SceneModel(kind=TUNNEL if enclosed else OPEN,
                      half_width=half_width, wall_support=support,
                      ceiling_height=ceiling, ceiling_support=ceiling_support,
                      width_spread=spread, bands_measured=n_populated)


class SceneTracker:
    """Решение о типе сцены по нескольким кадрам подряд.

    Переключать механизмы тракта по каждому кадру нельзя: на въезде в тоннель,
    у платформы и на стрелке признаки замкнутости то появляются, то пропадают,
    а вместе с ними менялись бы и длина коридора, и набор работающих фильтров.
    Решение принимается по большинству последних кадров, поэтому смена режима
    требует устойчивого изменения сцены, а не одного удачного кадра.
    """

    def __init__(self, window: int = 7):
        self.window = max(1, window)
        self._votes: Deque[str] = deque(maxlen=self.window)
        self._widths: Deque[float] = deque(maxlen=self.window)
        self.current = SceneModel()

    def reset(self) -> None:
        self._votes.clear()
        self._widths.clear()
        self.current = SceneModel()

    def update(self, model: SceneModel) -> SceneModel:
        if model.kind != UNKNOWN:
            self._votes.append(model.kind)
        if np.isfinite(model.half_width):
            self._widths.append(model.half_width)
        if not self._votes:
            self.current = model
            return self.current
        enclosed = sum(1 for kind in self._votes if kind == TUNNEL)
        kind = TUNNEL if enclosed * 2 > len(self._votes) else OPEN
        half_width = float(np.median(self._widths)) if self._widths else float("nan")
        self.current = SceneModel(
            kind=kind,
            half_width=half_width if kind == TUNNEL else model.half_width,
            wall_support=model.wall_support, ceiling_height=model.ceiling_height,
            ceiling_support=model.ceiling_support, width_spread=model.width_spread,
            bands_measured=model.bands_measured)
        return self.current

    @property
    def enclosed(self) -> bool:
        return self.current.enclosed

    @property
    def settled(self) -> bool:
        """Набралось ли кадров, чтобы решение считалось установившимся."""
        return len(self._votes) >= self.window
