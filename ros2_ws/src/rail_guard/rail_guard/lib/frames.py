"""Приведение облака к рабочей СК: X вперёд, Y влево, Z вверх (REP-103).

Лидар на поезде почти никогда не стоит осями по ходу движения: у записей
метрополитена продольная ось датчика — минус Y, у OSDaR23 — плюс X. Весь
алгоритм считает, что вперёд это X: по X идут обрезка рабочей зоны, ячейки
профиля пути, полосы прореживания. Облако в чужой СК не ломает пайплайн
громко — оно выбрасывается предфильтром почти целиком, и система рапортует
«путь свободен». Поэтому нормализация вынесена в отдельный этап с явной
диагностикой.

Ориентацию можно задать параметрами (`sensor.forward_axis`, `sensor.up_axis`),
а можно оставить `auto`: в тоннеле и на перегоне ориентация однозначно
восстанавливается из самого облака, и определяется она один раз — по первому
кадру, где данных достаточно.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import SensorConfig

AXIS_NAMES = ("x", "y", "z")
_AXIS_UNIT = {
    "x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
}
IDENTITY = np.eye(3, dtype=np.float32)


def axis_unit(name: str) -> np.ndarray:
    """Единичный вектор оси по имени вида `x`, `-y`."""
    key = name.strip().lower().replace("+", "")
    if key not in _AXIS_UNIT:
        raise ValueError(f"не ось: {name!r}; ожидается одно из {sorted(_AXIS_UNIT)}")
    return _AXIS_UNIT[key]


def rotation_from_axes(forward: str, up: str) -> np.ndarray:
    """Матрица M, для которой `xyz @ M` — облако в СК «X вперёд, Y влево, Z вверх».

    Столбцы M — это оси рабочей СК, выраженные в СК датчика. Влево получается
    векторным произведением: в правой тройке REP-103 Y = Z × X.
    """
    f = axis_unit(forward)
    u = axis_unit(up)
    if abs(float(np.dot(f, u))) > 1e-6:
        raise ValueError(f"оси «вперёд» ({forward}) и «вверх» ({up}) должны быть перпендикулярны")
    left = np.cross(u, f)
    return np.column_stack([f, left, u]).astype(np.float32)


def axes_name(vec: np.ndarray) -> str:
    """Обратное преобразование: вектор оси в имя (для сообщений в лог)."""
    idx = int(np.argmax(np.abs(vec)))
    return ("" if vec[idx] > 0 else "-") + AXIS_NAMES[idx]


def _axis_or_none(value) -> Optional[str]:
    """Имя оси или None, если ориентацию велено определить по кадру.

    `None` и `auto` означают здесь одно и то же — «измерить», как и во всех
    остальных полях профиля.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    return None if text in ("auto", "") else text


@dataclass
class FrameGuess:
    forward: str
    up: str
    confidence: float
    reason: str = ""

    @property
    def spec(self) -> str:
        return f"вперёд {self.forward}, вверх {self.up}"


def detect_axes(xyz: np.ndarray, min_points: int = 2000) -> Optional[FrameGuess]:
    """Определяет ориентацию датчика по геометрии облака.

    Опирается на два свойства любой записи с носа поезда, которые не зависят
    ни от среды, ни от модели лидара:

    * вертикаль — самая «сжатая» ось: тоннель и перегон тянутся на сотни
      метров вперёд и на десяток в стороны, но по высоте всегда единицы метров;
    * продольная ось — односторонняя: лидар смотрит вперёд, и почти все точки
      лежат по одну сторону от датчика, тогда как поперечная ось симметрична.

    Возвращает None, если признаки выражены слабо: тогда лучше оставить СК как
    есть и сказать об этом в лог, чем развернуть облако наугад.
    """
    if xyz.shape[0] < min_points:
        return None
    sample = xyz if xyz.shape[0] <= 200_000 else xyz[:: xyz.shape[0] // 200_000 + 1]
    finite = sample[np.isfinite(sample).all(axis=1)]
    # Нулевые точки — это «нет возврата», а не измерение в начале координат:
    # у Hesai в тоннеле их до 40 %, и они смещают все оценки к симметрии.
    finite = finite[np.abs(finite).sum(axis=1) > 1e-6]
    if finite.shape[0] < min_points:
        return None

    p01 = np.percentile(finite, 1, axis=0)
    p99 = np.percentile(finite, 99, axis=0)
    spread = p99 - p01
    up_idx = int(np.argmin(spread))

    best = None
    for idx in range(3):
        if idx == up_idx:
            continue
        col = finite[:, idx]
        positive = float((col > 0.0).mean())
        one_sided = abs(2.0 * positive - 1.0)          # 1.0 — всё по одну сторону
        reach = float(max(abs(p99[idx]), abs(p01[idx])))
        score = one_sided * reach
        if best is None or score > best[0]:
            best = (score, idx, one_sided, reach, 1.0 if positive >= 0.5 else -1.0)
    if best is None:
        return None
    _score, fwd_idx, one_sided, reach, fwd_sign = best

    # Порог намеренно мягкий: перепутать оси нельзя, а отказаться зря — можно.
    if one_sided < 0.7 or reach < 3.0 * max(spread[up_idx], 0.5):
        return FrameGuess(forward=("" if fwd_sign > 0 else "-") + AXIS_NAMES[fwd_idx],
                          up=AXIS_NAMES[up_idx], confidence=0.0,
                          reason=(f"односторонность {one_sided:.2f}, дальность {reach:.0f} м "
                                  f"при разбросе по вертикали {spread[up_idx]:.1f} м"))

    # Знак вертикали: снизу у облака жёсткая граница, сверху — хвост. Под
    # полотном отражаться нечему, поэтому распределение по вертикали упирается
    # в полотно и растягивается вверх — к стенам, своду, опорам. Признак не
    # зависит от того, где начало координат: на уровне датчика, как у лидара,
    # или на уровне пути, как в пересчитанных облаках.
    column = finite[:, up_idx]
    median = float(np.median(column))
    below = median - float(p01[up_idx])
    above = float(p99[up_idx]) - median
    if abs(above - below) > 0.15 * max(above + below, 1e-3):
        up_sign = 1.0 if above >= below else -1.0
    else:
        # Хвосты симметричны — редкий случай: полагаемся на то, что датчик
        # стоит над полотном и большая часть точек ниже него.
        up_sign = 1.0 if median <= 0.0 else -1.0
    confidence = float(min(1.0, one_sided) * min(1.0, reach / (10.0 * max(spread[up_idx], 0.5))))
    return FrameGuess(
        forward=("" if fwd_sign > 0 else "-") + AXIS_NAMES[fwd_idx],
        up=("" if up_sign > 0 else "-") + AXIS_NAMES[up_idx],
        confidence=max(confidence, 0.3),
        reason=f"односторонность {one_sided:.2f}, дальность {reach:.0f} м")


class FrameNormalizer:
    """Разворот облака в рабочую СК с фиксацией решения на всю запись.

    Автоопределение делается один раз: ориентация датчика в пределах проезда
    не меняется, а пересчёт по каждому кадру дал бы дрожание осей на кадрах,
    где тоннель пустой и признаки слабее.
    """

    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.matrix: Optional[np.ndarray] = None
        self.spec: str = ""
        self.source: str = ""
        self.auto_attempts = 0
        self._perm: Optional[np.ndarray] = None
        self._signs: Optional[np.ndarray] = None
        forward, up = _axis_or_none(cfg.forward_axis), _axis_or_none(cfg.up_axis)
        if forward is not None and up is not None:
            self._set_matrix(rotation_from_axes(forward, up),
                             f"вперёд {forward}, вверх {up}", "параметры")

    def _set_matrix(self, matrix: np.ndarray, spec: str, source: str) -> None:
        """Запоминает разворот сразу в виде перестановки со знаками.

        Матрица всегда знаковая перестановка, поэтому вместо матричного
        умножения достаточно выбрать столбцы и сменить знак: на облаке в
        900 тысяч точек это втрое дешевле, а результат тот же.
        """
        self.matrix = matrix
        self.spec = spec
        self.source = source
        self._perm = np.argmax(np.abs(matrix), axis=0)
        self._signs = np.array([matrix[self._perm[j], j] for j in range(3)], dtype=np.float32)

    @property
    def resolved(self) -> bool:
        return self.matrix is not None

    @property
    def axes(self):
        """Перестановка столбцов и знаки для приведения к рабочей СК."""
        if self._perm is None or self._signs is None:
            return None
        return self._perm, self._signs

    @property
    def forward_column(self):
        """Столбец исходного буфера, дающий продольную ось, и его знак.

        Нужен тому, кто разбирает сообщение: зная, где в сыром облаке лежит
        дальность, можно проредить ближнее поле и не тронуть дальнее — то
        самое, ради которого и борются за дальность обнаружения.
        """
        if self._perm is None or self._signs is None:
            return None
        return int(self._perm[0]), float(self._signs[0])

    @property
    def is_identity(self) -> bool:
        return self.matrix is not None and bool(np.allclose(self.matrix, IDENTITY))

    def apply(self, xyz: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Возвращает (облако в рабочей СК, было ли решение принято в этом вызове)."""
        decided_now = False
        if self.matrix is None:
            guess = detect_axes(xyz, self.cfg.auto_min_points)
            self.auto_attempts += 1
            if guess is not None and guess.confidence >= self.cfg.auto_min_confidence:
                self._set_matrix(rotation_from_axes(guess.forward, guess.up), guess.spec,
                                 f"автоопределение ({guess.reason})")
                decided_now = True
            elif self.auto_attempts >= self.cfg.auto_max_attempts:
                self._set_matrix(IDENTITY.copy(), "вперёд x, вверх z",
                                 "автоопределение не сработало"
                                 + (f": {guess.reason}" if guess is not None else ""))
                decided_now = True
            else:
                return xyz, False           # ждём кадр, по которому видно ориентацию

        if self.is_identity:
            return xyz, decided_now
        rotated = xyz[:, self._perm]
        if not np.allclose(self._signs, 1.0):
            rotated = rotated * self._signs
        return np.ascontiguousarray(rotated, dtype=np.float32), decided_now
