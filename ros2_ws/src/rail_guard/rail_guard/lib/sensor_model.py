"""Самокалибровка лидара по приходящим кадрам.

Всё, что алгоритму нужно знать о датчике — плотность лучей, угловое
разрешение, дальнобойность, — измеряется по самим облакам, а не задаётся
числом в профиле. Причина простая: число отражений от объекта определяет все
пороги детекции (сколько точек считать объектом, с какой дальности объект
вообще наблюдаем), и оно зависит от конкретного комплекта лидаров сильнее,
чем от чего-либо ещё. Вшитая константа плотности означает, что система
настроена на тот комплект, по которому её мерили, и молча деградирует на
любом другом: пороги окажутся либо недостижимыми (система ослепла), либо
достижимыми для шума (система тормозит по пыли).

Измеряемая величина — **плотность лучей в телесном угле**, точек на
стерадиан. Через неё число отражений от цели фронтальной площади A на
дальности r равно

    N = fill · k · A / r²

где k — измеренная плотность лучей (чистая геометрия: цель занимает
телесный угол A/r², в него попадает k·A/r² лучей), а fill — доля этих лучей,
возвращающая полезное отражение. Второй множитель нужен потому, что реальная
цель не есть сплошной прямоугольник своего габарита: человек занимает около
двух третей описанного прямоугольника, тёмная одежда и косой угол падения
съедают ещё часть. Измерение по разметке OSDaR23 (люди на 78 и 143 м) даёт
fill ≈ 0.5 — см. `scripts/measure_sensor.py`, которым это значение
перепроверяется на любом размеченном датасете.

**Где меряется плотность.** Не по всему полю зрения: у комплекта из
нескольких лидаров (OSDaR23 — Pandar64 на 360° плюс три Livox Tele-15 вперёд)
плотность в переднем секторе на порядок выше, чем по кругу, и средняя по
кадру величина не описывает ни одно реальное направление. Меряется сектор, в
котором вообще может находиться препятствие на пути, — угловой размер
габаритного коридора с запасом на уход оси в кривой. Он вычисляется из
профиля носителя (`beam_sector_for`), а не задаётся числом.

**Размер ячейки** сетки, по которой считается телесный угол, подбирается по
самому облаку: берётся наименьший, при котором в занятой ячейке набирается
несколько точек. Так оценка одинаково работает и для 16-лучевого лидара с
шагом в два градуса, и для сводного облака с нерегулярной сеткой. Лучи,
ушедшие в небо и не вернувшиеся, в знаменатель не попадают: считается
телесный угол только занятых ячеек.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np

# Размеры пробной ячейки сетки, от мелкой к крупной. Мельче 0.05° сканирующих
# лидаров не бывает; крупнее 3.2° в ячейку попадает уже не решётка лучей, а
# целый объект.
_CELL_STEPS_DEG = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)
# Сколько точек должно набраться в занятой ячейке, чтобы считать её
# описывающей решётку лучей, а не единичный луч. При меньшей населённости
# ячейка вырождается в «одна точка — одна ячейка», и оценка плотности
# сводится к обратному размеру ячейки, то есть к произволу.
_MIN_CELL_OCCUPANCY = 6
# Минимум точек в секторе, ниже которого оценке нельзя верить.
_MIN_SECTOR_POINTS = 800
# Дальность, начиная с которой препятствие должно попадать в измеряемый
# сектор. Ближе габаритный коридор занимает настолько широкий угол, что
# сектор перестал бы быть передним; при этом на такой дальности отражений
# всегда с избытком и пороги плотности ни на что не влияют.
SECTOR_REFERENCE_RANGE = 20.0


@dataclass
class SensorProfile:
    """Измеренные свойства датчика. Всё, что ниже, получено из облаков."""

    points_per_sr: float = 0.0     # плотность лучей в переднем секторе, точек/ср
    angular_step: float = 0.0      # рад, эквивалентный шаг решётки лучей
    max_range: float = 0.0         # наблюдаемая дальность, м
    sector: Tuple[float, float] = (0.0, 0.0)   # полуширина сектора (азимут, элевация), рад
    frames: int = 0                # по скольким кадрам измерено

    @property
    def valid(self) -> bool:
        return self.frames > 0 and self.points_per_sr > 0.0

    def expected_points(self, area: float, distance: float, fill: float = 1.0) -> float:
        """Сколько отражений ожидается от цели фронтальной площади `area`, м².

        `fill` — доля лучей, дающих полезный возврат (заполнение габаритного
        прямоугольника силуэтом цели и её отражательность).
        """
        if not self.valid:
            return 0.0
        return float(fill * self.points_per_sr * max(area, 0.0) / max(distance, 1.0) ** 2)

    def range_for_points(self, area: float, points: float, fill: float = 1.0) -> float:
        """Дальность, на которой от цели площади `area` останется `points`
        отражений. Обратная задача к expected_points: из неё получается
        предельная дальность, на которой система вправе заявлять обнаружение.
        """
        if not self.valid or points <= 0.0 or area <= 0.0:
            return 0.0
        return float(np.sqrt(fill * self.points_per_sr * area / points))

    def describe(self) -> str:
        if not self.valid:
            return "датчик не измерен"
        return (f"плотность лучей {self.points_per_sr:.3g} точек/ср "
                f"(шаг решётки {np.degrees(self.angular_step):.2f}°), "
                f"сектор ±{np.degrees(self.sector[0]):.0f}°/±{np.degrees(self.sector[1]):.0f}°, "
                f"дальность до {self.max_range:.0f} м, кадров {self.frames}")


def beam_sector_for(half_width: float, height: float, axis_uncertainty: float = 0.0,
                    reference_range: float = SECTOR_REFERENCE_RANGE,
                    grade: float = 0.05) -> Tuple[float, float]:
    """Угловое окно, в котором может находиться препятствие на пути.

    Считается из габарита носителя, а не подбирается: на опорной дальности
    коридор занимает по азимуту полуширину габарита плюс возможный уход оси
    в кривой, по элевации — высоту габарита плюс продольный уклон. Дальше
    опорной дальности окно только сужается, поэтому оценка плотности по нему
    консервативна — берётся чуть более широкая зона, то есть чуть меньшая
    плотность и чуть более строгие пороги.
    """
    ref = max(reference_range, 1.0)
    az = float(np.arctan2(max(half_width, 0.1) + max(axis_uncertainty, 0.0), ref))
    el = float(np.arctan2(max(height, 0.5) + grade * ref, ref))
    return az, el


def _forward_sector(xyz: np.ndarray, az_half: float, el_half: float
                    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Точки переднего сектора в сферических координатах.

    Невозвраты у многих лидаров записаны нулями, а не пропусками (в тоннеле
    их бывает до 40 % кадра). Нулевая точка — это «луч не вернулся»: считать
    её измерением значит завысить плотность и выдумать направление.
    """
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.any():
        return None
    pts = xyz[finite]
    r = np.linalg.norm(pts, axis=1)
    real = r > 1e-3
    if not real.any():
        return None
    pts, r = pts[real], r[real]
    az = np.arctan2(pts[:, 1], pts[:, 0])
    el = np.arcsin(np.clip(pts[:, 2] / r, -1.0, 1.0))
    sector = (np.abs(az) < az_half) & (np.abs(el) < el_half)
    if int(sector.sum()) < _MIN_SECTOR_POINTS:
        return None
    return r[sector], az[sector], el[sector]


def measure_beam_density(xyz: np.ndarray, az_half: float, el_half: float
                         ) -> Optional[SensorProfile]:
    """Измеряет плотность лучей переднего сектора по одному кадру.

    Возвращает None, если точек в секторе слишком мало: пустой кадр (закрытый
    датчик, участок без отражений, облако в чужой системе координат) о
    датчике ничего не говорит, и прежняя оценка лучше испорченной.
    """
    sector = _forward_sector(xyz, az_half, el_half)
    if sector is None:
        return None
    r, az, el = sector

    for step_deg in _CELL_STEPS_DEG:
        step = np.radians(step_deg)
        az_idx = np.floor(az / step).astype(np.int64)
        el_idx = np.floor(el / step).astype(np.int64)
        az_idx -= az_idx.min()
        el_idx -= el_idx.min()
        n_az = int(az_idx.max()) + 1
        counts = np.bincount(el_idx * n_az + az_idx)
        occupied = counts > 0
        if not occupied.any():
            continue
        if float(np.median(counts[occupied])) < _MIN_CELL_OCCUPANCY \
                and step_deg != _CELL_STEPS_DEG[-1]:
            continue
        # Телесный угол занятой части сектора: ячейки у горизонта шире, чем
        # у полюса, dΩ = cos(el)·daz·del.
        cell_el = (np.floor(np.arange(counts.size) / n_az) + 0.5) * step + el.min()
        solid = float((np.cos(cell_el[occupied]) * step * step).sum())
        if solid <= 0.0:
            continue
        density = float(r.size) / solid
        return SensorProfile(points_per_sr=density,
                             angular_step=float(np.sqrt(1.0 / density)),
                             max_range=float(np.percentile(r, 99.9)),
                             sector=(az_half, el_half), frames=1)
    return None


class SensorCalibrator:
    """Накапливает измерения датчика по кадрам и отдаёт устойчивую оценку.

    Плотность лучей — свойство установки, а не сцены, поэтому измерение
    делается по первым кадрам и затем освежается редко: считать его на каждом
    кадре значило бы тратить реальное время на величину, которая не меняется.
    Итог — медиана по измерениям: одиночный кадр в тупике или с закрытым
    датчиком не должен двигать пороги детекции.
    """

    def __init__(self, sector: Tuple[float, float],
                 warmup_frames: int = 5, refresh_every: int = 200,
                 history: int = 9):
        self.sector = sector
        self.warmup_frames = max(1, warmup_frames)
        self.refresh_every = max(0, refresh_every)
        self._density: Deque[float] = deque(maxlen=history)
        self._ranges: Deque[float] = deque(maxlen=history)
        self._frames = 0
        self._measured = 0
        self.profile = SensorProfile(sector=sector)

    @property
    def resolved(self) -> bool:
        """Есть ли хоть одно измерение — можно ли опираться на профиль."""
        return self.profile.valid

    @property
    def warm(self) -> bool:
        """Прогрет ли калибратор: измерений хватает для устойчивой медианы."""
        return self._measured >= self.warmup_frames

    def reset(self) -> None:
        self._density.clear()
        self._ranges.clear()
        self._frames = 0
        self._measured = 0
        self.profile = SensorProfile(sector=self.sector)

    def _due(self) -> bool:
        if self._measured < self.warmup_frames:
            return True
        if self.refresh_every <= 0:
            return False
        return self._frames % self.refresh_every == 0

    def update(self, xyz: np.ndarray) -> SensorProfile:
        """Обновляет профиль по очередному кадру и возвращает текущую оценку."""
        self._frames += 1
        if not self._due():
            return self.profile
        measured = measure_beam_density(xyz, self.sector[0], self.sector[1])
        if measured is None:
            return self.profile
        self._density.append(measured.points_per_sr)
        self._ranges.append(measured.max_range)
        self._measured += 1
        density = float(np.median(self._density))
        self.profile = SensorProfile(
            points_per_sr=density,
            angular_step=float(np.sqrt(1.0 / density)) if density > 0 else 0.0,
            max_range=float(np.median(self._ranges)),
            sector=self.sector, frames=self._measured)
        return self.profile
