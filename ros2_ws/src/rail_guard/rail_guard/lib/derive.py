"""Вывод рабочих параметров из физики носителя и измерений.

Здесь собрано всё, что раньше стояло числами в профиле и подбиралось по
записям: пороги числа точек, радиус связности кластеризации, длина
экстраполяции коридора, рабочая зона, полосы модели свободного пространства.
Ни одно из этих чисел не является свойством конкретной записи — каждое
выводится из того, что либо задано регламентом (габарит, колея, замедления,
минимальная цель), либо измерено по самим кадрам (плотность лучей,
дальнобойность, тип сцены).

Правило простое: поле профиля со значением `None` означает «вывести», и
именно вывод, а не подбор, — у каждой величины ниже есть формула и причина.
Значение, заданное в профиле числом, не трогается никогда: человек вправе
подавить автоматику, но тогда это его осознанное решение, а не умолчание.

Что откуда берётся:

* **дальность рабочей зоны и коридора** — наблюдаемая дальнобойность датчика,
  ограниченная тем, дальше чего тормозить всё равно не по чему;
* **пороги числа точек** — ожидаемое число отражений от минимальной цели на
  этой дальности (`lib/sensor_model.py`), умноженное на долю, которую
  система согласна принять за наблюдение;
* **связность кластера** — угловой шаг решётки лучей: на дальности r соседние
  лучи расходятся на r·Δφ, и связность обязана это покрывать;
* **длина экстраполяции коридора** — минимальный радиус кривой линии: за
  подтверждённой зоной ось известна с ошибкой Δ²/2R, и коридор строится
  ровно до той дальности, где эта ошибка ещё меньше его полуширины;
* **режимы тоннельных механизмов** — распознанная сцена (`lib/scene.py`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import PipelineConfig
from .scene import UNKNOWN, SceneModel
from .sensor_model import SensorProfile, beam_sector_for


@dataclass
class DerivedNote:
    """Одно выведенное значение — что, сколько и почему."""
    key: str
    value: object
    why: str

    def __str__(self) -> str:
        return f"{self.key} = {self.value} ({self.why})"


def min_target_area(cfg: PipelineConfig) -> float:
    """Фронтальная площадь минимальной цели, м²: её и «видит» лидар."""
    return float(max(cfg.safety.min_target_width, 1e-3)
                 * max(cfg.safety.min_target_height, 1e-3))


def braking_range(cfg: PipelineConfig) -> float:
    """Тормозной путь при эксплуатационной скорости линии, м.

    Считается по экстренному замедлению — той же формулой, что и решение
    (`lib/decision.py`), чтобы рабочая зона и решение опирались на одну
    физику, а не на два независимо подобранных числа.
    """
    v = max(cfg.decision.default_speed, 0.0)
    decel = max(cfg.decision.emergency_decel, 0.1)
    return float(v * cfg.decision.reaction_time + v * v / (2.0 * decel))


def detection_limit(cfg: PipelineConfig, sensor: SensorProfile) -> float:
    """Дальность, на которой минимальная цель ещё даёт измеримый кластер.

    Физический предел системы: дальше цель размером с `safety.min_target_*`
    не даёт даже минимального числа отражений, и никакая обработка этого не
    исправит. Величина публикуется в диагностике — это честный ответ на
    вопрос «с какой дальности вы видите препятствие», не зависящий от того,
    на какой записи его задали.
    """
    if not sensor.valid:
        return 0.0
    return sensor.range_for_points(min_target_area(cfg),
                                   max(cfg.cluster.abs_min_points, 1),
                                   cfg.safety.target_fill)


def sensor_sector(cfg: PipelineConfig) -> tuple:
    """Угловое окно, в котором меряется плотность лучей, — из габарита."""
    return beam_sector_for(half_width=cfg.gauge.half_width + cfg.gauge.lateral_margin,
                           height=cfg.gauge.height,
                           axis_uncertainty=cfg.track.max_axis_shift,
                           grade=cfg.ground.max_slope)


def _corridor_half_width(cfg: PipelineConfig) -> float:
    return float(cfg.gauge.half_width + cfg.gauge.lateral_margin)


def extrapolation_margin(cfg: PipelineConfig,
                         axis_noise: Optional[tuple] = None) -> float:
    """Насколько далеко за подтверждённой зоной ещё строить коридор, м.

    За концом зоны, где путь подтверждён рельсами или обделкой, ось — это
    продолжение дуги, и вопрос в том, насколько быстро продолжение расходится
    с настоящим путём. Есть два ответа.

    **Худший случай.** Если сразу за подтверждённой зоной начинается кривая
    минимального для линии радиуса, путь уходит от продолжения прямой на
    Δ²/2R. Это верхняя оценка, и она очень пессимистична: на прямом перегоне
    ничего подобного не происходит, а коридор она обрезает втрое.

    **Измеренная устойчивость оси.** Если оценки курса и кривизны от кадра к
    кадру согласуются, продолжение дуги можно вести дальше — ровно настолько,
    насколько позволяет собственный разброс оценки:

        σ_курса · Δ + σ_кривизны · Δ² ≤ полуширина коридора

    `axis_noise` — пара (σ_курса, σ_кривизны), измеренная по последним кадрам
    (`lib/pipeline.py`). Без неё остаётся худший случай: на первых кадрах
    записи разброс ещё не по чему считать.

    Нижняя граница — всё тот же худший случай: короче него коридор обрезать
    незачем, там ошибка заведомо мала.
    """
    radius = max(cfg.track.min_radius, 1.0)
    half = _corridor_half_width(cfg)
    worst_case = float(np.sqrt(2.0 * radius * half))
    if axis_noise is None:
        return worst_case
    sigma_lin, sigma_quad = (max(float(axis_noise[0]), 0.0), max(float(axis_noise[1]), 0.0))
    if sigma_quad <= 1e-12:
        if sigma_lin <= 1e-9:
            return float(cfg.preprocess.x_max or 250.0)
        measured = half / sigma_lin
    else:
        measured = (-sigma_lin + np.sqrt(sigma_lin ** 2 + 4.0 * sigma_quad * half)) \
            / (2.0 * sigma_quad)
    return float(np.clip(measured, worst_case, cfg.preprocess.x_max or 250.0))


def wall_clearance(cfg: PipelineConfig, scene: SceneModel) -> float:
    """Зазор от края габарита до боковой границы сечения, м.

    В замкнутом сечении — измеренный: сколько свободного места остаётся
    между габаритом поезда и обделкой. На открытом участке границы нет, и
    зазором считается сама полуширина габарита — то расстояние, на котором
    ошибка оси начинает уводить коридор с пути.
    """
    half = _corridor_half_width(cfg)
    if scene.enclosed and np.isfinite(scene.half_width):
        return float(max(scene.half_width - half, 0.15))
    return float(half)


class DerivedResolver:
    """Держит рабочий конфиг, в котором заполнены все автоматические поля.

    Пересчёт идёт не каждый кадр, а когда меняется то, из чего значения
    выводятся: появилось измерение датчика, сменился тип сцены, заметно
    изменилась ширина сечения. Так тракт не тратит время на арифметику,
    которая девяносто девять кадров из ста даёт тот же ответ, и — что важнее —
    пороги не дрожат от кадра к кадру.
    """

    def __init__(self, base: PipelineConfig):
        self.base = base
        self.effective = copy.deepcopy(base)
        self.notes: List[DerivedNote] = []
        self._sensor_key: Optional[tuple] = None
        self._scene_key: Optional[tuple] = None
        self._resolved = False

    @staticmethod
    def _sensor_signature(sensor: SensorProfile) -> tuple:
        if not sensor.valid:
            return (False,)
        # Округление: пересчитывать пороги из-за процентного колебания оценки
        # плотности незачем, а дрожание порогов между кадрами вредно.
        return (True, round(np.log10(max(sensor.points_per_sr, 1.0)), 2),
                round(sensor.max_range / 10.0))

    @staticmethod
    def _scene_signature(scene: SceneModel) -> tuple:
        width = scene.half_width if np.isfinite(scene.half_width) else -1.0
        return (scene.kind, round(float(width), 1))

    def update(self, sensor: SensorProfile, scene: SceneModel) -> PipelineConfig:
        """Возвращает рабочий конфиг с заполненными автоматическими полями."""
        sensor_key = self._sensor_signature(sensor)
        scene_key = self._scene_signature(scene)
        if self._resolved and sensor_key == self._sensor_key and scene_key == self._scene_key:
            return self.effective
        self._sensor_key, self._scene_key = sensor_key, scene_key
        self._resolved = True
        self._apply(sensor, scene)
        return self.effective

    # ------------------------------------------------------------------ вывод
    def _apply(self, sensor: SensorProfile, scene: SceneModel) -> None:
        base, eff = self.base, self.effective
        notes: List[DerivedNote] = []

        def note(key: str, value: object, why: str) -> None:
            notes.append(DerivedNote(key, value, why))

        area = min_target_area(base)
        fill = base.safety.target_fill
        brake = braking_range(base)

        # --- рабочая зона: докуда вообще смотреть ---------------------------
        if base.preprocess.x_max is None:
            # Дальше видимого лидаром — пусто по определению; дальше того, что
            # нужно для торможения с запасом на внимание, — незачем считать.
            far = brake * max(base.decision.attention_factor, 1.0)
            observed = sensor.max_range if sensor.valid else far
            eff.preprocess.x_max = float(np.clip(min(observed, far), 40.0, 400.0))
            note("preprocess.x_max", round(eff.preprocess.x_max),
                 f"дальнобойность {observed:.0f} м, тормозной путь {brake:.0f} м")

        if base.preprocess.y_abs_max is None:
            # Поперёк нужно видеть не только свой габарит: по боковой границе
            # распознаётся сцена и ведётся ось в тоннеле.
            width = 4.0 * _corridor_half_width(base)
            if np.isfinite(scene.half_width):
                width = max(width, 2.5 * scene.half_width)
            eff.preprocess.y_abs_max = float(np.clip(width, 8.0, 30.0))
            note("preprocess.y_abs_max", round(eff.preprocess.y_abs_max, 1),
                 "габарит с запасом на боковую границу сечения")

        if base.preprocess.max_input_points is None:
            # Без потолка: тракт сам уступит плотность, когда перестанет
            # укладываться в период лидара (GaugeConfig.adaptive_range).
            eff.preprocess.max_input_points = 0
            note("preprocess.max_input_points", 0,
                 "потолок вырабатывается по факту времени кадра")

        # --- коридор --------------------------------------------------------
        if base.gauge.max_range is None:
            far = brake * max(base.decision.attention_factor, 1.0)
            observed = sensor.max_range if sensor.valid else far
            eff.gauge.max_range = float(np.clip(min(observed, far), 40.0, 400.0))
            note("gauge.max_range", round(eff.gauge.max_range),
                 "предел видимости против запаса на торможение")

        if base.gauge.extrapolation_margin is None:
            eff.gauge.extrapolation_margin = round(extrapolation_margin(base), 1)
            note("gauge.extrapolation_margin", eff.gauge.extrapolation_margin,
                 f"ошибка дуги Δ²/2R достигает полуширины коридора "
                 f"{_corridor_half_width(base):.2f} м при R={base.track.min_radius:.0f} м")

        if base.gauge.adaptive_min_range is None:
            # Уступать дальность ниже тормозного пути бессмысленно: поезд
            # перестанет успевать остановиться по тому, что увидит. Но и выше
            # самой дальности коридора этот предел не поднимается — иначе
            # уступать было бы нечего, а на длинном тормозном пути система
            # осталась бы вовсе без механизма разгрузки.
            ceiling = float(eff.gauge.max_range or brake)
            eff.gauge.adaptive_min_range = round(min(max(brake, 30.0), 0.8 * ceiling), 1)
            note("gauge.adaptive_min_range", eff.gauge.adaptive_min_range,
                 f"тормозной путь {brake:.0f} м, но не выше 80 % дальности коридора")

        # --- полосы поиска: не дотягиваться до соседнего пути -----------------
        # Полоса поиска ограничена междупутьем: взятая шире, она захватывает
        # соседний путь, и по нему начинают считаться и ось, и уровень
        # полотна. Но сужать её сверх необходимого тоже нельзя — поиск идёт
        # вокруг оси предыдущего кадра, а та на дальности известна хуже, и в
        # кривой истинный путь оказывается за краем узкой полосы: рельсы
        # перестают попадать в дальние ячейки, и подгонка дуги слепнет.
        # Поэтому берётся почти всё междупутье, а в замкнутом сечении —
        # измеренная полуширина, если она меньше: за стеной пути нет.
        own_track = 0.9 * base.track.spacing
        if np.isfinite(scene.half_width) and scene.enclosed:
            own_track = min(own_track, max(scene.half_width, 1.5))
            why = (f"0.9 междупутья ({base.track.spacing} м), ограничено "
                   f"сечением {scene.half_width:.1f} м")
        else:
            why = f"0.9 междупутья ({base.track.spacing} м)"
        if base.track.search_half_width is None:
            eff.track.search_half_width = round(float(np.clip(own_track, 1.5, 6.0)), 2)
            note("track.search_half_width", eff.track.search_half_width, why)
        if base.ground.search_half_width is None:
            # Полотну — своя полоса, и она уже рельсовой. Поиск оси может
            # позволить себе широкую полосу: он ищет форму сечения и чужой
            # путь по ней отличит. Уровень полотна так не защищён — в полосу
            # попадает балласт соседнего пути, лежащий на другой высоте, и
            # оценка УГР уезжает вместе с ним. Своё полотно занимает не
            # больше половины междупутья: дальше начинается чужое.
            own_bed = 0.5 * base.track.spacing
            eff.ground.search_half_width = round(float(np.clip(own_bed, 1.5, 4.0)), 2)
            note("ground.search_half_width", eff.ground.search_half_width,
                 f"половина междупутья ({base.track.spacing} м): дальше чужое полотно")

        # --- пороги плотности ------------------------------------------------
        if base.cluster.ref_min_points is None:
            if sensor.valid:
                expected = sensor.expected_points(area, base.cluster.ref_range, fill)
                value = int(max(base.cluster.abs_min_points,
                                round(base.safety.detection_fraction * expected)))
                eff.cluster.ref_min_points = value
                note("cluster.ref_min_points", value,
                     f"цель {area:.2f} м² на {base.cluster.ref_range:.0f} м даёт "
                     f"{expected:.0f} отражений, принимаем "
                     f"{base.safety.detection_fraction:.0%}")
            else:
                eff.cluster.ref_min_points = max(base.cluster.abs_min_points, 8)
                note("cluster.ref_min_points", eff.cluster.ref_min_points,
                     "датчик ещё не измерен — минимально строгий порог")

        # Потолок, который ставит прореживание: он нужен порогу плотности,
        # иначе в ближней зоне тот требует точек больше, чем тракт способен
        # оставить после вокселя.
        eff.cluster.voxel_bands = tuple(base.preprocess.voxel_bands)
        eff.cluster.target_area = round(area, 4)
        eff.cluster.detection_fraction = base.safety.detection_fraction

        if base.cluster.eps0 is None:
            # Связность обязана покрывать не только собственную неоднородность
            # цели, но и шаг, который вносит само прореживание: после вокселя
            # с ребром L точки одной поверхности разнесены на L, а по
            # пространственной диагонали — на L·√3. Радиус меньше этого рвёт
            # кластер ровно там, где точек больше всего, — в ближней зоне.
            leaf = max((leaf for _upper, leaf in base.preprocess.voxel_bands),
                       default=0.0)
            eff.cluster.eps0 = round(max(0.5 * min(base.safety.min_target_width,
                                                   base.safety.min_target_depth),
                                         np.sqrt(3.0) * leaf), 3)
            note("cluster.eps0", eff.cluster.eps0,
                 f"диагональ вокселя прореживания ({leaf:.2f} м) против половины "
                 f"минимальной цели")

        if base.cluster.eps_per_meter is None:
            # Соседние лучи расходятся на r·Δφ — но это расстояние между их
            # следами только на поверхности, повёрнутой к лидару. Поверхность,
            # вытянутая вдоль пути (кабельный лоток, контактный рельс, стена),
            # видна под скользящим углом, и следы лучей на ней расходятся в
            # 1/sin(α) раз сильнее. Связность обязана покрывать и этот случай:
            # иначе протяжённое путевое оборудование дробится на куски по
            # полметра, перестаёт опознаваться как «тонкая полоса вдоль пути»
            # и выдаётся за препятствие — на записях метро это давало
            # экстренное торможение в трети кадров.
            step = sensor.angular_step if sensor.valid else 0.0035
            grazing = np.radians(15.0)   # острее поверхность уже не разрешается
            eff.cluster.eps_per_meter = round(step / np.sin(grazing), 5)
            note("cluster.eps_per_meter", eff.cluster.eps_per_meter,
                 f"шаг решётки {np.degrees(step):.2f}° на поверхности под углом 15°")

        if base.decision.min_points_brake is None:
            eff.decision.min_points_brake = int(base.safety.geometric_min_points)
            note("decision.min_points_brake", eff.decision.min_points_brake,
                 "минимум, при котором у кластера измеримы три габарита")

        # --- сцена: тоннельные механизмы ------------------------------------
        # Пока сцена не распознана (первые кадры, прямой вызов этапа), режим
        # остаётся неопределённым — и механизмы работают: модель свободного
        # пространства сама окажется пустой там, где границ нет, а ведение
        # оси само не найдёт опоры. Выключение — это решение, и принимать его
        # можно только по измерению: иначе система молча теряет главную
        # защиту от ложных торможений на обделке.
        enclosed = scene.enclosed
        if base.tunnel.enabled is None:
            # Модель свободного пространства строится всегда, а не только в
            # распознанном тоннеле. Она описывает не тоннель, а непрерывные
            # конструкции вдоль пути — обделку, платформу, опоры контактной
            # сети, — и там, где их нет, сама возвращает «не знаю» и ни на что
            # не влияет. Привязка к типу сцены стоила дорого: двухпутный
            # тоннель шириной десять метров классификатор считает открытым
            # участком, и главная защита от ложных торможений на обделке
            # молча отключалась ровно там, где она и нужна.
            eff.tunnel.enabled = True
            note("tunnel.enabled", True,
                 "модель строится всегда: где границ нет, она пуста и не мешает")
        if base.track.guide_enabled is None:
            # А вот ведение оси обделкой — механизм именно тоннельный: вести
            # путь по стене можно только там, где стена действительно идёт
            # вдоль него.
            eff.track.guide_enabled = bool(enclosed) if scene.kind != UNKNOWN else None
            note("track.guide_enabled",
                 eff.track.guide_enabled if scene.kind != UNKNOWN else "по сцене",
                 scene.describe())

        if base.tunnel.band is None:
            # Полоса вдвое короче максимального предмета: предмет не должен
            # заполнять собой столько полос, чтобы сойти за границу.
            eff.tunnel.band = round(max(base.objects.max_length / 2.0, 1.0), 1)
            note("tunnel.band", eff.tunnel.band,
                 f"половина максимальной длины предмета ({base.objects.max_length} м)")

        if base.tunnel.height_max is None:
            eff.tunnel.height_max = round(base.gauge.height + 0.5, 2)
            note("tunnel.height_max", eff.tunnel.height_max,
                 "верх габарита носителя с запасом")

        if base.objects.min_intrusion is None:
            # Заход внутрь свободного места должен превышать собственную
            # неровность границы; пока она не измерена — размер минимальной цели.
            eff.objects.min_intrusion = round(base.safety.min_target_width, 2)
            note("objects.min_intrusion", eff.objects.min_intrusion,
                 "поперечный размер минимальной цели")

        if base.scene.band is None:
            eff.scene.band = round(base.objects.max_length + 2.0, 1)
            note("scene.band", eff.scene.band,
                 "длиннее максимального предмета: предмет не подменяет границу")

        if base.scene.search_half_width is None:
            eff.scene.search_half_width = eff.preprocess.y_abs_max
            note("scene.search_half_width", eff.scene.search_half_width,
                 "по ширине рабочей зоны")

        # --- ведение оси обделкой -------------------------------------------
        if base.track.guide_band is None:
            eff.track.guide_band = round(base.objects.max_length + 2.0, 1)
            note("track.guide_band", eff.track.guide_band,
                 "длиннее максимального предмета у стены")

        if base.track.guide_max_half_width is None:
            if np.isfinite(scene.half_width):
                value = float(np.clip(2.0 * scene.half_width, 3.0, 20.0))
                why = f"вдвое шире измеренного сечения ({scene.half_width:.1f} м)"
            else:
                value = float(eff.preprocess.y_abs_max or 8.0)
                why = "сечение не измерено — по ширине рабочей зоны"
            eff.track.guide_max_half_width = round(value, 1)
            note("track.guide_max_half_width", eff.track.guide_max_half_width, why)

        self.effective.unknown_keys = list(base.unknown_keys)
        self.notes = notes

    def describe(self) -> str:
        """Человекочитаемая сводка: что выведено и почему."""
        if not self.notes:
            return "автоматических параметров нет: всё задано профилем"
        return "\n".join(f"   {note}" for note in self.notes)
