"""Сборка всего тракта обработки одного кадра лидара.

Модуль намеренно не зависит от ROS: тот же объект Pipeline крутится
и внутри ноды, и в офлайн-оценке качества на датасете — иначе метрики
меряют не то, что поедет на поезде.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .cluster import cluster
from .config import PipelineConfig
from .decision import GaugeDecision, decide
from .derive import (DerivedResolver, detection_limit, extrapolation_margin,
                     sensor_sector)
from .detect import Detection, detections_from_clusters, mark_linear_infrastructure
from .egomotion import EgoMotionEstimator
from .frames import FrameNormalizer
from .ground import GroundModel, estimate_ground
from .preprocess import limit_input, preprocess, radius_outlier_mask
from .scene import SceneModel, SceneTracker, classify_scene
from .sensor_model import SensorCalibrator, SensorProfile
from .track import Corridor, TrackModel, build_corridor, estimate_track, rail_point_mask
from .tracking import ObjectTracker, ObjectTrack
from .tunnel import FreeSpace, estimate_free_space


@dataclass
class FrameResult:
    timestamp: float
    obstacles: List[ObjectTrack]
    detections: List[Detection]
    corridor: Corridor
    track: TrackModel
    ground: GroundModel
    decision: GaugeDecision
    timings: Dict[str, float] = field(default_factory=dict)
    input_points: int = 0
    filtered_points: int = 0
    candidate_points: int = 0
    max_valid_range: float = 0.0
    # Скорость, по которой считался тормозной путь, и откуда она взята:
    # odometry — бортовая, estimated — оценка по облакам, default — из профиля.
    ego_speed: Optional[float] = None
    speed_source: str = "default"
    # Ориентация датчика: как она определена и не сменилась ли в этом кадре.
    frame_spec: str = ""
    frame_source: str = ""
    frame_decided: bool = False
    # Диагностика входа: кадр, от которого после фильтрации почти ничего не
    # осталось, — это не «путь свободен», а неверная СК или закрытый датчик.
    degenerate: bool = False
    free_space: Optional[FreeSpace] = None
    # Что система измерила о себе самой в этом кадре: датчик, сцена и
    # вытекающий из них предел обнаружения. Это не отладка, а ответ на
    # вопрос «на что система сейчас способна» — он зависит от лидара и
    # участка, а не от того, на какой записи её настраивали.
    sensor_profile: Optional[SensorProfile] = None
    scene: Optional[SceneModel] = None
    detection_limit: float = 0.0
    # Для отладочной визуализации (заполняются только при debug=True)
    filtered_xyz: Optional[np.ndarray] = None
    candidate_xyz: Optional[np.ndarray] = None
    ground_xyz: Optional[np.ndarray] = None

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings.values()))


class Pipeline:
    """Полный тракт: фильтрация → полотно → ось пути → габарит →
    кластеризация → объекты → треки → решение."""

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        # Профиль задаёт физику носителя; всё остальное система измеряет сама.
        # `sensor` меряет плотность лучей и дальнобойность, `scene` — тип
        # сечения, `derive` превращает то и другое в рабочие пороги. Базовый
        # конфиг при этом не меняется: в нём видно, что задал человек, а в
        # `eff` — с чем тракт работает на самом деле.
        self.sensor = SensorCalibrator(sensor_sector(self.cfg))
        self.scene = SceneTracker(self.cfg.scene.window)
        self.derive = DerivedResolver(self.cfg)
        self.eff = self.derive.update(self.sensor.profile, self.scene.current)
        self.tracker = ObjectTracker(self.cfg.tracker)
        self.frames = FrameNormalizer(self.cfg.sensor)
        self.egomotion = EgoMotionEstimator(self.cfg.egomotion)
        self._track_model: Optional[TrackModel] = None
        self._frame_index = 0
        self.route_prior: Optional[TrackModel] = None
        # Кандидаты дальней зоны с прошлых кадров: (время, точки). Сводятся к
        # текущему моменту по пройденному пути — см. AccumulationConfig.
        self._history: Deque[Tuple[float, np.ndarray]] = deque(maxlen=8)
        # Измеренная ось прошлого кадра для межкадрового сглаживания.
        self._guide_prev: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        # Оценки курса и кривизны последних кадров: по их разбросу видно,
        # насколько далеко за подтверждённой зоной продолжению дуги ещё можно
        # верить. На прямом перегоне разброс близок к нулю и коридор идёт до
        # предела видимости; в кривой, где оценка гуляет, он сам укорачивается.
        self._axis_history: Deque[Tuple[float, float]] = deque(maxlen=12)
        # Бюджет дальности и наблюдения за тем, успеваем ли за лидаром.
        self._range_budget = float(self.eff.gauge.max_range or 200.0)
        self._density_budget = int(self.eff.preprocess.max_input_points or 0)
        self._recent_ms: Deque[float] = deque(maxlen=20)
        self._recent_dt: Deque[float] = deque(maxlen=20)
        self._last_stamp: Optional[float] = None
        # Размер последнего входного облака: от него отсчитываются и пол
        # плотности, и шаг уступки — абсолютные числа здесь означали бы
        # настройку под конкретную модель лидара.
        self._last_raw_points = 0
        self._density_step = 0

    @property
    def density_budget(self) -> int:
        """Текущий потолок числа точек на кадр, 0 — без ограничения.

        Тот, кто разбирает сообщение лидара, может проредить облако сразу — до
        того, как оно превратится в массив. Бюджет живёт здесь, потому что
        вырабатывает его тракт: он один знает, укладывается ли в период кадров.
        """
        return int(self._density_budget or 0)

    def set_route_prior(self, lin: float, quad: float, valid_range: float = 200.0) -> None:
        """Ось пути из внешнего источника: путевая карта метро, маршрут,
        одометрия. На реальной линии геометрия пути известна заранее —
        лидару остаётся её подтвердить, а не выводить с нуля, и коридор
        остаётся достоверным на всю дальность обзора.
        """
        import numpy as _np
        self.route_prior = TrackModel(lin=lin, quad=quad, fit_range=valid_range,
                                      from_rails=False, bin_x=_np.zeros(0),
                                      bin_y=_np.zeros(0), bin_z=_np.zeros(0))

    def reset(self) -> None:
        self.tracker = ObjectTracker(self.cfg.tracker)
        self.egomotion.reset()
        self.scene.reset()
        self._track_model = None
        self._frame_index = 0
        self._history.clear()
        self._guide_prev = None
        self._axis_history.clear()
        self._range_budget = float(self.eff.gauge.max_range or 200.0)
        self._density_budget = int(self.eff.preprocess.max_input_points or 0)
        self._recent_ms.clear()
        self._recent_dt.clear()
        self._last_stamp = None
        # Ориентация датчика намеренно не сбрасывается: в пределах одной
        # установки она не меняется, а повторное автоопределение дало бы
        # дрожание осей на кадрах с пустым тоннелем.

    def note_frame_cost(self, frame_ms: float) -> None:
        """Сообщает тракту полную стоимость кадра, включая разбор сообщения.

        Сам тракт видит только своё время, а на 128-луче разбор PointCloud2
        стоит ещё четверть периода. Без этой поправки система считает, что
        успевает, и не уступает дальность, хотя кадры уже теряются.
        """
        if self._recent_ms:
            self._recent_ms[-1] = float(frame_ms)

    def _update_range_budget(self, frame_ms: float, timestamp: float) -> None:
        """Подстраивает дальность коридора под то, успеваем ли за лидаром.

        Период берётся из штампов кадров, а не из настроек: на контрольной
        записи частота может быть любой. Решение принимается по медиане
        последних кадров, чтобы единичный выброс не сдвигал дальность.
        """
        cfg = self.eff.gauge
        if self._last_stamp is not None:
            dt = timestamp - self._last_stamp
            if 1e-3 < dt < 1.0:
                self._recent_dt.append(dt)
        self._last_stamp = timestamp
        self._recent_ms.append(frame_ms)
        if not cfg.adaptive_range or len(self._recent_ms) < self._recent_ms.maxlen \
                or not self._recent_dt:
            return
        # Период лидара — по нижнему перцентилю интервалов, а не по медиане:
        # кадры, до которых тракт не добрался, увеличивают интервал вдвое, и
        # по медиане перегруженная система выглядит успевающей.
        period_ms = 1e3 * float(np.percentile(np.asarray(self._recent_dt), 10))
        typical = float(np.median(np.asarray(self._recent_ms)))
        pre = self.eff.preprocess
        # Пол плотности — доля исходного облака, а не абсолютное число: у
        # 64- и 128-луча размер кадра различается втрое, и одно и то же число
        # означало бы для них разную степень прореживания.
        density_floor = int(pre.min_input_fraction * max(self._last_raw_points, 1))
        if typical > cfg.adaptive_busy * period_ms:
            # Сначала плотность ближнего поля: там точек избыток, и их потеря
            # дальность обнаружения не трогает. Дальность уступаем, только
            # когда прореживать уже некуда.
            # Потолок плотности вырабатывается по факту: пока его нет, первым
            # шагом становится само текущее облако, дальше — шаги вниз.
            if self._density_budget <= 0:
                self._density_budget = max(density_floor, self._last_raw_points)
            if self._density_budget > density_floor:
                self._density_budget = max(density_floor,
                                           self._density_budget - self._density_step)
            else:
                self._range_budget = max(float(cfg.adaptive_min_range or 40.0),
                                         self._range_budget - cfg.adaptive_step)
        elif typical < cfg.adaptive_idle * period_ms:
            ceiling = float(cfg.max_range or self._range_budget)
            if self._range_budget < ceiling:
                self._range_budget = min(ceiling, self._range_budget + cfg.adaptive_step)
            elif 0 < self._density_budget < self._last_raw_points:
                self._density_budget = min(self._last_raw_points,
                                           self._density_budget + self._density_step)

    def _axis_noise(self) -> Optional[Tuple[float, float]]:
        """Разброс оценок курса и кривизны по последним кадрам.

        Берётся медианное абсолютное отклонение, а не дисперсия: один кадр с
        плохо распознанным сечением не должен схлопывать коридор. Пока кадров
        мало, разброса нет — и длина коридора считается по худшему случаю.
        """
        if len(self._axis_history) < max(4, self._axis_history.maxlen // 3):
            return None
        values = np.asarray(self._axis_history, dtype=float)
        deviation = np.abs(values - np.median(values, axis=0))
        # 1.4826 — перевод медианного отклонения в масштаб стандартного для
        # нормального распределения: дальше величина используется как «σ».
        sigma = 1.4826 * np.median(deviation, axis=0)
        return float(sigma[0]), float(sigma[1])

    def _smooth_guide(self, track: TrackModel, timestamp: float) -> None:
        """Сглаживает измеренную осью тоннеля часть между кадрами.

        Стена на сотне метров даёт считанные отражения, и центр полосы
        гуляет — коридор на дальности «дышит» на метр-полтора, а препятствие
        то попадает в него, то нет. Ось прошлого кадра сдвигается на
        пройденный поездом путь (то же преобразование, что и для накопления
        кандидатов) и смешивается с новой оценкой.
        """
        cfg = self.cfg.track
        previous = self._guide_prev
        if track.guide_s.size >= 2 and previous is not None and cfg.guide_smoothing < 1.0:
            prev_time, prev_s, prev_y = previous
            dt = timestamp - prev_time
            speed = self.egomotion.smoothed
            if 0.0 < dt <= 1.0 and speed is not None and prev_s.size >= 2:
                shifted_s = prev_s - float(speed) * dt
                inside = track.guide_s <= shifted_s[-1]
                if inside.any():
                    predicted = np.interp(track.guide_s[inside], shifted_s, prev_y)
                    blended = track.guide_y.copy()
                    blended[inside] = (cfg.guide_smoothing * track.guide_y[inside]
                                       + (1.0 - cfg.guide_smoothing) * predicted)
                    track.guide_y = blended
        if track.guide_s.size >= 2:
            self._guide_prev = (timestamp, track.guide_s.copy(), track.guide_y.copy())
        else:
            self._guide_prev = None

    def _accumulate(self, cand_pts: np.ndarray, intrusion: Optional[np.ndarray],
                    timestamp: float, speed_hint: Optional[float]):
        """Сводит кандидатов дальней зоны с прошлых кадров к текущему моменту.

        Препятствие неподвижно относительно тоннеля, поэтому за время dt его
        дальность сокращается ровно на пройденный путь. Сдвигаем прошлые точки
        на эту величину и кластеризуем вместе с текущими. Если скорость
        неизвестна или устарела, накопление не делается: смаз хуже, чем
        отсутствие точек.
        """
        cfg = self.cfg.accumulate
        far = cand_pts[:, 0] >= cfg.min_range if cand_pts.size else np.zeros(0, dtype=bool)
        merged = [cand_pts]
        merged_intrusion = [intrusion] if intrusion is not None else None
        frames = 1
        if speed_hint is not None:
            for stamp, points in self._history:
                dt = timestamp - stamp
                if not (0.0 < dt <= cfg.max_dt):
                    continue
                shifted = points.copy()
                shifted[:, 0] -= float(speed_hint) * dt
                shifted = shifted[shifted[:, 0] >= cfg.min_range]
                if shifted.size:
                    merged.append(shifted)
                    if merged_intrusion is not None:
                        # Про накопленные точки модель свободного пространства
                        # этого кадра ничего не знает — отдаём «неизвестно»,
                        # чтобы неизвестность не работала как признак помехи.
                        merged_intrusion.append(np.full(shifted.shape[0], np.inf, np.float32))
                    frames += 1
        if cand_pts.size and far.any():
            self._history.append((timestamp, cand_pts[far].copy()))
        while len(self._history) > max(1, cfg.frames - 1):
            self._history.popleft()
        if frames == 1:
            return cand_pts, intrusion, 1
        return (np.vstack(merged),
                np.concatenate(merged_intrusion) if merged_intrusion is not None else None,
                frames)

    def process(self, xyz: np.ndarray, intensity: Optional[np.ndarray] = None,
                timestamp: Optional[float] = None, ego_speed: Optional[float] = None) -> FrameResult:
        # Рабочий конфиг прошлого кадра: ранние этапы (прореживание, обрезка
        # рабочей зоны) идут до того, как этот кадр что-то расскажет о датчике
        # и сцене. Параметры меняются медленно, и отставание на кадр здесь
        # безвредно — зато пороги не зависят от того, с какого кадра начали.
        cfg = self.eff
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        if timestamp is None:
            timestamp = self._frame_index * 0.1
        self._frame_index += 1

        # 1. Рабочая СК: X вперёд, Y влево, Z вверх. До этого этапа никакие
        #    пороги по x и y смысла не имеют.
        t0 = time.perf_counter()
        raw_points = int(xyz.shape[0])
        # Прореживание — до поворота осей: поворачивать имеет смысл только то,
        # что пойдёт в обработку. На 128-луче это 250 тысяч точек вместо 920.
        # Ось дальности в сырой СК подсказывает нормализатор, когда ориентация
        # датчика уже определена; на первых кадрах её ещё нет, и прореживание
        # делается равномерным шагом.
        self._last_raw_points = raw_points
        if self._density_step <= 0:
            # Шаг уступки плотности — восьмая часть кадра: достаточно крупно,
            # чтобы сойтись за несколько кадров, и достаточно мелко, чтобы не
            # обрушить дальность одним движением.
            self._density_step = max(raw_points // 8, 10000)
        xyz, intensity = limit_input(xyz, intensity, cfg.preprocess, self.frames.axes,
                                     max_points=self._density_budget)
        xyz, frame_decided = self.frames.apply(xyz)
        # Датчик меряется по развёрнутому облаку: передний сектор, в котором
        # считается плотность лучей, определён в рабочей СК.
        if self.frames.resolved:
            self.sensor.update(xyz)
        cfg = self.eff = self.derive.update(self.sensor.profile, self.scene.current)
        timings["frames_ms"] = (time.perf_counter() - t0) * 1e3

        # 2. Фильтрация (с потолком на размер облака — по бюджету времени)
        t0 = time.perf_counter()
        used_points = int(xyz.shape[0])
        pts, inten, _kept = preprocess(xyz, intensity, cfg.preprocess)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1e3
        degenerate = bool(
            used_points >= cfg.sensor.auto_min_points
            and pts.shape[0] < cfg.preprocess.degenerate_fraction * used_points)

        # 3. Полотно пути. Первый проход — вдоль прошлой оси (или прямо вперёд).
        t0 = time.perf_counter()
        prior_y = None
        if self._track_model is not None:
            edges = np.arange(0.0, cfg.preprocess.x_max + cfg.ground.bin_size, cfg.ground.bin_size)
            prior_y = self._track_model.y_at(0.5 * (edges[:-1] + edges[1:]))
        ground = estimate_ground(pts, cfg.ground, centerline_y=prior_y,
                                 x_min=0.0, x_max=cfg.preprocess.x_max)
        timings["ground_ms"] = (time.perf_counter() - t0) * 1e3

        # 4. Ось пути по рельсам
        t0 = time.perf_counter()
        # Пройденный с прошлого кадра путь: на него сдвигается всё, что было
        # известно о пути впереди, если в этом кадре сечение не распознано.
        speed_prior = self.egomotion.smoothed
        travelled = 0.0
        if speed_prior is not None and self._last_stamp is not None:
            dt = timestamp - self._last_stamp
            if 0.0 < dt <= 1.0:
                travelled = float(speed_prior) * dt
        track = estimate_track(pts, inten, ground, cfg.track, previous=self._track_model,
                               prior=self.route_prior, travelled=travelled)
        if self.route_prior is not None:
            # С внешней осью зона доверия задаётся картой, а не видимостью рельсов.
            track.fit_range = max(track.fit_range, self.route_prior.fit_range)
        self._smooth_guide(track, timestamp)
        self._track_model = track
        if track.from_rails:
            # В историю идут только кадры, где сечение пути действительно
            # распознано: перенесённая с прошлого кадра ось ничего нового о
            # разбросе не говорит и занижала бы его.
            self._axis_history.append((float(track.lin), float(track.quad)))
        if self.cfg.gauge.extrapolation_margin is None:
            cfg.gauge.extrapolation_margin = extrapolation_margin(
                self.cfg, self._axis_noise())
        timings["track_ms"] = (time.perf_counter() - t0) * 1e3

        # 5. Габаритный коридор и высоты над УГР
        t0 = time.perf_counter()
        corridor = build_corridor(track, cfg.gauge, self._range_budget)
        heights = ground.height_above_rail(pts)
        lateral = corridor.lateral(pts)
        candidate_mask = corridor.inside(pts, heights, lateral) & (pts[:, 0] <= corridor.reach)
        candidate_mask &= ~rail_point_mask(pts, corridor, heights, cfg.track.gauge,
                                           cfg.track.rail_tolerance)
        # Поверхность полотна убираем по измеренной высоте над самим полотном,
        # а не по уровню УГР: иначе кластеризация склеивает балласт и шпалы в
        # змею вдоль пути. Предмет остаётся — он поднимается над полотном.
        # Высота над полотном получается из уже посчитанной высоты над УГР:
        # УГР — это и есть поверхность плюс высота головки рельса.
        candidate_mask &= (heights + ground.rail_offset) > cfg.objects.ground_band
        cand_idx = np.where(candidate_mask)[0]
        timings["gauge_ms"] = (time.perf_counter() - t0) * 1e3

        # 6. Сцена: замкнутое сечение или открытый участок. От этого зависит,
        #    работают ли тоннельные механизмы — и решает это кадр, а не
        #    профиль, написанный до поездки.
        t0 = time.perf_counter()
        if cfg.scene.enabled and pts.shape[0]:
            self.scene.update(classify_scene(
                pts[:, 0], lateral, heights,
                gauge_height=cfg.gauge.height,
                max_object_length=cfg.objects.max_length,
                band=float(cfg.scene.band or 10.0),
                max_range=float(cfg.preprocess.x_max or 0.0),
                min_side_points=cfg.scene.min_side_points,
                search_half_width=float(cfg.scene.search_half_width or 20.0),
                quantile=cfg.scene.quantile,
                relevant_half_width=cfg.track.spacing,
                corridor_half_width=cfg.gauge.half_width + cfg.gauge.lateral_margin))
        timings["scene_ms"] = (time.perf_counter() - t0) * 1e3

        # 7. Модель свободного пространства тоннеля. Строится по всему кадру,
        #    а спрашивается только про точки внутри коридора.
        t0 = time.perf_counter()
        # Модель свободного пространства описывает тоннель, а не коридор, и
        # строится на всю дальность кадра. Привязка к длине коридора делала её
        # зависимой от того, насколько далеко в этом кадре подтверждена ось:
        # стоило коридору удлиниться, как менялся набор полос, а с ним и то,
        # какие слои высоты признаются границей — вплоть до того, что стена в
        # ближней зоне перестаёт отсекаться.
        free_space = estimate_free_space(
            pts[:, 0], lateral, heights, cfg.tunnel,
            max(float(pts[:, 0].max()) if pts.shape[0] else 0.0, cfg.gauge.step),
            max_object_length=cfg.objects.max_length)
        if free_space is not None and self.cfg.objects.min_intrusion is None:
            # Порог захода внутрь свободного места — от собственного шума
            # измеренной границы: три её разброса. Меньший порог ничего не
            # доказывает, больший половины коридора — отвергал бы настоящие
            # предметы у края габарита.
            half = cfg.gauge.half_width + cfg.gauge.lateral_margin
            cfg.objects.min_intrusion = float(np.clip(
                3.0 * free_space.boundary_noise,
                self.cfg.safety.min_target_width, 0.5 * half))
        intrusion = None
        if free_space is not None and cand_idx.size:
            intrusion = free_space.slack(pts[cand_idx, 0], lateral[cand_idx], heights[cand_idx])
        timings["tunnel_ms"] = (time.perf_counter() - t0) * 1e3

        # 8. Снятие одиночных точек — только внутри коридора: там их мало,
        #    а цена ошибки (ложное торможение) максимальна.
        t0 = time.perf_counter()
        if cand_idx.size > 8:
            keep = radius_outlier_mask(pts[cand_idx])
            cand_idx = cand_idx[keep]
            if intrusion is not None:
                intrusion = intrusion[keep]
        timings["denoise_ms"] = (time.perf_counter() - t0) * 1e3

        # 9. Кластеризация и объекты. В дальней зоне кластеризуются кандидаты
        #    не одного кадра, а нескольких, сведённых к текущему моменту:
        #    иначе цели на сотне метров не хватает точек на кластер.
        t0 = time.perf_counter()
        cand_pts = pts[cand_idx]
        intrusion_pts = intrusion
        merged_frames = 1
        if cfg.accumulate.enabled and cfg.accumulate.frames > 1:
            cand_pts, intrusion_pts, merged_frames = self._accumulate(
                cand_pts, intrusion, timestamp, speed_hint=self.egomotion.smoothed)
        labels = cluster(cand_pts, cfg.cluster)
        timings["cluster_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        merged_heights = ground.height_above_rail(cand_pts) if merged_frames > 1 \
            else heights[cand_idx]
        detections = detections_from_clusters(
            cand_pts, labels, merged_heights, corridor,
            cfg.cluster, cfg.objects, keep_rejected=cfg.debug, intrusion=intrusion_pts,
            point_scale=merged_frames, scale_beyond=cfg.accumulate.min_range)
        # Линии вдоль пути: контактный рельс, лоток, короб. Их кластеры по
        # отдельности неотличимы от предметов, а вместе выстраиваются в линию
        # на одном смещении и одной высоте — этим и опознаются.
        mark_linear_infrastructure(detections, cfg.objects.max_length)

        # Уверенность падает там, где положение оси пути уже не подтверждено:
        # заявлять о вторжении в габарит на дальности, где неизвестно, где путь,
        # нельзя — это главный источник ложных торможений.
        accepted = [d for d in detections if not d.reject_reason]
        for det in accepted:
            det.confidence *= float(track.confidence_at(
                det.distance, cfg.track.confidence_decay, cfg.track.guide_confidence))
        timings["detect_ms"] = (time.perf_counter() - t0) * 1e3

        # 10. Скорость носителя: бортовая, а если её нет — оценка по облакам.
        #    Без скорости тормозной путь считается по значению из профиля, и
        #    решение вырождается: любой объект ближе 200 м даёт экстренное.
        t0 = time.perf_counter()
        estimated_speed = self.egomotion.update(pts, heights, timestamp)
        if ego_speed is not None:
            speed_used, speed_source = float(ego_speed), "odometry"
        elif estimated_speed is not None:
            speed_used, speed_source = float(estimated_speed), "estimated"
        else:
            speed_used, speed_source = None, "default"
        timings["egomotion_ms"] = (time.perf_counter() - t0) * 1e3

        # 11. Сопровождение и решение
        t0 = time.perf_counter()
        confirmed = self.tracker.update(accepted, timestamp,
                                        ego_speed=0.0 if speed_used is None else speed_used)
        max_range = float(pts[:, 0].max()) if pts.shape[0] else 0.0
        detection_range = min(max_range, corridor.reach)
        decision = decide(confirmed, speed_used, detection_range, cfg.decision,
                          speed_measured=speed_used is not None,
                          confident_range=min(track.fit_range, detection_range),
                          safety_margin=cfg.gauge.lateral_margin)
        timings["track_decide_ms"] = (time.perf_counter() - t0) * 1e3
        timings["total_ms"] = (time.perf_counter() - t_start) * 1e3
        self._update_range_budget(timings["total_ms"], timestamp)

        result = FrameResult(
            timestamp=timestamp, obstacles=confirmed, detections=detections,
            corridor=corridor, track=track, ground=ground, decision=decision,
            timings=timings, input_points=raw_points, filtered_points=int(pts.shape[0]),
            candidate_points=int(cand_idx.size), max_valid_range=max_range,
            ego_speed=speed_used, speed_source=speed_source,
            frame_spec=self.frames.spec, frame_source=self.frames.source,
            frame_decided=frame_decided, degenerate=degenerate, free_space=free_space,
            sensor_profile=self.sensor.profile, scene=self.scene.current,
            detection_limit=detection_limit(cfg, self.sensor.profile),
        )
        if cfg.debug:
            result.filtered_xyz = pts
            result.candidate_xyz = cand_pts
            ground_mask = np.abs(heights) < 0.15
            result.ground_xyz = pts[ground_mask]
        return result
