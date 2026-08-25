"""Сборка всего тракта обработки одного кадра лидара.

Модуль намеренно не зависит от ROS: тот же объект Pipeline крутится
и внутри ноды, и в офлайн-оценке качества на датасете — иначе метрики
меряют не то, что поедет на поезде.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .cluster import cluster
from .config import PipelineConfig
from .decision import GaugeDecision, decide
from .detect import Detection, detections_from_clusters
from .ground import GroundModel, estimate_ground
from .preprocess import preprocess, radius_outlier_mask
from .track import Corridor, TrackModel, build_corridor, estimate_track, rail_point_mask
from .tracking import ObjectTracker, ObjectTrack


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
        self.tracker = ObjectTracker(self.cfg.tracker)
        self._track_model: Optional[TrackModel] = None
        self._frame_index = 0
        self.route_prior: Optional[TrackModel] = None

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
        self._track_model = None
        self._frame_index = 0

    def process(self, xyz: np.ndarray, intensity: Optional[np.ndarray] = None,
                timestamp: Optional[float] = None, ego_speed: Optional[float] = None) -> FrameResult:
        cfg = self.cfg
        timings: Dict[str, float] = {}
        t_start = time.perf_counter()
        if timestamp is None:
            timestamp = self._frame_index * 0.1
        self._frame_index += 1

        # 1. Фильтрация
        t0 = time.perf_counter()
        pts, inten, _kept = preprocess(xyz, intensity, cfg.preprocess)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1e3

        # 2. Полотно пути. Первый проход — вдоль прошлой оси (или прямо вперёд).
        t0 = time.perf_counter()
        prior_y = None
        if self._track_model is not None:
            edges = np.arange(0.0, cfg.preprocess.x_max + cfg.ground.bin_size, cfg.ground.bin_size)
            prior_y = self._track_model.y_at(0.5 * (edges[:-1] + edges[1:]))
        ground = estimate_ground(pts, cfg.ground, centerline_y=prior_y,
                                 x_min=0.0, x_max=cfg.preprocess.x_max)
        timings["ground_ms"] = (time.perf_counter() - t0) * 1e3

        # 3. Ось пути по рельсам
        t0 = time.perf_counter()
        track = estimate_track(pts, inten, ground, cfg.track, previous=self._track_model,
                               prior=self.route_prior)
        if self.route_prior is not None:
            # С внешней осью зона доверия задаётся картой, а не видимостью рельсов.
            track.fit_range = max(track.fit_range, self.route_prior.fit_range)
        self._track_model = track
        timings["track_ms"] = (time.perf_counter() - t0) * 1e3

        # 4. Габаритный коридор и высоты над УГР
        t0 = time.perf_counter()
        corridor = build_corridor(track, cfg.gauge)
        heights = ground.height_above_rail(pts)
        candidate_mask = corridor.inside(pts, heights) & (pts[:, 0] <= cfg.gauge.max_range)
        candidate_mask &= ~rail_point_mask(pts, corridor, heights, cfg.track.gauge,
                                           cfg.track.rail_tolerance)
        cand_idx = np.where(candidate_mask)[0]
        timings["gauge_ms"] = (time.perf_counter() - t0) * 1e3

        # 5. Снятие одиночных точек — только внутри коридора: там их мало,
        #    а цена ошибки (ложное торможение) максимальна.
        t0 = time.perf_counter()
        if cand_idx.size > 8:
            keep = radius_outlier_mask(pts[cand_idx])
            cand_idx = cand_idx[keep]
        timings["denoise_ms"] = (time.perf_counter() - t0) * 1e3

        # 6. Кластеризация и объекты
        t0 = time.perf_counter()
        cand_pts = pts[cand_idx]
        labels = cluster(cand_pts, cfg.cluster)
        timings["cluster_ms"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        detections = detections_from_clusters(
            cand_pts, labels, heights[cand_idx], corridor,
            cfg.cluster, cfg.objects, keep_rejected=cfg.debug)
        # Уверенность падает там, где положение оси пути уже не подтверждено:
        # заявлять о вторжении в габарит на дальности, где неизвестно, где путь,
        # нельзя — это главный источник ложных торможений.
        accepted = [d for d in detections if not d.reject_reason]
        for det in accepted:
            det.confidence *= float(track.confidence_at(det.distance, cfg.track.confidence_decay))
        timings["detect_ms"] = (time.perf_counter() - t0) * 1e3

        # 7. Сопровождение и решение
        t0 = time.perf_counter()
        confirmed = self.tracker.update(accepted, timestamp,
                                        ego_speed=0.0 if ego_speed is None else ego_speed)
        max_range = float(pts[:, 0].max()) if pts.shape[0] else 0.0
        detection_range = min(max_range, cfg.gauge.max_range)
        decision = decide(confirmed, ego_speed, detection_range, cfg.decision)
        timings["track_decide_ms"] = (time.perf_counter() - t0) * 1e3
        timings["total_ms"] = (time.perf_counter() - t_start) * 1e3

        result = FrameResult(
            timestamp=timestamp, obstacles=confirmed, detections=detections,
            corridor=corridor, track=track, ground=ground, decision=decision,
            timings=timings, input_points=int(xyz.shape[0]), filtered_points=int(pts.shape[0]),
            candidate_points=int(cand_idx.size), max_valid_range=max_range,
        )
        if cfg.debug:
            result.filtered_xyz = pts
            result.candidate_xyz = cand_pts
            ground_mask = np.abs(heights) < 0.15
            result.ground_xyz = pts[ground_mask]
        return result
