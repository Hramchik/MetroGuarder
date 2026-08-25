"""Межкадровое сопровождение объектов.

Главный инструмент борьбы с ложными срабатываниями: реальное препятствие
видно в каждом кадре и приближается ровно со скоростью поезда, а всплеск
шума живёт один кадр и появляется в случайном месте. Тревога выдаётся
только по треку, подтверждённому N раз из последних M кадров.
Побочный полезный продукт — скорость сближения и время до контакта.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .config import TrackerConfig
from .detect import Detection


@dataclass
class ObjectTrack:
    id: int
    detection: Detection
    timestamp: float
    hits: int = 1
    misses: int = 0
    age: int = 1
    history: Deque[Tuple[float, float]] = field(default_factory=deque)   # (t, distance)
    recent_hits: Deque[bool] = field(default_factory=deque)
    confidence: float = 0.0

    @property
    def confirmed(self) -> bool:
        return self.hits >= 1 and self.confidence > 0.0

    def closing_speed(self) -> float:
        """МНК по истории дальности: >0 — объект приближается."""
        if len(self.history) < 2:
            return 0.0
        t = np.array([h[0] for h in self.history])
        d = np.array([h[1] for h in self.history])
        t = t - t[0]
        if float(t[-1]) <= 1e-3:
            return 0.0
        slope = float(np.polyfit(t, d, 1)[0])
        return -slope

    def time_to_contact(self) -> float:
        speed = self.closing_speed()
        if speed <= 0.1:
            return float("inf")
        return float(self.detection.distance / speed)


class ObjectTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: List[ObjectTrack] = []
        self._next_id = 1
        self._last_time: Optional[float] = None

    def update(self, detections: List[Detection], timestamp: float,
               ego_speed: float = 0.0) -> List[ObjectTrack]:
        dt = 0.0 if self._last_time is None else max(0.0, timestamp - self._last_time)
        self._last_time = timestamp

        # Предсказание: статичное препятствие приближается со скоростью поезда.
        predicted = []
        for tr in self.tracks:
            shift = ego_speed * dt if ego_speed > 0.0 else tr.closing_speed() * dt
            pos = tr.detection.centroid.copy()
            pos[0] -= shift
            predicted.append(pos)

        used_det = set()
        used_track = set()
        if self.tracks and detections:
            det_pos = np.array([d.centroid for d in detections])
            pred_pos = np.array(predicted)
            cost = np.linalg.norm(pred_pos[:, None, :] - det_pos[None, :, :], axis=2)
            # Ворота ассоциации расширяются с дальностью: на 150 м центроид
            # кластера гуляет на метры просто из-за смены видимых граней.
            gate = self.cfg.max_assoc_distance + 0.02 * np.array([d.distance for d in detections])
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for ti, di in order:
                if ti in used_track or di in used_det:
                    continue
                if cost[ti, di] > gate[di]:
                    continue
                used_track.add(int(ti))
                used_det.add(int(di))
                self._hit(self.tracks[int(ti)], detections[int(di)], timestamp)

        for di, det in enumerate(detections):
            if di in used_det:
                continue
            track = ObjectTrack(id=self._next_id, detection=det, timestamp=timestamp)
            track.history.append((timestamp, det.distance))
            track.recent_hits.append(True)
            self._next_id += 1
            self._update_confidence(track)
            self.tracks.append(track)

        for ti, track in enumerate(self.tracks):
            if ti in used_track or track.timestamp == timestamp:
                continue
            track.misses += 1
            track.age += 1
            track.recent_hits.append(False)
            while len(track.recent_hits) > self.cfg.confirm_window:
                track.recent_hits.popleft()
            self._update_confidence(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_misses]
        return self.confirmed_tracks()

    def _hit(self, track: ObjectTrack, det: Detection, timestamp: float) -> None:
        track.detection = det
        track.timestamp = timestamp
        track.hits += 1
        track.age += 1
        track.misses = 0
        track.history.append((timestamp, det.distance))
        while len(track.history) > self.cfg.history:
            track.history.popleft()
        track.recent_hits.append(True)
        while len(track.recent_hits) > self.cfg.confirm_window:
            track.recent_hits.popleft()
        self._update_confidence(track)

    def _update_confidence(self, track: ObjectTrack) -> None:
        hits_in_window = sum(1 for h in track.recent_hits if h)
        support = min(1.0, hits_in_window / max(1, self.cfg.min_hits))
        track.confidence = float(np.clip(track.detection.confidence * (0.35 + 0.65 * support), 0.0, 1.0))

    def confirmed_tracks(self) -> List[ObjectTrack]:
        """Треки, набравшие N подтверждений из последних M кадров."""
        out = []
        for track in self.tracks:
            hits_in_window = sum(1 for h in track.recent_hits if h)
            if hits_in_window >= self.cfg.min_hits and track.misses == 0:
                out.append(track)
        out.sort(key=lambda t: t.detection.distance)
        return out
