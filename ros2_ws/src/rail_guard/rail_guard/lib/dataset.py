"""Доступ к последовательностям OSDaR23 (реальные записи лидара с поезда).

Датасет: «Open Sensor Data for Rail 2023», DZSF / DB Netz AG, CC-BY 4.0,
записан в Гамбурге. Шесть лидаров сведены в одно облако (поле
sensor_index), система координат — X вперёд, Y влево, Z вверх,
начало на уровне головок рельсов под носом поезда.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .pcio import GtObject, OpenLabelScene, read_pcd

# sensor_index в PCD → модель лидара (см. readme последовательности)
SENSOR_NAMES = {
    0: "pandar64",       # средняя дальность, 360 гр.
    1: "tele15_right",   # дальнобойные Livox Tele-15, узкий FOV вперёд
    2: "tele15_center",
    3: "tele15_left",
    4: "honeycomb_left",  # ближняя зона Waymo Honeycomb
    5: "honeycomb_right",
}
LONG_RANGE_SENSORS = (1, 2, 3)


@dataclass
class Frame:
    index: int
    timestamp: float
    pcd_path: str
    gt_objects: List[GtObject] = field(default_factory=list)
    speed: float = float("nan")     # м/с из INSPVA, nan если нет данных
    gt_centerline: Optional[np.ndarray] = None   # (M,3) эталонная ось своего пути


class Osdar23Sequence:
    """Одна последовательность: кадры, разметка, скорость поезда."""

    def __init__(self, path: str, use_cache: bool = True):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path.rstrip("/"))
        self.use_cache = use_cache
        self.cache_dir = os.path.join(self.path, ".npy_cache")
        label_files = sorted(glob.glob(os.path.join(self.path, "*_labels.json")))
        self.scene: Optional[OpenLabelScene] = OpenLabelScene(label_files[0]) if label_files else None
        self.speeds = self._read_speeds()
        self.frames: List[Frame] = []

        if self.scene is not None:
            for gt in self.scene.frames:
                pcd = self.scene.pcd_path(gt)
                self.frames.append(Frame(index=gt.index, timestamp=gt.timestamp,
                                         pcd_path=pcd, gt_objects=gt.objects,
                                         speed=self.speeds.get(gt.index, float("nan")),
                                         gt_centerline=gt.ego_centerline()))
        else:
            for pcd in sorted(glob.glob(os.path.join(self.path, "lidar", "*.pcd"))):
                base = os.path.basename(pcd)
                idx = int(base.split("_")[0])
                stamp = float(base.split("_", 1)[1].rsplit(".", 1)[0])
                self.frames.append(Frame(index=idx, timestamp=stamp, pcd_path=pcd,
                                         speed=self.speeds.get(idx, float("nan"))))
        self.frames.sort(key=lambda f: f.index)

    def _read_speeds(self) -> Dict[int, float]:
        speeds: Dict[int, float] = {}
        for csv_path in glob.glob(os.path.join(self.path, "novatel_oem7_inspva", "*.csv")):
            try:
                with open(csv_path, "r", encoding="utf-8") as fh:
                    header = fh.readline().strip().split(",")
                    values = fh.readline().strip().split(",")
                if len(values) < len(header):
                    continue
                row = dict(zip(header, values))
                north = float(row["north_velocity"])
                east = float(row["east_velocity"])
                speeds[int(float(row["frame_idx"]))] = float(np.hypot(north, east))
            except (ValueError, KeyError, OSError):
                continue
        return speeds

    def __len__(self) -> int:
        return len(self.frames)

    def load(self, i: int, sensors: Optional[Tuple[int, ...]] = None
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Возвращает (xyz, intensity, sensor_index) i-го кадра.

        Разбор ASCII-PCD на 140 тысяч точек стоит около 0.4 с, поэтому
        результат кешируется в .npy рядом с последовательностью —
        воспроизведение в реальном времени иначе невозможно.
        """
        frame = self.frames[i]
        cache = os.path.join(self.cache_dir, os.path.basename(frame.pcd_path) + ".npz")
        if self.use_cache and os.path.exists(cache):
            data = np.load(cache)
            xyz, intensity, sidx = data["xyz"], data["intensity"], data["sensor_index"]
        else:
            cloud = read_pcd(frame.pcd_path)
            xyz = cloud.xyz
            intensity = cloud.intensity if cloud.intensity is not None \
                else np.zeros(len(cloud), dtype=np.float32)
            fields = cloud.fields or {}
            sidx = fields.get("sensor_index")
            sidx = np.zeros(len(cloud), dtype=np.int8) if sidx is None else sidx.astype(np.int8)
            if self.use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)
                # Без сжатия: распаковка zlib на 140 тысячах точек стоит
                # больше 200 мс и рушит воспроизведение в реальном темпе.
                np.savez(cache, xyz=xyz, intensity=intensity, sensor_index=sidx)
        if sensors is not None:
            mask = np.isin(sidx, sensors)
            xyz, intensity, sidx = xyz[mask], intensity[mask], sidx[mask]
        return xyz, intensity, sidx


def find_sequences(root: str) -> List[str]:
    """Все распакованные последовательности внутри каталога."""
    out = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "lidar")):
            out.append(path)
    return out
