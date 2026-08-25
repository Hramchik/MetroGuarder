"""Чтение облаков точек (.pcd) и разметки OpenLABEL (OSDaR23).

Namespace-пакет специально не тянет ROS: тот же код работает и в ноде,
и в офлайн-скриптах оценки качества.
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_PCD_TYPE = {
    ("F", 4): np.float32, ("F", 8): np.float64,
    ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32, ("U", 8): np.uint64,
    ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64,
}


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Распаковка LZF (формат binary_compressed в PCD). Чистый Python —
    вызывается редко, поэтому скорость здесь не критична."""
    out = bytearray()
    idx = 0
    n = len(data)
    while idx < n:
        ctrl = data[idx]
        idx += 1
        if ctrl < 32:                       # литералы
            length = ctrl + 1
            out += data[idx:idx + length]
            idx += length
        else:                               # обратная ссылка
            length = ctrl >> 5
            ref_off = ((ctrl & 0x1f) << 8)
            if length == 7:
                length += data[idx]
                idx += 1
            ref_off |= data[idx]
            idx += 1
            ref = len(out) - ref_off - 1
            if ref < 0:
                raise ValueError("Повреждённый LZF-поток в PCD")
            for _ in range(length + 2):
                out.append(out[ref])
                ref += 1
    if expected_size and len(out) != expected_size:
        raise ValueError(f"LZF: ожидалось {expected_size} байт, получено {len(out)}")
    return bytes(out)


@dataclass
class PointCloud:
    xyz: np.ndarray                     # (N, 3) float32
    intensity: Optional[np.ndarray] = None   # (N,) float32
    ring: Optional[np.ndarray] = None        # (N,) int32
    fields: Optional[Dict[str, np.ndarray]] = None

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def read_pcd(path: str) -> PointCloud:
    """Читает PCD: ascii, binary, binary_compressed."""
    with open(path, "rb") as fh:
        header_lines: List[str] = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: заголовок PCD не завершён")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text.upper().startswith("DATA"):
                break
        header: Dict[str, List[str]] = {}
        for text in header_lines:
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            header[parts[0].upper()] = parts[1:]

        names = header["FIELDS"]
        sizes = [int(v) for v in header["SIZE"]]
        types = [v.upper() for v in header["TYPE"]]
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(names))]
        n_points = int(header["POINTS"][0]) if "POINTS" in header else (
            int(header["WIDTH"][0]) * int(header["HEIGHT"][0]))
        data_kind = header["DATA"][0].lower()

        dtype_items = []
        for name, size, typ, cnt in zip(names, sizes, types, counts):
            np_type = _PCD_TYPE[(typ, size)]
            if cnt == 1:
                dtype_items.append((name, np_type))
            else:
                dtype_items.append((name, np_type, (cnt,)))
        dtype = np.dtype(dtype_items)

        if data_kind == "ascii":
            raw = fh.read().decode("ascii", errors="replace")
            rows = [r for r in raw.splitlines() if r.strip()]
            arr = np.zeros(len(rows), dtype=dtype)
            flat = np.array([r.split() for r in rows], dtype=np.float64)
            for i, name in enumerate(names):
                arr[name] = flat[:, i].astype(arr[name].dtype)
        elif data_kind == "binary":
            arr = np.frombuffer(fh.read(n_points * dtype.itemsize), dtype=dtype, count=n_points)
        elif data_kind == "binary_compressed":
            compressed_size, uncompressed_size = struct.unpack("II", fh.read(8))
            blob = _lzf_decompress(fh.read(compressed_size), uncompressed_size)
            # В сжатом PCD данные лежат «по столбцам», а не по точкам.
            arr = np.zeros(n_points, dtype=dtype)
            offset = 0
            for name in names:
                field_dtype = arr[name].dtype
                nbytes = n_points * field_dtype.itemsize
                arr[name] = np.frombuffer(blob[offset:offset + nbytes], dtype=field_dtype, count=n_points)
                offset += nbytes
        else:
            raise ValueError(f"{path}: неизвестный DATA {data_kind}")

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    extra = {name: np.asarray(arr[name]) for name in names if name not in ("x", "y", "z")}
    for key in list(extra):
        extra[key] = extra[key][finite]
    intensity = None
    for key in ("intensity", "i", "reflectivity"):
        if key in extra:
            intensity = extra[key].astype(np.float32)
            break
    ring = None
    for key in ("ring", "laser_id", "channel"):
        if key in extra:
            ring = extra[key].astype(np.int32)
            break
    return PointCloud(xyz=xyz[finite], intensity=intensity, ring=ring, fields=extra)


# --------------------------------------------------------------------------
# OpenLABEL (разметка OSDaR23)
# --------------------------------------------------------------------------

@dataclass
class GtObject:
    uid: str
    name: str
    obj_type: str
    center: np.ndarray          # (3,)
    size: np.ndarray            # (3,) длина, ширина, высота
    quaternion: np.ndarray      # (4,) x, y, z, w
    attributes: Dict[str, object]

    @property
    def distance(self) -> float:
        return float(np.hypot(self.center[0], self.center[1]))


@dataclass
class GtFrame:
    index: int
    timestamp: float
    lidar_uri: str
    objects: List[GtObject]
    rails: List[np.ndarray] = field(default_factory=list)   # (M,3) полилинии рельсов

    def ego_centerline(self, max_offset: float = 1.2) -> Optional[np.ndarray]:
        """Ось собственного пути как среднее двух рельсов, проходящих под поездом.

        Служит эталоном при оценке точности поиска оси: сам алгоритм
        размеченными рельсами не пользуется.
        """
        ego = []
        for rail in self.rails:
            if rail.shape[0] < 2 or rail[:, 0].min() > 1.0:
                continue
            y0 = float(np.interp(0.0, rail[:, 0], rail[:, 1]))
            if abs(y0) < max_offset:
                ego.append(rail)
        if len(ego) < 2:
            return None
        ego.sort(key=lambda r: float(np.interp(0.0, r[:, 0], r[:, 1])))
        x_max = min(float(r[:, 0].max()) for r in ego[:2])
        xs = np.arange(0.0, x_max + 0.5, 1.0)
        ys = np.mean([np.interp(xs, r[:, 0], r[:, 1]) for r in ego[:2]], axis=0)
        zs = np.mean([np.interp(xs, r[:, 0], r[:, 2]) for r in ego[:2]], axis=0)
        return np.stack([xs, ys, zs], axis=1)


class OpenLabelScene:
    """Минимальный разбор OpenLABEL: кадры, uri облаков, 3D-кубоиды."""

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.raw = data.get("openlabel", data)
        self.root = os.path.dirname(os.path.abspath(path))
        self.objects_meta = self.raw.get("objects", {})
        self.coordinate_systems = self.raw.get("coordinate_systems", {})
        self.frames: List[GtFrame] = []
        for key, frame in sorted(self.raw.get("frames", {}).items(), key=lambda kv: int(kv[0])):
            props = frame.get("frame_properties", {})
            streams = props.get("streams", {}) or props.get("sensors", {})
            lidar_uri, timestamp = "", 0.0
            for stream_name, stream in streams.items():
                if "lidar" not in stream_name.lower():
                    continue
                sp = stream.get("stream_properties", {})
                sync = sp.get("sync", {})
                lidar_uri = stream.get("uri", "") or sync.get("uri", "")
                timestamp = float(sync.get("timestamp", props.get("timestamp", 0.0)) or 0.0)
                break
            if not lidar_uri:
                lidar_uri = props.get("uri", "")
            objs: List[GtObject] = []
            rails: List[np.ndarray] = []
            for uid, obj in (frame.get("objects", {}) or {}).items():
                meta_type = self.objects_meta.get(uid, {}).get("type", "")
                if meta_type == "track":
                    for poly in (obj.get("object_data", {}) or {}).get("poly3d", []) or []:
                        val = np.asarray(poly.get("val", []), dtype=np.float64)
                        if val.size >= 6:
                            rails.append(val.reshape(-1, 3))
                for cuboid in (obj.get("object_data", {}) or {}).get("cuboid", []) or []:
                    if "lidar" not in str(cuboid.get("coordinate_system", "lidar")).lower():
                        continue
                    val = cuboid.get("val", [])
                    if len(val) < 10:
                        continue
                    meta = self.objects_meta.get(uid, {})
                    attrs = {}
                    for group in (cuboid.get("attributes", {}) or {}).values():
                        if isinstance(group, list):
                            for item in group:
                                attrs[item.get("name", "")] = item.get("val")
                    objs.append(GtObject(
                        uid=uid,
                        name=meta.get("name", uid),
                        obj_type=meta.get("type", "unknown"),
                        center=np.array(val[0:3], dtype=np.float64),
                        quaternion=np.array(val[3:7], dtype=np.float64),
                        size=np.array(val[7:10], dtype=np.float64),
                        attributes=attrs,
                    ))
            self.frames.append(GtFrame(index=int(key), timestamp=timestamp,
                                       lidar_uri=lidar_uri, objects=objs, rails=rails))

    def pcd_path(self, frame: GtFrame) -> str:
        uri = frame.lidar_uri.lstrip("./")
        candidate = os.path.join(self.root, uri)
        if os.path.exists(candidate):
            return candidate
        return os.path.join(self.root, os.path.basename(uri))


def find_scene_json(sequence_dir: str) -> Optional[str]:
    """Находит файл разметки внутри распакованной последовательности OSDaR23."""
    best = None
    for name in sorted(os.listdir(sequence_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(sequence_dir, name)
        if re.search(r"label|openlabel|osdar|_\d+\.\d+\.json$", name, re.I) or best is None:
            best = path
    return best


def list_pcd_files(sequence_dir: str, sensor: str = "lidar") -> List[str]:
    """Резервный путь: просто все .pcd нужного сенсора по порядку."""
    lidar_dir = os.path.join(sequence_dir, sensor)
    if not os.path.isdir(lidar_dir):
        matches = []
        for root, _dirs, files in os.walk(sequence_dir):
            matches += [os.path.join(root, f) for f in files if f.endswith(".pcd")]
        return sorted(matches)
    return sorted(os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if f.endswith(".pcd"))
