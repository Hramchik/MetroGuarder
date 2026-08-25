#!/usr/bin/env python3
"""Точность поиска оси пути относительно размеченных рельсов OSDaR23.

Алгоритм ищет ось только по облаку точек; poly3d-разметка рельсов из
датасета используется исключительно как эталон для замера ошибки.
"""
import argparse
import os

import numpy as np

import _bootstrap  # noqa: F401
from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.dataset import Osdar23Sequence
from rail_guard.lib.pipeline import Pipeline

BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--data-root", default=_bootstrap.DATA_ROOT)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    path = args.sequence if os.path.isdir(args.sequence) \
        else os.path.join(args.data_root, args.sequence)
    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    seq = Osdar23Sequence(path)
    pipe = Pipeline(cfg)

    errors = {band: [] for band in BANDS}
    print(f"{'кадр':>5} {'ось,м':>6} {'R,м':>7}  ошибка оси по дальности, м")
    for i in range(len(seq)):
        xyz, intensity, _ = seq.load(i)
        fr = seq.frames[i]
        speed = None if np.isnan(fr.speed) else fr.speed
        res = pipe.process(xyz, intensity, timestamp=fr.timestamp, ego_speed=speed)
        gt = fr.gt_centerline
        if gt is None:
            print(f"{fr.index:>5}  нет эталонной оси в разметке")
            continue
        est_y = res.track.y_at(gt[:, 0])
        err = np.abs(est_y - gt[:, 1])
        row = []
        for band in BANDS:
            m = (gt[:, 0] >= band[0]) & (gt[:, 0] < band[1])
            if m.sum():
                errors[band].append(float(err[m].mean()))
                row.append(f"{band[0]}-{band[1]}м: {err[m].mean():.2f}")
        radius = res.track.curvature_radius
        print(f"{fr.index:>5} {res.track.fit_range:>6.0f} "
              f"{(radius if np.isfinite(radius) else 9999):>7.0f}  " + "  ".join(row))

    print("\nСредняя ошибка оси пути по всей последовательности:")
    for band, values in errors.items():
        if values:
            print(f"  {band[0]:>3}-{band[1]:<3} м: {np.mean(values):.2f} м "
                  f"(макс {np.max(values):.2f} м, кадров {len(values)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
