#!/usr/bin/env python3
"""Прогон пайплайна по последовательности OSDaR23 без ROS.

Нужен для двух вещей: отладки алгоритма на реальных данных и замера
времени обработки кадра — то же самое, что будет делать ROS-нода, но
без накладных расходов транспорта.

    python3 scripts/run_offline.py --sequence 7_approach_underground_station_7.1
"""

# _bootstrap идёт раньше остальных импортов намеренно: он добавляет пакет
# rail_guard в sys.path и, если на хосте нет numpy/scipy, перезапускает
# скрипт в контейнере с ROS. После `import numpy` было бы уже поздно.
import _bootstrap  # noqa: E402
import argparse
import os
import sys

import numpy as np

from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.dataset import Osdar23Sequence
from rail_guard.lib.pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sequence", required=True,
                        help="имя каталога последовательности или полный путь")
    parser.add_argument("--data-root", default=_bootstrap.DATA_ROOT)
    parser.add_argument("--config", default=None, help="YAML с параметрами пайплайна")
    parser.add_argument("--frames", type=int, default=0, help="0 — все кадры")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--route-prior", action="store_true",
                        help="ось пути из разметки — эмуляция путевой карты метро")
    args = parser.parse_args()

    path = args.sequence if os.path.isdir(args.sequence) \
        else os.path.join(args.data_root, args.sequence)
    if not os.path.isdir(path):
        print(f"Нет такой последовательности: {path}", file=sys.stderr)
        return 1

    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    cfg.debug = args.debug or cfg.debug
    seq = Osdar23Sequence(path)
    pipeline = Pipeline(cfg)

    n = len(seq) if args.frames <= 0 else min(args.frames, len(seq))
    print(f"Последовательность: {seq.name}   кадров: {len(seq)}   обрабатываем: {n}")
    print(f"{'кадр':>5} {'точек':>8} {'после':>7} {'канд':>6} {'ось,м':>6} {'R,м':>7} "
          f"{'объектов':>8} {'ближ,м':>7} {'решение':<16} {'мс':>6}")
    total_ms = []
    actions = {0: 0, 1: 0, 2: 0, 3: 0}
    for i in range(n):
        xyz, intensity, _sidx = seq.load(i)
        frame = seq.frames[i]
        if args.route_prior and frame.gt_centerline is not None \
                and frame.gt_centerline.shape[0] >= 3:
            gt = frame.gt_centerline
            design = np.stack([gt[:, 0], gt[:, 0] ** 2], axis=1)
            coef, *_ = np.linalg.lstsq(design, gt[:, 1], rcond=None)
            pipeline.set_route_prior(float(coef[0]), float(coef[1]), valid_range=200.0)
        speed = None if np.isnan(frame.speed) else frame.speed
        res = pipeline.process(xyz, intensity, timestamp=frame.timestamp, ego_speed=speed)
        total_ms.append(res.timings["total_ms"])
        actions[res.decision.action] = actions.get(res.decision.action, 0) + 1
        radius = res.track.curvature_radius
        nearest = res.decision.nearest_distance
        if not args.quiet:
            print(f"{frame.index:>5} {res.input_points:>8} {res.filtered_points:>7} "
                  f"{res.candidate_points:>6} {res.track.fit_range:>6.0f} "
                  f"{(radius if np.isfinite(radius) else 9999):>7.0f} "
                  f"{len(res.obstacles):>8} "
                  f"{(nearest if np.isfinite(nearest) else -1):>7.1f} "
                  f"{res.decision.action_name:<16} {res.timings['total_ms']:>6.1f}")
    ms = np.array(total_ms)
    print(f"\nВремя кадра: среднее {ms.mean():.1f} мс, медиана {np.median(ms):.1f} мс, "
          f"максимум {ms.max():.1f} мс  →  {1000.0 / max(ms.mean(), 1e-6):.1f} Гц")
    print(f"Решения: свободно {actions[0]}, внимание {actions[1]}, "
          f"служебное торможение {actions[2]}, экстренное {actions[3]} (из {n} кадров)")
    if args.debug:
        stages = {}
        for key in res.timings:
            stages[key] = res.timings[key]
        print("Время по этапам последнего кадра: "
              + ", ".join(f"{k.replace('_ms','')} {v:.1f}" for k, v in stages.items() if k != "total_ms"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
