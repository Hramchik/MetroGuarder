#!/usr/bin/env python3
"""Отрисовка кадра: вид сверху и вид сбоку с коридором габарита,
найденной осью пути, детекциями и эталонной разметкой.

    python3 scripts/plot_frame.py --sequence 7_approach_underground_station_7.1 --frame 9
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import _bootstrap  # noqa: F401
from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.dataset import Osdar23Sequence
from rail_guard.lib.injection import box_target, inject, person_target
from rail_guard.lib.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--data-root", default=_bootstrap.DATA_ROOT)
    ap.add_argument("--frame", type=int, default=-1, help="-1 — последний кадр")
    ap.add_argument("--config", default=None)
    ap.add_argument("--range", type=float, default=200.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inject", type=float, default=0.0,
                    help="подставить цель на указанной дальности, м")
    ap.add_argument("--inject-target", default="person", choices=["person", "box40", "box20"])
    ap.add_argument("--route-prior", action="store_true",
                    help="ось пути из разметки — эмуляция путевой карты")
    args = ap.parse_args()

    path = args.sequence if os.path.isdir(args.sequence) \
        else os.path.join(args.data_root, args.sequence)
    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    cfg.debug = True
    seq = Osdar23Sequence(path)
    pipe = Pipeline(cfg)

    target = len(seq) - 1 if args.frame < 0 else args.frame
    res = None
    rng = np.random.default_rng(20260825)
    injected = None
    catalogue = {"person": person_target(), "box40": box_target(0.40), "box20": box_target(0.20)}
    for i in range(target + 1):                       # треки нужны «прогретые»
        xyz, intensity, _ = seq.load(i)
        fr = seq.frames[i]
        gt = fr.gt_centerline
        if (args.route_prior or args.inject > 0.0) and gt is not None and gt.shape[0] >= 3:
            design = np.stack([gt[:, 0], gt[:, 0] ** 2], axis=1)
            coef, *_ = np.linalg.lstsq(design, gt[:, 1], rcond=None)
            if args.route_prior:
                pipe.set_route_prior(float(coef[0]), float(coef[1]), valid_range=200.0)
            if args.inject > 0.0:
                grade = np.polyfit(gt[:, 0], gt[:, 2], 1)
                injected = np.array([args.inject,
                                     float(coef[0] * args.inject + coef[1] * args.inject ** 2),
                                     float(np.polyval(grade, args.inject))])
                xyz, intensity, _added = inject(xyz, intensity, catalogue[args.inject_target],
                                                injected, rng)
        speed = None if np.isnan(fr.speed) else fr.speed
        res = pipe.process(xyz, intensity, timestamp=fr.timestamp, ego_speed=speed)

    pts = res.filtered_xyz
    heights = res.ground.height_above_rail(pts)
    fig, (ax_top, ax_side) = plt.subplots(2, 1, figsize=(16, 11), gridspec_kw={"height_ratios": [2, 1]})

    sel = pts[:, 0] < args.range
    ax_top.scatter(pts[sel, 0], pts[sel, 1], s=0.25, c=np.clip(heights[sel], -1, 5),
                   cmap="viridis", alpha=0.55, linewidths=0)
    s = res.corridor.s
    y = res.corridor.y
    hw = res.corridor.half_width
    ax_top.plot(s, y, "r-", lw=1.2, label="ось пути (оценка)")
    ax_top.plot(s, y + hw, "r--", lw=1.0, label="граница габарита")
    ax_top.plot(s, y - hw, "r--", lw=1.0)
    if res.track.bin_x.size:
        ax_top.plot(res.track.bin_x, res.track.bin_y, "wo", ms=4, mec="k",
                    label="рельсовая пара по ячейкам")
    for gt in seq.frames[target].gt_objects:
        cx, cy = gt.center[0], gt.center[1]
        lx, ly = gt.size[0], gt.size[1]
        ax_top.add_patch(Rectangle((cx - lx / 2, cy - ly / 2), lx, ly, fill=False,
                                   ec="orange", lw=0.8))
        if abs(cy) < 6:
            ax_top.text(cx, cy + 1.2, gt.obj_type, color="orange", fontsize=7)
    for det in res.detections:
        colour = "lime" if not det.reject_reason else "grey"
        ax_top.add_patch(Rectangle((det.min_point[0], det.min_point[1]),
                                   max(det.size[0], 0.4), max(det.size[1], 0.4),
                                   fill=False, ec=colour, lw=1.6))
        label = det.class_name if not det.reject_reason else det.reject_reason
        ax_top.text(det.centroid[0], det.centroid[1] - 1.5,
                    f"{label} {det.point_count}т", color=colour, fontsize=7)
    if injected is not None:
        ax_top.plot(injected[0], injected[1], "m*", ms=16,
                    label=f"подставленная цель на {args.inject:.0f} м")
    ax_top.set_xlim(0, args.range)
    ax_top.set_ylim(-15, 15)
    ax_top.set_xlabel("x, м (вперёд)")
    ax_top.set_ylabel("y, м (влево)")
    ax_top.set_title(f"{seq.name}, кадр {seq.frames[target].index} — вид сверху. "
                     f"{res.decision.action_name}: {res.decision.reason}")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(alpha=0.2)

    near = sel & (np.abs(res.corridor.lateral(pts)) < 4.0)
    ax_side.scatter(pts[near, 0], pts[near, 2], s=0.3, c="steelblue", alpha=0.5, linewidths=0)
    ax_side.plot(res.ground.centers, res.ground.z0, "g-", lw=1.2, label="полотно (оценка)")
    ax_side.plot(res.ground.centers, res.ground.z0 + cfg.gauge.height, "r--", lw=1.0,
                 label="верх габарита")
    ax_side.plot(res.ground.centers, res.ground.z0 + cfg.gauge.bottom, "r--", lw=1.0)
    ax_side.set_xlim(0, args.range)
    ax_side.set_ylim(-4, 8)
    ax_side.set_xlabel("x, м")
    ax_side.set_ylabel("z, м")
    ax_side.legend(loc="upper right", fontsize=8)
    ax_side.grid(alpha=0.2)

    out = args.out or os.path.join("/tmp", f"{seq.name}_f{seq.frames[target].index}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"Сохранено: {out}")
    print(f"Решение: {res.decision.action_name} — {res.decision.reason}")
    for det in res.detections:
        print(f"  x={det.distance:7.1f} y={det.lateral_offset:6.2f} h={det.height_above_rail:5.2f}"
              f" размер {det.size[0]:.2f}x{det.size[1]:.2f}x{det.size[2]:.2f}"
              f" точек {det.point_count:4d} → {det.class_name}"
              f"{' (отклонён: ' + det.reject_reason + ')' if det.reject_reason else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
