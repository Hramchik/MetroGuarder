#!/usr/bin/env python3
"""Видео работы алгоритма по записи ROS 2: тоннель → облако → решение.

Собирает ту самую цепочку, которую требует ТЗ показать на демонстрации, но в
виде файла, а не записи экрана: кадры рисуются из результата пайплайна
(`filtered_xyz`, коридор, подтверждённые рельсами ячейки, рамки объектов),
поэтому на видео видно именно то, что видит алгоритм.

    python3 scripts/make_demo_video.py --bag /путь/doubleT_platform \\
            --frames 150:260 --out docs/demo.mp4

Запускать в окружении с ROS 2 (нужен rosbag2_py для чтения записи) — например,
внутри собранного контейнера:

    docker run --rm -v $PWD:/work -v /путь/к/бэгам:/bags:ro rail-guard \\
        python3 /work/scripts/make_demo_video.py --bag /bags/doubleT_platform \\
                --out /work/docs/demo.mp4
"""
import _bootstrap  # noqa: F401  (добавляет rail_guard в sys.path)
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from _bagreader import read_frames

from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.pipeline import FrameResult, Pipeline

CLASS_TEXT = {0: "неизв.", 1: "человек", 2: "крупный", 3: "мелкий", 4: "инфра"}
ACTION_TEXT = {0: "ПУТЬ СВОБОДЕН", 1: "ВНИМАНИЕ", 2: "СЛУЖЕБНОЕ ТОРМОЖЕНИЕ",
               3: "ЭКСТРЕННОЕ ТОРМОЖЕНИЕ"}
ACTION_COLOR = {0: "#1a7f37", 1: "#b7791f", 2: "#c2410c", 3: "#b42318"}


def draw(result: FrameResult, index: int, path: str, x_max: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    grid = fig.add_gridspec(1, 2, width_ratios=(3.1, 1.0), wspace=0.16)
    ax = fig.add_subplot(grid[0, 0])
    panel = fig.add_subplot(grid[0, 1])
    panel.axis("off")

    points = result.filtered_xyz if result.filtered_xyz is not None else np.zeros((0, 3))
    if points.shape[0]:
        order = np.argsort(points[:, 2])
        ax.scatter(points[order, 0], points[order, 1], s=0.7, c=points[order, 2],
                   cmap="viridis", vmin=-2.0, vmax=3.0, linewidths=0)
    corridor = result.corridor
    if corridor is not None and corridor.s.size:
        ax.plot(corridor.s, corridor.y, "-", color="#b42318", lw=1.4, label="ось пути")
        ax.plot(corridor.s, corridor.y + corridor.half_width, "-", color="#b42318",
                lw=1.0, alpha=0.75, label="границы габарита")
        ax.plot(corridor.s, corridor.y - corridor.half_width, "-", color="#b42318",
                lw=1.0, alpha=0.75)
    track = result.track
    if track is not None and track.bin_x is not None and track.bin_x.size:
        ax.plot(track.bin_x, track.bin_y, "o", ms=3.5, mfc="none", mec="#1a7f37",
                label="рельсы подтверждены")
    for obstacle in result.obstacles:
        det = obstacle.detection
        ax.add_patch(Rectangle((det.min_point[0], det.min_point[1]),
                               max(det.size[0], 0.4), max(det.size[1], 0.4),
                               fill=False, ec="#1f6feb", lw=2.0))
        ax.annotate(f"#{obstacle.id} {CLASS_TEXT.get(int(det.classification), '?')}"
                    f" {det.distance:.0f} м",
                    (det.max_point[0], det.max_point[1]), xytext=(4, 4),
                    textcoords="offset points", fontsize=9, color="#1f6feb")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-11, 11)
    ax.set_xlabel("вперёд, м")
    ax.set_ylabel("влево, м")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"rail-guard: кадр {index}, вид сверху", fontsize=11)

    decision = result.decision
    action = int(decision.action)
    nearest = "—" if not np.isfinite(decision.nearest_distance) \
        else f"{decision.nearest_distance:.0f} м"
    speed = "—" if result.ego_speed is None else f"{result.ego_speed:.1f} м/с"
    rows = [
        ("точек на входе", f"{result.input_points}"),
        ("после фильтрации", f"{result.filtered_points}"),
        ("дальность обзора", f"{result.max_valid_range:.0f} м"),
        ("коридор построен до", f"{corridor.reach:.0f} м" if corridor is not None else "—"),
        ("ось по рельсам до", f"{track.fit_range:.0f} м" if track is not None else "—"),
        ("скорость", f"{speed} ({result.speed_source})"),
        ("тормозной путь", f"{decision.braking_distance:.0f} м"),
        ("помех в габарите", f"{decision.obstacle_count}"),
        ("до ближайшей", nearest),
        ("время кадра", f"{result.timings['total_ms']:.0f} мс"),
    ]
    panel.text(0.0, 0.98, ACTION_TEXT.get(action, "?"), fontsize=15, fontweight="bold",
               color=ACTION_COLOR.get(action, "#000000"), va="top")
    panel.text(0.0, 0.92, decision.reason, fontsize=8.5, color="#374151", va="top",
               wrap=True)
    y = 0.82
    for name, value in rows:
        panel.text(0.0, y, name, fontsize=9, color="#6b7280", va="top")
        panel.text(1.0, y, value, fontsize=9.5, color="#111827", va="top", ha="right")
        y -= 0.052
    panel.text(0.0, y - 0.02, "цвет точек — высота над лидаром", fontsize=8,
               color="#9ca3af", va="top")

    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="каталог записи rosbag2")
    ap.add_argument("--topic", default="", help="топик облака (по умолчанию — первый найденный)")
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ros2_ws", "src", "rail_guard_bringup", "config", "metro.yaml"))
    ap.add_argument("--frames", default="", help="диапазон кадров вида 150:260")
    ap.add_argument("--out", default="docs/demo.mp4")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--x-max", type=float, default=100.0, help="глубина вида, м")
    ap.add_argument("--keep-frames", default="", help="куда сложить PNG (по умолчанию — временно)")
    args = ap.parse_args()

    first, last = 0, 10 ** 9
    if args.frames:
        parts = args.frames.split(":")
        first = int(parts[0] or 0)
        last = int(parts[1]) if len(parts) > 1 and parts[1] else 10 ** 9

    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    cfg.debug = True                     # нужны отфильтрованные точки для картинки
    pipeline = Pipeline(cfg)

    workdir = args.keep_frames or tempfile.mkdtemp(prefix="rail-guard-demo-")
    os.makedirs(workdir, exist_ok=True)
    rendered = 0
    for index, (xyz, intensity, stamp) in enumerate(read_frames(args.bag, args.topic)):
        if index > last:
            break
        result = pipeline.process(xyz, intensity, timestamp=stamp)
        if index < first:
            continue                     # тракт «прогревается»: треки и ось нужны с историей
        draw(result, index, os.path.join(workdir, f"frame_{rendered:05d}.png"), args.x_max)
        rendered += 1
        if rendered % 20 == 0:
            print(f"  нарисовано кадров: {rendered}", flush=True)

    if not rendered:
        print("Нечего рисовать: проверьте --frames и имя топика", file=sys.stderr)
        return 1
    if shutil.which("ffmpeg") is None:
        print(f"ffmpeg не найден: кадры остались в {workdir}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
           "-i", os.path.join(workdir, "frame_%05d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", args.out]
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Готово: {args.out} — {rendered} кадров, {rendered / args.fps:.0f} с, "
          f"{size_mb:.1f} МБ")
    if not args.keep_frames:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
