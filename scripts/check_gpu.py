#!/usr/bin/env python3
"""Проверка видеокарты: считает ли она то же самое и насколько быстрее.

Видеокарта в этом проекте — способ успеть к сроку, а не другой алгоритм.
Значит проверять надо два утверждения, и оба на настоящих кадрах, а не на
синтетике:

1. **Ответ совпадает.** Прогоняем одни и те же кадры дважды — на процессоре
   и на видеокарте — и сравниваем не время, а результат: решение, дальность
   до ближайшей помехи, длину коридора, число точек после фильтрации.
   Расхождение хотя бы в одном кадре означает, что на устройстве считается
   что-то другое, и пользоваться им нельзя.

2. **Выигрыш есть.** Печатается время по этапам для обоих путей: перенос
   кадра на устройство стоит миллисекунд, и на маленьком кадре видеокарта
   проигрывает. Видеть это надо до стенда, а не на стенде.

    python3 scripts/check_gpu.py --bag dataset/for_hackathon/doubleT_platform --frames 40
    python3 scripts/check_gpu.py --sequence 9_station_ruebenkamp_9.1
"""
import _bootstrap  # noqa: F401  (добавляет rail_guard в sys.path)
import argparse
import os
import sys
from typing import List

import numpy as np

from rail_guard.lib import report
from rail_guard.lib.backend import select_backend
from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.pipeline import Pipeline

# Что сравнивается в каждом кадре. Это выход системы, а не внутренние
# величины: совпасть должно именно то, что уходит поезду.
FIELDS = (
    ("решение", lambda r: float(r.decision.action), 0.0),
    ("ближайшая помеха, м", lambda r: float(r.decision.nearest_distance)
     if np.isfinite(r.decision.nearest_distance) else -1.0, 0.05),
    ("длина коридора, м", lambda r: float(r.corridor.reach), 0.01),
    ("ось подтверждена, м", lambda r: float(r.track.fit_range), 0.01),
    ("точек после фильтра", lambda r: float(r.filtered_points), 0.0),
    ("точек-кандидатов", lambda r: float(r.candidate_points), 0.0),
    ("объектов", lambda r: float(len(r.obstacles)), 0.0),
)


def load_frames(args) -> List[tuple]:
    """Кадры из записи ROS 2 или из последовательности OSDaR23."""
    if args.bag:
        from _bagreader import read_frames
        out = []
        for index, frame in enumerate(read_frames(args.bag, args.topic)):
            if index >= args.frames:
                break
            out.append(frame)
        return out
    from rail_guard.lib.dataset import Osdar23Sequence
    path = args.sequence if os.path.isdir(args.sequence) \
        else os.path.join(args.data_root, args.sequence)
    sequence = Osdar23Sequence(path)
    out = []
    for index in range(min(args.frames, len(sequence))):
        xyz, intensity, _ = sequence.load(index)
        out.append((xyz, intensity, sequence.frames[index].timestamp))
    return out


def run(frames: List[tuple], cfg: PipelineConfig, device: str) -> tuple:
    """Прогон всех кадров на заданном устройстве."""
    cfg = PipelineConfig.from_dict(cfg.to_dict())
    cfg.compute.device = device
    pipeline = Pipeline(cfg)
    values: List[List[float]] = []
    stages: dict = {}
    for xyz, intensity, stamp in frames:
        result = pipeline.process(xyz, intensity, timestamp=stamp)
        values.append([getter(result) for _name, getter, _tol in FIELDS])
        for key, value in result.timings.items():
            stages.setdefault(key, []).append(value)
    return np.asarray(values), stages, pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--bag", help="каталог записи rosbag2")
    source.add_argument("--sequence", help="последовательность OSDaR23")
    ap.add_argument("--topic", default="", help="топик облака в записи")
    ap.add_argument("--data-root", default=_bootstrap.DATA_ROOT)
    ap.add_argument("--config", default=None, help="профиль линии")
    ap.add_argument("--frames", type=int, default=40, help="сколько кадров сравнить")
    args = ap.parse_args()

    style = report.Style()
    backend = select_backend("auto")
    print(report.panel("rail-guard · проверка видеокарты", [
        f"устройство: {backend.describe()}",
        f"кадров для сравнения: {args.frames}",
    ], style))
    if not backend.on_gpu:
        print(style("\nВидеокарта недоступна — сравнивать не с чем.", "orange"))
        print("Соберите образ с --build-arg WITH_GPU=true и запустите с --gpus all,")
        print("либо поставьте cupy-cuda12x в окружение.")
        return 2

    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    frames = load_frames(args)
    if not frames:
        print("Кадров не прочитано", file=sys.stderr)
        return 1

    print(style("\nСчитаю на процессоре...", "grey"), flush=True)
    cpu_values, cpu_stages, _ = run(frames, cfg, "cpu")
    print(style("Считаю на видеокарте...", "grey"), flush=True)
    gpu_values, gpu_stages, gpu_pipeline = run(frames, cfg, "gpu")

    print(style("\n Совпадение результата", "bold"))
    ok = True
    for column, (name, _getter, tolerance) in enumerate(FIELDS):
        diff = np.abs(cpu_values[:, column] - gpu_values[:, column])
        worst = float(diff.max()) if diff.size else 0.0
        passed = worst <= tolerance
        ok &= passed
        mark = style("совпадает", "green") if passed else style("РАСХОДИТСЯ", "red", "bold")
        note = "" if passed else f"  (максимум расхождения {worst:.3f}, допуск {tolerance})"
        print(f"  {name:<24} {mark}{note}")

    if gpu_pipeline.gpu_fallback:
        print(style(f"\n  Видеокарта отключилась в работе: {gpu_pipeline.gpu_fallback}",
                    "orange"))
        ok = False

    print(style("\n Время по этапам, мс (медиана)", "bold"))
    print(f"  {'этап':<18} {'процессор':>10} {'видеокарта':>12} {'выигрыш':>10}")
    for key in sorted(set(cpu_stages) | set(gpu_stages)):
        if key == "total_ms":
            continue
        cpu_ms = float(np.median(cpu_stages.get(key, [0.0])))
        gpu_ms = float(np.median(gpu_stages.get(key, [0.0])))
        if max(cpu_ms, gpu_ms) < 0.05:
            continue
        gain = cpu_ms / gpu_ms if gpu_ms > 1e-6 else float("inf")
        colour = "green" if gain > 1.2 else ("orange" if gain < 0.9 else "grey")
        print(f"  {key:<18} {cpu_ms:>10.1f} {gpu_ms:>12.1f} "
              + style(f"{gain:>9.2f}×", colour))
    cpu_total = float(np.median(cpu_stages["total_ms"]))
    gpu_total = float(np.median(gpu_stages["total_ms"]))
    print(f"  {'весь кадр':<18} {cpu_total:>10.1f} {gpu_total:>12.1f} "
          + style(f"{cpu_total / max(gpu_total, 1e-6):>9.2f}×", "bold"))

    print()
    if ok:
        print(style("Видеокарту можно использовать: результат тот же.", "green", "bold"))
        return 0
    print(style("Результаты расходятся — считайте на процессоре (device:=cpu).",
                "red", "bold"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
