#!/usr/bin/env python3
"""Измерение датчика по записи и проверка модели числа отражений.

Все пороги детекции считаются из одной величины — плотности лучей в телесном
угле, точек на стерадиан. Она измеряется по самим кадрам, а не задаётся
константой, и этот скрипт показывает, что именно система измерила на ваших
данных и насколько эта модель сходится с действительностью.

Проверка (только там, где есть разметка): для каждого размеченного объекта
считается, сколько точек реально лежит внутри его кубоида, и сравнивается с
предсказанием k·A/r². Отношение факта к предсказанию — это и есть `fill`,
доля лучей, дающих полезный возврат: силуэт занимает не весь габаритный
прямоугольник, тёмная поверхность и косой угол падения съедают часть. Если
медиана по вашим данным заметно отличается от `safety.target_fill` в профиле,
её и надо туда поставить — это единственный коэффициент модели, и он
измеряется, а не подбирается.

    python3 scripts/measure_sensor.py --sequence 9_station_ruebenkamp_9.1
    python3 scripts/measure_sensor.py --bag /путь/к/записи --frames 20
"""
import _bootstrap  # noqa: F401  (добавляет rail_guard в sys.path)
import argparse
import os
import sys
from collections import defaultdict
from typing import List, Optional

import numpy as np

from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.derive import detection_limit, min_target_area, sensor_sector
from rail_guard.lib.frames import FrameNormalizer
from rail_guard.lib.sensor_model import SensorCalibrator


def _load_frames(args) -> List[np.ndarray]:
    """Кадры из последовательности OSDaR23, каталога .pcd или записи rosbag2."""
    if args.bag:
        from _bagreader import read_frames
        clouds = []
        for index, (xyz, _intensity, _stamp) in enumerate(read_frames(args.bag, args.topic)):
            if index >= args.frames:
                break
            clouds.append(xyz)
        return clouds
    from rail_guard.lib.dataset import Osdar23Sequence
    path = args.sequence if os.path.isdir(args.sequence) \
        else os.path.join(args.data_root, args.sequence)
    sequence = Osdar23Sequence(path)
    return sequence, [sequence.load(i)[0] for i in range(min(args.frames, len(sequence)))]


def _check_against_labels(sequence, frames: List[np.ndarray], profile,
                          limit: int) -> None:
    """Сравнивает предсказанное число отражений с фактом по разметке."""
    by_type = defaultdict(list)
    for index, xyz in enumerate(frames[:limit]):
        for obj in sequence.frames[index].gt_objects:
            if float(obj.size.min()) <= 0.0:
                continue
            distance = obj.distance
            if distance < 5.0:
                continue
            inside = np.all(np.abs(xyz - obj.center) <= obj.size / 2.0, axis=1)
            actual = int(inside.sum())
            if actual < 4:
                continue
            area = float(obj.size[1] * obj.size[2])
            predicted = profile.expected_points(area, distance, fill=1.0)
            if predicted <= 0.0:
                continue
            by_type[obj.obj_type].append(actual / predicted)

    if not by_type:
        print("\nРазметки в этой последовательности нет — проверить модель не по чему.")
        return

    print("\nПроверка модели по разметке: доля лучей, давших возврат (fill)")
    print(f"{'тип объекта':<24}{'объектов':>9}{'fill медиана':>14}{'разброс p25-p75':>20}")
    everything: List[float] = []
    for name, values in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        arr = np.asarray(values)
        everything.extend(values)
        print(f"{name:<24}{arr.size:>9}{np.median(arr):>14.2f}"
              f"{np.percentile(arr, 25):>12.2f}-{np.percentile(arr, 75):<8.2f}")
    combined = np.asarray(everything)
    print(f"\nПо всем объектам: fill медиана {np.median(combined):.2f} "
          f"({combined.size} измерений)")
    print("Столбчатые конструкции (опоры, сигналы) дают fill заметно ниже: столб "
          "занимает малую долю своего кубоида. Для порогов детекции ориентируйтесь "
          "на компактные цели — людей и предметы.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence", help="последовательность OSDaR23 или каталог .pcd")
    source.add_argument("--bag", help="каталог записи rosbag2")
    ap.add_argument("--topic", default="", help="топик облака в записи")
    ap.add_argument("--data-root", default=_bootstrap.DATA_ROOT)
    ap.add_argument("--config", default=None, help="профиль носителя (для сектора и целей)")
    ap.add_argument("--frames", type=int, default=10, help="сколько кадров измерять")
    args = ap.parse_args()

    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    problems = cfg.validate()
    if problems:
        print("Замечания к профилю:", file=sys.stderr)
        for item in problems:
            print(f"   {item}", file=sys.stderr)

    loaded = _load_frames(args)
    sequence, frames = loaded if isinstance(loaded, tuple) else (None, loaded)
    if not frames:
        print("Кадров не прочитано", file=sys.stderr)
        return 1

    # Облака приводятся к рабочей СК: передний сектор, в котором меряется
    # плотность, определён относительно направления движения.
    normalizer = FrameNormalizer(cfg.sensor)
    sector = sensor_sector(cfg)
    calibrator = SensorCalibrator(sector, warmup_frames=len(frames), refresh_every=0)
    for xyz in frames:
        rotated, _decided = normalizer.apply(xyz)
        calibrator.update(rotated)

    profile = calibrator.profile
    if not profile.valid:
        print("Плотность лучей измерить не удалось: в переднем секторе слишком "
              "мало точек", file=sys.stderr)
        return 1

    print(f"Кадров измерено: {len(frames)}   СК датчика: "
          f"{normalizer.spec or 'не определена'} ({normalizer.source or '—'})")
    print(f"Сектор измерения: ±{np.degrees(sector[0]):.1f}° по азимуту, "
          f"±{np.degrees(sector[1]):.1f}° по элевации — угловой размер габарита "
          f"на опорной дальности")
    print(f"Датчик: {profile.describe()}")

    area = min_target_area(cfg)
    fill = cfg.safety.target_fill
    print(f"\nМинимальная цель из профиля: {cfg.safety.min_target_width:.2f} × "
          f"{cfg.safety.min_target_height:.2f} м (фронтальная площадь {area:.2f} м²)")
    print(f"{'дальность':>10}{'ожидается отражений':>22}")
    for distance in (20.0, 40.0, 60.0, 80.0, 100.0, 140.0, 200.0):
        expected = profile.expected_points(area, distance, fill)
        if expected < 0.5:
            break
        print(f"{distance:>9.0f}м{expected:>22.0f}")
    print(f"\nПредел обнаружения минимальной цели: {detection_limit(cfg, profile):.0f} м "
          f"(дальше от неё приходит меньше {cfg.cluster.abs_min_points} отражений — "
          f"это предел датчика, а не алгоритма)")

    if sequence is not None:
        _check_against_labels(sequence, frames, profile, len(frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
