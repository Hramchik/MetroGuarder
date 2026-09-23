#!/usr/bin/env python3
"""Дальность детекции: цель подставляется в реальные кадры и проверяется,
с какой дистанции комплекс её видит и с какой — принимает решение о тормозе.

Плотность точек цели рассчитывается по модели, измеренной на этом же
датасете (см. rail_guard/lib/injection.py), поэтому цифра дальности
относится к конкретному лидарному комплекту, а не к абстрактному лидару.

    python3 scripts/eval_detection.py --sequence 9_station_ruebenkamp_9.1
"""

# _bootstrap идёт раньше остальных импортов намеренно: он добавляет пакет
# rail_guard в sys.path и, если на хосте нет numpy/scipy, перезапускает
# скрипт в контейнере с ROS. После `import numpy` было бы уже поздно.
import _bootstrap  # noqa: E402
import argparse
import copy
import os
import sys
from typing import List, Optional

import numpy as np

from rail_guard.lib.config import PipelineConfig, load_yaml
from rail_guard.lib.dataset import Osdar23Sequence
from rail_guard.lib.decision import ACTION_NAMES
from rail_guard.lib.derive import sensor_sector
from rail_guard.lib.sensor_model import measure_beam_density
from rail_guard.lib.injection import box_target, inject, person_target
from rail_guard.lib.pipeline import Pipeline


def without_kinematic_gate(cfg: PipelineConfig) -> PipelineConfig:
    """Копия конфигурации без кинематического гейта (см. run_case)."""
    relaxed = copy.deepcopy(cfg)
    relaxed.decision.kinematic_gate = 0.0
    return relaxed


def gt_arc(frame) -> Optional[np.ndarray]:
    """Коэффициенты дуги (b, a) по размеченным рельсам: y = b*x + a*x^2."""
    gt = frame.gt_centerline
    if gt is None or gt.shape[0] < 3:
        return None
    design = np.stack([gt[:, 0], gt[:, 0] ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(design, gt[:, 1], rcond=None)
    return coef


def run_case(seq: Osdar23Sequence, cfg: PipelineConfig, target, distance: float,
             lateral: float, fill: float, speed_override=None,
             route_prior: bool = False, sensor=None) -> dict:
    """Прогоняет всю последовательность с подставленной целью.

    Цель стоит на одной и той же дальности во всех кадрах: так замер
    дальности детекции не смешивается с замером точности сопровождения.
    Из-за этого приходится отключать кинематический гейт: относительно
    поезда такая цель «летит» с его же скоростью, а гейт именно это и
    отсеивает как артефакт коридора. Сам гейт проверяется на реальных
    записях (docs/EXPERIMENTS.md) и отдельным тестом.
    """
    cfg = without_kinematic_gate(cfg)
    pipeline = Pipeline(cfg)
    rng = np.random.default_rng(20260825)
    detected_frames = 0
    braking_frames = 0
    countable = 0
    confidences: List[float] = []
    injected_points: List[int] = []

    for i in range(len(seq)):
        xyz, intensity, _ = seq.load(i)
        frame = seq.frames[i]
        # Цель ставится на РЕАЛЬНУЮ ось пути (по размеченным рельсам), а не
        # на оценённую: иначе замер дальности детекции превратился бы в замер
        # точности поиска оси, а это разные вещи и меряются они отдельно.
        axis_y, base_z = 0.0, 0.0
        gt = frame.gt_centerline
        if gt is not None and gt.shape[0] >= 3:
            # Размеченная ось короче нужной дальности почти всегда: продолжаем
            # её дугой y = b*x + a*x^2 через начало — так же, как реально идёт путь.
            design = np.stack([gt[:, 0], gt[:, 0] ** 2], axis=1)
            coef, *_ = np.linalg.lstsq(design, gt[:, 1], rcond=None)
            axis_y = float(coef[0] * distance + coef[1] * distance ** 2)
            grade = np.polyfit(gt[:, 0], gt[:, 2], 1)
            base_z = float(np.polyval(grade, min(distance, gt[:, 0].max() * 1.5)))
        elif pipeline._track_model is not None:
            axis_y = float(pipeline._track_model.y_at(distance))
        position = np.array([distance, axis_y + lateral, base_z])
        # Плотность точек цели — из измерения того самого кадра, в который она
        # вставляется, и в направлении, куда она ставится (lib/sensor_model.py).
        # Константа здесь означала бы, что дальность меряется не для этого
        # лидара, а для того, на котором её когда-то сняли.
        xyz, intensity, added = inject(xyz, intensity, target, position, rng,
                                       sensor=sensor, fill=fill)
        injected_points.append(added)

        if route_prior:
            # Ось пути из «карты»: на метрополитене геометрия маршрута известна,
            # и коридор не зависит от того, разрешил ли лидар рельсы.
            coef = gt_arc(frame)
            if coef is not None:
                pipeline.set_route_prior(float(coef[0]), float(coef[1]), valid_range=200.0)
        speed = speed_override if speed_override is not None else (
            None if np.isnan(frame.speed) else frame.speed)
        result = pipeline.process(xyz, intensity, timestamp=frame.timestamp, ego_speed=speed)

        # Сопоставляем по абсолютному положению цели, а не по её смещению
        # от оценённой оси: иначе замер дальности зависел бы от точности
        # поиска оси, которая меряется отдельно.
        hit = None
        for track in result.obstacles:
            det = track.detection
            centre = 0.5 * (det.min_point + det.max_point)
            if np.hypot(centre[0] - position[0], centre[1] - position[1]) < 2.5:
                hit = track
                break
        if i < cfg.tracker.min_hits - 1:
            # Первые кадры трек физически не может быть подтверждён —
            # они не в счёт ни как пропуск, ни как обнаружение.
            continue
        countable += 1
        if hit is not None:
            detected_frames += 1
            confidences.append(hit.confidence)
            if result.decision.action >= 2 and result.decision.nearest_id == hit.id:
                braking_frames += 1

    return {
        "distance": distance,
        "frames": max(countable, 1),
        "detected": detected_frames,
        "braking": braking_frames,
        "confidence": float(np.mean(confidences)) if confidences else 0.0,
        "points": int(np.median(injected_points)) if injected_points else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--data-root", default=_bootstrap.DATA_ROOT)
    ap.add_argument("--config", default=None)
    ap.add_argument("--distances", default="20,40,60,80,100,120,140,160,180,200")
    ap.add_argument("--lateral", type=float, default=0.0, help="смещение цели от оси, м")
    ap.add_argument("--fill", type=float, default=None,
                    help="доля лучей цели, дающих возврат (по умолчанию — из профиля)")
    ap.add_argument("--targets", default="person,box40,box20")
    ap.add_argument("--route-prior", action="store_true",
                    help="ось пути берётся из разметки (эмуляция путевой карты метро)")
    ap.add_argument("--speed", type=float, default=None,
                    help="скорость поезда для расчёта решения, м/с (по умолчанию — из записи)")
    args = ap.parse_args()

    path = args.sequence if os.path.isdir(args.sequence) \
        else os.path.join(args.data_root, args.sequence)
    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    seq = Osdar23Sequence(path)

    catalogue = {
        "person": person_target(),
        "box40": box_target(0.40),
        "box20": box_target(0.20),
        "box60": box_target(0.60),
    }
    distances = [float(d) for d in args.distances.split(",")]

    print(f"Последовательность: {seq.name}, кадров {len(seq)}")
    print("Ось пути: " + ("из разметки (эмуляция путевой карты)" if args.route_prior
                          else "оценивается по облаку точек"))
    # Плотность лучей меряется по кадрам этой же последовательности: она
    # свойство комплекта лидаров, и подставленная цель обязана получить
    # ровно столько точек, сколько дал бы этот комплект.
    fill = args.fill if args.fill is not None else cfg.safety.target_fill
    probe = measure_beam_density(seq.load(0)[0], *sensor_sector(cfg))
    if probe is None:
        print("Не удалось измерить плотность лучей по первому кадру", file=sys.stderr)
        return 1
    print(f"Датчик (измерен по записи): {probe.describe()}")
    print(f"Доля возвращающих лучей цели: {fill:.2f} — проверяется "
          f"scripts/measure_sensor.py по разметке")
    if args.speed is not None:
        from rail_guard.lib.decision import braking_distance
        print(f"Скорость поезда задана: {args.speed:.1f} м/с ({args.speed * 3.6:.0f} км/ч), "
              f"экстренный тормозной путь {braking_distance(args.speed, cfg.decision):.0f} м")
    print()
    summary = {}
    for name in args.targets.split(","):
        target = catalogue[name.strip()]
        print(f"=== цель: {target.name}, фронтальная площадь {target.frontal_area:.2f} м²")
        print(f"{'дальность':>10} {'точек':>7} {'детекций':>9} {'торможений':>11} "
              f"{'уверенность':>12}")
        max_detected = 0.0
        max_braking = 0.0
        for distance in distances:
            res = run_case(seq, cfg, target, distance, args.lateral, fill,
                           args.speed, args.route_prior)
            rate = res["detected"] / max(res["frames"], 1)
            print(f"{distance:>9.0f}м {res['points']:>7d} {res['detected']:>4d}/{res['frames']:<4d} "
                  f"{res['braking']:>6d}/{res['frames']:<4d} {res['confidence']:>12.2f}")
            if rate >= 0.6:
                max_detected = max(max_detected, distance)
            if res["braking"] / max(res["frames"], 1) >= 0.6:
                max_braking = max(max_braking, distance)
        summary[target.name] = (max_detected, max_braking)
        print()

    print("Итог (устойчивая детекция — не менее 60 % кадров):")
    for name, (detect, brake) in summary.items():
        print(f"  {name:20s} обнаружение до {detect:5.0f} м, "
              f"команда торможения до {brake:5.0f} м")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
