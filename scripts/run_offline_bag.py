#!/usr/bin/env python3
"""Прогон пайплайна по записи ROS 2 без ROS-транспорта.

Тот же объект `Pipeline`, что крутится в ноде, но кадры читаются из записи
напрямую. Нужен для двух вещей: обрабатываются **все** кадры (в ROS-цепочке
часть теряется в транспорте на 8-24 МБ сообщениях, и это мешает сравнивать
варианты алгоритма между собой), и любой параметр можно переопределить из
командной строки — на этом скрипте измерены все цифры docs/EXPERIMENTS.md.

    python3 scripts/run_offline_bag.py --bag /путь/doubleT_platform
    python3 scripts/run_offline_bag.py --bag /путь/doubleT_obstacle \\
            --topic /sensing/lidar/hesai128/pointcloud --quiet --reasons
    python3 scripts/run_offline_bag.py --bag /путь/doubleT_platform --quiet \\
            --set tunnel.enabled=false gauge.extrapolation_margin=0

Запускать в окружении с ROS 2 (нужен `rosbag2_py` для чтения записи).
"""
import _bootstrap  # noqa: F401  (добавляет rail_guard в sys.path)
import argparse
import os
import time

import numpy as np
from _bagreader import read_frames

from rail_guard.lib.config import PipelineConfig, load_yaml, parse_param
from rail_guard.lib.pipeline import Pipeline

ACTIONS = {0: "свободен", 1: "внимание", 2: "служебное", 3: "ЭКСТРЕННОЕ"}
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ros2_ws", "src", "rail_guard_bringup", "config", "metro.yaml")


def apply_overrides(cfg: PipelineConfig, overrides) -> None:
    """Переопределение параметров вида `секция.поле=значение`.

    Поле, которое система определяет сама, имеет умолчание `None`, и тип
    восстанавливается по самому значению: `auto` возвращает автоматику,
    число или true/false её подавляют.
    """
    for item in overrides:
        key, value = item.split("=", 1)
        section, field = key.split(".", 1)
        target = getattr(cfg, section)
        current = getattr(target, field)
        if current is None:
            parsed = parse_param(value)
        elif isinstance(current, bool):
            parsed = value.strip().lower() in ("1", "true", "yes", "on")
        else:
            parsed = type(current)(value)
        setattr(target, field, parsed)
        print(f"переопределено: {key} = {parsed} (было {current})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="каталог записи rosbag2")
    ap.add_argument("--topic", default="", help="топик облака (по умолчанию — первый найденный)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="профиль параметров")
    ap.add_argument("--frames", type=int, default=0, help="0 — все кадры записи")
    ap.add_argument("--speed", type=float, default=None,
                    help="скорость на /train/speed, м/с (по умолчанию — оценка по облакам)")
    ap.add_argument("--quiet", action="store_true", help="только итоговая сводка")
    ap.add_argument("--reasons", action="store_true", help="гистограмма причин отсева кластеров")
    ap.add_argument("--stages", action="store_true", help="время по стадиям тракта")
    ap.add_argument("--set", nargs="*", default=[], metavar="секция.поле=значение")
    args = ap.parse_args()

    cfg = load_yaml(args.config) if args.config else PipelineConfig()
    apply_overrides(cfg, args.set)
    cfg.debug = cfg.debug or args.reasons
    pipeline = Pipeline(cfg)

    speed_note = "из облаков" if args.speed is None else f"{args.speed} м/с"
    print(f"запись: {os.path.basename(args.bag.rstrip('/'))}   "
          f"профиль: {os.path.basename(args.config)}   скорость: {speed_note}")
    if not args.quiet:
        print(f"{'кадр':>5} {'точек':>8} {'после':>7} {'ось,м':>6} {'R,м':>7} {'кор,м':>6} "
              f"{'помех':>6} {'ближ,м':>7} {'v,м/с':>6} {'источник':>9} "
              f"{'решение':<10} {'мс':>6}")

    actions = {0: 0, 1: 0, 2: 0, 3: 0}
    latency, fit_range, reach, nearest, speeds = [], [], [], [], []
    sources, reasons, stages = {}, {}, {}
    rails = degenerate = detections = frames_with_objects = 0

    for index, (xyz, intensity, stamp) in enumerate(read_frames(args.bag, args.topic)):
        if args.frames and index >= args.frames:
            break
        started = time.perf_counter()
        result = pipeline.process(xyz, intensity, timestamp=stamp, ego_speed=args.speed)
        latency.append((time.perf_counter() - started) * 1e3)

        actions[int(result.decision.action)] += 1
        rails += int(result.track.from_rails)
        fit_range.append(float(result.track.fit_range))
        reach.append(float(result.corridor.reach))
        degenerate += int(result.degenerate)
        detections += len(result.obstacles)
        frames_with_objects += int(bool(result.obstacles))
        sources[result.speed_source] = sources.get(result.speed_source, 0) + 1
        if result.ego_speed is not None:
            speeds.append(float(result.ego_speed))
        if result.obstacles:
            nearest.append(float(result.decision.nearest_distance))
        for key, value in result.timings.items():
            stages.setdefault(key, []).append(value)
        if cfg.debug:
            for det in result.detections:
                key = det.reject_reason or "принят"
                reasons[key] = reasons.get(key, 0) + 1
        if result.frame_decided:
            print(f"      СК датчика: {result.frame_spec} — {result.frame_source}")
        if not args.quiet:
            radius = result.track.curvature_radius
            # Значения готовим заранее: перенос строки внутри фигурных скобок
            # f-строки до Python 3.12 — синтаксическая ошибка, а в контейнере 3.10.
            radius_text = abs(radius) if np.isfinite(radius) else 0.0
            near = result.decision.nearest_distance
            near_text = near if np.isfinite(near) else 0.0
            speed_text = result.ego_speed if result.ego_speed is not None else 0.0
            print(f"{index:5d} {result.input_points:8d} {result.filtered_points:7d} "
                  f"{result.track.fit_range:6.0f} {radius_text:7.0f} "
                  f"{result.corridor.reach:6.0f} {len(result.obstacles):6d} "
                  f"{near_text:7.1f} {speed_text:6.1f} "
                  f"{result.speed_source:>9} {ACTIONS[int(result.decision.action)]:<10} "
                  f"{latency[-1]:6.1f}")

    total = sum(actions.values())
    if not total:
        print("Кадров не прочитано: проверьте путь к записи и имя топика")
        return 1

    print(f"\nкадров: {total}")
    print("решения: " + ", ".join(f"{ACTIONS[key]}={value} ({100 * value / total:.0f} %)"
                                  for key, value in actions.items() if value))
    lat = np.array(latency)
    print(f"время кадра, мс: медиана {np.median(lat):.1f}  p95 {np.percentile(lat, 95):.1f}  "
          f"максимум {lat.max():.1f}; дольше 100 мс — {int((lat > 100).sum())} кадров "
          f"({100 * (lat > 100).mean():.0f} %)")
    print(f"ось по рельсам: {rails}/{total} кадров, подтверждена до "
          f"{np.median(fit_range):.0f} м (медиана), максимум {max(fit_range):.0f} м")
    print(f"длина коридора, м: медиана {np.median(reach):.0f}, максимум {max(reach):.0f}")
    print(f"детекций: {detections} в {frames_with_objects} кадрах"
          + (f"; ближайшая помеха медиана {np.median(nearest):.1f} м" if nearest else ""))
    if speeds:
        print(f"скорость носителя: медиана {np.median(speeds):.1f} м/с, "
              f"максимум {max(speeds):.1f} м/с; источник: "
              + ", ".join(f"{key}={value}" for key, value in sources.items()))
    if degenerate:
        print(f"ВЫРОЖДЕННЫХ КАДРОВ: {degenerate} — облако почти целиком уходит в фильтр")
    if args.stages:
        print("время по стадиям, мс (медиана):")
        for key, values in sorted(stages.items(), key=lambda kv: -float(np.median(kv[1]))):
            if key != "total_ms":
                print(f"   {key:<18} {np.median(values):6.1f}")
    if reasons:
        clusters = sum(reasons.values())
        print("кластеры в коридоре по причинам отсева:")
        for key, value in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"   {key:<16} {value:6d} ({100 * value / clusters:4.1f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
