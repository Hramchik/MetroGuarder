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

from rail_guard.lib import report
from rail_guard.lib.config import PipelineConfig, load_yaml, parse_param
from rail_guard.lib.pipeline import Pipeline

ACTIONS = report.ACTION_TEXT
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

    style = report.Style()
    speed_note = "оценивается по облакам" if args.speed is None else f"{args.speed} м/с (задана)"
    print(report.panel("rail-guard · офлайн-прогон", [
        f"запись:   {os.path.basename(args.bag.rstrip('/'))}",
        f"профиль:  {os.path.basename(args.config)} · "
        f"габарит {2 * cfg.gauge.half_width:.2f} × {cfg.gauge.height:.2f} м · "
        f"колея {cfg.track.gauge:.3f} м",
        f"скорость: {speed_note}",
        f"счёт:     {pipeline.backend_note}",
    ], style))
    if not args.quiet:
        print(report.frame_header(style))
        print(style.rule())

    actions = {0: 0, 1: 0, 2: 0, 3: 0}
    latency, fit_range, reach, nearest, speeds = [], [], [], [], []
    sources, reasons, stages = {}, {}, {}
    stamps, limits, decision_latency, scenes = [], [], [], {}
    beam_density = 0.0
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
        stamps.append(float(stamp))
        decision_latency.append(float(result.latency))
        if result.detection_limit > 0.0:
            limits.append(float(result.detection_limit))
        if result.sensor_profile is not None and result.sensor_profile.valid:
            beam_density = float(result.sensor_profile.points_per_sr)
        if result.scene is not None:
            scenes[result.scene.kind] = scenes.get(result.scene.kind, 0) + 1
        if cfg.debug:
            for det in result.detections:
                key = det.reject_reason or "принят"
                reasons[key] = reasons.get(key, 0) + 1
        if result.frame_decided:
            print(f"      СК датчика: {result.frame_spec} — {result.frame_source}")
        if not args.quiet:
            print(report.frame_line(
                index=index, points=result.input_points,
                confirmed_range=result.track.fit_range,
                corridor=result.corridor.reach,
                obstacles=len(result.obstacles),
                nearest=result.decision.nearest_distance,
                speed=result.ego_speed,
                latency_ms=latency[-1], action=int(result.decision.action),
                style=style))

    total = sum(actions.values())
    if not total:
        print("Кадров не прочитано: проверьте путь к записи и имя топика")
        return 1

    width = report.terminal_width()
    span = 0.0
    if stamps:
        span = max(stamps) - min(stamps)
    lat = np.array(latency)
    print()
    print(style("═" * width, "cyan"))
    print(style(f"  ИТОГ ПРОГОНА · {total} кадров · {report.duration(span)} записи", "bold"))
    print(style("═" * width, "cyan"))

    print(style("\n Решения", "bold"))
    for line in report.decisions_block(actions, total, style):
        print(line)

    print(style("\n Реакция", "bold"))
    period = 1e3 * float(np.median(np.diff(np.sort(stamps)))) if len(stamps) > 2 else 100.0
    late = int((lat > period).sum())
    verdict = "укладывается в период лидара" if np.median(lat) <= period \
        else style("медленнее периода лидара", "orange")
    print(f"  счёт кадра                     {np.median(lat):>6.0f} мс "
          f"(p95 {np.percentile(lat, 95):.0f}, максимум {lat.max():.0f}) — {verdict}")
    print(f"  кадров дольше периода          {late:>6} из {total} "
          f"({100.0 * late / total:.0f} %), период {period:.0f} мс")
    if decision_latency:
        print(f"  задержка «съёмка → команда»    {1e3 * np.median(decision_latency):>6.0f} мс "
              f"— входит в тормозной путь")
    print(f"  вычисления                     {pipeline.backend_note}")
    if pipeline.gpu_fallback:
        print("  " + style(f"видеокарта отключилась: {pipeline.gpu_fallback}", "orange"))

    print(style("\n Обзор", "bold"))
    print(f"  ось по рельсам                 {rails:>6} из {total} кадров, "
          f"подтверждена до {np.median(fit_range):.0f} м (максимум {max(fit_range):.0f})")
    print(f"  длина коридора                 {np.median(reach):>6.0f} м "
          f"(максимум {max(reach):.0f})")
    if limits:
        print(f"  предел обнаружения цели        {np.median(limits):>6.0f} м "
              f"(плотность лучей {beam_density:.2g} точек/ср, измерена по кадрам)")
    if scenes:
        names = {"tunnel": "замкнутое сечение", "open": "открытый участок",
                 "unknown": "не определена"}
        parts = ", ".join(f"{names.get(k, k)} {100.0 * v / total:.0f} %"
                          for k, v in sorted(scenes.items(), key=lambda kv: -kv[1]))
        print(f"  сцена по кадрам                {parts}")
    if speeds:
        src = ", ".join(f"{key}={value}" for key, value in sources.items())
        print(f"  скорость носителя              {np.median(speeds):>6.1f} м/с "
              f"(максимум {max(speeds):.1f}); источник: {src}")

    print(style("\n Найденное", "bold"))
    print(f"  детекций                       {detections:>6} в {frames_with_objects} кадрах"
          + (f", ближайшая {np.median(nearest):.0f} м (медиана)" if nearest else ""))
    if degenerate:
        print("  " + style(f"вырожденных кадров: {degenerate} — облако уходит в фильтр "
                           f"почти целиком", "red"))

    if args.stages:
        print(style("\n Время по стадиям, мс (медиана)", "bold"))
        ordered = sorted(((k, float(np.median(v))) for k, v in stages.items()
                          if k != "total_ms"), key=lambda kv: -kv[1])
        worst = ordered[0][1] if ordered else 1.0
        for key, value in ordered:
            print(f"  {key:<18} {value:>6.1f}  {report.bar(value / max(worst, 1e-6), 20, style)}")
    if reasons:
        clusters = sum(reasons.values())
        print(style("\n Кластеры в коридоре по причинам отсева", "bold"))
        for key, value in sorted(reasons.items(), key=lambda kv: -kv[1]):
            share = value / clusters
            colour = "green" if key == "принят" else "grey"
            print(f"  {style(f'{key:<18}', colour)} {value:>6} "
                  f"{report.bar(share, 16, style)} {100 * share:>4.1f} %")
    print(style("═" * width, "cyan"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
