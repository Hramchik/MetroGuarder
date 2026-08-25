"""Проверки библиотеки обработки на синтетических сценах.

Сцены собираются так, чтобы ответ был известен заранее: прямой участок
с рельсами, кривая заданного радиуса, предмет в габарите и рядом с ним.
"""
import numpy as np
import pytest

from rail_guard.lib.config import PipelineConfig
from rail_guard.lib.decision import (ACTION_ATTENTION, ACTION_CLEAR, ACTION_EMERGENCY_BRAKE,
                                     braking_distance)
from rail_guard.lib.detect import CLASS_PERSON
from rail_guard.lib.ground import estimate_ground
from rail_guard.lib.injection import box_target, expected_points, inject, person_target
from rail_guard.lib.pipeline import Pipeline
from rail_guard.lib.preprocess import banded_downsample, workspace_mask
from rail_guard.lib.track import build_corridor, estimate_track


GAUGE = 1.59


def synthetic_scene(length: float = 120.0, curvature: float = 0.0, grade: float = 0.0,
                    seed: int = 7) -> np.ndarray:
    """Путь: балласт, шпалы и две рельсовые нитки.

    curvature — 1/R (положительное значение уводит путь влево),
    grade — продольный уклон.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(3.0, length, 0.05)
    axis_y = 0.5 * curvature * x ** 2
    bed_z = grade * x

    points = []
    # Балласт: полоса шириной 4 м вокруг оси
    for _ in range(6):
        offset = rng.uniform(-2.0, 2.0, size=x.size)
        points.append(np.stack([x, axis_y + offset, bed_z + rng.normal(0, 0.01, x.size)], axis=1))
    # Головки рельсов: 0.25 м над балластом
    for sign in (-1.0, 1.0):
        for _ in range(2):
            points.append(np.stack([x, axis_y + sign * GAUGE / 2 + rng.normal(0, 0.01, x.size),
                                    bed_z + 0.25 + rng.normal(0, 0.005, x.size)], axis=1))
    return np.vstack(points).astype(np.float32)


def add_box(cloud: np.ndarray, centre_x: float, centre_y: float, base_z: float,
            size=(0.4, 0.5, 1.7), density: int = 400, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    box = np.stack([
        rng.uniform(-size[0] / 2, size[0] / 2, density) + centre_x,
        rng.uniform(-size[1] / 2, size[1] / 2, density) + centre_y,
        rng.uniform(0.0, size[2], density) + base_z,
    ], axis=1)
    return np.vstack([cloud, box.astype(np.float32)])


def test_workspace_mask_removes_own_train():
    cfg = PipelineConfig().preprocess
    pts = np.array([[0.5, 0.0, 0.5], [50.0, 0.0, 0.0], [-20.0, 0.0, 0.0]], dtype=np.float32)
    keep = workspace_mask(pts, cfg)
    assert not keep[0], "точка на носу поезда должна отбрасываться"
    assert keep[1]
    assert not keep[2], "точки позади поезда не относятся к зоне мониторинга"


def test_banded_downsample_keeps_far_points():
    cfg = PipelineConfig().preprocess
    near = np.random.default_rng(1).uniform([5, -1, 0], [10, 1, 0.5], size=(4000, 3))
    far = np.random.default_rng(2).uniform([90, -1, 0], [100, 1, 0.5], size=(200, 3))
    cloud = np.vstack([near, far]).astype(np.float32)
    idx = banded_downsample(cloud, cfg)
    kept_far = (cloud[idx][:, 0] > 90).sum()
    assert kept_far == 200, "дальние точки прореживать нельзя — на них держится дальность"
    assert (cloud[idx][:, 0] < 10).sum() < 4000


def test_ground_follows_grade():
    cloud = synthetic_scene(grade=0.02)
    ground = estimate_ground(cloud, PipelineConfig().ground, x_min=0.0, x_max=120.0)
    probe = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    assert abs(float(ground.ground_z(probe)[0]) - 2.0) < 0.25


def test_track_converges_to_curvature():
    """Ось выходит на кривую за несколько кадров, а не за один.

    Ограничение непрерывности (max_axis_shift) намеренно не даёт оценке
    прыгнуть на произвольную кривизну в одном кадре: именно так она не
    цепляется за соседний путь. Плата — сходимость за 3-4 кадра, то есть
    за треть секунды при 10 Гц.
    """
    cfg = PipelineConfig()
    model = None
    for frame in range(5):
        cloud = synthetic_scene(curvature=-1.0 / 400.0, seed=frame)
        prior_y = None if model is None else model.y_at(
            np.arange(2.5, 250.0, cfg.ground.bin_size))
        ground = estimate_ground(cloud, cfg.ground, centerline_y=prior_y,
                                 x_min=0.0, x_max=120.0)
        model = estimate_track(cloud, None, ground, cfg.track, previous=model)
    assert model.from_rails
    assert 250.0 < model.curvature_radius < 700.0
    assert abs(float(model.y_at(60.0)) - (-0.5 * 60.0 ** 2 / 400.0)) < 1.0


def test_object_in_gauge_is_detected_and_outside_is_not():
    cfg = PipelineConfig()
    pipeline = Pipeline(cfg)
    for frame in range(5):
        cloud = synthetic_scene(seed=frame)
        cloud = add_box(cloud, 40.0, 0.0, 0.25)          # человек между рельсами
        cloud = add_box(cloud, 45.0, 4.0, 0.25, seed=9)  # предмет вне габарита
        result = pipeline.process(cloud, None, timestamp=frame * 0.1, ego_speed=10.0)
    distances = [t.detection.distance for t in result.obstacles]
    assert any(abs(d - 39.8) < 1.5 for d in distances), "предмет в габарите не найден"
    assert all(t.detection.lateral_offset < 2.0 for t in result.obstacles), \
        "объект в четырёх метрах от оси не должен попадать в габарит"
    assert not result.decision.clear


def test_person_classification_and_ttc():
    cfg = PipelineConfig()
    pipeline = Pipeline(cfg)
    for frame in range(6):
        cloud = synthetic_scene(seed=frame)
        cloud = add_box(cloud, 40.0 - frame * 1.0, 0.0, 0.25, size=(0.35, 0.5, 1.75))
        result = pipeline.process(cloud, None, timestamp=frame * 0.1, ego_speed=10.0)
    assert result.obstacles, "приближающийся объект должен вести к подтверждённому треку"
    track = result.obstacles[0]
    assert track.detection.classification == CLASS_PERSON
    assert track.closing_speed() > 5.0
    assert track.time_to_contact() < 10.0


def test_clean_track_gives_no_alarm():
    cfg = PipelineConfig()
    pipeline = Pipeline(cfg)
    for frame in range(5):
        result = pipeline.process(synthetic_scene(seed=frame), None,
                                  timestamp=frame * 0.1, ego_speed=10.0)
    assert result.decision.clear, f"ложная тревога на чистом пути: {result.decision.reason}"
    assert result.decision.action == ACTION_CLEAR


def test_decision_scales_with_speed():
    cfg = PipelineConfig().decision
    assert braking_distance(5.0, cfg) < braking_distance(20.0, cfg)
    assert braking_distance(20.0, cfg) > 150.0     # 72 км/ч требуют полутора сотен метров


def test_injection_density_falls_as_inverse_square():
    target = person_target()
    near = expected_points(target, 50.0)
    far = expected_points(target, 100.0)
    assert 3.6 < near / far < 4.4, "плотность точек должна падать как 1/r²"
    assert expected_points(box_target(0.4), 100.0) < expected_points(target, 100.0)


def test_injection_adds_points_and_shadows_background():
    cloud = synthetic_scene()
    background = np.array([[80.0, 0.0, 0.9]], dtype=np.float32)   # точка за целью
    cloud = np.vstack([cloud, background])
    out, intensity, added = inject(cloud, None, person_target(),
                                   np.array([60.0, 0.0, 0.0]),
                                   np.random.default_rng(0))
    assert added > 50
    assert len(out) == len(intensity)
    survived = np.any(np.all(np.abs(out - background) < 1e-3, axis=1))
    assert not survived, "цель должна затенять фон, иначе дальность детекции завышается"
