"""Проверки библиотеки обработки на синтетических сценах.

Сцены собираются так, чтобы ответ был известен заранее: прямой участок
с рельсами, кривая заданного радиуса, предмет в габарите и рядом с ним.
"""
import numpy as np
import pytest

from rail_guard.lib.config import PipelineConfig
from rail_guard.lib.decision import (ACTION_ATTENTION, ACTION_CLEAR, ACTION_EMERGENCY_BRAKE,
                                     ACTION_SERVICE_BRAKE, braking_distance, decide,
                                     safe_speed)
from rail_guard.lib.detect import CLASS_PERSON
from rail_guard.lib.ground import estimate_ground
from rail_guard.lib.injection import box_target, expected_points, inject, person_target
from rail_guard.lib.scene import TUNNEL, OPEN, classify_scene, support_threshold
from rail_guard.lib.sensor_model import SensorProfile, measure_beam_density
from rail_guard.lib.pipeline import Pipeline
from rail_guard.lib.preprocess import banded_downsample, workspace_mask
from rail_guard.lib.track import build_corridor, estimate_track


GAUGE = 1.59


def config() -> PipelineConfig:
    """Профиль с заполненными автоматическими полями.

    Этапы обработки вызываются здесь напрямую, без тракта, а он-то и выводит
    пороги из измерений. `resolved()` даёт то же заполнение по одной физике —
    так тест проверяет алгоритм, а не забытое `None`.
    """
    return PipelineConfig().resolved()


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
    cfg = config().preprocess
    pts = np.array([[0.5, 0.0, 0.5], [50.0, 0.0, 0.0], [-20.0, 0.0, 0.0]], dtype=np.float32)
    keep = workspace_mask(pts, cfg)
    assert not keep[0], "точка на носу поезда должна отбрасываться"
    assert keep[1]
    assert not keep[2], "точки позади поезда не относятся к зоне мониторинга"


def test_banded_downsample_keeps_far_points():
    cfg = config().preprocess
    near = np.random.default_rng(1).uniform([5, -1, 0], [10, 1, 0.5], size=(4000, 3))
    far = np.random.default_rng(2).uniform([90, -1, 0], [100, 1, 0.5], size=(200, 3))
    cloud = np.vstack([near, far]).astype(np.float32)
    idx = banded_downsample(cloud, cfg)
    kept_far = (cloud[idx][:, 0] > 90).sum()
    assert kept_far == 200, "дальние точки прореживать нельзя — на них держится дальность"
    assert (cloud[idx][:, 0] < 10).sum() < 4000


def test_ground_follows_grade():
    cloud = synthetic_scene(grade=0.02)
    ground = estimate_ground(cloud, config().ground, x_min=0.0, x_max=120.0)
    probe = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    assert abs(float(ground.ground_z(probe)[0]) - 2.0) < 0.25


def test_track_converges_to_curvature():
    """Ось выходит на кривую за несколько кадров, а не за один.

    Ограничение непрерывности (max_axis_shift) намеренно не даёт оценке
    прыгнуть на произвольную кривизну в одном кадре: именно так она не
    цепляется за соседний путь. Плата — сходимость за 3-4 кадра, то есть
    за треть секунды при 10 Гц.
    """
    cfg = config()
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
    cfg = config()
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
    cfg = config()
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
    cfg = config()
    pipeline = Pipeline(cfg)
    for frame in range(5):
        result = pipeline.process(synthetic_scene(seed=frame), None,
                                  timestamp=frame * 0.1, ego_speed=10.0)
    assert result.decision.clear, f"ложная тревога на чистом пути: {result.decision.reason}"
    assert result.decision.action == ACTION_CLEAR


def test_decision_scales_with_speed():
    cfg = config().decision
    assert braking_distance(5.0, cfg) < braking_distance(20.0, cfg)
    assert braking_distance(20.0, cfg) > 150.0     # 72 км/ч требуют полутора сотен метров


def test_injection_density_falls_as_inverse_square():
    sensor = SensorProfile(points_per_sr=1.0e6, angular_step=0.001,
                           max_range=200.0, frames=5)
    target = person_target()
    near = expected_points(target, 50.0, sensor)
    far = expected_points(target, 100.0, sensor)
    assert 3.6 < near / far < 4.4, "плотность точек должна падать как 1/r²"
    assert expected_points(box_target(0.4), 100.0, sensor) \
        < expected_points(target, 100.0, sensor)


def test_injection_requires_measured_sensor():
    """Без измеренного датчика подставлять цель нельзя: число её точек —
    свойство конкретного лидара, и константа здесь означала бы замер
    дальности не для того датчика, чей кадр перед нами."""
    with pytest.raises(ValueError):
        expected_points(person_target(), 50.0, None)


def test_injection_scales_with_sensor_density():
    """Вдвое более плотный лидар даёт вдвое больше отражений от той же цели."""
    sparse = SensorProfile(points_per_sr=2.0e5, angular_step=0.003,
                           max_range=150.0, frames=5)
    dense = SensorProfile(points_per_sr=8.0e5, angular_step=0.0015,
                          max_range=150.0, frames=5)
    target = person_target()
    assert expected_points(target, 80.0, dense) \
        == pytest.approx(4 * expected_points(target, 80.0, sparse), rel=0.05)


def test_injection_adds_points_and_shadows_background():
    cloud = synthetic_scene()
    background = np.array([[80.0, 0.0, 0.9]], dtype=np.float32)   # точка за целью
    cloud = np.vstack([cloud, background])
    sensor = SensorProfile(points_per_sr=1.0e6, angular_step=0.001,
                           max_range=200.0, frames=5)
    out, intensity, added = inject(cloud, None, person_target(),
                                   np.array([60.0, 0.0, 0.0]),
                                   np.random.default_rng(0), sensor=sensor)
    assert added > 50
    assert len(out) == len(intensity)
    survived = np.any(np.all(np.abs(out - background) < 1e-3, axis=1))
    assert not survived, "цель должна затенять фон, иначе дальность детекции завышается"

# ─────────────────────────────── приведение системы координат

def test_frame_auto_detection_recovers_orientation():
    """Развёрнутый лидар должен распознаваться, а не выбрасывать облако."""
    from rail_guard.lib.frames import FrameNormalizer, detect_axes
    from rail_guard.lib.config import SensorConfig

    scene = add_box(synthetic_scene(), 40.0, 0.0, 0.25)
    assert detect_axes(scene).forward == "x"
    assert detect_axes(scene).up == "z"

    # Та же сцена в СК записей метро: вперёд -Y, влево +X.
    rotated = np.stack([scene[:, 1], -scene[:, 0], scene[:, 2]], axis=1)
    guess = detect_axes(rotated)
    assert (guess.forward, guess.up) == ("-y", "z"), guess.spec

    # Перевёрнутая установка: вертикаль тоже должна определиться.
    flipped = np.stack([scene[:, 0], -scene[:, 1], -scene[:, 2]], axis=1)
    assert detect_axes(flipped).up == "-z"

    # Нормализатор возвращает облако в рабочей СК с точностью до знака нуля.
    norm = FrameNormalizer(SensorConfig())
    restored, decided = norm.apply(rotated)
    assert decided
    assert np.allclose(restored, scene, atol=1e-4)


def test_rotated_cloud_gives_same_detection():
    """Результат не должен зависеть от того, как повёрнут датчик."""
    results = {}
    for name, transform in (("прямая", lambda c: c),
                            ("метро", lambda c: np.stack([c[:, 1], -c[:, 0], c[:, 2]], axis=1))):
        cfg = config()
        cfg.tunnel.enabled = False        # синтетика — не тоннель
        cfg.gauge.extrapolation_margin = 0.0
        pipeline = Pipeline(cfg)
        for frame in range(5):
            cloud = add_box(synthetic_scene(seed=frame), 40.0, 0.0, 0.25)
            result = pipeline.process(transform(cloud), None, timestamp=frame * 0.1,
                                      ego_speed=10.0)
        results[name] = sorted(t.detection.distance for t in result.obstacles)
    assert results["прямая"], "объект не найден в прямой СК"
    assert len(results["прямая"]) == len(results["метро"])
    for a, b in zip(results["прямая"], results["метро"]):
        assert abs(a - b) < 0.5, results


def test_degenerate_frame_is_reported():
    """Облако в чужой СК — не «путь свободен», а диагностируемый отказ."""
    cfg = config()
    cfg.sensor.forward_axis = "x"          # запрещаем автоопределение
    cfg.sensor.up_axis = "z"
    pipeline = Pipeline(cfg)
    # Сцена, развёрнутая так, что вперёд смотрит -Y: при фиксированной СК
    # предфильтр обрежет почти всё.
    cloud = add_box(synthetic_scene(), 40.0, 0.0, 0.25)
    rotated = np.stack([cloud[:, 1], -cloud[:, 0], cloud[:, 2]], axis=1)
    result = pipeline.process(rotated, None, timestamp=0.0)
    assert result.degenerate, "вырожденный кадр должен быть помечен"
    assert result.decision.clear, "по такому кадру решения быть не может"


# ─────────────────────────────── свободное пространство тоннеля

def _cluster_points(x0: float, length: float, lateral: float, h_from: float, h_to: float,
                    width: float, seed: int) -> np.ndarray:
    """Точки одного кластера в рабочей СК."""
    rng = np.random.default_rng(seed)
    n = 120
    return np.stack([
        x0 + rng.uniform(0.0, length, n),
        lateral + rng.uniform(-0.5 * width, 0.5 * width, n),
        rng.uniform(h_from, h_to, n),
    ], axis=1).astype(np.float32)


def test_tunnel_surface_is_rejected_and_object_is_kept():
    """Кластер на границе свободного места — конструкция, в глубине — предмет.

    Оба кластера проходят все остальные фильтры: по размеру, плотности и
    положению они неотличимы, и решает именно заход внутрь свободного места.
    """
    from rail_guard.lib.detect import CLASS_INFRASTRUCTURE, detections_from_clusters
    from rail_guard.lib.track import Corridor

    cfg = config()

    # Толщина конструкции взята «стеновой» (0.8 м), а не тонкой: полосу тоньше
    # 0.3 м отсекает более дешёвый признак «тонкая полоса вдоль пути», и тогда
    # проверялся бы уже не тот механизм.
    surface = _cluster_points(40.0, 4.0, 1.0, 0.8, 1.3, 0.8, seed=1)
    obstacle = _cluster_points(30.0, 0.4, 0.0, 0.0, 1.7, 0.5, seed=2)
    xyz = np.concatenate([surface, obstacle])
    labels = np.concatenate([np.zeros(surface.shape[0], dtype=np.int64),
                             np.ones(obstacle.shape[0], dtype=np.int64)])
    heights = xyz[:, 2].copy()

    s_axis = np.arange(0.0, 160.5, 0.5)
    corridor = Corridor(s=s_axis, y=np.zeros_like(s_axis),
                        half_width=np.full_like(s_axis, 1.65), top=3.75, bottom=-0.10)

    # Модель свободного пространства: у конструкции запаса нет, у предмета — метры.
    intrusion = np.concatenate([np.full(surface.shape[0], 0.05, dtype=np.float32),
                                np.full(obstacle.shape[0], 2.0, dtype=np.float32)])

    detections = detections_from_clusters(xyz, labels, heights, corridor, cfg.cluster,
                                          cfg.objects, keep_rejected=True,
                                          intrusion=intrusion)
    by_distance = sorted(detections, key=lambda d: d.distance)
    assert len(by_distance) == 2, [(round(d.distance, 1), d.reject_reason) for d in by_distance]
    obj, surf = by_distance

    assert not obj.reject_reason, obj.reject_reason
    assert obj.intrusion_depth > 1.0
    assert surf.reject_reason == "tunnel_surface", surf.reject_reason
    assert surf.classification == CLASS_INFRASTRUCTURE

    # Без модели свободного пространства ровно тот же кластер проходит: именно
    # она и даёт различение, ни один другой фильтр его не ловит.
    no_model = detections_from_clusters(xyz, labels, heights, corridor, cfg.cluster,
                                        cfg.objects, keep_rejected=True)
    assert all(not d.reject_reason for d in no_model), \
        [(round(d.distance, 1), d.reject_reason) for d in no_model]


def test_free_space_boundary_is_measured_where_it_exists():
    """Модель описывает границу там, где та непрерывна, и молчит там, где нет."""
    from rail_guard.lib.config import TunnelConfig
    from rail_guard.lib.tunnel import estimate_free_space

    rng = np.random.default_rng(5)
    x = np.arange(2.0, 80.0, 0.1)
    # Стена на 2.5 м по всей длине и одинокий предмет на 40 м у самой оси.
    wall_x = np.tile(x, 12)
    wall_lat = np.full(wall_x.size, 2.5) + rng.normal(0, 0.01, wall_x.size)
    wall_h = np.tile(np.arange(0.0, 3.6, 0.3), x.size)[: wall_x.size]
    obj_x = np.full(200, 40.0)
    obj_lat = np.full(200, 0.2)
    obj_h = np.linspace(0.0, 1.7, 200)

    xs = np.concatenate([wall_x, obj_x])
    lat = np.concatenate([wall_lat, obj_lat])
    hs = np.concatenate([wall_h, obj_h])
    free = estimate_free_space(xs, lat, hs, TunnelConfig(), x_max=80.0)
    assert free is not None and free.valid

    # У стены запаса нет, у предмета — больше двух метров.
    slack = free.slack(np.array([40.0, 40.0]), np.array([2.5, 0.2]), np.array([1.5, 1.5]))
    assert abs(float(slack[0])) < 0.2, float(slack[0])
    assert float(slack[1]) > 2.0, float(slack[1])


# ─────────────────────────────── скорость носителя

def test_ego_speed_is_estimated_from_clouds():
    """Скорость восстанавливается из сдвига облаков между кадрами."""
    from rail_guard.lib.config import EgoMotionConfig
    from rail_guard.lib.egomotion import EgoMotionEstimator

    rng = np.random.default_rng(11)
    # Кабельные кронштейны вдоль стен через неравные промежутки: именно на
    # такой продольной неоднородности и держится корреляция. Однородная стена
    # не даёт сдвигу ни одного признака — и оценки по ней быть не может.
    posts = np.sort(rng.uniform(0.0, 220.0, 260))
    per_post = 160
    base_x = np.repeat(posts, per_post) + rng.normal(0, 0.02, posts.size * per_post)
    base_y = (rng.choice([-2.4, 2.4], size=base_x.size)
              + rng.normal(0, 0.02, base_x.size))
    base_z = rng.uniform(0.2, 2.5, base_x.size)

    speed = 12.0
    est = EgoMotionEstimator(EgoMotionConfig())
    got = None
    for frame in range(8):
        shift = speed * frame * 0.1
        xyz = np.stack([base_x - shift, base_y, base_z], axis=1).astype(np.float32)
        keep = xyz[:, 0] > 0.0
        got = est.update(xyz[keep], base_z[keep], timestamp=frame * 0.1)
    assert got is not None, "оценка скорости не получена"
    assert abs(got - speed) < 1.5, got


def test_kinematic_gate_blocks_non_closing_track():
    """Трек, стоящий относительно поезда, не должен тормозить поезд."""
    from rail_guard.lib.decision import ACTION_ATTENTION, decide
    from rail_guard.lib.config import DecisionConfig
    from rail_guard.lib.detect import Detection
    from rail_guard.lib.tracking import ObjectTrack

    def track_with(distances):
        det = Detection(centroid=np.array([distances[-1], 0.0, 0.5]),
                        min_point=np.array([distances[-1], -0.2, 0.0]),
                        max_point=np.array([distances[-1] + 0.4, 0.2, 1.7]),
                        size=np.array([0.4, 0.4, 1.7]), distance=distances[-1],
                        lateral_offset=0.0, height_above_rail=0.0, top_height=1.7,
                        clearance_margin=-0.8, point_count=90)
        det.in_gauge = True
        track = ObjectTrack(id=1, detection=det, timestamp=0.1 * len(distances))
        track.confidence = 0.9
        for i, d in enumerate(distances):
            track.history.append((0.1 * i, d))
        return track

    cfg = DecisionConfig()
    closing = track_with([30.0, 28.8, 27.6, 26.4, 25.2])      # сближается на 12 м/с
    standing = track_with([25.2, 25.2, 25.2, 25.2, 25.2])     # стоит рядом с поездом

    assert decide([closing], 12.0, 120.0, cfg, speed_measured=True).action > ACTION_ATTENTION
    gated = decide([standing], 12.0, 120.0, cfg, speed_measured=True)
    assert gated.action == ACTION_ATTENTION, gated.reason
    # Без измеренной скорости проверка не применяется: у стоящего поезда она
    # отсеяла бы реальные препятствия.
    assert decide([standing], 12.0, 120.0, cfg, speed_measured=False).action > ACTION_ATTENTION


def test_corridor_is_clamped_to_confirmed_axis():
    """Коридор не строится дальше зоны доверия оси плюс запас."""
    from rail_guard.lib.config import GaugeConfig
    from rail_guard.lib.track import TrackModel, build_corridor

    empty = np.zeros(0)
    model = TrackModel(lin=0.0, quad=0.0, fit_range=30.0, from_rails=True,
                       bin_x=empty, bin_y=empty, bin_z=empty)
    cfg = GaugeConfig()
    cfg.max_range = 150.0
    cfg.extrapolation_margin = 25.0
    assert abs(build_corridor(model, cfg).reach - 55.0) < cfg.step + 1e-6

    cfg.extrapolation_margin = 0.0
    assert abs(build_corridor(model, cfg).reach - 150.0) < cfg.step + 1e-6


# --------------------------------------------------------------- тоннель ведёт ось
def tunnel_walls(cloud: np.ndarray, half_width: float = 2.3, curvature: float = 0.0,
                 length: float = 140.0, seed: int = 11) -> np.ndarray:
    """Добавляет к сцене обделку: две вертикальные стены вдоль пути.

    Стены — то, что видно дальше рельсов, и по ним достраивается зона доверия
    оси. Плотность взята редкой (шаг 10 см по высоте), чтобы проверка не
    зависела от того, сколько точек даёт настоящий лидар.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(3.0, length, 0.2)
    axis_y = 0.5 * curvature * x ** 2
    points = []
    for sign in (-1.0, 1.0):
        for z in np.arange(0.7, 2.4, 0.1):
            points.append(np.stack([
                x, axis_y + sign * half_width + rng.normal(0, 0.01, x.size),
                np.full(x.size, z) + rng.normal(0, 0.01, x.size)], axis=1))
    return np.vstack([cloud] + [p.astype(np.float32) for p in points])


def _track_of(cloud: np.ndarray, cfg: PipelineConfig):
    ground = estimate_ground(cloud, cfg.ground, x_min=0.0, x_max=cfg.preprocess.x_max)
    return estimate_track(cloud, None, ground, cfg.track)


def test_tunnel_walls_extend_confident_range():
    """Обделка продлевает зону доверия оси за пределы видимости рельсов."""
    cfg = config()
    bare = synthetic_scene(length=60.0)          # рельсы кончаются на 60 м
    with_walls = tunnel_walls(bare, length=140.0)

    without = _track_of(bare, cfg)
    with_tube = _track_of(with_walls, cfg)

    assert without.guide_range == 0.0            # стен нет — вести нечем
    assert with_tube.guide_range > with_tube.fit_range
    assert with_tube.guide_range >= 80.0
    assert with_tube.guide_source                # труба назвала опору
    # Коридор строится до продлённой зоны, а не до рельсовой.
    assert build_corridor(with_tube, cfg.gauge).reach > build_corridor(without, cfg.gauge).reach


def test_tunnel_guide_rejects_tube_that_is_not_our_path():
    """Труба, убегающая от пути быстрее, чем путь может повернуть, отвергается.

    Критерий физический: за подтверждённой рельсами зоной путь вправе уйти от
    продолжения дуги на Δ²/2R — не больше, где R — минимальный радиус кривой
    линии. Труба, расходящаяся быстрее, принадлежит не нашему пути (соседний
    тоннель, камера съезда, стена платформы), и вести по ней ось нельзя.

    Прежде здесь стоял подобранный предел наклона (0.08 на полосу), и он
    отвергал заодно трубы, расхождение которых физика допускает: на линии с
    кривыми радиусом 150 м путь за тридцать метров уходит почти на три метра,
    и отличить такую кривую от чужой трубы по одному кадру нельзя.
    """
    cfg = config()
    cloud = synthetic_scene(length=60.0, curvature=1.0 / 400.0)
    # Труба уходит в другую сторону вчетверо круче минимального радиуса линии.
    shifted = tunnel_walls(cloud, half_width=2.3, curvature=-1.0 / 40.0, length=140.0)
    with_tube = _track_of(shifted, cfg)
    assert with_tube.guide_range == 0.0, with_tube.guide_source


# --------------------------------------------------------- реакция и допустимая скорость
def test_safe_speed_is_inverse_of_braking_distance():
    cfg = config().decision
    for distance in (20.0, 50.0, 95.0, 200.0):
        v = safe_speed(distance, cfg)
        assert braking_distance(v, cfg, emergency=True) == pytest.approx(distance, rel=1e-6)


def test_speed_limit_reported_without_alarm():
    """Превышение допустимой по обзору скорости — не тревога, а ограничение."""
    cfg = config().decision
    decision = decide([], speed=20.0, detection_range=40.0, cfg=cfg,
                      speed_measured=True, confident_range=40.0)
    assert decision.action == ACTION_CLEAR and decision.clear
    assert decision.speed_limited
    assert decision.max_safe_speed < 20.0
    assert "обзор достоверен" in decision.reason


def test_object_beyond_rail_zone_is_not_emergency():
    """За зоной рельсов выдаётся служебное торможение, внутри — экстренное."""
    cfg = config()
    pipeline = Pipeline(cfg)
    base = synthetic_scene(length=120.0)
    # Предмет приближается со скоростью поезда: иначе его отсечёт
    # кинематический гейт, и это правильно — неподвижная относительно поезда
    # помеха препятствием не является.
    last = None
    for step in range(5):
        scene = add_box(base, centre_x=50.0 - 1.5 * step, centre_y=0.0, base_z=0.3,
                        size=(0.5, 0.5, 1.6), density=900)
        last = pipeline.process(scene, None, timestamp=0.1 * step, ego_speed=15.0)
    obstacles = [t for t in last.obstacles if t.detection.in_gauge]
    assert obstacles, "предмет в габарите должен быть найден"
    near_zone = decide(obstacles, speed=15.0, detection_range=120.0, cfg=cfg.decision,
                       speed_measured=True, confident_range=120.0)
    far_zone = decide(obstacles, speed=15.0, detection_range=120.0, cfg=cfg.decision,
                      speed_measured=True, confident_range=20.0)
    assert near_zone.action == ACTION_EMERGENCY_BRAKE
    assert far_zone.action == ACTION_SERVICE_BRAKE


# ------------------------------------------------------------------- полотно и склейки
def test_track_bed_is_not_an_obstacle():
    """Поверхность полотна в кандидаты не попадает, а предмет на ней — попадает."""
    cfg = config()
    scene = synthetic_scene(length=80.0)
    pipeline = Pipeline(cfg)
    for _ in range(4):
        empty = pipeline.process(scene, None, ego_speed=10.0)
    assert not [t for t in empty.obstacles if t.detection.in_gauge]

    with_object = add_box(scene, centre_x=25.0, centre_y=0.0, base_z=0.0,
                          size=(0.4, 0.4, 0.45), density=600)
    pipeline = Pipeline(cfg)
    for _ in range(4):
        found = pipeline.process(with_object, None, ego_speed=10.0)
    assert [t for t in found.obstacles if t.detection.in_gauge], \
        "предмет высотой 45 см должен находиться и после снятия полотна"


def test_long_cluster_is_rejected_as_infrastructure():
    """Кластер длиной 12 м вдоль пути — конструкция, а не предмет."""
    cfg = config()
    cfg.debug = True
    scene = add_box(synthetic_scene(length=80.0), centre_x=20.0, centre_y=0.4, base_z=0.2,
                    size=(12.0, 0.5, 1.8), density=3000)
    pipeline = Pipeline(cfg)
    result = pipeline.process(scene, None, ego_speed=10.0)
    long_ones = [d for d in result.detections if d.size[0] > 8.0]
    assert long_ones, "длинный кластер должен был собраться"
    assert all(d.reject_reason for d in long_ones)
    assert not [t for t in result.obstacles if t.detection.in_gauge]


def test_voxel_grid_is_stable_between_frames():
    """Решётка вокселей привязана к координатам, а не к границам облака."""
    cfg = config().preprocess
    cloud = synthetic_scene(length=60.0)
    first = banded_downsample(cloud, cfg)
    # Добавляем одну далёкую точку: границы облака меняются, решётка — нет.
    extended = np.vstack([cloud, np.array([[55.0, 7.5, 3.0]], dtype=np.float32)])
    second = banded_downsample(extended, cfg)
    assert np.array_equal(first, second[second < cloud.shape[0]])


# ------------------------------------------------- устойчивость оси, скорости, треков
def test_axis_is_carried_forward_not_reset():
    """Кадр без узнаваемого сечения не обнуляет знание о пути.

    Зона доверия сокращается на пройденный путь, а не на фиксированные 10 м:
    иначе три-четыре плохих кадра подряд обрезают коридор до нуля, и система
    слепнет там, где путь давно измерен.
    """
    cfg = config()
    scene = tunnel_walls(synthetic_scene(length=60.0), length=140.0)
    known = _track_of(scene, cfg)
    assert known.fit_range > 20.0 and known.guide_range > known.fit_range

    # Кадр, в котором пути не видно вовсе: только стены, без рельсов.
    empty = tunnel_walls(np.zeros((0, 3), dtype=np.float32), length=140.0)
    ground = estimate_ground(empty, cfg.ground, x_min=0.0, x_max=cfg.preprocess.x_max)
    carried = estimate_track(empty, None, ground, cfg.track, previous=known, travelled=1.5)

    assert carried.fit_range == pytest.approx(known.fit_range - 1.5, abs=1e-6)
    assert carried.guide_range == pytest.approx(known.guide_range - 1.5, abs=1e-6)
    assert carried.guide_s.size >= 2, "измеренная ось должна переноситься вперёд"
    # Коридор остаётся длинным, а не падает до минимума.
    assert build_corridor(carried, cfg.gauge).reach > 40.0


def test_speed_estimate_rejects_impossible_jump():
    """Оценка скорости, требующая невозможного ускорения, отбрасывается."""
    from rail_guard.lib.egomotion import EgoMotionEstimator

    cfg = config()
    estimator = EgoMotionEstimator(cfg.egomotion)
    scene = synthetic_scene(length=80.0)
    heights = scene[:, 2].copy()

    # Три кадра с постоянным сдвигом 1.5 м за 0.1 с — это 15 м/с.
    speed = None
    for step in range(4):
        shifted = scene.copy()
        shifted[:, 0] -= 1.5 * step
        speed = estimator.update(shifted, heights, timestamp=0.1 * step)
    assert speed is not None and 10.0 < speed < 20.0

    # Кадр, «прыгнувший» на 4 м (40 м/с): физически невозможно за 100 мс.
    jumped = scene.copy()
    jumped[:, 0] -= 1.5 * 3 + 4.0
    after = estimator.update(jumped, heights, timestamp=0.4)
    assert after is not None
    assert abs(after - speed) < 3.0, "скачок скорости не должен попадать в оценку"


def test_track_class_is_voted_over_frames():
    """Класс объекта берётся по всему треку, а не по последнему кадру."""
    from rail_guard.lib.detect import CLASS_LARGE_OBJECT, CLASS_PERSON, Detection
    from rail_guard.lib.tracking import ObjectTracker

    tracker = ObjectTracker(PipelineConfig().tracker)
    track = None
    for step, klass in enumerate((CLASS_PERSON, CLASS_PERSON, CLASS_PERSON,
                                 CLASS_LARGE_OBJECT)):
        det = Detection(centroid=np.array([50.0 - 1.5 * step, 0.0, 0.8]),
                        min_point=np.array([50.0 - 1.5 * step, -0.2, 0.0]),
                        max_point=np.array([50.3 - 1.5 * step, 0.2, 1.7]),
                        size=np.array([0.3, 0.4, 1.7]), distance=50.0 - 1.5 * step,
                        lateral_offset=0.0, height_above_rail=0.0, top_height=1.7,
                        clearance_margin=-1.2, point_count=40, classification=klass,
                        confidence=0.9)
        confirmed = tracker.update([det], timestamp=0.1 * step, ego_speed=15.0)
        if confirmed:
            track = confirmed[0]
    assert track is not None
    assert track.detection.classification == CLASS_LARGE_OBJECT  # последний кадр
    assert track.classification == CLASS_PERSON                  # но трек — человек


def test_full_height_cluster_is_infrastructure():
    """Кластер от полотна до свода — конструкция, а не препятствие."""
    from rail_guard.lib.config import ClusterConfig, ObjectFilterConfig
    from rail_guard.lib.detect import CLASS_INFRASTRUCTURE, detections_from_clusters
    from rail_guard.lib.track import Corridor

    wall = _cluster_points(45.0, 1.0, 0.3, 0.0, 3.5, 0.9, seed=5)
    person = _cluster_points(30.0, 0.35, 0.0, 0.0, 1.75, 0.5, seed=6)
    xyz = np.concatenate([wall, person])
    labels = np.concatenate([np.zeros(wall.shape[0], dtype=np.int64),
                             np.ones(person.shape[0], dtype=np.int64)])
    s_axis = np.arange(0.0, 160.5, 0.5)
    corridor = Corridor(s=s_axis, y=np.zeros_like(s_axis),
                        half_width=np.full_like(s_axis, 1.65), top=3.75, bottom=0.05)
    detections = detections_from_clusters(xyz, labels, xyz[:, 2].copy(), corridor,
                                          ClusterConfig(), ObjectFilterConfig(),
                                          keep_rejected=True)
    by_distance = sorted(detections, key=lambda d: d.distance)
    assert len(by_distance) == 2
    obj, tall = by_distance
    assert not obj.reject_reason, obj.reject_reason
    assert tall.reject_reason == "full_height"
    assert tall.classification == CLASS_INFRASTRUCTURE


# ─────────────────────────────── самокалибровка датчика

def beam_grid_cloud(az_step_deg: float = 0.2, el_step_deg: float = 0.2,
                    distance: float = 30.0, az_half: float = 20.0,
                    el_half: float = 10.0) -> np.ndarray:
    """Облако с заранее известной решёткой лучей: по лучу на каждый узел сетки.

    Плотность такого облака известна точно — 1/(Δaz·Δel) точек на стерадиан,
    поэтому измерение можно проверить, а не поверить ему на слово.
    """
    az = np.radians(np.arange(-az_half, az_half, az_step_deg))
    el = np.radians(np.arange(-el_half, el_half, el_step_deg))
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
    az_grid, el_grid = az_grid.ravel(), el_grid.ravel()
    return np.stack([distance * np.cos(el_grid) * np.cos(az_grid),
                     distance * np.cos(el_grid) * np.sin(az_grid),
                     distance * np.sin(el_grid)], axis=1).astype(np.float32)


def test_beam_density_matches_known_grid():
    """Измеренная плотность должна совпасть с решёткой, из которой сделано облако."""
    step = 0.2
    cloud = beam_grid_cloud(az_step_deg=step, el_step_deg=step)
    profile = measure_beam_density(cloud, np.radians(15.0), np.radians(8.0))
    assert profile is not None and profile.valid
    expected = 1.0 / np.radians(step) ** 2
    assert profile.points_per_sr == pytest.approx(expected, rel=0.25), \
        f"измерено {profile.points_per_sr:.3g}, решётка даёт {expected:.3g}"


def test_beam_density_scales_with_beam_count():
    """Вдвое более редкая решётка — вчетверо меньшая плотность лучей."""
    dense = measure_beam_density(beam_grid_cloud(0.2, 0.2), np.radians(15.0), np.radians(8.0))
    sparse = measure_beam_density(beam_grid_cloud(0.4, 0.4), np.radians(15.0), np.radians(8.0))
    assert dense is not None and sparse is not None
    assert dense.points_per_sr == pytest.approx(4 * sparse.points_per_sr, rel=0.3)


def test_detection_thresholds_follow_sensor_density():
    """Порог числа точек обязан ехать за плотностью датчика.

    Это главное свойство универсальности: один и тот же профиль носителя на
    редком и плотном лидаре должен давать разные пороги — иначе на редком
    система ослепнет, а на плотном станет тормозить по шуму.
    """
    from rail_guard.lib.derive import DerivedResolver
    from rail_guard.lib.scene import SceneModel

    base = PipelineConfig()
    scene = SceneModel(kind=OPEN)
    sparse = SensorProfile(points_per_sr=2.0e5, angular_step=0.003, max_range=150.0, frames=5)
    dense = SensorProfile(points_per_sr=1.6e6, angular_step=0.001, max_range=150.0, frames=5)

    low = DerivedResolver(base).update(sparse, scene).cluster.ref_min_points
    high = DerivedResolver(base).update(dense, scene).cluster.ref_min_points
    assert high > low, "на плотном лидаре порог должен быть выше"
    assert high == pytest.approx(8 * low, rel=0.3), "порог пропорционален плотности лучей"


# ─────────────────────────────── распознавание сцены

def tunnel_cloud(half_width: float = 2.25, ceiling: float = 4.3,
                 length: float = 120.0, seed: int = 5) -> tuple:
    """Замкнутое сечение: полотно, две стены и свод."""
    rng = np.random.default_rng(seed)
    x = np.arange(3.0, length, 0.05)
    parts = [np.stack([x, rng.uniform(-2, 2, x.size), rng.normal(0, 0.01, x.size)], axis=1)]
    for sign in (-1.0, 1.0):
        parts.append(np.stack([x, np.full(x.size, sign * half_width),
                               rng.uniform(0.5, ceiling - 0.5, x.size)], axis=1))
    parts.append(np.stack([x, rng.uniform(-half_width, half_width, x.size),
                           np.full(x.size, ceiling)], axis=1))
    cloud = np.vstack(parts)
    return cloud[:, 0], cloud[:, 1], cloud[:, 2]


def test_scene_recognises_enclosed_section():
    x, lateral, height = tunnel_cloud()
    scene = classify_scene(x, lateral, height, gauge_height=3.75)
    assert scene.kind == TUNNEL, scene.describe()
    assert scene.half_width == pytest.approx(2.25, abs=0.3)
    assert scene.has_ceiling


def test_scene_recognises_open_track():
    """Открытый перегон: полотно есть, боковых границ нет."""
    rng = np.random.default_rng(3)
    x = np.arange(3.0, 120.0, 0.05)
    x_all = np.concatenate([x, x])
    lateral = np.concatenate([rng.uniform(-3, 3, x.size), rng.uniform(-12, 12, x.size)])
    height = np.concatenate([rng.normal(0, 0.02, x.size), rng.normal(0.1, 0.05, x.size)])
    scene = classify_scene(x_all, lateral, height, gauge_height=4.3)
    assert scene.kind == OPEN, scene.describe()


def test_support_threshold_follows_object_length():
    """Порог непрерывности границы выводится из длины предмета, а не задан числом."""
    short = support_threshold(max_object_length=4.0, band=4.0, n_bands=20)
    long_object = support_threshold(max_object_length=16.0, band=4.0, n_bands=20)
    assert long_object > short


# ─────────────────────────────── профиль носителя

def test_profile_validation_catches_impossible_gauge():
    cfg = config()
    cfg.gauge.half_width = 0.4          # уже половины колеи — так не бывает
    problems = cfg.validate()
    assert any("half_width" in item for item in problems), problems


def test_auto_tokens_become_none():
    """`auto` в профиле означает «измерить», и это должно доходить до конфига."""
    cfg = PipelineConfig.from_dict({"tunnel": {"enabled": "auto"},
                                    "gauge": {"max_range": "auto", "half_width": 1.5}})
    assert cfg.tunnel.enabled is None
    assert cfg.gauge.max_range is None
    assert cfg.gauge.half_width == 1.5


def test_unknown_profile_keys_are_reported():
    """Параметр, которого нет в схеме, не должен молча пропадать."""
    cfg = PipelineConfig.from_dict({"tunnel": {"quantile": 0.95}, "нет_такой_секции": {}})
    assert "tunnel.quantile" in cfg.unknown_keys
    assert "нет_такой_секции" in cfg.unknown_keys


def test_extrapolation_margin_follows_curve_radius():
    """Длина экстраполяции коридора — геометрия линии, а не настройка.

    На линии с более крутыми кривыми продолжение дуги расходится с путём
    быстрее, и коридор обязан обрываться раньше.
    """
    from rail_guard.lib.derive import extrapolation_margin

    tight, wide = PipelineConfig(), PipelineConfig()
    tight.track.min_radius = 150.0
    wide.track.min_radius = 600.0
    assert extrapolation_margin(tight) < extrapolation_margin(wide)


def test_scene_ignores_distant_boundary():
    """Непрерывная граница в десятке метров — не тоннель.

    Насыпь выемки и лесополоса вдоль перегона дают такую же непрерывную
    боковую границу, как обделка, но коридор их не достанет никогда. Если
    считать их замкнутым сечением, на открытом перегоне включатся тоннельные
    механизмы — и коридор начнёт обрываться там, где обрываться не должен.
    """
    x, lateral, height = tunnel_cloud(half_width=12.0, ceiling=6.0)
    near = classify_scene(x, lateral, height, gauge_height=4.3,
                          search_half_width=20.0, relevant_half_width=4.5)
    assert near.kind == OPEN, near.describe()
    # Та же геометрия, но сечение узкое — это уже тоннель.
    x2, lateral2, height2 = tunnel_cloud(half_width=2.3, ceiling=4.3)
    assert classify_scene(x2, lateral2, height2, gauge_height=3.75,
                          search_half_width=20.0,
                          relevant_half_width=4.0).kind == TUNNEL


def test_point_threshold_respects_voxel_grid():
    """Порог не может требовать больше точек, чем оставляет прореживание.

    В ближней зоне облако режется вокселем, и цель размером с минимальную
    даёт десяток точек, сколько бы лучей в неё ни попало. Порог, посчитанный
    по одним лучам, там недостижим — предмет перед поездом остаётся невидим.
    """
    from rail_guard.lib.cluster import min_points_for_range

    cfg = config().cluster
    cfg.ref_min_points = 500          # заведомо больше, чем даст воксель
    cfg.voxel_bands = ((30.0, 0.12), (1e9, 0.0))
    cfg.target_area = 0.15
    cfg.detection_fraction = 0.1
    near = float(min_points_for_range(20.0, cfg))
    cap = 0.1 * 0.15 / 0.12 ** 2
    # Ниже абсолютного минимума порог не опускается: по двум точкам о форме
    # кластера говорить нечего, сколько бы их ни съел воксель.
    assert near <= max(np.ceil(cap), cfg.abs_min_points) + 1e-6, \
        f"порог {near} выше потолка прореживания {cap}"
    # За полосой прореживания потолка нет — работает оценка по лучам.
    assert float(min_points_for_range(80.0, cfg)) > near


def test_linear_infrastructure_is_rejected():
    """Кластеры, выстроенные в линию вдоль пути, — конструкция, а не предметы.

    Контактный рельс и кабельный лоток лидар видит рвано: на низкой
    горизонтальной поверхности соседние кольца ложатся через метр и больше, и
    связать их кластеризацией нельзя, не склеив заодно всё остальное. Зато
    сами кластеры выстраиваются в линию на одном поперечном смещении и одной
    высоте — по этому они и опознаются.
    """
    from rail_guard.lib.detect import CLASS_INFRASTRUCTURE, Detection, mark_linear_infrastructure

    def piece(distance: float, lateral: float, height: float) -> Detection:
        return Detection(centroid=np.array([distance, lateral, height]),
                         min_point=np.array([distance, lateral - 0.08, height]),
                         max_point=np.array([distance + 0.8, lateral + 0.08, height + 0.2]),
                         size=np.array([0.8, 0.16, 0.2]), distance=distance,
                         lateral_offset=lateral, height_above_rail=height,
                         top_height=height + 0.2, clearance_margin=-0.2, point_count=9)

    line = [piece(24.0, 1.45, 0.15), piece(28.0, 1.47, 0.14), piece(33.0, 1.43, 0.16)]
    assert mark_linear_infrastructure(line, max_object_length=8.0) == 3
    assert all(d.reject_reason == "track_line" for d in line)
    assert all(d.classification == CLASS_INFRASTRUCTURE for d in line)
    assert all(not d.in_gauge for d in line)


def test_single_obstacle_survives_line_check():
    """Одиночный предмет и предметы на разных высотах линией не считаются."""
    from rail_guard.lib.detect import Detection, mark_linear_infrastructure

    def piece(distance: float, lateral: float, height: float) -> Detection:
        return Detection(centroid=np.array([distance, lateral, height]),
                         min_point=np.array([distance, lateral - 0.2, height]),
                         max_point=np.array([distance + 0.4, lateral + 0.2, height + 1.7]),
                         size=np.array([0.4, 0.4, 1.7]), distance=distance,
                         lateral_offset=lateral, height_above_rail=height,
                         top_height=height + 1.7, clearance_margin=-0.9, point_count=40)

    alone = [piece(30.0, 0.1, 0.05)]
    assert mark_linear_infrastructure(alone, max_object_length=8.0) == 0
    assert not alone[0].reject_reason

    # Три предмета на одном смещении, но на разной высоте — не линия.
    mixed = [piece(24.0, 1.45, 0.10), piece(28.0, 1.45, 0.90), piece(33.0, 1.45, 1.80)]
    assert mark_linear_infrastructure(mixed, max_object_length=8.0) == 0

    # Три куска одного предмета, стоящие вплотную, тоже не линия.
    tight = [piece(24.0, 1.45, 0.10), piece(25.5, 1.45, 0.10), piece(27.0, 1.45, 0.10)]
    assert mark_linear_infrastructure(tight, max_object_length=8.0) == 0


def test_brake_requires_entering_the_body_gauge():
    """Объект, задевший только запас на раскачку, тормозить поезд не должен."""
    from rail_guard.lib.decision import ACTION_ATTENTION, decide
    from rail_guard.lib.detect import Detection
    from rail_guard.lib.tracking import ObjectTrack

    def track_at(clearance: float) -> ObjectTrack:
        det = Detection(centroid=np.array([28.0, 1.6, 0.6]),
                        min_point=np.array([28.0, 1.5, 0.1]),
                        max_point=np.array([28.4, 1.7, 1.2]),
                        size=np.array([0.4, 0.2, 1.1]), distance=28.0,
                        lateral_offset=1.6, height_above_rail=0.1, top_height=1.2,
                        clearance_margin=clearance, point_count=30)
        det.in_gauge = True
        track = ObjectTrack(id=1, detection=det, timestamp=0.5)
        track.confidence = 0.9
        for i, d in enumerate((32.0, 30.8, 29.6, 28.4, 28.0)):
            track.history.append((0.1 * i, d))
        return track

    cfg = config().decision
    # Зашёл на 3 см при запасе 25 см — только внимание.
    edge = decide([track_at(-0.03)], 12.0, 120.0, cfg, speed_measured=True, safety_margin=0.25)
    assert edge.action == ACTION_ATTENTION, edge.reason
    assert "запас габарита" in edge.reason
    # Зашёл в габарит кузова — полноценная реакция.
    inside = decide([track_at(-0.6)], 12.0, 120.0, cfg, speed_measured=True, safety_margin=0.25)
    assert inside.action > ACTION_ATTENTION, inside.reason
