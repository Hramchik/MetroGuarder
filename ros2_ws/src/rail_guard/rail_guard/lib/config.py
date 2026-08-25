"""Параметры пайплайна детекции. Одна dataclass — один этап обработки.

Значения по умолчанию рассчитаны на габарит вагона метро (ширина 2.7 м,
высота 3.7 м от уровня головок рельсов) и лидар, установленный на носу
поезда на высоте ~2 м. Все длины в метрах, углы в радианах.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple


@dataclass
class SensorConfig:
    # Положение лидара относительно base_link (нос поезда, УГР). Нужно только
    # если облако приходит в собственной СК лидара и не преобразовано через TF.
    mount_height: float = 2.0
    # Угловое разрешение — из него считается ожидаемое число точек на объекте.
    azimuth_res: float = 0.0035      # рад (~0.2 град)
    elevation_res: float = 0.0035    # рад
    max_range: float = 250.0
    min_range: float = 1.5


@dataclass
class PreprocessConfig:
    # Обрезка рабочей области: смотрим только вперёд и в разумной полосе.
    x_min: float = 2.0
    x_max: float = 250.0
    y_abs_max: float = 25.0
    z_min: float = -3.0
    z_max: float = 8.0
    # Габарит самого поезда — точки внутри отбрасываются (отражения от носа).
    ego_box: Tuple[float, float, float, float, float, float] = (-5.0, 2.5, -2.0, 2.0, -3.0, 4.0)
    # Воксельная дискретизация по полосам дальности: дальние точки не прореживаем,
    # иначе теряется как раз то, ради чего борются за дальность.
    voxel_bands: Tuple[Tuple[float, float], ...] = ((30.0, 0.12), (60.0, 0.06), (1e9, 0.0))
    # Отсечка по интенсивности для дальних возвратов (пыль/капли/выхлоп).
    # 0 = выключено; включать только зная шкалу интенсивности конкретного лидара.
    min_intensity_far: float = 0.0
    intensity_far_range: float = 60.0


@dataclass
class GroundConfig:
    bin_size: float = 5.0            # длина продольной ячейки для оценки полотна
    search_half_width: float = 3.0   # полоса поиска полотна вокруг оси пути
    low_percentile: float = 20.0     # перцентиль z, принимаемый за уровень балласта
    plane_tolerance: float = 0.12    # допуск при уточнении плоскости МНК
    max_slope: float = 0.05          # макс. правдоподобный продольный уклон (5 %)
    follow_tolerance: float = 0.45   # коридор слежения за поверхностью между ячейками
    min_bin_points: int = 20
    rail_head_offset: float = 0.25   # головка рельса над уровнем балласта


@dataclass
class TrackConfig:
    gauge: float = 1.59              # расстояние между ЦЕНТРАМИ головок рельсов:
                                     # колея + ширина головки (1520 + ~70 мм для РФ,
                                     # 1435 + ~70 мм = 1.505 для колеи 1435 мм)
    rail_height_min: float = 0.10    # головка рельса над уровнем балласта
    rail_height_max: float = 0.32
    near_start: float = 4.0          # ближе поезд закрывает путь собственным носом
    max_fit_range: float = 120.0     # дальше сечение пути уже не разрешается
    search_half_width: float = 3.0   # полоса поиска вокруг опорной оси
    min_rail_points: int = 200
    profile_bin: float = 4.0         # продольная ячейка профиля
    lat_bin: float = 0.05            # поперечная ячейка профиля
    center_weight: float = 0.6       # вес «провала» между рельсами в шаблоне
    outer_offset: float = 0.45       # где шаблон ожидает пустоту снаружи колеи
    outer_weight: float = 0.4
    rail_tolerance: float = 0.12     # допуск «точка легла на нитку», м
    min_bin_points: int = 6          # отражений на нитках, чтобы ячейка считалась подтверждённой
    max_bin_gap: int = 2             # сколько ячеек подряд можно не узнать
    min_fit_range: float = 15.0      # короче — считаем, что путь не распознан
    min_radius: float = 150.0        # минимальный радиус кривой в сетке гипотез
    lin_span: float = 0.05           # диапазон поправки курса относительно опорной оси
    coarse_lin: int = 13             # сетка гипотез: курс
    coarse_quad: int = 41            # сетка гипотез: кривизна
    fine_steps: int = 9              # уточняющая сетка
    prior_weight: float = 0.0002     # штраф за отход от опорной оси
    max_axis_shift: float = 1.5      # насколько ось может уйти от опорной за кадр, м
    shift_check_range: float = 50.0  # на какой дальности проверяется это ограничение
    row_weight_range: float = 60.0   # дальность, на которой вес ячейки падает вдвое
    smoothing: float = 0.5           # доля новой оценки при межкадровом сглаживании
    confidence_decay: float = 120.0  # на сколько метров хватает экстраполяции дуги


@dataclass
class GaugeConfig:
    """Габарит поезда — та самая зона, вторжение в которую и ищем."""
    half_width: float = 1.45         # половина ширины подвижного состава
    lateral_margin: float = 0.25     # запас на раскачку/износ пути/ошибку оси
    height: float = 3.75             # верх габарита над УГР
    bottom: float = -0.10            # низ: чуть ниже УГР, чтобы видеть предметы между рельсами
    # Расширение габарита в кривой (вынос кузова). Приближение: dW = L^2/(8R).
    car_length: float = 19.2
    max_range: float = 200.0
    step: float = 0.5                # шаг дискретизации оси коридора


@dataclass
class ClusterConfig:
    eps0: float = 0.30               # базовый радиус связности на нулевой дальности
    eps_per_meter: float = 0.012     # прирост радиуса с дальностью (луч расходится)
    eps_max: float = 1.5
    min_cluster_points: int = 3
    # Порог числа точек масштабируется как 1/r^2 от значения на опорной дальности.
    ref_range: float = 20.0
    ref_min_points: int = 18
    abs_min_points: int = 3


@dataclass
class ObjectFilterConfig:
    # Собственная высота объекта. Ниже порога уверенно отделить предмет от
    # путевого оборудования и от ошибки оценки полотна невозможно ни на
    # какой дальности, поэтому порог задан явно, а не подобран.
    min_size_z: float = 0.20
    min_top_height: float = -0.08    # верх объекта относительно УГР
    min_volume: float = 0.0015
    # Плоские протяжённые кластеры — это стрелка, кабельный лоток, край
    # платформы или «протёкшее» в коридор полотно, но не препятствие.
    flat_length: float = 3.0         # длина, с которой кластер уже «протяжённый»
    flat_height: float = 0.40        # и при этом низкий — путевое оборудование
    sheet_height: float = 1.00       # низкий, но широкий —
    sheet_width: float = 1.50        # это поверхность, а не предмет
    min_footprint: float = 0.05      # м^2; меньше — вырожденный кластер
    sliver_points: int = 15
    max_length: float = 25.0
    # Признаки инфраструктуры вдоль пути:
    wall_length: float = 3.0         # длинный вдоль пути и узкий поперёк
    wall_width: float = 0.7
    wall_edge_margin: float = 0.25   # и прижат к границе коридора
    overhead_clearance: float = 2.6  # низ выше этой высоты — портал, мост, свод тоннеля


@dataclass
class TrackerConfig:
    max_assoc_distance: float = 2.0  # ворота ассоциации, м
    min_hits: int = 3                # N из M: подтверждение трека
    max_misses: int = 3
    history: int = 8                 # длина истории для оценки скорости сближения
    confirm_window: int = 5


@dataclass
class DecisionConfig:
    reaction_time: float = 0.6       # с, задержка тракта принятия решения
    service_decel: float = 0.9       # м/с^2
    emergency_decel: float = 1.3     # м/с^2
    attention_factor: float = 2.5    # во сколько тормозных дистанций начинать внимание
    default_speed: float = 15.0      # м/с, если одометрия не подключена (~54 км/ч)
    min_confidence: float = 0.35     # ниже — только информируем, не тормозим


@dataclass
class PipelineConfig:
    sensor: SensorConfig = field(default_factory=SensorConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    gauge: GaugeConfig = field(default_factory=GaugeConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    objects: ObjectFilterConfig = field(default_factory=ObjectFilterConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineConfig":
        cfg = PipelineConfig()
        for section, values in (d or {}).items():
            if not hasattr(cfg, section):
                continue
            target = getattr(cfg, section)
            if not hasattr(target, "__dataclass_fields__"):
                setattr(cfg, section, values)
                continue
            for key, value in (values or {}).items():
                if hasattr(target, key):
                    setattr(target, key, value)
        return cfg


def load_yaml(path: str) -> "PipelineConfig":
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    # Поддерживаем как «плоский» файл, так и ros2-обёртку /**: ros__parameters:
    for key in ("/**", "rail_guard"):
        if key in data:
            data = data[key]
    if "ros__parameters" in data:
        data = data["ros__parameters"]
    return PipelineConfig.from_dict(data)
