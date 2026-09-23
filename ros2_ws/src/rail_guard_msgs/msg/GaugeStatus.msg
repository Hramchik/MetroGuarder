# Сводное состояние зоны габарита — вход системы принятия решений поезда.
uint8 ACTION_CLEAR=0            # путь свободен
uint8 ACTION_ATTENTION=1        # объект обнаружен далеко, снизить скорость
uint8 ACTION_SERVICE_BRAKE=2    # служебное торможение
uint8 ACTION_EMERGENCY_BRAKE=3  # экстренное торможение

std_msgs/Header header
bool clear
uint8 action
float32 nearest_distance        # до ближайшей помехи в габарите, м (inf, если помех нет)
uint32 nearest_obstacle_id
uint32 obstacle_count
float32 braking_distance        # расчётная тормозная дистанция при текущей скорости, м
float32 detection_range         # фактическая дальность мониторинга в этом кадре, м
float32 confident_range         # дальность, на которой положение пути подтверждено рельсами, м
float32 max_safe_speed          # скорость, с которой поезд успевает остановиться в этой зоне, м/с
bool speed_limited              # текущая скорость выше допустимой по дальности обзора
float32 ego_speed               # скорость, по которой считался тормозной путь, м/с
string speed_source             # откуда она взята: odometry | estimated | default
string reason                   # человекочитаемое обоснование решения

# Что система измерила о себе и об участке. Эти величины не настраиваются, а
# определяются по кадрам, и именно они задают все пороги детекции: на другом
# лидаре или другом участке они другие, и наружу это видно сразу.
float32 detection_limit         # дальность, на которой минимальная цель ещё различима, м
float32 beam_density            # измеренная плотность лучей датчика, точек на стерадиан
string scene                    # распознанная сцена: tunnel | open | unknown
