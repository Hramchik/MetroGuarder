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
string reason                   # человекочитаемое обоснование решения
