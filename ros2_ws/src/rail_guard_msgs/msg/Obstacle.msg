# Посторонний объект, обнаруженный в зоне габарита поезда.
# Система координат — как в header.frame_id (по умолчанию base_link:
# X — вперёд по ходу движения, Y — влево, Z — вверх, REP-103).

uint8 CLASS_UNKNOWN=0
uint8 CLASS_PERSON=1
uint8 CLASS_LARGE_OBJECT=2      # автомобиль, вагон, крупный предмет
uint8 CLASS_SMALL_OBJECT=3      # предмет ниже 0.5 м
uint8 CLASS_INFRASTRUCTURE=4    # опознано как элемент инфраструктуры (не помеха)

uint32 id                       # идентификатор трека, стабилен между кадрами
uint8 classification
float32 confidence              # 0..1, растёт с числом подтверждений и плотностью точек

geometry_msgs/Point centroid
geometry_msgs/Vector3 size      # габариты AABB объекта, м
geometry_msgs/Point min_point   # угол AABB
geometry_msgs/Point max_point   # угол AABB

float32 distance                # дальность до ближайшей точки объекта вдоль оси пути, м
float32 lateral_offset          # смещение центра объекта от оси пути, м (влево +)
float32 height_above_rail       # высота нижней точки над плоскостью головок рельсов, м
float32 clearance_margin        # запас до границы габарита, м; <0 — объект внутри габарита

uint32 point_count
uint32 age                      # сколько кадров существует трек
uint32 hits                     # в скольких кадрах объект подтверждён
float32 closing_speed           # скорость сближения, м/с (>0 — приближается)
float32 time_to_contact         # время до контакта, с (inf, если сближения нет)

bool in_gauge                   # объект пересекает габарит поезда
