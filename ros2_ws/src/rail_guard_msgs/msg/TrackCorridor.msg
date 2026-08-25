# Оценённая ось пути и построенный по ней коридор габарита.
std_msgs/Header header
geometry_msgs/Point[] centerline   # точки оси пути с шагом step_m, от носа поезда вперёд
float32[] half_width               # полуширина габарита в каждой точке оси, м
float32[] height                   # высота габарита в каждой точке оси, м
float32 curvature_radius           # оценка радиуса кривой, м (inf — прямая)
bool estimated_from_rails          # true — ось найдена по рельсам, false — экстраполяция
float32 valid_range                # дальность, до которой ось достоверна, м
