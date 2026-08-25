# Результат детекции за один кадр лидара.
std_msgs/Header header
Obstacle[] obstacles

float32 processing_time_ms      # время обработки кадра, мс (контроль реального времени)
uint32 input_point_count        # точек на входе
uint32 filtered_point_count     # точек после фильтрации
float32 max_valid_range         # дальность, до которой в этом кадре были валидные данные, м
