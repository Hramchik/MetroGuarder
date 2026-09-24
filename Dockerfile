# Образ для стенда: Ubuntu 22.04 + ROS 2 Humble, как требует ТЗ.
# Собирается одной командой, запускается без ручной установки зависимостей.
#
#   docker build -t rail-guard .                       # только процессор
#   docker build --build-arg WITH_GPU=true -t rail-guard:gpu .
#
#   docker run --rm -v /путь/к/бэгам:/bags:ro rail-guard demo /bags/doubleT_platform
#   docker run --rm --gpus all -v /путь/к/бэгам:/bags:ro rail-guard:gpu demo /bags/запись
#
# Образ с видеокартой работает и без неё: если устройство не проброшено или
# драйвер недоступен, тракт считает на процессоре и пишет об этом в лог.
# Обратное тоже верно — обычный образ не станет хуже от `--gpus all`.
FROM ros:humble-ros-base

# В базовом образе уже есть numpy, PyYAML, colcon и rosbag2 — не хватает
# только scipy (KD-дерево и связные компоненты). Версия закреплена под
# numpy 1.21 из образа: подтягивание более свежей scipy потянуло бы за собой
# другой numpy и сломало бы ABI сообщений ROS.
#
# pip в ros:humble-ros-base отсутствует и ставится штатным get-pip: так
# сборка зависит только от PyPI и не зависит от состояния apt-зеркал.
RUN curl -sSf -o /tmp/get-pip.py https://bootstrap.pypa.io/pip/get-pip.py \
    && python3 /tmp/get-pip.py --no-cache-dir --root-user-action=ignore \
    && rm /tmp/get-pip.py \
    && python3 -m pip install --no-cache-dir --root-user-action=ignore "scipy==1.8.1" \
    && python3 -c "import scipy, numpy; print('scipy', scipy.__version__, 'numpy', numpy.__version__)"

# Поддержка видеокарты — отдельным слоем, чтобы обычный образ не тяжелел.
# Ставится cupy под CUDA 12: рантайм приезжает колесом, от хоста нужен только
# драйвер (на стенде 580.x, он обратно совместим с 12.x). Сама сборка
# видеокарты не требует — cupy проверяет устройство при первом обращении,
# уже в работе.
# Версия закреплена по той же причине, что и у scipy: в образе numpy 1.21, а
# cupy 13.x собран под numpy 1.22+ и падает на несовпадении ABI
# («numpy.dtype size changed»). Ветка 12.x — последняя, работающая с numpy
# этого поколения, и она уже поддерживает CUDA 12.
ARG WITH_GPU=false
#
# Импортом при сборке пакет не проверяется намеренно: cupy тянет libcuda.so.1
# из драйвера, а на машине сборки видеокарты может не быть вовсе — и образ,
# собранный на ней, обязан оставаться рабочим. Устройство проверяется в
# рантайме (lib/backend.py) и при старте контейнера.
RUN if [ "$WITH_GPU" = "true" ]; then \
        python3 -m pip install --no-cache-dir --root-user-action=ignore \
            "cupy-cuda12x==12.3.0" \
        && python3 -m pip show cupy-cuda12x | head -2; \
    fi
ENV RAIL_GUARD_DEVICE=auto

# RViz в образ не входит: он тянет около 400 МБ и нужен только там, где есть
# дисплей. На стенде с дисплеем собирайте с --build-arg WITH_RVIZ=true, либо
# запускайте RViz на хосте — контейнер публикует маркеры в общий домен DDS.
ARG WITH_RVIZ=false
RUN if [ "$WITH_RVIZ" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends ros-humble-rviz2 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /ws
COPY ros2_ws/src /ws/src
RUN . /opt/ros/humble/setup.sh \
    && colcon build --symlink-install --event-handlers console_cohesion+

# Тесты алгоритма прогоняются на сборке: образ, который их не проходит,
# до стенда доезжать не должен.
RUN . /opt/ros/humble/setup.sh && . /ws/install/setup.sh \
    && cd /ws/src/rail_guard && PYTHONPATH=. python3 -m pytest test/ -q

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/fastdds_profile.xml /etc/rail_guard/fastdds_profile.xml
RUN chmod +x /usr/local/bin/entrypoint.sh

# Записи ожидаются в /bags (примонтировать томом). Каталог создаётся заранее,
# чтобы запуск без тома не падал на несуществующем пути.
RUN mkdir -p /bags
ENV RAIL_GUARD_CONFIG=/ws/install/rail_guard_bringup/share/rail_guard_bringup/config/metro.yaml
# Вывод питона не буферизуется: иначе построчный результат в `docker logs`
# появляется пачками по 4 КБ, и «реальное время» в демонстрации не видно.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["demo"]
