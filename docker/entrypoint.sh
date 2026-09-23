#!/bin/bash
# Точка входа контейнера. Три режима, чтобы инженеру не приходилось помнить
# ни имён нод, ни имён топиков:
#
#   docker run ... rail-guard demo /bags/doubleT_platform   — проиграть запись
#                                                             и показать результат
#
# Переменные окружения: BAG, RATE (темп), LOOP (по кругу), OFFSET (с какой
# секунды записи начинать — на двадцатиминутных проездах без этого не обойтись),
# RVIZ, POINTS_TOPIC.
#   docker run ... rail-guard detect                        — ждать облако извне
#                                                             (ros2 bag play на хосте)
#   docker run ... rail-guard <любая команда>                — как есть, в окружении ROS
#
# Без аргументов работает demo: если в /bags лежит ровно одна запись, она и
# будет проиграна; если записей несколько — они перечисляются, и надо выбрать.
set -e

source /opt/ros/humble/setup.bash
source /ws/install/setup.bash

CONFIG="${RAIL_GUARD_CONFIG:-/ws/install/rail_guard_bringup/share/rail_guard_bringup/config/metro.yaml}"

# Собственный домен DDS, чтобы контейнер не подхватывал чужой трафик; при
# проигрывании записи с хоста домен должен совпадать (ROS_DOMAIN_ID).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# Транспорт под 24-мегабайтные облака 128-луча: UDP с расширенными буферами.
# Разделяемая память не используется сознательно — проверено, что в
# контейнере FastDDS не создаёт сегмент даже при --shm-size=512m
# («SHM Transport is not supported in the current platform»), а с SHM первым
# в списке транспортов участник поднимается с ошибками. Свой профиль
# подставляется через ту же переменную.
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/etc/rail_guard/fastdds_profile.xml}"

find_single_bag() {
    local found=()
    while IFS= read -r meta; do
        found+=("$(dirname "$meta")")
    done < <(find /bags -maxdepth 2 -name metadata.yaml 2>/dev/null | sort)
    if [ "${#found[@]}" -eq 1 ]; then
        echo "${found[0]}"
        return 0
    fi
    if [ "${#found[@]}" -eq 0 ]; then
        echo "В /bags нет записей ROS 2 (не найден metadata.yaml)." >&2
        echo "Примонтируйте каталог с записями: -v /путь/к/бэгам:/bags:ro" >&2
    else
        echo "В /bags найдено несколько записей — укажите нужную:" >&2
        printf '  %s\n' "${found[@]}" >&2
    fi
    return 1
}

# ros2 launch не принимает аргумент с пустым значением, поэтому топик
# добавляется в командную строку только если он задан. Пустой топик — это
# штатный режим: детектор находит облако сам.
LAUNCH_EXTRA=()
if [ -n "${POINTS_TOPIC:-}" ]; then
    LAUNCH_EXTRA+=("points:=${POINTS_TOPIC}")
fi

case "${1:-demo}" in
    demo)
        BAG="${2:-${BAG:-}}"
        if [ -z "$BAG" ]; then
            BAG="$(find_single_bag)" || exit 1
        fi
        # Проверяем запись до запуска: иначе ros2 bag play падает своей
        # справкой по аргументам, и непонятно, что дело в томе. Самая частая
        # причина — docker создал пустой каталог из неверного пути на хосте.
        if [ ! -f "$BAG/metadata.yaml" ]; then
            echo "Записи «$BAG» нет (не найден metadata.yaml)." >&2
            AVAILABLE="$(find /bags -maxdepth 2 -name metadata.yaml -printf '  %h\n' 2>/dev/null | sort)"
            if [ -n "$AVAILABLE" ]; then
                echo "В /bags доступно:" >&2
                echo "$AVAILABLE" >&2
            else
                echo "Каталог /bags пуст — проверьте, что том смонтирован по" >&2
                echo "абсолютному пути: -v /полный/путь/к/for_hackathon:/bags:ro" >&2
            fi
            exit 1
        fi
        echo "Запись: $BAG"
        echo "Профиль: $CONFIG"
        exec ros2 launch rail_guard_bringup metro.launch.py \
            config:="$CONFIG" bag:="$BAG" \
            rviz:="${RVIZ:-false}" rate:="${RATE:-1.0}" loop:="${LOOP:-false}" \
            offset:="${OFFSET:-0.0}" \
            "${LAUNCH_EXTRA[@]}"
        ;;
    detect)
        echo "Ожидаю облако точек из ROS 2 (домен $ROS_DOMAIN_ID)."
        echo "Профиль: $CONFIG"
        exec ros2 launch rail_guard_bringup metro.launch.py \
            config:="$CONFIG" rviz:="${RVIZ:-false}" \
            "${LAUNCH_EXTRA[@]}"
        ;;
    *)
        exec "$@"
        ;;
esac
