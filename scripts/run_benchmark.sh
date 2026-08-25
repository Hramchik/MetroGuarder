#!/usr/bin/env bash
# Полный прогон метрик по реальным данным. Результат — docs/benchmark_output.txt.
#   ./scripts/run_benchmark.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$ROOT/ros2_ws/src/rail_guard_bringup/config/osdar23.yaml"
OUT="$ROOT/docs/benchmark_output.txt"
SEQUENCES=(7_approach_underground_station_7.1 9_station_ruebenkamp_9.1 15_construction_vehicle_15.1)
PY=${PYTHON:-python3}

# numpy и scipy стоят только в контейнере с ROS — если их нет, перезапускаем
# весь прогон внутри него одним заходом, а не по разу на каждый скрипт.
CONTAINER=${RAIL_GUARD_CONTAINER:-ros-gazebo}
if ! "$PY" -c "import numpy, scipy" >/dev/null 2>&1; then
  if [ -n "${RAIL_GUARD_REEXEC:-}" ]; then
    echo "numpy/scipy недоступны даже в контейнере «$CONTAINER»" >&2
    exit 1
  fi
  if ! command -v distrobox >/dev/null 2>&1; then
    echo "Нужны numpy и scipy: запустите прогон в окружении с ROS 2" >&2
    exit 1
  fi
  echo "[rail-guard] numpy на хосте нет — прогон идёт в distrobox «$CONTAINER»" >&2
  exec distrobox enter "$CONTAINER" -- bash -lc \
    "env RAIL_GUARD_REEXEC=1 $(printf '%q' "${BASH_SOURCE[0]}") $(printf '%q ' "$@")"
fi

exec > >(tee "$OUT") 2>&1
echo "Прогон метрик $(date -Iseconds)"
echo "Конфигурация: $CFG"
echo

for seq in "${SEQUENCES[@]}"; do
  echo "############ $seq"
  echo "--- точность оси пути (эталон — размеченные рельсы)"
  $PY "$ROOT/scripts/eval_track.py" --sequence "$seq" --config "$CFG"
  echo "--- ложные срабатывания: путь свободен, тревог быть не должно"
  $PY "$ROOT/scripts/run_offline.py" --sequence "$seq" --config "$CFG"
  echo "--- то же с осью пути из путевой карты"
  $PY "$ROOT/scripts/run_offline.py" --sequence "$seq" --config "$CFG" --route-prior --quiet
  echo
done

echo "############ дальность детекции (подставленные цели, 80 км/ч)"
for mode in "" "--route-prior"; do
  echo "=== ось пути: $([ -z "$mode" ] && echo 'по облаку точек' || echo 'из путевой карты')"
  $PY "$ROOT/scripts/eval_detection.py" --sequence 15_construction_vehicle_15.1 \
      --config "$CFG" --targets person,box40 --speed 22 \
      --distances 20,40,60,80,100,120,140,160,180,200 $mode
done
