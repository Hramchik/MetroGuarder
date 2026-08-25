#!/usr/bin/env bash
# Загрузка последовательностей датасета OSDaR23 (DZSF / DB Netz AG, CC-BY 4.0).
# Полный датасет — около 200 ГБ, поэтому качаем только нужные последовательности
# и распаковываем из них лидар, разметку и навигацию (изображения не нужны).
#
#   ./scripts/download_osdar23.sh 7_approach_underground_station_7.1 [ещё...]
#
# Без аргументов берутся три последовательности, на которых снимались метрики.
set -euo pipefail

BASE_URL="https://download.data.fid-move.de/dzsf/osdar23"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data/osdar23"
SEQ_DIR="$DATA_DIR/sequences"

DEFAULT_SEQUENCES=(
  "7_approach_underground_station_7.1"   # спуск к подземной станции, кривая R≈250 м
  "9_station_ruebenkamp_9.1"             # платформа, люди рядом с путём
  "15_construction_vehicle_15.1"         # прямой участок, техника у пути
)

sequences=("$@")
if [ ${#sequences[@]} -eq 0 ]; then
  sequences=("${DEFAULT_SEQUENCES[@]}")
fi

mkdir -p "$DATA_DIR" "$SEQ_DIR"
for name in "${sequences[@]}"; do
  archive="$DATA_DIR/$name.zip"
  if [ ! -f "$archive" ]; then
    echo "Скачиваю $name.zip"
    curl -fSL -C - -o "$archive" "$BASE_URL/$name.zip"
  fi
  echo "Распаковываю лидар и разметку $name"
  unzip -q -o "$archive" -d "$SEQ_DIR/$name" \
    'lidar/*' 'novatel_oem7_inspva/*' '*.json' '*.txt' '*.md' 'rgb_center/*'
  echo "  кадров: $(ls "$SEQ_DIR/$name/lidar" | wc -l)"
done
echo "Готово: $SEQ_DIR"
