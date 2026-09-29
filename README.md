# rail-guard

Обнаружение посторонних объектов в габарите тоннеля метро по 3D-лидару.

## Запуск

```bash
# образ только под процессор
docker build -t rail-guard .

# образ с поддержкой видеокарты (cupy под CUDA 12; от хоста нужен драйвер)
docker build --build-arg WITH_GPU=true -t rail-guard:gpu .

# запуск с видеокартой
docker run --rm --gpus all -v $PWD/dataset/for_hackathon:/bags:ro \
    rail-guard:gpu demo /bags/doubleT_platform

# то же, но счёт принудительно на процессоре
docker run --rm -e RAIL_GUARD_DEVICE=cpu -v $PWD/dataset/for_hackathon:/bags:ro \
    rail-guard:gpu demo /bags/doubleT_platform
```

Для процессорного образа — та же команда без `--gpus all` и с тегом `rail-guard`.
Если в `/bags` лежит ровно одна запись, имя можно не указывать.
