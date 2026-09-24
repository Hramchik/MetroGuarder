"""Выбор вычислительного бэкенда: процессор или видеокарта.

Тракт считает одно и то же двумя способами. На процессоре массивы — это
numpy, на видеокарте — cupy; оба модуля дают почти совпадающий набор
операций, поэтому этапы написаны один раз и принимают модуль массивов
аргументом (`xp`), а не выбирают его сами.

Три правила, которым подчинён этот модуль.

**Результат не должен зависеть от того, где считали.** Видеокарта — способ
успеть к сроку, а не другой алгоритм. Поэтому на GPU перенесены только те
этапы, которые состоят из поэлементных операций, сортировок и гистограмм:
у них нет ни итеративной логики, ни накопленного состояния, и ответ
совпадает с процессорным с точностью до порядка суммирования. Проверяется
это `scripts/check_gpu.py` на реальном кадре.

**Отказ видеокарты не должен останавливать поезд.** Любая ошибка на
устройстве — нет драйвера, кончилась память, несовместимая версия cupy —
переводит тракт на процессор навсегда, с записью в лог. Система при этом
продолжает работать: медленнее, но без перерыва.

**Ничего не переносится «на всякий случай».** Перенос кадра на устройство
стоит времени (24 МБ на 128-луче — это миллисекунды на шине), и этапы,
работающие с сотнями точек, на видеокарте только теряют. Кластеризация и
сопровождение остаются на процессоре сознательно.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

CPU = "cpu"
GPU = "gpu"


@dataclass
class Backend:
    """Выбранный бэкенд и всё, что о нём нужно знать вызывающему коду."""

    name: str = CPU
    xp: Any = np
    device_name: str = "процессор"
    reason: str = ""

    @property
    def on_gpu(self) -> bool:
        return self.name == GPU

    def describe(self) -> str:
        if self.on_gpu:
            return f"видеокарта: {self.device_name}"
        return f"процессор{f' ({self.reason})' if self.reason else ''}"


def _probe_gpu() -> tuple:
    """Пробует поднять cupy и выделить память. Возвращает (модуль, имя, причина).

    Проверка не ограничивается импортом: cupy импортируется и без драйвера, а
    падает уже на первом обращении к устройству. Поэтому здесь делается
    настоящая операция — выделение и сложение маленького массива.
    """
    import importlib.util
    installed = importlib.util.find_spec("cupy") is not None
    try:
        import cupy  # noqa: F401
    except Exception as exc:                       # pragma: no cover — нет cupy
        if installed:
            # Пакет есть, но подгрузить не вышло — обычно это отсутствие
            # библиотек драйвера: контейнер запущен без проброса устройства.
            return None, "", "устройство не проброшено (запустите с --gpus all)"
        return None, "", "образ собран без поддержки видеокарты"
    try:
        probe = cupy.arange(8, dtype=cupy.float32)
        float((probe * 2).sum())
        properties = cupy.cuda.runtime.getDeviceProperties(
            cupy.cuda.runtime.getDevice())
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        return cupy, name, ""
    except Exception as exc:                       # pragma: no cover — нет устройства
        return None, "", f"видеокарта недоступна: {type(exc).__name__}"


def select_backend(preference: str = "auto") -> Backend:
    """Выбирает бэкенд по настройке профиля.

    `auto` — взять видеокарту, если она есть, иначе процессор; `gpu` —
    потребовать видеокарту (без неё остаётся процессор, но причина
    записывается в профиль бэкенда и попадёт в лог); `cpu` — не трогать
    устройство вовсе.
    """
    choice = (preference or "auto").strip().lower()
    if choice == CPU:
        return Backend(name=CPU, xp=np, reason="выбран профилем")
    module, device_name, reason = _probe_gpu()
    if module is None:
        return Backend(name=CPU, xp=np, reason=reason)
    return Backend(name=GPU, xp=module, device_name=device_name)


def to_device(array: Optional[np.ndarray], backend: Backend):
    """Кладёт массив на устройство выбранного бэкенда."""
    if array is None or not backend.on_gpu:
        return array
    return backend.xp.asarray(array)


def to_host(array) -> Optional[np.ndarray]:
    """Возвращает массив в память процессора независимо от того, где он был."""
    if array is None or isinstance(array, np.ndarray):
        return array
    get = getattr(array, "get", None)
    return get() if callable(get) else np.asarray(array)


def scalar(value) -> float:
    """Число из массива-скаляра любого бэкенда."""
    return float(to_host(value)) if hasattr(value, "shape") else float(value)


def sliding_median(values, radius: int, xp) -> Any:
    """Скользящая медиана по первой оси с игнорированием пустых ячеек.

    Написана через сдвиги, а не через `sliding_window_view`: представление с
    произвольными шагами есть не во всех версиях cupy, а сдвиг и укладка в
    новую ось работают одинаково на обоих бэкендах.
    """
    if radius <= 0:
        return values
    window = 2 * radius + 1
    padded = xp.full((values.shape[0] + 2 * radius,) + values.shape[1:],
                     xp.nan, dtype=values.dtype)
    padded[radius:radius + values.shape[0]] = values
    stack = xp.stack([padded[i:i + values.shape[0]] for i in range(window)], axis=-1)
    result = xp.full(values.shape, xp.nan, dtype=values.dtype)
    # Медиана считается только там, где в окне вообще есть данные: срез из
    # одних NaN — нормальное явление (на высоте свода границы может не быть
    # вовсе), но numpy на него ругается, а cupy возвращает мусор.
    has_data = xp.isfinite(stack).any(axis=-1)
    if bool(has_data.any()):
        result[has_data] = xp.nanmedian(stack[has_data], axis=-1)
    return result
