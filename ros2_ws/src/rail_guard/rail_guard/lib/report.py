"""Текстовый вывод для человека, который смотрит за работой системы.

За трактом следят глазами — на стенде, в кабине, при разборе записи. Поэтому
вывод здесь не «лог отладки», а приборная панель: главное видно сразу,
единицы подписаны, состояние отличается цветом, а не только словом.

Три правила.

**Решение — самое заметное на строке.** Оператор ищет глазами момент, когда
система потребовала торможения; он должен находиться без чтения остальных
колонок.

**Числа подписаны и округлены до осмысленного.** Дальность в метрах с
точностью до метра, скорость в м/с до десятой, задержка в миллисекундах:
доли метра и микросекунды здесь ничего не значат и только мешают.

**Цвет отключается сам.** Если вывод идёт в файл или переменная `NO_COLOR`
задана, управляющие последовательности не печатаются: в журнале они
превращаются в мусор.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Dict, Iterable, Optional, Sequence

# Действия системы — те же коды, что в lib/decision.py и в сообщении GaugeStatus.
ACTION_TEXT = {
    0: "СВОБОДЕН",
    1: "ВНИМАНИЕ",
    2: "СЛУЖЕБНОЕ",
    3: "ЭКСТРЕННОЕ",
}
_ACTION_COLOR = {0: "green", 1: "yellow", 2: "orange", 3: "red"}

_CODES = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[32m", "yellow": "\033[33m", "orange": "\033[38;5;208m",
    "red": "\033[31m", "cyan": "\033[36m", "grey": "\033[38;5;245m",
    "white": "\033[97m", "bg_red": "\033[41m\033[97m",
}


def colour_enabled(stream=None) -> bool:
    """Печатать ли управляющие последовательности."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("RAIL_GUARD_COLOR") == "always":
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


class Style:
    """Раскраска, которая сама выключается там, где цвет не нужен."""

    def __init__(self, enabled: Optional[bool] = None, stream=None):
        self.enabled = colour_enabled(stream) if enabled is None else enabled

    def __call__(self, text: str, *names: str) -> str:
        if not self.enabled or not names:
            return text
        prefix = "".join(_CODES.get(name, "") for name in names)
        return f"{prefix}{text}{_CODES['reset']}" if prefix else text

    def action(self, action: int) -> str:
        """Название решения в его цвете; экстренное — на красном поле."""
        text = ACTION_TEXT.get(int(action), "?")
        if int(action) == 3:
            return self(f" {text} ", "bg_red", "bold")
        return self(text, _ACTION_COLOR.get(int(action), "white"), "bold")

    def rule(self, width: Optional[int] = None, char: str = "─") -> str:
        return self(char * (width or terminal_width()), "grey")


def terminal_width(default: int = 100) -> int:
    try:
        return max(60, min(shutil.get_terminal_size((default, 24)).columns, 160))
    except OSError:                                    # pragma: no cover
        return default


def bar(fraction: float, width: int = 12, style: Optional[Style] = None,
        colour: str = "cyan") -> str:
    """Полоса заполнения: доли читаются взглядом быстрее, чем проценты."""
    style = style or Style()
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return style("█" * filled, colour) + style("░" * (width - filled), "grey")


def panel(title: str, lines: Sequence[str], style: Optional[Style] = None,
          width: Optional[int] = None) -> str:
    """Рамка с заголовком — ею открывается работа и закрывается сводка."""
    style = style or Style()
    width = width or terminal_width()
    inner = width - 2
    head = f"┌─ {title} " + "─" * max(0, inner - len(title) - 3) + "┐"
    body = [f"│ {line:<{inner - 2}} │" for line in lines]
    tail = "└" + "─" * inner + "┘"
    return "\n".join([style(head, "cyan")] + body + [style(tail, "cyan")])


# Колонки построчного вывода: заголовок, ширина, выравнивание.
_COLUMNS = (
    ("кадр", 5, ">"), ("точек", 8, ">"), ("путь", 7, ">"), ("коридор", 8, ">"),
    ("помех", 6, ">"), ("ближняя", 9, ">"), ("скорость", 9, ">"),
    ("задержка", 9, ">"), ("решение", 11, "<"),
)


def frame_header(style: Optional[Style] = None) -> str:
    style = style or Style()
    cells = [f"{name:{align}{width}}" for name, width, align in _COLUMNS]
    return style(" ".join(cells), "bold")


def frame_line(index: int, points: int, confirmed_range: float, corridor: float,
               obstacles: int, nearest: float, speed: Optional[float],
               latency_ms: float, action: int, style: Optional[Style] = None) -> str:
    """Одна строка приборной панели.

    Прочерк вместо числа значит «нечего показывать»: помех нет, скорость не
    измерена. Ноль в этих колонках читался бы как измеренное значение.
    """
    style = style or Style()
    near = f"{nearest:.0f} м" if nearest and nearest < 1e6 else "—"
    fast = f"{speed:.1f} м/с" if speed is not None else "—"
    cells = [
        f"{index:>5}", f"{points:>8,}".replace(",", " "),
        f"{confirmed_range:>5.0f} м", f"{corridor:>6.0f} м",
        f"{obstacles:>6}", f"{near:>9}", f"{fast:>9}",
        style(f"{latency_ms:>6.0f} мс", "red" if latency_ms > 150 else "grey"),
    ]
    return " ".join(cells) + " " + style.action(action)


def decisions_block(counts: Dict[int, int], total: int,
                    style: Optional[Style] = None) -> Iterable[str]:
    """Доли решений полосами — так видно распределение, а не только числа."""
    style = style or Style()
    for action in (0, 1, 2, 3):
        count = counts.get(action, 0)
        if not count and action not in (0, 3):
            continue
        share = count / max(total, 1)
        yield (f"  {style.action(action):<22} {bar(share, 14, style):<14} "
               f"{count:>5} ({share * 100:>3.0f} %)")


def duration(seconds: float) -> str:
    """Человеческая запись длительности: секунды, минуты, часы."""
    if seconds < 90:
        return f"{seconds:.0f} с"
    if seconds < 5400:
        return f"{seconds / 60:.1f} мин"
    return f"{seconds / 3600:.1f} ч"


def metres(value: float) -> str:
    return "—" if value is None or value != value else f"{value:.0f} м"
