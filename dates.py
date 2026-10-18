# -*- coding: utf-8 -*-
"""
dates.py: инструмент call_date_agent.

  resolve_date(expr)          -> "18.10.2026" или None, если дату понять не удалось
  validate_booking_date(date) -> текст ошибки для пользователя или None
  moscow_today()              -> сегодняшняя дата по Москве

Если resolve_date вернул None, оркестратор просит пользователя
назвать дату иначе.

"""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Optional

MAX_BOOKING_DAYS_AHEAD = 90

WEEKDAYS = {  # префиксы
    "понедельн": 0, "вторник": 1, "сред": 2, "четверг": 3,
    "пятниц": 4, "суббот": 5, "воскресен": 6,
}
MONTHS = {"янв": 1, "фев": 2, "мар": 3, "апр": 4, "июн": 6, "июл": 7,
          "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12}
NUM_WORDS = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
             "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10}

_MONTH_WORD = r"(?:янв\w*|фев\w*|мар\w*|апр\w*|ма[яй]\w*|июн\w*|июл\w*|авг\w*|сен\w*|окт\w*|ноя\w*|дек\w*)"
_RE_DM_NAME = re.compile(rf"(?<!\d)(\d{{1,2}})(?:\s*-?\s*(?:го|е))?\s+({_MONTH_WORD})(?:\s+(\d{{4}}))?(?!\w*\d)")
_RE_NUMERIC = re.compile(r"(?<![\d:])(\d{1,2})[./\-](\d{1,2})(?:[./\-](\d{2}|\d{4}))?(?![\d:])")
_RE_THROUGH = re.compile(r"через\s+(\d{1,2}|[а-я]+)\s+(дн\w*|день|недел\w*|месяц\w*)")
_RE_DAY_OF_MONTH = re.compile(r"(?<!\d)(\d{1,2})\s*(?:-?\s*(?:го|е))?\s+числа")


def moscow_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Moscow")).replace(tzinfo=None)
    except Exception:
        return datetime.now()


def moscow_today() -> date:
    return moscow_now().date()


def _month_from_word(w: str) -> Optional[int]:
    if w.startswith(("мая", "май")):
        return 5
    return MONTHS.get(w[:3])


def _make(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _with_year(d: int, m: int, year: Optional[int], today: date) -> Optional[date]:
    """Дата без года -> этот год, а если уже прошла -> следующий."""
    if year is not None:
        return _make(year, m, d)
    cand = _make(today.year, m, d)
    if cand is not None and cand >= today:
        return cand
    return _make(today.year + 1, m, d)  


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def resolve_date(expression: str, today: Optional[date] = None) -> Optional[str]:
    """Русское выражение даты -> 'ДД.ММ.ГГГГ' или None."""
    today = today or moscow_today()
    s = re.sub(r"\s+", " ", (expression or "").lower().replace("ё", "е")).strip()
    if not s:
        return None
    result: Optional[date] = None

    m = _RE_DM_NAME.search(s)                                  # «22 октября», «7 янв 2027»
    if m:
        mon = _month_from_word(m.group(2))
        if mon:
            result = _with_year(int(m.group(1)), mon, int(m.group(3)) if m.group(3) else None, today)
            return _fmt(result)

    m = _RE_NUMERIC.search(s)                                  # «24.12», «22.10.2026», «5/11/26»
    if m and 1 <= int(m.group(2)) <= 12:
        year = None
        if m.group(3):
            year = int(m.group(3)) + (2000 if len(m.group(3)) == 2 else 0)
        return _fmt(_with_year(int(m.group(1)), int(m.group(2)), year, today))

    if "послепослезавтра" in s:
        return _fmt(today + timedelta(days=3))
    if "послезавтра" in s:
        return _fmt(today + timedelta(days=2))
    if "завтра" in s:
        return _fmt(today + timedelta(days=1))
    if "сегодня" in s:
        return _fmt(today)

    m = _RE_THROUGH.search(s)                                  # «через неделю», «через 3 дня», «через две недели»
    if m:
        raw, unit = m.group(1), m.group(2)
        n = int(raw) if raw.isdigit() else NUM_WORDS.get(raw)
        if n is None and raw in ("неделю", "день", "месяц"):
            n = 1
        if n is None:
            return None
        if unit.startswith("дн") or unit == "день":
            return _fmt(today + timedelta(days=n))
        if unit.startswith("недел"):
            return _fmt(today + timedelta(weeks=n))
        return _fmt(_add_months(today, n))
    if re.search(r"через\s+(неделю|месяц)", s):                # «через неделю» без числа
        return _fmt(today + timedelta(weeks=1)) if "недел" in s else _fmt(_add_months(today, 1))

    for prefix, wd in WEEKDAYS.items():                        # «в пятницу», «на четверг», «во вторник»
        pat = r"сред(?:а|у|е|ой)(?!\w)" if prefix == "сред" else rf"{prefix}\w*"      # не «средний»/«среди»
        if re.search(rf"(?<!\w){pat}", s):
            delta = (wd - today.weekday()) % 7 or 7
            return _fmt(today + timedelta(days=delta))

    m = _RE_DAY_OF_MONTH.search(s)                             # «25 числа» -> ближайшее 25-е
    if m:
        day = int(m.group(1))
        for k in range(0, 13):
            base = _add_months(today.replace(day=1), k)
            cand = _make(base.year, base.month, day)
            if cand is not None and cand >= today:
                return _fmt(cand)
    return None                                                


def _fmt(d: Optional[date]) -> Optional[str]:
    return d.strftime("%d.%m.%Y") if d else None


def validate_booking_date(date_str: str, today: Optional[date] = None,
                          max_days: int = MAX_BOOKING_DAYS_AHEAD) -> Optional[str]:
    """Текст ошибки для пользователя (прошла дата / слишком далеко) или None, если дата годится."""
    today = today or moscow_today()
    try:
        d = datetime.strptime(date_str, "%d.%m.%Y").date()
    except ValueError:
        return "Не удалось разобрать дату. Напишите её, пожалуйста, например: 25 октября."
    if d < today:
        return f"Дата {date_str} уже прошла. Напишите, пожалуйста, другую дату."
    if (d - today).days > max_days:
        return f"Бронировать можно не более чем на {max_days} дней вперёд. Выберите, пожалуйста, более раннюю дату."
    return None
