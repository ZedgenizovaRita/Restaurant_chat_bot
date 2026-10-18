# -*- coding: utf-8 -*-
"""slots.py — разбор брони из слов гостя правилами: дата, время, число гостей, имя.

    extract(text, today, expecting=None) -> {"date": "ДД.ММ.ГГГГ", "time": "ЧЧ:ММ", "guests": 4, "name": "Рита"}  (только найденное)

expecting — какое поле бот только что спросил ('date'|'time'|'guests'|'name'|None): помогает понять
короткие ответы вроде «19», «четверо», «рита».
"""
import re
from datetime import date
from typing import Optional

from dates import resolve_date

_MONTH = r"(?:янв\w*|фев\w*|мар\w*|апр\w*|ма[яй]\w*|июн\w*|июл\w*|авг\w*|сен\w*|окт\w*|ноя\w*|дек\w*)"
_PERIOD = r"(?:\s+(?:часов|часа|час))?(?:\s+(утра|дня|вечера|ночи))?"
_NOT_GUEST = r"(?!\s*(?:человек|чел\b|персон|гост|взросл|дет))"

HOUR_WORDS = {"час": 1, "часу": 1, "один": 1, "одному": 1, "два": 2, "двум": 2, "три": 3, "трем": 3, "четыре": 4, "четырем": 4,
              "пять": 5, "пяти": 5, "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9,
              "девяти": 9, "десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11, "двенадцать": 12, "двенадцати": 12}
GUEST_WORDS = {"один": 1, "одного": 1, "одна": 1, "одному": 1, "двое": 2, "двоих": 2, "два": 2, "две": 2, "двух": 2, "вдвоем": 2,
               "трое": 3, "троих": 3, "три": 3, "трех": 3, "втроем": 3, "четверо": 4, "четверых": 4, "четыре": 4, "четырех": 4,
               "вчетвером": 4, "пятеро": 5, "пятерых": 5, "пять": 5, "пяти": 5, "впятером": 5, "шестеро": 6, "шестерых": 6,
               "шесть": 6, "шести": 6, "вшестером": 6, "семеро": 7, "семерых": 7, "семь": 7, "семи": 7, "восьмеро": 8,
               "восьмерых": 8, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
               "одиннадцать": 11, "двенадцать": 12, "двенадцати": 12}
_GW = "|".join(sorted(GUEST_WORDS, key=len, reverse=True))
_HW = "|".join(sorted(HOUR_WORDS, key=len, reverse=True))

STOP_NAMES = {x for x in """да нет ок окей хорошо спасибо привет здравствуйте добрый день вечер утро хочу нужен нужна нужно можно забронируйте
забронируй запишите запиши нам бронь давайте есть сегодня завтра послезавтра тогда ладно конечно пожалуйста столик стол меня зовут имя
понедельник вторник среда среду четверг пятница пятницу суббота субботу воскресенье январь февраль март апрель май июнь июль август
сентябрь октябрь ноябрь декабрь привет! здравствуйте! хай бронировать забронировать подскажите скажите напишите отмените отмена один
одна двое трое четверо пятеро шестеро нас вечером утром днем ночью обед ужин на в к""".split()}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("ё", "е")).strip()


def _hour(h: int, period: Optional[str], bare: bool) -> int:
    if period == "вечера":
        return h + 12 if h < 12 else h
    if period == "дня":
        return h + 12 if 1 <= h < 12 else h
    if period == "утра":
        return 0 if h == 12 else h
    if period == "ночи":
        return 0 if h == 12 else h
    if bare and 1 <= h <= 9:                  # ресторан с 10:00, значит вечер
        return h + 12
    return h


def _strip_dates(t: str) -> str:
    t = re.sub(rf"(?<!\d)\d{{1,2}}\s*(?:-?\s*(?:го|е))?\s+{_MONTH}(?:\s+\d{{4}})?", " ", t)     
    t = re.sub(r"(?<!\d)\d{1,2}\s*(?:-?\s*(?:го|е))?\s+числа", " ", t)
    return t


def _date_dots(t: str):
    """Числовые даты 24.10 / 24.10.2026 (но не время 19.30)."""
    out = []
    for m in re.finditer(r"(?<![\d:.])(\d{1,2})[./](\d{1,2})(?:[./](\d{2}|\d{4}))?(?![\d:.])", t):
        a, b = int(m.group(1)), int(m.group(2))
        before = t[max(0, m.start() - 8):m.start()]
        timeish = bool(re.search(r"(?:^|\s)(?:в|к|около|ровно)\s*$", before)) and b in (0, 15, 30, 45) and a <= 23
        if 1 <= b <= 12 and 1 <= a <= 31 and not timeish:
            out.append(m.span())
    return out


def extract_time(t: str) -> Optional[str]:
    t = _strip_dates(t)
    for a, b in reversed(_date_dots(t)):
        t = t[:a] + " " + t[b:]
    m = re.search(rf"(?<!\d)([01]?\d|2[0-3])\s*[:.]\s*([0-5]\d)(?!\d){_PERIOD}", t)
    if m:
        h = _hour(int(m.group(1)), m.group(3), False)
        return f"{h:02d}:{m.group(2)}"
    if re.search(r"в полдень|к полудню", t):
        return "12:00"
    m = re.search(rf"(?<![\w:.])(?:в|к|около|ровно|до)\s+(\d{{1,2}})(?![\d:.])(?:\s+час\w*)?(?:\s+(утра|дня|вечера|ночи))?{_NOT_GUEST}", t)
    if m and int(m.group(1)) <= 24:
        h = _hour(int(m.group(1)) % 24, m.group(2), True)
        return f"{h:02d}:00"
    m = re.search(rf"(?<![\w:.])на\s+(\d{{1,2}})(?![\d:.])(?:\s+час\w*)(?:\s+(утра|дня|вечера|ночи))?|"
                  rf"(?<![\w:.])на\s+(\d{{1,2}})\s+(утра|дня|вечера|ночи)", t)
    if m:
        h = int(m.group(1) or m.group(3)) % 24
        return f"{_hour(h, m.group(2) or m.group(4), True):02d}:00"
    m = re.search(rf"(?<![\w:.])(?:в|к|около|на)\s+({_HW})\b(?:\s+час\w*)?(?:\s+(утра|дня|вечера|ночи))?", t)
    if m:
        h = HOUR_WORDS[m.group(1)]
        if m.group(1) == "час" and not m.group(2):
            return "13:00"
        return f"{_hour(h, m.group(2), True):02d}:00"
    return None


def extract_guests(t: str, expecting: Optional[str]) -> Optional[int]:
    t = _strip_dates(t)
    for a, b in reversed(_date_dots(t)):
        t = t[:a] + " " + t[b:]
    t = re.sub(r"(?<!\d)([01]?\d|2[0-3])\s*[:.]\s*[0-5]\d(?!\d)(?:\s+(?:утра|дня|вечера|ночи))?", " ", t)              
    t = re.sub(rf"(?<![\w:.])(?:в|к|около|ровно|до)\s+\d{{1,2}}(?![\d:.])(?:\s+час\w*)?(?:\s+(?:утра|дня|вечера|ночи))?{_NOT_GUEST}", " ", t)
    t = re.sub(r"(?<![\w:.])на\s+\d{1,2}\s+(?:час\w*|утра|дня|вечера|ночи)", " ", t)
    t = re.sub(rf"(?<![\w:.])(?:в|к|около|на)\s+(?:{_HW})\s+(?:час\w*|утра|дня|вечера|ночи)", " ", t)
    m = re.search(r"(?<!\d)(\d{1,2})\s*(?:человек\w*|чел\b|персон\w*|гост\w*|взросл\w*)", t)
    if m:
        return int(m.group(1))
    m = re.search(rf"(?<![\w])({_GW})\s+(?:человек\w*|персон\w*|гост\w*|взросл\w*)", t)
    if m:
        return GUEST_WORDS[m.group(1)]
    m = re.search(rf"(?<![\w])(?:нас|будет|компанией из|компания из|группа из|группой из)\s+(?:будет\s+)?(\d{{1,2}}|{_GW})(?![\d:.])", t)
    if m:
        x = m.group(1)
        return int(x) if x.isdigit() else GUEST_WORDS[x]
    m = re.search(rf"(?<![\w])(?:на|для)\s+(\d{{1,2}})(?![\d:.\w])", t)
    if m:
        return int(m.group(1))
    m = re.search(rf"(?<![\w])(?:на|для)\s+({_GW})(?![\w])|(?<![\w])(вдвоем|втроем|вчетвером|впятером|вшестером)(?![\w])|(?<![\w])я\s+(один|одна)(?![\w])", t)
    if m:
        x = next(g for g in m.groups() if g)
        return GUEST_WORDS[x]
    if expecting == "guests":
        m = re.fullmatch(r"\D*(\d{1,2})\D*", t.strip())
        if m:
            return int(m.group(1))
        w = re.findall(r"\w+", t)
        if len(w) <= 2 and w and w[-1] in GUEST_WORDS:
            return GUEST_WORDS[w[-1]]
    return None


_CAP = r"(?<![\w\-])[А-ЯЁA-Z][а-яёa-z\-]{1,}"
_QWORDS = set("где как что когда сколько какой какая какие какое почему зачем куда откуда можно ли есть это а но и или подскажите скажите расскажите покажите".split())
_NAME_STOP = STOP_NAMES | set(GUEST_WORDS) | _QWORDS | {"wi", "fi", "wifi", "вайфай"}


def _is_stop(w: str) -> bool:
    return _norm(w) in _NAME_STOP


def extract_name(text: str, expecting: Optional[str]) -> Optional[str]:
    if expecting == "name" and ("?" in text or len(text.split()) > 4):
        expecting = None                                    # вопрос или длинная фраза — не имя
    m = re.search(r"(?:меня зовут|зовут|мо[её] имя|на имя|имя)\s+([а-яёa-z\-]{2,})", text, re.I)
    if m and not _is_stop(m.group(1)):
        return m.group(1).capitalize()
    m = re.search(rf"(?<![а-яёА-ЯЁ])[яЯ]\s+({_CAP})(?!\w)", text)
    if m and not _is_stop(m.group(1)):
        return m.group(1)
    cands = [(x.start(), x.group(0)) for x in re.finditer(_CAP, text) if not _is_stop(x.group(0))]
    if cands:
        pos, tok = cands[-1]
        before = text[:pos].rstrip()
        if before.endswith(",") or before.endswith(":") or pos + len(tok) >= len(text.strip().rstrip("!.?")) or expecting == "name":
            if pos > 0 or expecting == "name" or len(text.split()) == 1:
                return tok
    if expecting == "name":
        w = re.findall(r"[а-яёa-z\-]{2,}", text.lower())
        if len(w) == 1 and not _is_stop(w[0]):
            return w[0].capitalize()
    return None


def extract(text: str, today: Optional[date] = None, expecting: Optional[str] = None) -> dict:
    t = _norm(text)
    out = {}
    d = resolve_date(t, today)
    if d:
        out["date"] = d
    tm = extract_time(t)
    if tm:
        out["time"] = tm
    elif expecting == "time":
        m = re.fullmatch(r"\D*?(\d{1,2})\D*", t)
        if m and int(m.group(1)) <= 24:
            out["time"] = f"{_hour(int(m.group(1)) % 24, None, True):02d}:00"
    g = extract_guests(t, expecting)
    if g:
        out["guests"] = g
    nm = extract_name(text, expecting)
    if nm:
        out["name"] = nm
    return out
