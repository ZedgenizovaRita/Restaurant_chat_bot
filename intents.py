# -*- coding: utf-8 -*-
"""intents.py — простые проверки на стороне кода: «да/нет», намерение отменить бронь,
слова про бронирование и задание параметров (имя, число гостей, время должны быть в словах гостя)."""
import re
from typing import Optional

MONTHS = "январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().replace("ё", "е")).strip()


def _first(t: str) -> str:
    m = re.match(r"[^\w+]*([\w+]+)", t)
    return m.group(1) if m else ""


YES = {"да", "даа", "ага", "угу", "конечно", "подтверждаю", "подтверждаем", "верно", "точно", "давайте", "давай", "ок", "окей",
       "хорошо", "согласен", "согласна", "+", "отлично", "ладно", "именно", "уверен", "уверена"}
YES_CANCEL = YES | {"отменяйте", "отмена", "отменяй", "отмените", "отмени", "отменим"}
NO = {"нет", "не", "стоп", "отставить", "ни"}


def is_yes(text: str, cancel: bool = False) -> bool:
    t = norm(text)
    return _first(t) in (YES_CANCEL if cancel else YES) or bool(re.match(r"^(все|всё) верно", t))


def is_no(text: str) -> bool:
    return _first(norm(text)) in NO


CANCEL_RE = re.compile(r"отмен|сня(ть|ти|тие)\s+(мою\s+)?(бронь|стол)|не\s+(придем|придём|сможем\s+прийти|смогу\s+прийти)")
CANCEL_FAQ_RE = re.compile(r"^(как|каким образом|можно ли|платн|что будет|нужно ли|а если)")
BOOKING_RE = re.compile(r"брон|стол|записа|свобод|(?<!\w)мест[оаеу]?\b|посид|резерв")


def is_cancel_intent(text: str) -> bool:
    t = norm(text)
    return bool(CANCEL_RE.search(t)) and not CANCEL_FAQ_RE.search(t)


def has_booking_words(text: str) -> bool:
    return bool(BOOKING_RE.search(norm(text)))


# ---------------------------------------------------------------- заземление параметров
_NUM = {"один": 1, "одного": 1, "одна": 1, "одному": 1, "два": 2, "две": 2, "двух": 2, "двое": 2, "двоих": 2, "вдвоем": 2, "парой": 2,
        "три": 3, "трех": 3, "трое": 3, "троих": 3, "втроем": 3, "четыре": 4, "четырех": 4, "четверо": 4, "четверых": 4, "вчетвером": 4,
        "пять": 5, "пяти": 5, "пятеро": 5, "пятерых": 5, "впятером": 5, "шесть": 6, "шести": 6, "шестеро": 6, "шестерых": 6, "вшестером": 6,
        "семь": 7, "семи": 7, "семеро": 7, "семерых": 7, "восемь": 8, "восьми": 8, "восьмеро": 8, "восьмерых": 8,
        "девять": 9, "девяти": 9, "десять": 10, "десяти": 10, "одиннадцать": 11, "двенадцать": 12, "двенадцати": 12}


def _strip_dates_times(t: str) -> str:
    t = re.sub(r"\d{1,2}[:.]\d{2}([:.]\d{2,4})?", " ", t)                
    t = re.sub(rf"\d{{1,2}}\s*(?:{MONTHS})\w*", " ", t)                     
    return t


def numbers_in(text: str) -> set[int]:
    t = _strip_dates_times(norm(text))
    nums = {int(x) for x in re.findall(r"(?<!\d)\d{1,2}(?!\d)", t)}
    nums |= {_NUM[w] for w in re.findall(r"\w+", t) if w in _NUM}
    return nums


def guests_ok(guests: int, text: str) -> bool:
    return guests in numbers_in(text)


def time_ok(hhmm: str, text: str) -> bool:
    t = norm(text)
    h, m = int(hhmm[:2]), int(hhmm[3:])
    if re.search(rf"(?<!\d)0?{h}\s*[:.\s]\s*{m:02d}(?!\d)", t):
        return True
    if m != 0:
        return False
    nums = numbers_in(t)
    if re.search(r"\b(в|к|на|до)\s+час\b", t):
        nums.add(1)
    return (h in nums or (h > 12 and h - 12 in nums) or (h == 12 and "полдень" in t)
            or (h == 0 and "полноч" in t) or (h == 13 and 1 in nums))


def name_ok(name: str, text: str) -> bool:
    n = norm(name).split(" ")[0]
    if not n:
        return False
    stem = n if len(n) <= 3 else n[:max(3, len(n) - 2)]
    return stem in norm(text)


def ungrounded(params: dict, user_text: str, known_dates: set) -> list[str]:
    """Какие параметры брони модель выдумала: подмножество ['date','time','guests','name']."""
    bad = []
    if params["DATE"] not in known_dates:
        bad.append("date")
    if not time_ok(params["TIME"], user_text):
        bad.append("time")
    if not guests_ok(int(params["GUESTS"]), user_text):
        bad.append("guests")
    if not name_ok(params["NAME"], user_text):
        bad.append("name")
    return bad


# ---------------------------------------------------------------- намерение «хочу забронировать» (а не вопрос про бронь)
BOOK_REQ_RE = re.compile(r"хоч|хотел|хотим|нужн|нужен|надо|нам бы|мне бы|забронир|заброни|запиши|записа|давайте стол|давай стол|"
                         r"есть (свобод|мест)|свободн\w* (стол|мест)|бронь|брон\w* (на|стол)|столик")
BOOK_FAQ_RE = re.compile(r"^(как|что нужно|на сколько|сколько|платн|нужен ли|нужно ли|когда|до скольки|какой|где|можно ли)|"
                         r"несколько|два стол|сразу|депозит|предоплат|отмен")


def is_booking_request(text: str) -> bool:
    t = norm(text)
    return bool(BOOK_REQ_RE.search(t)) and bool(BOOKING_RE.search(t)) and not BOOK_FAQ_RE.search(t)


ABORT_RE = re.compile(r"передумал|не надо|не нужно|не хочу|отбой|оставь|забудь|не будем|ничего не нужно|отмена$|отмените$")


def is_abort(text: str) -> bool:
    return bool(ABORT_RE.search(norm(text)))


# ---------------------------------------------------------------- обработка опечаток
import difflib

_BL = ["бизнес", "ланч", "ланчи", "lunch", "бизнесланч"]
FILTER_WORDS_RE = re.compile(r"остр|пряно|пикант|жгуч|веган|вегет|постн|мяс")


def looks_like_business_lunch(text: str) -> bool:
    words = re.findall(r"[а-яa-z]+", norm(text))
    return any(difflib.get_close_matches(w, _BL, n=1, cutoff=0.75) for w in words if len(w) >= 3)


def looks_like_menu(text: str) -> bool:
    words = re.findall(r"[а-яa-z]+", norm(text))
    return any(difflib.get_close_matches(w, ["меню"], n=1, cutoff=0.8) for w in words if len(w) >= 3)
