# -*- coding: utf-8 -*-
"""
protocol.py —  формат обмена «модель <-> код».

Датасет — это текстовые диалоги, один диалог на строку:

    User: <реплика> -> Bot: <ответ> || User: ... -> Bot: ... || Tool: RESULT: <результат> -> Bot: ...

Модель дописывает текст после последнего "-> Bot:". Если в её ответе есть
"ACTION: имя | КЛЮЧ: значение | ...", код выполняет действие и (для действий из
TOOLS_WITH_RESULT) дописывает " || Tool: RESULT: ... -> Bot:" и зовёт модель ещё раз.

  - sanitize_user_text  защита формата от подделки пользователем;
  - clean_generation    обрезка того, что модель «нагаллюцинировала» после ответа;
  - parse_reply         разбор ответа на текст + действие + параметры;
  - validate_action     проверка параметров (модель не должна выдумывать формат);
  - DialogHistory       сборка промпта из истории диалога.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

SEG_SEP = " || "
ARROW = " -> Bot:"
MAX_USER_CHARS = 600

# ---------------------------------------------------------------- допустимые значения
CATEGORIES = {
    "all", "appetizers", "salads", "sushi_rolls", "soups", "noodles",
    "main_courses", "street_food", "desserts", "drinks", "signature",
}
FILTERS = {"vegan", "vegetarian", "spicy", "not_spicy"}
PAYMENTS = {"СБП", "карта", "наличные"}

# Обязательные параметры каждого действия
ACTION_PARAMS: dict[str, tuple[str, ...]] = {
    "show_business_lunch": (),
    "show_menu": ("CATEGORY",),
    "call_menu_filter_agent": ("FILTER",),
    "call_price_agent": ("ITEM",),
    # бронирование: дата -> время -> свободные столики -> бронь
    "call_date_agent": ("EXPRESSION",),
    "call_booking_time_agent": ("DATE", "TIME", "GUESTS", "NAME"),
    "call_free_tables_agent": ("DATE", "TIME", "GUESTS", "NAME"),
    "book_table": ("DATE", "TIME", "GUESTS", "NAME"),
    "check_active_booking": (),
    "cancel_booking": ("CONFIRMED",),
    # отзывы
    "notify_manager": (),
}

# После этих действий код возвращает модели результат и диалог продолжается.
TOOLS_WITH_RESULT = {"call_date_agent", "call_booking_time_agent", "call_free_tables_agent",
                     "book_table", "check_active_booking", "cancel_booking"}
# Остальные действия завершают шаг: код выполняет их сам (показывает меню и т.д.).
TERMINAL_ACTIONS = set(ACTION_PARAMS) - TOOLS_WITH_RESULT

# Если в ответе без ACTION есть эти слова — бот ждёт ответа пользователя, диалог не закрываем.
REQUEST_MARKERS = ("Уточните", "Подскажите", "Напишите", "Скажите", "Назовите", "?")


# ---------------------------------------------------------------- защита ввода
_FORMAT_WORDS = re.compile(r"\b(ACTION|Tool|Bot|User|RESULT)\s*:", re.IGNORECASE)


def sanitize_user_text(text: str, max_chars: int = MAX_USER_CHARS) -> str:
    """Убирает из пользовательского текста всё, чем можно сломать формат
    (разделители диалога, «поддельные» Tool/ACTION), и обрезает длину."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = text.replace("||", "|").replace("->", "-")
    text = _FORMAT_WORDS.sub(lambda m: m.group(1) + " -", text)
    return text[:max_chars].strip()


# ---------------------------------------------------------------- разбор ответа модели
@dataclass
class Reply:
    raw: str                                   # очищенная генерация
    text: str = ""                             # что показать пользователю (может быть пустым)
    action: Optional[str] = None               # имя действия или None
    params: dict[str, Optional[str]] = field(default_factory=dict)   # "null" -> None
    review: Optional[dict[str, str]] = None    # для строк «СУТЬ: ... | ТОНАЛЬНОСТЬ: ...»

    @property
    def has_action(self) -> bool:
        return self.action is not None


def clean_generation(raw: str) -> str:
    """Оставляет только первый ответ бота: обрезает по переводу строки, по '||'
    (дальше модель уже придумывает за пользователя/инструмент) и по <|endoftext|>."""
    raw = raw.split("<|endoftext|>")[0]
    raw = raw.split("\n")[0]
    raw = re.split(r"\s*\|\|", raw)[0]
    return raw.strip()


def parse_reply(raw: str) -> Reply:
    raw = clean_generation(raw)
    idx = raw.find("ACTION:")
    if idx == -1:
        return Reply(raw=raw, text=raw)

    head = raw[:idx].strip().rstrip("|").strip()
    parts = [p.strip() for p in raw[idx + len("ACTION:"):].strip().split(" | ")]
    m = re.match(r"\w+", parts[0])
    action = m.group(0) if m else None

    params: dict[str, Optional[str]] = {}
    for p in parts[1:]:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        v = v.strip()
        params[k.strip()] = None if v == "null" else v

    review = None
    if head.startswith("СУТЬ:"):          # строки-отзывы: служебная классификация, не для пользователя
        review = {}
        for chunk in head.split(" | "):
            if ":" in chunk:
                k, v = chunk.split(":", 1)
                review[k.strip()] = v.strip()
        head = ""
    return Reply(raw=raw, text=head, action=action, params=params, review=review)


# ---------------------------------------------------------------- валидация действия
_DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
# В заказах TIME бывает "17.10.2026 18:15" (основной формат) или просто "18:15" (тогда дата = сегодня).
_DT_RE = re.compile(r"^(\d{2}\.\d{2}\.\d{4} )?([01]\d|2[0-3]):[0-5]\d$")
_ITEM_RE = re.compile(r"^.+ x[1-9]\d?$")


def _valid_date(s: Optional[str]) -> bool:
    if not s or not _DATE_RE.match(s):
        return False
    try:
        datetime.strptime(s, "%d.%m.%Y")
        return True
    except ValueError:
        return False


def validate_action(r: Reply) -> list[str]:
    """Возвращает список ошибок (пустой список = действие корректно).
    Нужна, чтобы код не выполнял то, что модель выдумала: неизвестное действие,
    пропущенные параметры, кривая дата, 'самовывоз' вместо адреса и т.п."""
    if r.action is None:
        return []
    if r.action not in ACTION_PARAMS:
        return [f"неизвестное действие: {r.action}"]

    errs: list[str] = []
    need = ACTION_PARAMS[r.action]
    p = r.params
    errs += [f"нет параметра {k}" for k in need if k not in p]
    errs += [f"лишний параметр {k}" for k in p if k not in need]
    if errs:
        return errs

    a = r.action
    if a == "show_menu" and p["CATEGORY"] not in CATEGORIES:
        errs.append(f"CATEGORY={p['CATEGORY']}")
    elif a == "call_menu_filter_agent" and p["FILTER"] not in FILTERS:
        errs.append(f"FILTER={p['FILTER']}")
    elif a == "call_price_agent" and not p["ITEM"]:
        errs.append("пустой ITEM")
    elif a == "call_date_agent" and not p["EXPRESSION"]:
        errs.append("пустой EXPRESSION")
    elif a == "cancel_booking" and p["CONFIRMED"] != "true":
        errs.append(f"CONFIRMED={p['CONFIRMED']}")
    elif a in ("book_table", "call_free_tables_agent", "call_booking_time_agent"):
        g = p["GUESTS"]
        if not (g and g.isdigit() and 1 <= int(g) <= 50):
            errs.append(f"GUESTS={g}")
        if not _valid_date(p["DATE"]):
            errs.append(f"DATE={p['DATE']}")
        if not (p["TIME"] and _TIME_RE.match(p["TIME"])):
            errs.append(f"TIME={p['TIME']}")
        if not p["NAME"]:
            errs.append("пустое NAME")
    return errs


def expects_answer(r: Reply) -> bool:
    """True, если бот в ответе что-то спросил у пользователя (диалог держим открытым)."""
    t = r.text.lower()
    return r.action is None and any(m.lower() in t for m in REQUEST_MARKERS)


# ---------------------------------------------------------------- история диалога
class DialogHistory:
    """Копит завершённые сегменты диалога и собирает промпт для модели.

    Использование:
        h = DialogHistory()
        prompt = h.prompt_for_user("Хочу столик")   # -> строка для модели
        ...генерация...  reply = parse_reply(generated)
        h.commit(reply.raw)
        # если reply.action in TOOLS_WITH_RESULT и код выполнил инструмент:
        prompt = h.prompt_for_tool("18.10.2026")
        ...генерация...  h.commit(...)
    """

    def __init__(self) -> None:
        self.done: list[str] = []      # завершённые сегменты "<заголовок> -> Bot: <ответ>"
        self._open: Optional[str] = None

    def prompt_for_user(self, text: str) -> str:
        self._open = f"User: {sanitize_user_text(text)}{ARROW}"
        return self._prompt()

    def prompt_for_tool(self, result: str) -> str:
        self._open = f"Tool: RESULT: {sanitize_user_text(result)}{ARROW}"
        return self._prompt()

    def commit(self, reply_raw: str) -> None:
        """Записывает ответ бота (то, что реально показали/выполнили)."""
        if self._open is None:
            raise RuntimeError("commit() без prompt_for_*()")
        self.done.append(f"{self._open} {clean_generation(reply_raw)}".rstrip())
        self._open = None

    def trim(self, max_chars: int) -> None:
        """Выкидывает самые старые сегменты, пока промпт не влезет в max_chars.
        Первым всегда остаётся сегмент 'User:' (как в обучающих диалогах)."""
        while self.done and len(self._prompt()) > max_chars:
            self.done.pop(0)
            while self.done and not self.done[0].startswith("User:"):
                self.done.pop(0)

    def _prompt(self) -> str:
        return SEG_SEP.join(self.done + ([self._open] if self._open else []))

    def reset(self) -> None:
        self.done.clear()
        self._open = None
