# -*- coding: utf-8 -*-
"""
menu.py — меню, бизнес-ланчи, цены, фильтры

Инструменты для оркестратора (все возвращают текст в формате Telegram HTML,
отправлять с parse_mode="HTML"):

  tool_show_menu(category)   -> list[str]   ACTION: show_menu | CATEGORY: ...
  tool_show_business_lunch() -> list[str]   ACTION: show_business_lunch
  tool_price(item)           -> str         ACTION: call_price_agent | ITEM: ...
  tool_filter(name)          -> list[str]   ACTION: call_menu_filter_agent | FILTER: ...

Принципы:
  - цену бот берёт ТОЛЬКО из menu.json; если price = null, честно говорит, что цены нет в базе;
  - блюдо ищется по названию: «Филадельфию», «ролл Филадельфия», «том яма»,
    «удон», «tokyo lunch», опечатка «филадельфея» — всё находит нужное блюдо;
  - если запрос подходит нескольким блюдам, бот покажет все, а не выберет наугад;
  - в фильтры попадают только блюда с явно заданными флагами (null = «не знаю» = не попадает);
  - файл перечитывается автоматически при изменении — перезапуск бота ради новой цены не нужен.

Файлы: menu.json (основное) и business_lunch.json (бизнес-ланчи, необязателен; пути можно задать MENU_PATH / LUNCH_PATH).
Проверка файлов:  python menu.py menu.json business_lunch.json
"""
from __future__ import annotations

import difflib
import html
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import Optional

_DIR = os.path.dirname(os.path.abspath(__file__))
MENU_PATH = os.environ.get("MENU_PATH") or os.path.join(_DIR, "menu.json")
LUNCH_PATH = os.environ.get("LUNCH_PATH") or os.path.join(_DIR, "business_lunch.json")   # необязателен
BUSINESS_LUNCH_SCHEDULE = "с понедельника по пятницу с 12:00 до 16:00"

CATEGORY_TITLES = {
    "appetizers": "Закуски", "salads": "Салаты", "sushi_rolls": "Суши и роллы", "soups": "Супы",
    "noodles": "Лапша и вок", "main_courses": "Горячие блюда", "street_food": "Стритфуд",
    "desserts": "Десерты", "drinks": "Напитки",
}
CATEGORY_ORDER = list(CATEGORY_TITLES)
FILTER_TITLES = {
    "vegan": "Веганские блюда", "vegetarian": "Вегетарианские блюда",
    "spicy": "Острые блюда", "not_spicy": "Неострые блюда",
}
NOT_FOUND_SUFFIX = "Можете посмотреть варианты в меню."


# ---------------------------------------------------------------- данные
@dataclass(frozen=True)
class Item:
    name: str
    category: Optional[str] = None
    price: Optional[float] = None
    description: str = ""
    weight: Optional[str] = None
    spicy: Optional[bool] = None
    vegan: Optional[bool] = None
    vegetarian: Optional[bool] = None
    business_lunch: bool = False
    signature: bool = False
    aliases: tuple = ()


class Menu:
    def __init__(self, items: list[Item]) -> None:
        self.items = items

    @staticmethod
    def _read(path: str, lunch: bool) -> list[Item]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        rows = raw["items"] if isinstance(raw, dict) else raw
        items = []
        for d in rows:
            d = {k: v for k, v in d.items() if not k.startswith("_") and k in Item.__dataclass_fields__}
            d["aliases"] = tuple(d.get("aliases") or ())
            d["description"] = d.get("description") or ""             # "description": null в menu.json не должен ронять форматирование
            if lunch:
                d["business_lunch"], d["category"] = True, None        # всё из файла бизнес-ланчей — бизнес-ланчи
            if d.get("vegetarian") is None and d.get("vegan") is True:
                d["vegetarian"] = True                 # веганское всегда вегетарианское
            d["signature"] = bool(d.get("signature")) or d["name"].upper().startswith("YUQI")
            items.append(Item(**d))
        return items

    @classmethod
    def load(cls, path: str, lunch_path: Optional[str] = None) -> "Menu":
        """path — основное меню; lunch_path — отдельный файл бизнес-ланчей (если есть)."""
        items = cls._read(path, False)
        if lunch_path and os.path.exists(lunch_path):
            items += cls._read(lunch_path, True)
        return cls(items)


_cache: dict = {"key": None, "menu": None}
_lock = threading.Lock()


def get_menu(path: Optional[str] = None, lunch_path: Optional[str] = None) -> Menu:
    """Меню из двух файлов (основное + бизнес-ланчи); перечитывается, если любой из них изменился."""
    path = path or MENU_PATH
    lunch_path = lunch_path or (LUNCH_PATH if path == MENU_PATH else None)
    with _lock:
        key = (path, os.path.getmtime(path), lunch_path, os.path.getmtime(lunch_path) if lunch_path and os.path.exists(lunch_path) else None)
        if _cache["key"] != key:
            _cache.update(key=key, menu=Menu.load(path, lunch_path))
        return _cache["menu"]


def check_menu(menu: Menu) -> list[str]:
    """Предупреждения для того, кто заполняет menu.json."""
    w, seen = [], set()
    for it in menu.items:
        if it.name in seen:
            w.append(f"дубликат названия: {it.name}")
        seen.add(it.name)
        if it.price is None:
            w.append(f"нет цены: {it.name}")
        if not it.business_lunch and it.category not in CATEGORY_TITLES:
            w.append(f"неизвестная категория у «{it.name}»: {it.category!r}")
    return w


# ---------------------------------------------------------------- поиск блюда по названию
_STOP = {"с", "со", "и", "в", "на", "по", "из", "а"}


def norm(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _tokens(s: str) -> list[str]:
    return [w for w in norm(s).split() if w not in _STOP]


def _same(a: str, b: str) -> bool:
    """Слова совпадают с точностью до окончания: филадельфия/филадельфию, ям/яма, удон/удона."""
    if a == b:
        return True
    if min(len(a), len(b)) >= 5:
        return a[:5] == b[:5]
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 2 and long_.startswith(short) and len(long_) - len(short) <= 2


def _match_score(q: list[str], t: list[str]) -> int:
    if not q or not t:
        return 0
    mq = sum(any(_same(x, y) for y in t) for x in q)
    mi = sum(any(_same(x, y) for y in q) for x in t)
    if mq == len(q) or mi == len(t):       # весь запрос найден в названии ИЛИ всё название найдено в запросе
        return mq + mi
    return 0


def is_business_lunch_query(query: str) -> bool:
    q = norm(query)
    return "бизнес" in q or "ланч" in q or "обед" in q


def find_items(query: str, menu: Optional[Menu] = None) -> list[Item]:
    """Блюда, подходящие под запрос (лучшие совпадения; несколько — если запрос неоднозначен)."""
    menu = menu or get_menu()
    qn = norm(query)
    if not qn:
        return []
    names = {it: [it.name, *it.aliases] for it in menu.items}

    exact = [it for it, ns in names.items() if any(norm(n) == qn for n in ns)]
    if exact:
        return exact[:1]

    q = _tokens(query)
    scored = [(max(_match_score(q, _tokens(n)) for n in ns), it) for it, ns in names.items()]
    best = max((s for s, _ in scored), default=0)
    if best > 0:
        return [it for s, it in scored if s == best]

    flat = {norm(n): it for it, ns in names.items() for n in ns}      # опечатки
    close = difflib.get_close_matches(qn, list(flat), n=3, cutoff=0.78)
    out: list[Item] = []
    for c in close:
        if flat[c] not in out:
            out.append(flat[c])
    return out


# ---------------------------------------------------------------- форматирование
def fmt_price(p: Optional[float]) -> str:
    if p is None:
        return ""
    return f"{int(p):,}".replace(",", " ") + " ₽" if float(p).is_integer() else f"{p:g} ₽"


def _item_line(it: Item) -> str:
    e = html.escape
    head = f"• <b>{e(it.name)}</b>" + (f" — {fmt_price(it.price)}" if it.price is not None else "")
    extra = ", ".join(x for x in (e(it.description), e(it.weight or "")) if x)
    return head + (f"\n   {extra}" if extra else "")


def split_message(text: str, limit: int = 4000) -> list[str]:
    """Режет длинный текст по пустым строкам (лимит Telegram — 4096 символов)."""
    parts, cur = [], ""
    for para in text.split("\n\n"):
        while len(para) > limit:                       # аварийно: один абзац длиннее лимита
            parts.append((cur + "\n\n" + para[:limit]).strip() if cur else para[:limit])
            cur, para = "", para[limit:]
        if cur and len(cur) + 2 + len(para) > limit:
            parts.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        parts.append(cur)
    return parts


def _section(title: str, items: list[Item]) -> str:
    return f"<b>{html.escape(title)}</b>\n" + "\n".join(_item_line(i) for i in items)


# ---------------------------------------------------------------- инструменты
def tool_show_menu(category: str, menu: Optional[Menu] = None) -> list[str]:
    menu = menu or get_menu()
    if category == "all":
        blocks = [_section(CATEGORY_TITLES[c], its) for c in CATEGORY_ORDER
                  if (its := [i for i in menu.items if i.category == c and not i.business_lunch])]
        if not blocks:
            return ["Меню пока пусто."]
        blocks.append("Бизнес-ланчи показываю по команде «Покажи меню бизнес-ланчей».")
        return split_message("\n\n".join(blocks))
    if category == "signature":
        title, items = "Фирменные блюда YUQI", [i for i in menu.items if i.signature]
    else:
        title = CATEGORY_TITLES.get(category, category)
        items = [i for i in menu.items if i.category == category and not i.business_lunch]
    if not items:
        return [f"В разделе «{html.escape(title)}» пока нет блюд. {NOT_FOUND_SUFFIX}"]
    return split_message(_section(title, items))


def tool_show_business_lunch(menu: Optional[Menu] = None) -> list[str]:
    menu = menu or get_menu()
    items = [i for i in menu.items if i.business_lunch]
    head = f"<b>Бизнес-ланчи</b>\nДоступны {BUSINESS_LUNCH_SCHEDULE}."
    if not items:
        return [head + "\n\nМеню бизнес-ланчей пока пусто."]
    return split_message(head + "\n\n" + "\n".join(_item_line(i) for i in items))


def tool_price(item_query: str, menu: Optional[Menu] = None) -> str:
    """Реальная цена из menu.json. Никогда не придумывает: нет блюда/цены — так и говорит."""
    menu = menu or get_menu()
    e = html.escape
    found = find_items(item_query, menu)
    if not found and is_business_lunch_query(item_query):
        found = [i for i in menu.items if i.business_lunch]
        if found:
            found_prefix = f"Бизнес-ланчи у нас {BUSINESS_LUNCH_SCHEDULE}.\n"
        else:
            found_prefix = ""
    else:
        found_prefix = ""
    if not found:
        similar = difflib.get_close_matches(norm(item_query), [norm(i.name) for i in menu.items], n=3, cutoff=0.5)
        names = [i.name for i in menu.items if norm(i.name) in similar]
        hint = f" Возможно, вы имели в виду: {', '.join(e(n) for n in names)}." if names else ""
        return f"Не нашёл блюдо «{e(item_query)}» в меню.{hint} {NOT_FOUND_SUFFIX}"

    lines = []
    for it in found:
        if it.price is None:
            lines.append(f"• <b>{e(it.name)}</b> — цены пока нет в базе")
        else:
            lines.append(f"• <b>{e(it.name)}</b> — {fmt_price(it.price)}")
    if len(found) == 1:
        it = found[0]
        return (f"{e(it.name)} — {fmt_price(it.price)}" if it.price is not None
                else f"Цены на «{e(it.name)}» пока нет в базе. Уточните, пожалуйста, у администратора.")
    return found_prefix + "Нашёл несколько подходящих блюд:\n" + "\n".join(lines)


def tool_filter(filter_name: str, menu: Optional[Menu] = None) -> list[str]:
    menu = menu or get_menu()
    title = FILTER_TITLES.get(filter_name, filter_name)
    if filter_name == "vegan":
        items = [i for i in menu.items if i.vegan is True]
    elif filter_name == "vegetarian":
        items = [i for i in menu.items if i.vegetarian is True]
    elif filter_name == "spicy":
        items = [i for i in menu.items if i.spicy is True]
    elif filter_name == "not_spicy":
        items = [i for i in menu.items if i.spicy is False and i.category not in ("drinks", "desserts")]
    else:
        items = []
    if not items:
        return [f"По данным меню подходящих блюд («{html.escape(title.lower())}») не нашлось. {NOT_FOUND_SUFFIX}"]
    return split_message(_section(title, items))


# ---------------------------------------------------------------- проверка файла
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else MENU_PATH
    m = get_menu(path, sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"блюд: {len(m.items)}, из них бизнес-ланчей: {sum(i.business_lunch for i in m.items)}")
    for w in check_menu(m):
        print("  !", w)
