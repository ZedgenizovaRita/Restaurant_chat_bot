# -*- coding: utf-8 -*-
"""menu_router.py — вопросы про меню отвечает код по menu.json, а не модель (модель путает блюда и выдумывает).

    route(text, menu, rng) -> list[str] | None     # готовые сообщения (Telegram HTML) или None — пусть решает модель

Что умеет:
  • фильтры: «вегетарианские блюда», «что-нибудь острое», «неострое» (опечатки тоже: «вегетерианкие»)
  • рекомендации: «посоветуйте», «что взять на двоих», «что-нибудь для ребёнка» — случайно из меню, с учётом фильтров
  • «какие у вас есть удоны / роллы / десерты / лапша» — поиск по названиям, составу и категориям
  • состав блюда: «из чего Филадельфия», «что в составе Том Яма»
  • цена блюда: «сколько стоит удон»
"""
from __future__ import annotations

import html
import random
import re
from typing import Optional

from menu import (CATEGORY_TITLES, is_business_lunch_query, tool_show_business_lunch, Item, Menu, _item_line, _same, _tokens, fmt_price, find_items, norm,
                  split_message, tool_filter, tool_price, tool_show_menu)

CATEGORY_STEMS = {
    "appetizers": ("закуск",), "salads": ("салат",), "sushi_rolls": ("ролл", "суши", "сашими", "маки"),
    "soups": ("суп", "бульон"), "noodles": ("лапш", "фунчоз"),
    "main_courses": ("горяч", "основн"), "street_food": ("стрит", "уличн"),
    "desserts": ("десерт", "сладк", "торт", "мороженое", "пирожн"), "drinks": ("напит", "чай", "кофе", "лимонад", "сок"),
}
REC_RE = re.compile(r"посовет|порекоменд|рекоменд|что (?:бы )?(?:взять|заказать|выбрать|попробовать|вкусн)|"
                    r"на что обратить|что.{0,15}(?:популярн|хит)|чем (?:перекус|поужин)|что у вас вкусн|не могу выбрать|"
                    r"что.{0,12}из меню|что.{0,12}на (?:ужин|обед|ланч|завтрак)")
PRICE_RE = re.compile(r"сколько (?:стоит|стоят|будет стоить)|цена|цены|почем|стоимост|по чем")
COMPOSE_RE = re.compile(r"из чего|состав|что входит|что (?:там )?в (?:составе|нем|ней|нём)|что внутри|из каких")
LIST_RE = re.compile(r"какие|какая|какой|какое|что у вас|есть ли|есть у вас|у вас есть|покажи|расскажи|что за|что.нибудь|"
                     r"подскажи|хочу|хотел|можно|а .*есть|^есть\b|найди|предлож")
REF_RE = re.compile(r"оттуда|из них|из эт(?:их|ого|ой)|из списка|из перечисленн|из предложенн|из показанн|из того|среди них|из вышеперечисленн")
SKIP_RE = re.compile(r"брон|столик|адрес|где вы|как добрат|парковк|wi.?fi|вайфай|розетк|час\w* работы|до скольки|"
                     r"отзыв|жалоб|алкогол|доставк|самовывоз|бизнес|ланч")
FILTER_TOKEN_RE = re.compile(r"^(?:вег|неостр|остр|пряно|пикант|жгуч|постн)")
STOPWORDS = {"какие", "какая", "какой", "есть", "вас", "меню", "блюда", "блюдо", "ваше", "ваши", "покажите", "расскажите",
             "подскажите", "хочу", "хотел", "хотела", "можно", "пожалуйста", "что", "что-нибудь", "нибудь", "еще", "ещё",
             "посоветуйте", "порекомендуйте", "есть", "ли"}


MEAT_RE = re.compile(r"кури|курин|свин|говяд|утк|утин|бекон|мяс|колбас|ветчин|баранин|телят|индейк|карбонад")


def detect_filter(text: str) -> Optional[str]:
    t = norm(text)
    if re.search(r"без мяс|безмяс|не ем мяс|без мясн|мяса не|не мясн|без курицы|без свинин|без говядин", t):
        return "no_meat"
    if re.search(r"(?<!\w)не\s?(?:очень\s)?(?:остр|пряно|пикант|жгуч)|неостр|без остро|нежгуч|нейтральн|для дет|ребен|ребён|детям", t):
        return "not_spicy"
    for w in re.findall(r"[а-я]+", t):
        if w.startswith("веган") or w.startswith("постн"):
            return "vegan"
        if w.startswith("вег") and len(w) >= 6:                        # вегетарианское, вегетерианкие, вегетарианцам
            return "vegetarian"
    if re.search(r"остр|пряно|пикант|жгуч", t):
        return "spicy"
    return None


def _apply(items: list[Item], flt: Optional[str]) -> list[Item]:
    if flt == "no_meat":                                               # рыба и морепродукты можно, мяса и птицы нет
        return [i for i in items if i.category not in ("drinks", "desserts") and not MEAT_RE.search(norm(i.name + " " + i.description))]
    if flt == "vegan":
        return [i for i in items if i.vegan is True]
    if flt == "vegetarian":
        return [i for i in items if i.vegetarian is True]
    if flt == "spicy":
        return [i for i in items if i.spicy is True]
    if flt == "not_spicy":
        return [i for i in items if i.spicy is False and i.category not in ("drinks", "desserts")]
    return items


def _regular(menu: Menu) -> list[Item]:
    return [i for i in menu.items if not i.business_lunch]


def _category_of(tokens: list[str]) -> Optional[str]:
    for cat, stems in CATEGORY_STEMS.items():
        if any(t.startswith(s) for t in tokens for s in stems):
            return cat
    return None


def _dish_hits(text: str, menu: Menu) -> list[Item]:
    """Блюда, у которых в названии/псевдонимах есть слова из вопроса (удоны -> Удон с курицей)."""
    cat_stems = tuple(s for ss in CATEGORY_STEMS.values() for s in ss)
    q = [w for w in _tokens(text) if len(w) >= 4 and w not in STOPWORDS and not FILTER_TOKEN_RE.match(w)
         and not w.startswith(cat_stems)]                               # «лапша», «роллы» — это раздел, а не блюдо
    hits = []
    for it in _regular(menu):
        name_tokens = [w for n in (it.name, *it.aliases) for w in _tokens(n) if len(w) >= 3]
        if any(_same(a, b) for a in q for b in name_tokens):
            hits.append(it)
    return hits


def _ingredient_hits(text: str, menu: Menu) -> list[Item]:
    q = [w for w in _tokens(text) if len(w) >= 5 and w not in STOPWORDS and not FILTER_TOKEN_RE.match(w)]
    out = []
    for it in _regular(menu):
        dt = [w for w in _tokens(it.description) if len(w) >= 5]
        if any(_same(a, b) for a in q for b in dt):
            out.append(it)
    return out


def _list_message(title: str, items: list[Item]) -> list[str]:
    return split_message(f"<b>{html.escape(title)}</b>\n" + "\n".join(_item_line(i) for i in items))


def _recommend(text: str, menu: Menu, rng: random.Random, flt: Optional[str], ctx: Optional[dict] = None) -> list[str]:
    t = norm(text)
    names = set((ctx or {}).get("names") or ())
    if REF_RE.search(t) and names:                                      # «посоветуйте что-нибудь оттуда» — из только что показанного
        pool = [i for i in _regular(menu) if i.name in names]
        flt = flt or (ctx or {}).get("flt")
        pool = _apply(pool, flt) or pool
        return _pick_message(pool, rng, 3 if re.search(r"двоих|компани|вчетвером|втроем|втроём", t) else 2)
    pool = _apply([i for i in _regular(menu) if i.category not in ("drinks", "desserts")], flt)
    cat = _category_of(_tokens(text))
    if cat:
        pool = [i for i in pool if i.category == cat] or pool
    if re.search(r"десерт|сладк", t):
        pool = _apply([i for i in _regular(menu) if i.category == "desserts"], flt)
    elif re.search(r"напит|попить|выпить", t):
        pool = _apply([i for i in _regular(menu) if i.category == "drinks"], flt)
    elif re.search(r"сытн|ужин|обед|голодн|плотн", t) and not cat:
        pool = [i for i in pool if i.category in ("noodles", "main_courses", "soups")] or pool
    elif re.search(r"перекус|легк|закуск", t) and not cat:
        pool = [i for i in pool if i.category in ("appetizers", "salads", "street_food")] or pool
    if not pool:
        title = {"no_meat": "блюд без мяса", "vegan": "веганских", "vegetarian": "вегетарианских", "spicy": "острых", "not_spicy": "неострых"}.get(flt or "", "подходящих")
        return [f"К сожалению, {title} блюд в меню не нашлось. Можете посмотреть остальные варианты в меню."]
    return _pick_message(pool, rng, 3 if re.search(r"двоих|компани|вчетвером|втроем|втроём|нас \d|на \d", t) else 2)


def _pick_message(pool: list[Item], rng: random.Random, n: int) -> list[str]:
    sig = [i for i in pool if i.signature]
    picks: list[Item] = []
    if sig and rng.random() < 0.5:
        picks.append(rng.choice(sig))
    rest = [i for i in pool if i not in picks]
    rng.shuffle(rest)
    seen = {p.category for p in picks}
    for i in rest:                                                      # разные категории — разнообразнее
        if len(picks) >= n:
            break
        if i.category not in seen:
            picks.append(i)
            seen.add(i.category)
    for i in rest:
        if len(picks) >= n:
            break
        if i not in picks:
            picks.append(i)
    lines = [_item_line(i) for i in picks]
    return split_message("Рекомендую:\n" + "\n".join(lines) + "\n\nОстальные варианты можно посмотреть в меню.")


def route(text: str, menu: Menu, rng: Optional[random.Random] = None, ctx: Optional[dict] = None) -> Optional[list[str]]:
    """ctx — память чата о последнем показанном списке {"names": [...], "flt": ...}; route обновляет её сам."""
    out = _route(text, menu, rng or random, ctx if ctx is not None else {})
    return out


def _remember(ctx: dict, items: list[Item], flt: Optional[str]) -> None:
    ctx["names"], ctx["flt"] = [i.name for i in items], flt


def _route(text: str, menu: Menu, rng, ctx: dict) -> Optional[list[str]]:
    t = norm(text)
    prev_topic, ctx["topic"] = ctx.get("topic"), None                    # тема прошлого вопроса («а сколько стоит?»)
    if not t:
        return None
    if re.search(r"бизнес|ланч|lunch", t) and not re.search(r"брон|столик", t):
        ctx["topic"] = "lunch"
        if PRICE_RE.search(t):                                           # «сколько стоит бизнес ланч» — цены из business_lunch.json
            return tool_show_business_lunch(menu)
        return None                                                      # часы работы и т.п. — по датасету
    if PRICE_RE.search(t) and not SKIP_RE.search(t):
        rest = re.sub(PRICE_RE, " ", t)
        if not find_items(rest, menu) and not _dish_hits(rest, menu) \
                and not [w for w in _tokens(rest) if len(w) >= 3 and w not in STOPWORDS]:   # «а сколько стоит?» без названия
            if prev_topic == "lunch":
                ctx["topic"] = "lunch"
                return tool_show_business_lunch(menu)
            names = ctx.get("names") or []
            if len(names) == 1:
                return [tool_price(names[0], menu)]
            return ["Уточните, пожалуйста, цену какого блюда вас интересует? Например: «сколько стоит Том Ям» или «сколько стоит удон»."]
    if SKIP_RE.search(t):
        return None
    flt = detect_filter(text)
    tokens = _tokens(text)

    # цена
    if PRICE_RE.search(t):
        q = re.sub(PRICE_RE, " ", t)
        if find_items(q, menu) and not _is_business(find_items(q, menu)):
            return [tool_price(q, menu)]
        return None

    # фирменное
    if re.search(r"фирменн|авторск|визитн|шеф", t):
        sig = [i for i in _regular(menu) if i.signature]
        if sig:
            return tool_show_menu("signature", menu)

    # состав
    if COMPOSE_RE.search(t):
        found = _dish_hits(text, menu)
        if not found:
            return None
        e = html.escape
        _remember(ctx, found[:3], None)
        lines = [f"<b>{e(i.name)}</b>: {e(i.description)}" + (f" ({e(i.weight)})" if i.weight else "")
                 if i.description else f"<b>{e(i.name)}</b>: подробного состава в меню пока нет." for i in found[:3]]
        return ["\n".join(lines)]

    # рекомендации
    if REC_RE.search(t):
        return _recommend(text, menu, rng, flt, ctx)

    # «что-нибудь острое / вегетарианские блюда / неострое»
    dish = _dish_hits(text, menu)
    cat = _category_of(tokens)
    ingr = [] if (dish or cat) else _ingredient_hits(text, menu)
    asks = bool(LIST_RE.search(t)) or len(tokens) <= 3
    if dish and asks:
        items = _apply(dish, flt)
        if items:
            _remember(ctx, items, flt)
        return _list_message("Вот что есть в меню", items) if items else None
    if cat and asks:
        items = _apply([i for i in _regular(menu) if i.category == cat], flt)
        if items:
            _remember(ctx, items, flt)
        title = CATEGORY_TITLES[cat]
        return _list_message(title, items) if items else [f"В разделе «{html.escape(title)}» подходящих блюд не нашлось. Можете посмотреть остальное меню."]
    if flt == "no_meat" and asks:
        items = _apply(_regular(menu), flt)
        if items:
            _remember(ctx, items, flt)
        return _list_message("Блюда без мяса (рыба и морепродукты — да, мясо и птица — нет)", items) if items else None
    if flt and asks:
        _remember(ctx, _apply(_regular(menu), flt), flt)
        return tool_filter(flt, menu)
    if ingr and asks:
        items = _apply(ingr, flt)
        _remember(ctx, items, flt)
        return _list_message("Подходящие блюда", items)
    return None


def _is_business(items: list[Item]) -> bool:
    return bool(items) and all(i.business_lunch for i in items)
