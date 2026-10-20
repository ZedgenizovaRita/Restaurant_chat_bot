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
PRICE_RE = re.compile(r"сколько (?:он |она |оно |они )?(?:стоит|стоят|будет стоить)|цена|цены|почем|стоимост|по чем")
COMPOSE_RE = re.compile(r"из чего|состав|что входит|что (?:там )?в (?:составе|нем|ней|нём)|что внутри|из каких")
LIST_RE = re.compile(r"какие|какая|какой|какое|что у вас|есть ли|есть у вас|у вас есть|покажи|расскажи|что за|что.нибудь|"
                     r"подскажи|хочу|хотел|можно|а .*есть|^есть\b|найди|предлож|что такое|подробнее")
REF_RE = re.compile(r"оттуда|отсюда|из перечн|из данн|из тех|из выбранн|из найденн|из них|из эт(?:их|ого|ой)|из списка|из перечисленн|из предложенн|из показанн|из того|среди них|из вышеперечисленн")
MENU_REQ_WORDS = {"меню", "отправь", "отправьте", "пришли", "пришлите", "покажи", "покажите", "скинь", "скиньте", "дай", "дайте", "можно",
                  "хочу", "хотел", "хотела", "посмотреть", "увидеть", "ваше", "ваш", "все", "всё", "полное", "мне", "пожалуйста", "а",
                  "у", "вас", "есть", "какое", "ещё", "еще", "основное", "целиком", "полностью", "и", "пока"}
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
    """Блюда, у которых в названии/псевдонимах есть слова из вопроса (удоны -> Удон с курицей; «Том Ям» — два коротких слова)."""
    cat_stems = tuple(s for ss in CATEGORY_STEMS.values() for s in ss)
    all_q = [w for w in _tokens(text) if w not in STOPWORDS and not FILTER_TOKEN_RE.match(w) and not w.startswith(cat_stems)]
    q = [w for w in all_q if len(w) >= 4]                               # «лапша», «роллы» — это раздел, а не блюдо
    hits = []
    for it in _regular(menu):
        for n in (it.name, *it.aliases):
            name_tokens = [w for w in _tokens(n) if len(w) >= 2]
            long_hit = any(_same(a, b) for a in q for b in name_tokens if len(b) >= 3)
            short_hits = sum(any(_same(a, b) for a in all_q) for b in name_tokens)   # «том» + «ям» — оба слова названия
            if long_hit or (len(name_tokens) >= 2 and short_hits >= 2):
                score = sum(any(_same(a, b) for b in name_tokens) for a in all_q)
                hits.append((score, it))
                break
    best = max((sc for sc, _ in hits), default=0)
    return [it for sc, it in hits if sc == best]                        # «Корейский пибимпап» — только он, а не всё «корейское»


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
        return _pick_message(pool, rng, 3 if re.search(r"двоих|компани|вчетвером|втроем|втроём", t) else 2, ctx, flt)
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
    if not pool and flt == "vegan":
        veg = _apply([i for i in _regular(menu) if i.category not in ("drinks", "desserts")], "vegetarian")
        if veg:
            msgs = _pick_message(veg, rng, 3, ctx, "vegetarian")
            return ["Строго веганских блюд с такой пометкой в нашем меню нет, но есть вегетарианские — без мяса и рыбы. Могу предложить:\n" + msgs[0].split("\n", 1)[-1]] + msgs[1:]
    if not pool:
        title = {"no_meat": "блюд без мяса", "vegan": "веганских", "vegetarian": "вегетарианских", "spicy": "острых", "not_spicy": "неострых"}.get(flt or "", "подходящих")
        return [f"К сожалению, {title} блюд в меню не нашлось. Можете посмотреть остальные варианты в меню."]
    return _pick_message(pool, rng, 3 if re.search(r"двоих|компани|вчетвером|втроем|втроём|нас \d|на \d", t) else 2, ctx, flt)


def _pick_message(pool: list[Item], rng: random.Random, n: int, ctx: Optional[dict] = None, flt: Optional[str] = None) -> list[str]:
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
    if ctx is not None:
        crit = {"no_meat": "из блюд без мяса", "vegan": "из веганских блюд", "vegetarian": "из вегетарианских блюд",
                "spicy": "из острых блюд", "not_spicy": "из неострых блюд"}.get(flt or "", "из меню")
        ctx["names"], ctx["flt"], ctx["fresh"] = [i.name for i in picks], flt, True
        ctx["rec"], ctx["rec_fresh"] = {"pool": [i.name for i in pool], "shown": [i.name for i in picks], "flt": flt}, True
        ctx["why"] = _why_for_items(picks, f"Конечно! Я выбирал {crit}, старался предложить разные позиции, а фирменные советую чаще. Вот что отмечено в меню:")
    lines = [_item_line(i) for i in picks]
    intro = rng.choice(["С удовольствием подскажу! Попробуйте:", "Могу предложить вот что:", "Из нашего меню советую:", "Давайте я предложу пару вариантов:"])
    outro = rng.choice(["Если хотите, подберу что-нибудь ещё — например, острое или без мяса.", "Остальные варианты можно посмотреть в меню — скажите, если нужен другой вариант.", "Если не то, что хотелось, скажите — подскажу другое."])
    return split_message(intro + "\n" + "\n".join(lines) + "\n\n" + outro)


WHY_WORDS = {"почему", "отчего", "зачем"}
MORE_RE = re.compile(r"давай друго|друго[ейе]|друг\w+ вариант|ещё|еще|не то|не нравится|не подходит|не хочу это|предложи друго|что.нибудь друго|без этого|заменит")
ABOUT_RE = re.compile(r"расскаж\w* (?:мне )?(?:про|о|об) (?:ваш\w* |этот )?(?:ресторан|заведени|вас|месте)|что (?:у вас )?за (?:ресторан|заведени|место)|что такое yuqi|кто вы|о вас|о ресторане|про ресторан|концепци")
ABOUT_TEXTS = [
    "YUQI — паназиатский ресторан: у нас готовят блюда японской, корейской, китайской, тайской и вьетнамской кухни. Мы находимся в Москве, на Большой Никитской, 24/1, и каждый день открыты с 10:00 до 23:00. По будням с 12:00 до 16:00 у нас есть бизнес-ланчи. Могу подсказать по меню, подобрать блюдо или помочь забронировать столик.",
    "Рады рассказать! YUQI — это паназиатская кухня: японские, корейские, китайские, тайские и вьетнамские блюда. Ждём вас по адресу Москва, ул. Большая Никитская, 24/1, каждый день с 10:00 до 23:00, а в будни с 12:00 до 16:00 подаём бизнес-ланчи. Если хотите, посоветую, что попробовать, или помогу забронировать столик.",
]
DEMO_RE = re.compile(r"(?<!\w)(?:это|этот|эта|эти|этого)(?!\w)")
NOT_ON_MENU = {  # блюда, которых у нас нет (азиатская кухня): основа слова -> родительный падеж, раздел для подсказки
    "борщ": ("борща", "soups"), "щи": ("щей", "soups"), "солянк": ("солянки", "soups"), "уха": ("ухи", "soups"), "ухи": ("ухи", "soups"),
    "харчо": ("харчо", "soups"), "окрошк": ("окрошки", "soups"), "пицц": ("пиццы", None), "пастa": ("пасты", None), "паста": ("пасты", None),
    "бургер": ("бургеров", None), "стейк": ("стейка", None), "шашлык": ("шашлыка", None), "пельмен": ("пельменей", None),
    "картошк": ("картошки", None), "картофел": ("картофеля", None), "цезар": ("салата «Цезарь»", "salads"), "оливье": ("оливье", "salads"),
    "плов": ("плова", None), "котлет": ("котлет", None), "блин": ("блинов", "desserts"), "шаурм": ("шаурмы", None), "хот-дог": ("хот-догов", None),
}
PRONOUN_RE = re.compile(r"(?<!\w)(?:он|она|оно|они|его|её|ее|нём|нем|ней|них)(?!\w)")
OPINION_RE = re.compile(r"вкусн|нравит|понравит|стоит (?:ли )?(?:попробовать|заказать|брать|взять)|хорош|советуеш|рекомендуеш")
BEST_RE = re.compile(r"самое вкусн|самый вкусн|самая вкусн|что вкуснее|вкуснее всего|самое популярн|что лучше всего|чем славит|ваш хит|ваши хиты|лучшее у вас|лучшее в меню")
SPICY_Q_RE = re.compile(r"остр|пряно|пикант|жгуч")
VEG_Q_RE = re.compile(r"веган|вегет|постн")


def _traits(it: Item) -> list[str]:
    out = []
    if it.signature:
        out.append("фирменное блюдо ресторана")
    if it.spicy is True:
        out.append("острое")
    elif it.spicy is False:
        out.append("неострое")
    if it.vegan is True:
        out.append("веганское")
    elif it.vegetarian is True:
        out.append("вегетарианское")
    return out


def _answer_dish(items: list[Item], t: str, with_text: bool, rng=random) -> str:
    """Живой ответ про блюдо: описание, прямой ответ на «острый ли?», «вегетарианское ли?», цена и порция."""
    e = html.escape
    parts = []
    for it in items[:3]:
        sents = []
        if with_text and it.description:
            d = it.description.rstrip(".")
            if len(d) > 1 and d[0].isupper() and d[1].islower():
                d = d[0].lower() + d[1:]
            sents.append(f"<b>{e(it.name)}</b> — {e(d)}.")
        else:
            sents.append(rng.choice([f"Про блюдо «<b>{e(it.name)}</b>» могу сказать вот что.", f"Если про блюдо «<b>{e(it.name)}</b>», то вот что могу сказать.", f"Отвечаю про «<b>{e(it.name)}</b>»."]))
        if SPICY_Q_RE.search(t):
            if it.spicy is True:
                sents.append(rng.choice(["Да, блюдо острое — если не любите остренькое, лучше взять что-нибудь помягче.",
                                         "Острота там есть, так что учитывайте это при заказе.",
                                         "Да, оно острое, но многим как раз это и нравится."]))
            elif it.spicy is False:
                sents.append(rng.choice(["Нет, острого там нет — блюдо мягкое.",
                                         "Оно неострое, так что подойдёт и тем, кто не любит остренькое.",
                                         "Нет, не переживайте, остроты в нём нет."]))
            else:
                sents.append("В меню для этого блюда нет отметки об остроте — лучше уточнить у официанта.")
        if VEG_Q_RE.search(t):
            if it.vegan is True:
                sents.append("И оно веганское, так что подойдёт даже без молочного и яиц.")
            elif it.vegetarian is True:
                sents.append("Блюдо вегетарианское, хотя не полностью веганское.")
            else:
                sents.append("Вегетарианским это блюдо не назвать — в нём есть мясо или рыба.")
        if it.price is not None:
            sents.append(rng.choice(["Стоит", "По цене —", "Цена —"]) + f" {fmt_price(it.price)}" + (f", порция — {e(it.weight)}." if it.weight else "."))
        parts.append(" ".join(sents))
    tail = rng.choice(["", "", "Хотите, расскажу про что-нибудь ещё?", "Если нужно, подскажу и другие блюда."])
    return "\n\n".join(parts) + (f"\n\n{tail}" if tail else "")


def _why_for_items(items: list[Item], intro: str) -> str:
    e = html.escape
    lines = []
    for i in items[:4]:
        tr = ", ".join(_traits(i))
        line = f"• <b>{e(i.name)}</b> — " + (tr if tr else "позиция из меню")
        if len(items) <= 2 and i.description:                           # по одному-двум блюдам можно привести и описание
            line += f" (в описании: «{e(i.description)}»)"
        lines.append(line)
    return intro + "\n" + "\n".join(lines) + "\n\nЕсли хочется другой вариант — просто скажите, подберу."


def _price_reply(q: str, menu: Menu, rng) -> str:
    found = find_items(q, menu)
    if len(found) == 1 and found[0].price is not None:
        n, p = html.escape(found[0].name), fmt_price(found[0].price)
        return rng.choice([f"{n} стоит {p}.", f"Сейчас {n} стоит {p}.", f"{n} — {p}."]) + rng.choice(["", "", " Что-нибудь ещё подсказать?"])
    return tool_price(q, menu)


def _opinion(it: Item, rng) -> str:
    """Честный ответ на «он вкусный?»: за гостей не решаем, но фирменное отмечаем и напоминаем состав."""
    e = html.escape
    d = it.description.rstrip(".") if it.description else ""
    if d and len(d) > 1 and d[0].isupper() and d[1].islower():
        d = d[0].lower() + d[1:]
    if it.signature:
        return rng.choice([f"«{e(it.name)}» — одно из наших фирменных блюд, так что вполне можно начать с него.",
                           f"Это наше фирменное блюдо — «{e(it.name)}», его у нас советуют попробовать в первую очередь."]) + (f" По описанию в меню — {e(d)}." if d else "")
    return rng.choice([f"Вкусы у всех разные, но по составу «{e(it.name)}» — {e(d)}. Если любите такое сочетание, думаю, вам понравится.",
                       f"Скажу честно: за каждого гостя решать не берусь, но «{e(it.name)}» — {e(d)}. Если такое вам по душе, смело пробуйте."]) if d else \
        f"Вкусы у всех разные, но «{e(it.name)}» стоит попробовать, если он вам по составу подходит."


def _opinion_many(items: list[Item], rng) -> str:
    if len(items) == 1:
        return _opinion(items[0], rng)
    e = html.escape
    lines = []
    for i in items:
        d = (i.description or "").rstrip(".")
        if d and len(d) > 1 and d[0].isupper() and d[1].islower():
            d = d[0].lower() + d[1:]
        lines.append(f"• <b>{e(i.name)}</b> — " + ("наше фирменное блюдо" if i.signature else (d or "позиция из меню")))
    lead = rng.choice(["Вкусы у всех разные, но вот что могу сказать:", "За каждого гостя решать не берусь, зато скажу, что это за блюда:"])
    return lead + "\n" + "\n".join(lines) + "\n\nЕсли любите такие вкусы — смело пробуйте."


def _more(menu: Menu, rng, ctx: dict) -> list[str]:
    """«Давай другое» — ещё пара блюд из того же набора, без уже названных."""
    rec = ctx["rec"]
    if set(rec["pool"]) & set(ctx.get("lunch_names") or ()):
        ctx["topic"] = "lunch"                                           # речь всё ещё о бизнес-ланчах («а сколько стоит?»)
    pool = [i for i in menu.items if i.name in rec["pool"] and i.name not in rec["shown"]]
    if not pool:
        ctx["rec"] = {"pool": rec["pool"], "shown": [], "flt": rec["flt"]}
        ctx["rec_fresh"] = True
        return ["Это всё, что у меня нашлось под ваш запрос — другие варианты закончились. Хотите, подберу по-другому: острое, без мяса, что-нибудь сладкое или из определённого раздела?"]
    msgs = _pick_message(pool, rng, 2, ctx, rec["flt"])
    ctx["rec"] = {"pool": rec["pool"], "shown": rec["shown"] + ctx["names"], "flt": rec["flt"]}
    ctx["rec_fresh"] = True
    lead = rng.choice(["Хорошо, вот другие варианты:", "Конечно, тогда попробуйте вот это:", "Давайте посмотрим, что ещё есть:"])
    msgs[0] = lead + "\n" + (msgs[0].split("\n", 1)[1] if "\n" in msgs[0] else msgs[0])
    return msgs


def _lunch_recommend(menu: Menu, rng, ctx: dict) -> list[str]:
    lunch = [i for i in menu.items if i.business_lunch]
    if not lunch:
        return ["Меню бизнес-ланчей пока пустое — загляните позже или посмотрите основное меню."]
    msgs = _pick_message(lunch, rng, 2, ctx, None)
    ctx["topic"] = "lunch"
    ctx["lunch_names"] = [i.name for i in lunch]
    msgs[0] = rng.choice(["Из бизнес-ланчей могу посоветовать:", "Если выбирать из бизнес-ланчей, я бы предложил:"]) + "\n" + (msgs[0].split("\n", 1)[1] if "\n" in msgs[0] else msgs[0])
    return msgs


def is_followup(text: str, ctx: dict) -> bool:
    """Реплика опирается на то, что код только что показал (список, блюдо, рекомендацию) — её сначала разбирает код."""
    t = norm(text)
    words = [w for w in t.split() if w not in ("а", "и", "но", "ну")]
    if not words or not (ctx.get("fresh") or ctx.get("rec_fresh") or ctx.get("why") or ctx.get("topic")):
        return False
    if REF_RE.search(t) or words[0] in WHY_WORDS:
        return True
    if ctx.get("rec_fresh") and (MORE_RE.search(t) and len(words) <= 5 or detect_filter(text) and len(words) <= 7):
        return True
    if (PRONOUN_RE.search(t) or DEMO_RE.search(t)) and len(words) <= 7:
        return True
    if PRICE_RE.search(t) and len(words) <= 4:
        return True
    return len(words) <= 2 and bool(OPINION_RE.search(t))


def expire(ctx: dict) -> None:
    """Ход прошёл без кода меню — «свежесть» прошлого показа больше не действует."""
    for k in ("fresh", "rec_fresh", "why", "topic"):
        ctx.pop(k, None)


def note_shown(ctx: dict, menu: Menu, *, category: Optional[str] = None, flt: Optional[str] = None,
               item: Optional[str] = None, lunch: bool = False, text: str = "") -> None:
    """Модель вызвала инструмент меню (или назвала блюда в ответе) — запоминаем, что гость увидел, для «оттуда», «а почему?», «он острый?»."""
    expire(ctx)
    if lunch:
        ctx["topic"], ctx["lunch_names"] = "lunch", [i.name for i in menu.items if i.business_lunch]
        return
    items: list[Item] = []
    if category and category not in ("all", "signature"):
        items = [i for i in _regular(menu) if i.category == category]
    elif category == "signature":
        items = [i for i in _regular(menu) if i.signature]
    elif flt:
        items = _apply(_regular(menu), flt)
    elif item:
        items = find_items(item, menu)[:1]
    elif text:                                                          # «Рекомендую «Филадельфия» и «Том Ям…»»
        named = {norm(x) for x in re.findall(r"«([^»]+)»", text)}
        items = [i for i in _regular(menu) if norm(i.name) in named or any(norm(a) in named for a in i.aliases)]
        if not items:
            items = [i for i in _regular(menu) if norm(i.name) in norm(text)]
        if len(items) >= 2:                                              # это рекомендация — «давай другое» возьмёт другие блюда
            ctx["rec"] = {"pool": [i.name for i in _regular(menu) if i.category not in ("drinks", "desserts")],
                          "shown": [i.name for i in items], "flt": None}
            ctx["rec_fresh"] = True
    if items:
        _remember(ctx, items, flt)


def route(text: str, menu: Menu, rng: Optional[random.Random] = None, ctx: Optional[dict] = None) -> Optional[list[str]]:
    """ctx — память чата о последнем показанном списке {"names": [...], "flt": ...}; route обновляет её сам."""
    out = _route(text, menu, rng or random, ctx if ctx is not None else {})
    return out


FLT_WHY = {"no_meat": "без мяса", "vegan": "веганские", "vegetarian": "вегетарианские", "spicy": "острые", "not_spicy": "неострые"}


def _remember(ctx: dict, items: list[Item], flt: Optional[str]) -> None:
    ctx["names"], ctx["flt"], ctx["fresh"] = [i.name for i in items], flt, True
    intro = (f"Конечно, объясню! Я отобрал блюда по отметкам в меню — {FLT_WHY[flt]}. Вот, например:" if flt in FLT_WHY
             else "Конечно, объясню! Я показал то, что нашлось в меню по вашему вопросу. Вот, например:")
    ctx["why"] = _why_for_items(items[:4], intro)


def _route(text: str, menu: Menu, rng, ctx: dict) -> Optional[list[str]]:
    t = norm(text)
    prev_topic, ctx["topic"] = ctx.get("topic"), None                    # тема прошлого вопроса («а сколько стоит?»)
    prev_why = ctx.pop("why", None)                                      # чем объяснить прошлый ответ («а почему?»)
    prev_fresh = ctx.pop("fresh", False)                                 # прошлый ответ был про блюдо(а) из меню
    prev_rec = ctx.pop("rec_fresh", False)                               # прошлый ответ был рекомендацией («давай другое»)
    if not t:
        return None
    words = [w for w in t.split() if w not in ("а", "и", "но", "ну")]
    if words and words[0] in WHY_WORDS and len(words) <= 4:
        if prev_why:
            ctx["why"], ctx["fresh"] = prev_why, prev_fresh               # можно спросить «почему» ещё раз
            return [prev_why]
        return ["Простите, не уловил, о чём вы спрашиваете. Уточните, пожалуйста, — подскажу по блюдам меню, ценам, бизнес-ланчам, адресу и часам работы, а ещё помогу забронировать столик."]
    if prev_rec and ctx.get("rec") and MORE_RE.search(t) and len(t.split()) <= 5 and not SKIP_RE.search(t):
        return _more(menu, rng, ctx)
    if prev_rec and detect_filter(text) and len(t.split()) <= 7 and not SKIP_RE.search(t) and not LIST_RE.search(t.replace("есть", "")):
        return _recommend(text, menu, rng, detect_filter(text), ctx)          # «а среди блюд без мяса?», «а что-то неострое?»
    if re.fullmatch(r"(?:ну )?(?:давай|а|и)?\s*(?:друго[ейе]|не то|предложи друго[ейе]|друг\w+ вариант\w*)", t):
        return ["Что именно предложить вместо этого? Могу подобрать острое, без мяса, что-нибудь сладкое или блюдо из любого раздела — скажите, что по вкусу."]
    names = ctx.get("names") or []
    if (prev_fresh and names and OPINION_RE.search(t) and not REC_RE.search(t) and not BEST_RE.search(t)
            and (PRONOUN_RE.search(t) or DEMO_RE.search(t) or len(t.split()) <= 2)
            and not _dish_hits(text, menu) and not _category_of(_tokens(text))):
        ctx["fresh"] = True                                              # «а это вкусно?», «они вкусные?» — про только что показанное
        items = [i for i in _regular(menu) if i.name in names][:3]
        if items:
            return [_opinion_many(items, rng)]
    if prev_fresh and len(names) == 1 and PRONOUN_RE.search(t) and not _dish_hits(text, menu):
        ctx["fresh"] = True                                              # «он острый?», «сколько он стоит?» — это про последнее блюдо
        it = next((i for i in _regular(menu) if i.name == names[0]), None)
        if it and OPINION_RE.search(t) and not REC_RE.search(t):
            return [_opinion(it, rng)]
        text = f"{text} {names[0]}"
        t = norm(text)
    elif PRONOUN_RE.search(t) and OPINION_RE.search(t) and not REC_RE.search(t) and len(t.split()) <= 5:
        return ["Про какое блюдо вы спрашиваете? Назовите его, пожалуйста — расскажу, что в нём, и подскажу, подойдёт ли оно вам."]
    if prev_topic == "lunch" and (REF_RE.search(t) or len(t.split()) <= 4) and (REC_RE.search(t) or BEST_RE.search(t)) and not _category_of(_tokens(text)):
        return _lunch_recommend(menu, rng, ctx)                          # «что отсюда посоветуешь?» сразу после меню ланчей
    if ABOUT_RE.search(t) and not SKIP_RE.search(t):                     # «расскажи про ресторан»
        return [rng.choice(ABOUT_TEXTS)]
    if re.search(r"бизнес|ланч|lunch", t) and not re.search(r"брон|столик", t):
        ctx["topic"] = "lunch"
        ctx["lunch_names"] = [i.name for i in menu.items if i.business_lunch]
        if REC_RE.search(t) or BEST_RE.search(t):                        # «что посоветуешь из бизнес-ланчей»
            return _lunch_recommend(menu, rng, ctx)
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
                return [_price_reply(names[0], menu, rng)]
            return ["Уточните, пожалуйста, цену какого блюда вас интересует? Например: «сколько стоит Том Ям» или «сколько стоит удон»."]
    if SKIP_RE.search(t):
        return None
    for stem, (gen, cat_hint) in NOT_ON_MENU.items():                    # «у вас есть борщ?» — честно: такого нет, вот ближайшее
        if any(w.startswith(stem) for w in t.split()) and not _dish_hits(text, menu):
            lead = rng.choice([f"К сожалению, {gen} у нас нет — у нас паназиатская кухня.", f"Увы, {gen} в нашем меню нет: мы готовим блюда паназиатской кухни."])
            if cat_hint:
                items = [i for i in _regular(menu) if i.category == cat_hint]
                if items:
                    _remember(ctx, items, None)
                    return split_message(f"{lead} Зато из раздела «{CATEGORY_TITLES[cat_hint]}» могу предложить:\n" + "\n".join(_item_line(i) for i in items))
            return [lead + " Могу подсказать, что выбрать из нашего меню — скажите, что вам по вкусу: острое, без мяса, лёгкое или посытнее."]
    if "меню" in _tokens(text) and set(_tokens(text)) <= MENU_REQ_WORDS:          # «отправь меню» — просто всё меню
        return tool_show_menu("all", menu)
    flt = detect_filter(text)
    tokens = _tokens(text)

    # цена
    if PRICE_RE.search(t):
        q = re.sub(PRICE_RE, " ", t)
        if find_items(q, menu) and not _is_business(find_items(q, menu)):
            return [_price_reply(q, menu, rng)]
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
        _remember(ctx, found[:3], None)
        ctx["why"] = _why_for_items(found[:3], "Я опираюсь на карточку блюда в нашем меню:")
        return [_answer_dish(found, t, with_text=True, rng=rng)]

    # «моти сладкий?», «рамен сытный?» — по разделу меню и описанию, честно
    if re.search(r"сладк|сытн|наед|порци|большой|большая|сладост", t):
        found = _dish_hits(text, menu)
        if found and len(found) <= 2:
            _remember(ctx, found, None)
            e = html.escape
            outs = []
            for it in found:
                d = it.description.rstrip(".") if it.description else ""
                d = d[0].lower() + d[1:] if len(d) > 1 and d[0].isupper() and d[1].islower() else d
                if re.search(r"сладк|сладост", t):
                    ans = ("Да, это десерт, так что сладкий." if it.category == "desserts" else
                           "Скорее нет — это не десерт." if it.category not in ("drinks",) else "Отдельной отметки о сладости в меню нет.")
                else:
                    ans = ("Да, это сытное блюдо." if it.category in ("soups", "noodles", "main_courses") else
                           "Скорее лёгкий вариант — для сытного обеда можно добавить суп или лапшу.")
                outs.append(f"<b>{e(it.name)}</b> — {e(d)}. {ans}" + (f" Порция — {e(it.weight)}." if it.weight else ""))
            ctx["why"] = _why_for_items(found, "Я сужу по разделу меню и описанию блюда:")
            return ["\n\n".join(outs)]

    # вопрос про свойство конкретного блюда: «Том Ям острый?», «Филадельфия вегетарианская?»
    if SPICY_Q_RE.search(t) or VEG_Q_RE.search(t):
        found = _dish_hits(text, menu)
        if found and len(found) <= 2 and not re.search(r"какие|какой|какая|посовет|что.нибудь|блюда без", t):
            _remember(ctx, found, None)
            ctx["why"] = _why_for_items(found, "Я опираюсь на карточку блюда в нашем меню:")
            return [_answer_dish(found, t, with_text=False, rng=rng)]

    # «что самое вкусное» — начинаем с фирменных блюд
    if BEST_RE.search(t):
        bcat = _category_of(tokens)
        base = [i for i in _regular(menu) if (i.category == bcat if bcat else i.category not in ("drinks", "desserts"))]
        sig = [i for i in base if i.signature]
        msgs = _pick_message(sig or base or [i for i in _regular(menu)], rng, 3 if not bcat else 2, ctx, None)
        msgs[0] = msgs[0].split("\n", 1)[1] if "\n" in msgs[0] else msgs[0]
        return [rng.choice(["Вкусы у всех разные, но я бы начал с этого:" if sig and bcat else "Вкусы у всех разные, но я бы начал с наших фирменных блюд:",
                        "Тут всё дело вкуса, а я бы посоветовал начать вот с чего:" if bcat else "Тут всё дело вкуса, а из нашего меню я бы посоветовал начать с фирменных:"]) + "\n" + msgs[0]] + msgs[1:]

    # рекомендации
    if REC_RE.search(t):
        return _recommend(text, menu, rng, flt, ctx)

    # «что-нибудь острое / вегетарианские блюда / неострое»
    dish = _dish_hits(text, menu)
    cat = _category_of(tokens)
    ingr = [] if (dish or cat) else _ingredient_hits(text, menu)
    asks = bool(LIST_RE.search(t)) or len(tokens) <= 3 or bool(OPINION_RE.search(t))
    if dish and OPINION_RE.search(t) and len(dish) <= 3 and not flt:      # «а токпокки вкусные?» — про это блюдо, а не список
        _remember(ctx, dish, None)
        return [_opinion_many(dish, rng)]
    if dish and asks:
        items = _apply(dish, flt)
        if items:
            _remember(ctx, items, flt)
        return _list_message(rng.choice(["Вот что у нас есть", "Смотрите, что нашлось", "Вот подходящие варианты"]), items) if items else None
    if cat and asks:
        items = _apply([i for i in _regular(menu) if i.category == cat], flt)
        if items:
            _remember(ctx, items, flt)
        title = CATEGORY_TITLES[cat]
        if items and OPINION_RE.search(t):                               # «а лимонады у вас вкусные?» — честно: вкус личный, вот что есть
            lead = rng.choice(["Вкусы у всех разные, но выбор такой — смотрите, что из этого вам по душе:", "За каждого гостя решать не берусь, зато могу показать, что у нас есть:"])
            return split_message(f"{lead}\n" + "\n".join(_item_line(i) for i in items))
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
        return _list_message(rng.choice(["Вот подходящие блюда", "Смотрите, что нашлось"]), items)
    return _followup(text, t, menu, ctx)


EXPLAIN_RE = re.compile(r"что такое|что это|это (?:тогда |же |вообще )?что|что значит|что за|подробнее|поясни|объясни|а .{0,20}это")


def _followup(text: str, t: str, menu: Menu, ctx: dict) -> Optional[list[str]]:
    """«А мини версия это тогда что?» — ищем слова вопроса в описаниях того, что гостю только что показали (в т.ч. бизнес-ланчи)."""
    if not EXPLAIN_RE.search(t):
        return None
    names = set(ctx.get("names") or ()) | set(ctx.get("lunch_names") or ())
    q = [w for w in _tokens(text) if len(w) >= 4 and w not in STOPWORDS and w not in ("тогда", "такое", "значит", "подробнее", "поясни", "объясни", "расскажи")]
    hits = [i for i in menu.items if i.name in names and any(_same(a, b) for a in q for b in _tokens(i.description) if len(b) >= 4)]
    if not hits:
        return None
    e = html.escape
    return ["\n".join(f"<b>{e(i.name)}</b> — {fmt_price(i.price)}: {e(i.description)}" if i.price is not None
                      else f"<b>{e(i.name)}</b>: {e(i.description)}" for i in hits[:3])]


def _is_business(items: list[Item]) -> bool:
    return bool(items) and all(i.business_lunch for i in items)
