# -*- coding: utf-8 -*-
"""
orchestrator.py — агентный цикл бота

Один ход:
  1. Реплика гостя добавляется в историю диалога (формат датасета: «User: … -> Bot:»).
  2. Модель пишет ответ: текст и/или команду ACTION (protocol.py).
  3. Код проверяет только форму: ответ не оборван, команда существует и с верными параметрами,
     модель уверена (средняя log-вероятность). Иначе — честное «не понял».
  4. Команды с результатом (инструменты брони) код выполняет и возвращает модели «Tool: RESULT: …»,
     модель продолжает. Конечные команды (меню, фильтр, цена, бизнес-ланчи, notify_manager)
     код выполняет и показывает гостю.

Инструменты:
  call_date_agent(EXPRESSION)                 -> ДД.ММ.ГГГГ               «завтра», «в пятницу» -> дата
  call_booking_time_agent(DATE,TIME,…)        -> time_valid | invalid_time
  call_free_tables_agent(DATE,TIME,GUESTS,…)  -> available | unavailable | active_booking_exists
  book_table(DATE,TIME,GUESTS,NAME)           -> booked | table_busy      база заново проверяет всё сама
  check_active_booking / cancel_booking       -> бронь гостя / cancelled
  show_menu, call_menu_filter_agent, call_price_agent, show_business_lunch — данные только из menu.json

"""
from __future__ import annotations

import asyncio
import collections
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from dates import moscow_now, resolve_date, validate_booking_date
from db import BookingDB, check_slot
from llm import GenResult, PromptTooLong
from menu import Menu, find_items, fmt_price, get_menu, tool_filter, tool_price, tool_show_business_lunch, tool_show_menu
from slots import extract as extract_slots
from protocol import DialogHistory, Reply, TOOLS_WITH_RESULT, expects_answer, parse_reply, sanitize_user_text, validate_action

log = logging.getLogger("yuqi.orchestrator")
BOOKING_TOOLS = {"call_date_agent", "call_booking_time_agent", "call_free_tables_agent", "book_table"}

FALLBACK_TEXT = ("Извините, я вас не понял. Я могу рассказать про меню, цены, бизнес-ланчи, адрес и часы работы, "
                 "а также забронировать столик. Попробуйте, пожалуйста, переформулировать.")
BAD_DATE_TEXT = "Не удалось понять дату. Напишите её, пожалуйста, например: 25 октября."    # так же в датасете
BOOKING_WORDS = re.compile(r"брон|стол|запи[сш]|посидеть|резерв")
FIELDS = (("DATE", "date", "на какую дату", "дата"), ("TIME", "time", "на какое время", "время"),
          ("GUESTS", "guests", "сколько человек будет", "гостей"), ("NAME", "name", "на чьё имя оформить бронь", "имя"))
NEED = {"call_booking_time_agent": ("DATE", "TIME"), "call_free_tables_agent": ("DATE", "TIME", "GUESTS"),
        "book_table": ("DATE", "TIME", "GUESTS", "NAME")}
CLAIM_BOOKED = re.compile(r"бронь (?:подтвержден|оформлен|создан)|столик забронирован|забронировал[аи]? (?:для вас|столик)")
CLAIM_CANCELLED = re.compile(r"бронь (?:отменен|отменён|снят)")
CLAIM_EXISTS = re.compile(r"у вас (?:уже )?есть (?:активная )?бронь")
TIME_SAID = re.compile(r"\d{1,2}:\d{2}|(?<![\w])(?:в|к|на|до)\s+\d{1,2}(?![\d:.\-/])(?!\s*(?:человек|чел|гост|персон|янв|фев|мар|апр|ма[йя]|июн|июл|авг|сен|окт|ноя|дек))"
                       r"|утра|вечера|часов|час(?:а|у)?\b|полдень|полноч")
OK_LEAD = {"готово", "отлично", "конечно", "хорошо", "спасибо", "супер", "замечательно", "прекрасно", "всё", "все", "итак",
           "принято", "понял", "записал", "увы", "к", "извините", "простите", "да", "нет", "сейчас", "секунду", "минутку"}
YES = ("да", "ага", "угу", "конечно", "подтверждаю", "верно", "давай", "давайте", "ок", "окей")
KEEP_EXCHANGES = 2          # после завершённого ответа модель видит 2 прошлых обмена (если контекст ей помогает, см. _turn)
WEEKDAYS = {"понедельник": 1, "вторник": 1, "сред": 1, "четверг": 1, "пятниц": 1, "суббот": 0, "воскресен": 0, "выходн": 0, "будн": 1}
LUNCH_SCHED = "с понедельника по пятницу с 12:00 до 16:00"


def booking_line(d: str, t: str, g: int, n: str) -> str:
    w = "человека" if g % 10 in (2, 3, 4) and g % 100 not in (12, 13, 14) else "человек"
    return f"{d}, {t}, {g} {w}, имя {n}"


@dataclass
class Message:
    text: str
    html: bool = False            # True -> отправлять с parse_mode="HTML"


@dataclass
class ManagerNotice:
    """Тихое уведомление администратору (жалоба в отзыве). Гость его не видит."""
    chat_id: int
    user_name: Optional[str]
    text: str


@dataclass
class TurnResult:
    messages: list[Message] = field(default_factory=list)
    notices: list[ManagerNotice] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)   # отладка: что генерировала модель
    source: str = ""                                  # «модель», «модель + инструмент», «запасной ответ»


@dataclass
class Session:
    history: DialogHistory = field(default_factory=DialogHistory)
    last_active: Optional[datetime] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_armed: bool = False      # бот показал гостю его бронь -> cancel_booking разрешён
    cancelled: object = None        # только что отменённая бронь (для сверки слов модели)
    prev_raws: set = field(default_factory=set)      # что модель сгенерировала в прошлый ход (ловим повторы)
    prev_user: str = ""
    prev_msgs: list = field(default_factory=list)    # что бот отправил в прошлый ход
    used_ctx: bool = False
    pending_book: Optional[dict] = None   # модель хочет book_table — ждём «да» (или имя) от гостя
    slots: dict = field(default_factory=dict)       # дата/время/гости/имя, найденные КОДОМ в словах гостя
    last_slots: dict = field(default_factory=dict)  # то же только из последней реплики
    booking_ok: bool = False        # гость действительно говорит о брони (проверено один раз за разговор)
    ctx_closed: bool = False        # прошлый вопрос закрыт: контекст — только подсказка, модель может начать с чистого листа
    user_texts: list = field(default_factory=list)   # реплики гостя в текущем разговоре (проверка имени и числа гостей)

    def reset(self) -> None:
        self.history.reset()
        self.cancel_armed = False
        self.pending_book = None
        self.slots, self.last_slots = {}, {}
        self.booking_ok = False
        self.ctx_closed = False
        self.user_texts = []

    def close_dialog(self) -> None:
        """Вопрос закрыт: в истории остаются только последние KEEP_EXCHANGES обменов (контекст для «а в выходные?»)."""
        segs, starts = self.history.done, [i for i, s in enumerate(self.history.done) if s.startswith("User:")]
        self.history.done = [] if not KEEP_EXCHANGES else segs[starts[-KEEP_EXCHANGES]:] if len(starts) >= KEEP_EXCHANGES else segs[:]
        self.ctx_closed = True
        self.slots = {}


class Orchestrator:
    def __init__(self, llm, booking_db: BookingDB, *, menu: Optional[Menu] = None, min_logprob: Optional[float] = -1.5,
                 session_ttl_min: int = 30, max_tool_steps: int = 6, now_fn: Callable[[], datetime] = moscow_now,
                 confirm_booking: bool = True, grounding: bool = True, context_choice: bool = True, **_old_flags) -> None:
        """llm — объект с async agenerate(prompt) -> GenResult (llm.LLM).
        min_logprob — порог уверенности модели; ниже -> «не понял». None отключает проверку.
        confirm_booking — перед записью брони гость подтверждает данные «да».
        grounding — проверки брони кодом (имя из слов гостя, бронь только по просьбе, сверка слов модели с базой)."""
        self.llm, self.db, self.menu = llm, booking_db, menu
        self.confirm_booking, self.grounding, self.context_choice = confirm_booking, grounding, context_choice
        self.min_logprob, self.ttl_min, self.max_tool_steps, self.now = min_logprob, session_ttl_min, max_tool_steps, now_fn
        self._sessions: dict[int, Session] = {}
        self.stats = collections.Counter()           # сколько ответов дала модель, а сколько — запасной ответ

    # ------------------------------------------------------------ сессии
    def _session(self, chat_id: int) -> Session:
        return self._sessions.setdefault(chat_id, Session())

    def reset(self, chat_id: int) -> None:
        """Сбросить диалог (команда /start)."""
        if chat_id in self._sessions:
            self._sessions[chat_id].reset()

    def wipe(self, chat_id: int) -> tuple[int, int]:
        """/reset для тестов: забыть диалог и стереть брони и отзывы этого чата. -> (броней, отзывов)."""
        self._sessions.pop(chat_id, None)
        return self.db.clear_chat(chat_id)

    def _menu(self) -> Menu:
        return self.menu or get_menu()

    # ------------------------------------------------------------ главный вход
    async def handle_text(self, chat_id: int, text: str, user_name: Optional[str] = None, on_progress=None) -> TurnResult:
        """on_progress — async-функция(Message): реплики модели перед инструментом («Проверяю время.») уходят гостю сразу."""
        sess = self._session(chat_id)
        async with sess.lock:                                            # сообщения одного чата — по очереди
            now = self.now()
            if sess.last_active and (now - sess.last_active).total_seconds() > self.ttl_min * 60:
                sess.reset()                                             # диалог «остыл»
            sess.last_active = now
            clean = sanitize_user_text(text) or "?"
            sess.user_texts = (sess.user_texts + [clean])[-8:]
            expecting = self._expecting(sess)
            sess.last_slots = extract_slots(clean, now.date(), expecting)
            if expecting != "time" and not TIME_SAID.search(clean.lower()):
                sess.last_slots.pop("time", None)                        # «нас будет 4» — это гости, а не 16:00
            if len(sess.last_slots.get("name", "")) < 3:
                sess.last_slots.pop("name", None)                        # «Не», «Да» — не имена
            sess.slots.update(sess.last_slots)
            if sess.pending_book:
                res = await self._confirm(chat_id, sess, clean, user_name, on_progress)
            else:
                res = await self._turn(chat_id, sess, clean, user_name, on_progress)
            if self.grounding:
                await self._check_facts(chat_id, sess, res)
                await self._check_claims(chat_id, res)
            texts = [m.text for m in res.messages]
            shown = [m.text for m in res.messages if m.html] or texts      # для меню сравниваем сами списки блюд
            if shown and shown == sess.prev_msgs and sess.used_ctx and clean.lower() != sess.prev_user and texts != [FALLBACK_TEXT] \
                    and not res.source.startswith("код"):                # ответ из базы повторять можно
                log.warning("chat=%s: бот повторил прошлое сообщение на другой вопрос -> «не понял»", chat_id)
                sess.reset()                                             # одно и то же дважды подряд не отправляем
                res = TurnResult([Message(FALLBACK_TEXT)], steps=res.steps, source="запасной ответ")
                shown = [FALLBACK_TEXT]
            sess.prev_msgs = shown
            sess.prev_raws, sess.prev_user = {st["raw"].strip() for st in res.steps if st.get("raw")}, clean.lower()
            self.stats[res.source] += 1
            log.info("chat=%s источник ответа: %s", chat_id, res.source)
            return res

    async def _turn(self, chat_id: int, sess: Session, user_text: str, user_name: Optional[str], on_progress) -> TurnResult:
        h, first = sess.history, None
        prompt = h.prompt_for_user(user_text)
        if self.context_choice and sess.ctx_closed and h.done:
            # Новый вопрос после закрытого: модель отвечает дважды — с прошлым контекстом и без него, — берём ответ,
            # в котором она увереннее. «А сколько стоит?» без контекста непонятен, а «Скинь меню» контекст только сбивает.
            try:
                with_ctx = await self.llm.agenerate(prompt)
                saved, h.done = h.done, []
                fresh_prompt = h.prompt_for_user(user_text)
                fresh = await self.llm.agenerate(fresh_prompt)
            except PromptTooLong:
                saved, with_ctx, fresh = [], None, None
                fresh_prompt = h.prompt_for_user(user_text)
            if with_ctx and self._score(with_ctx, sess, user_text, ctx=True) > self._score(fresh, sess, user_text):
                h.done, first = saved, with_ctx
                h.prompt_for_user(user_text)
            else:
                prompt, first = fresh_prompt, fresh
                sess.user_texts, sess.booking_ok, sess.slots = [user_text], False, dict(sess.last_slots)
        sess.ctx_closed = False
        sess.used_ctx = bool(h.done)                 # модель видела прошлые реплики (только тогда повтор — подозрителен)
        return await self._loop(chat_id, sess, user_text, user_name, on_progress, prompt, TurnResult(source="модель"), first)

    def _score(self, gen: Optional[GenResult], sess: Session, user_text: str, ctx: bool = False) -> float:
        if gen is None:
            return float("-inf")
        r = parse_reply(gen.text)
        if ctx and r.raw.strip() in sess.prev_raws and user_text.lower() != sess.prev_user:
            return float("-inf")    # с контекстом модель повторила прошлый ответ на другой вопрос («А есть харчо?» -> снова удон)
        if self._problem(gen, r) or (self.grounding and r.action in BOOKING_TOOLS and not self._booking_wanted(sess)) \
                or (r.action == "cancel_booking" and not sess.cancel_armed) \
                or (r.action == "call_price_agent" and not self._item_said(sess.user_texts if ctx else [user_text], r.params["ITEM"])):
            return float("-inf")
        return gen.mean_logprob

    async def _loop(self, chat_id: int, sess: Session, user_text: str, user_name: Optional[str], on_progress,
                    prompt: str, out: TurnResult, first: Optional[GenResult] = None) -> TurnResult:
        h = sess.history
        for _ in range(self.max_tool_steps + 1):
            try:
                gen: GenResult = first or await self.llm.agenerate(prompt)
                first = None
            except PromptTooLong:
                return self._fallback(sess, out, "слишком длинный диалог")
            reply = parse_reply(gen.text)
            out.steps.append({"prompt": prompt, "raw": reply.raw, "logprob": round(gen.mean_logprob, 3), "action": reply.action})
            problem = self._problem(gen, reply)
            if problem:
                return self._fallback(sess, out, f"chat={chat_id}: {problem} | raw={reply.raw!r}")

            # --- просто текст
            if not reply.has_action:
                text = reply.text
                if "Продолжим бронь" in text and not h.done:
                    text = text.split("Продолжим бронь")[0].strip() or text   # модель «вспомнила» бронь, которой нет
                text = self._lunch_day(user_text, text)
                if "не по теме" in text:
                    text = self._dish_on_menu(user_text) or re.sub(r"^Да, это", "Извините, это", text)             # «А у вас есть фунчоза?» — блюдо есть в меню
                if self.grounding and re.match(r"(Понял|Записал|Принял):", text) and not self._ack_ok(sess, text):
                    text = self._ask_missing(sess, [f[0] for f in FIELDS if f[1] not in sess.slots]) or text   # не верим цифрам модели
                out.messages.append(Message(text))
                h.commit(reply.raw)
                if not expects_answer(reply):
                    sess.close_dialog()
                return out

            out.source = "модель + инструмент"
            # --- инструмент с результатом: код выполняет, модель продолжает
            if self.grounding and reply.action in BOOKING_TOOLS:
                if not self._booking_wanted(sess):
                    return self._fallback(sess, out, f"chat={chat_id}: модель начала бронь без просьбы гостя: {reply.raw!r}")
                sess.booking_ok = True
                missing = self._ground(sess, reply)                      # факты брони — только из слов гостя
                if missing:
                    ask = self._ask_missing(sess, missing)
                    out.messages.append(Message(ask))
                    h.commit(ask)
                    return out
            if reply.action == "book_table" and self.confirm_booking:      # записывает бронь только после «да» гостя
                h.commit(reply.raw)
                return self._ask_confirm(sess, dict(reply.params), out)
            if reply.action in TOOLS_WITH_RESULT:
                if reply.text:                                           # «Проверяю время.» — сразу, до выполнения
                    if on_progress:
                        await on_progress(Message(reply.text))
                    else:
                        out.messages.append(Message(reply.text))
                try:
                    result, direct_text = await self._run_tool(chat_id, sess, reply)
                except Exception:
                    log.exception("chat=%s: инструмент %s упал", chat_id, reply.action)
                    return self._fallback(sess, out, "инструмент упал")
                if direct_text:                                          # код сам объясняет гостю (дата не понята и т.п.)
                    out.messages.append(Message(direct_text))
                    h.commit(direct_text)
                    if not expects_answer(Reply(raw=direct_text, text=direct_text)):
                        sess.close_dialog()
                    return out
                h.commit(reply.raw)
                out.steps[-1]["result"] = result
                prompt = h.prompt_for_tool(result)
                continue

            # --- конечное действие: меню, фильтр, цена, бизнес-ланчи, жалоба администратору
            if reply.action == "call_menu_filter_agent" and reply.params["FILTER"] == "spicy" \
                    and re.search(r"не\s*остр", f"{user_text} {reply.text}".lower()):
                reply.params["FILTER"] = "not_spicy"                     # «А неострые?» — модель написала «Неострые…», а команду перепутала
                reply.raw = reply.raw.replace("FILTER: spicy", "FILTER: not_spicy")
            if reply.action == "call_price_agent" and not self._item_said(sess.user_texts, reply.params["ITEM"]):
                return self._fallback(sess, out, f"chat={chat_id}: блюдо {reply.params['ITEM']!r} гость не называл")
            if reply.action == "call_menu_filter_agent":
                reply.params["FILTER"] = self._spicy_filter(user_text, reply)
            if reply.text:
                out.messages.append(Message(reply.text))
            self._run_terminal(chat_id, user_name, user_text, reply, out)
            h.commit(reply.raw)
            sess.close_dialog()
            return out

        return self._fallback(sess, out, f"chat={chat_id}: слишком много шагов инструментов")

    @staticmethod
    def _fallback(sess: Session, out: TurnResult, why: str) -> TurnResult:
        log.warning(why)
        sess.reset()
        return TurnResult([Message(FALLBACK_TEXT)], steps=out.steps, source="запасной ответ")

    # ------------------------------------------------------------ бронь под защитой кода
    def _ask_confirm(self, sess: Session, p: dict, out: TurnResult) -> TurnResult:
        sess.pending_book = p
        out.messages.append(Message(f"Проверьте, пожалуйста: {booking_line(p['DATE'], p['TIME'], int(p['GUESTS']), p['NAME'])}. "
                                    "Подтверждаете бронь? Напишите «да» или «нет»."))
        return out

    async def _confirm(self, chat_id: int, sess: Session, text: str, user_name, on_progress) -> TurnResult:
        """Ответ на «Подтверждаете бронь?»: «да» — код пишет бронь в базу, итог по результату пишет модель."""
        p, sess.pending_book = sess.pending_book, None
        words = text.lower().replace(",", " ").replace("!", " ").replace(".", " ").split()
        if not (words and words[0] in YES):                              # не подтвердил — бронь не пишем, разговор с начала
            sess.reset()
            if words and words[0] in ("нет", "не", "неа"):
                return TurnResult([Message("Хорошо, бронь не оформляю. Напишите, на какую дату, время и сколько гостей — "
                                           "и я проверю ещё раз.")], source="код: бронь")
            return await self._turn(chat_id, sess, text, user_name, on_progress)
        return await self._book(chat_id, sess, p, text, user_name, on_progress, TurnResult(source="модель + инструмент"))

    async def _book(self, chat_id: int, sess: Session, p: dict, text: str, user_name, on_progress, out: TurnResult) -> TurnResult:
        """Запись в базу (база сама заново проверяет время и столы); итог по результату пишет модель."""
        result, direct_text = await self._run_tool(chat_id, sess, Reply(raw="", action="book_table", params=p))
        if direct_text:
            sess.reset()
            return TurnResult([Message(direct_text)], source="код: бронь")
        out.steps.append({"action": "book_table", "result": result, "raw": "(подтверждено гостем)"})
        return await self._loop(chat_id, sess, text, user_name, on_progress, sess.history.prompt_for_tool(result), out)

    @staticmethod
    def _spicy_filter(user_text: str, reply: Reply) -> str:
        """«Какие блюда НЕ острые?» — модель путает spicy и not_spicy; отрицание в словах гостя (или в её же тексте) решает код."""
        f = reply.params["FILTER"]
        if f not in ("spicy", "not_spicy"):
            return f
        t = (user_text + " " + reply.text).lower()
        neg = re.search(r"не\s*остр|без\s+остр|не\s+сильно\s+остр|мягк", t)
        return "not_spicy" if neg else "spicy" if "остр" in user_text.lower() else f

    @staticmethod
    def _item_said(texts: list, item: str) -> bool:
        """Цену какого блюда спрашивают — гость должен был назвать это блюдо (сейчас или в прошлом вопросе)."""
        said = " ".join(texts).lower().replace("ё", "е")
        words = [w for w in re.findall(r"[а-яёa-z]{3,}", item.lower().replace("ё", "е"))]
        return not words or any(w[:4] in said for w in words)

    def _booking_wanted(self, sess: Session) -> bool:
        """Гость сам говорит о брони: слова «бронь/столик/записаться» или хотя бы две детали (дата, время, гости, имя)."""
        last = sess.user_texts[-1] if sess.user_texts else ""
        return (sess.booking_ok or bool(BOOKING_WORDS.search(" ".join(sess.user_texts).lower())) or len(sess.slots) >= 2
                or ("date" in sess.last_slots and "?" not in last and len(last.split()) <= 3))     # «25 октября»

    def _ground(self, sess: Session, reply: Reply) -> list[str]:
        """Подставляет в команду модели дату/время/гостей/имя из слов гостя (их нашёл код).
        Возвращает поля, которых гость ещё не назвал, — их бот спросит, а не придумает."""
        p, s, said = reply.params, sess.slots, " ".join(sess.user_texts).lower()
        if reply.action == "call_date_agent":
            if s.get("date"):
                p["EXPRESSION"] = s["date"]
                return []
            return [] if p["EXPRESSION"].lower()[:4] in said else ["DATE"]
        tool_dates = re.findall(r"Tool: RESULT: (\d\d\.\d\d\.\d{4})", " ".join(sess.history.done))
        model_hour = re.match(r"(\d{1,2})", p.get("TIME", "")) or None
        vals = {"DATE": s.get("date") or (p.get("DATE") if p.get("DATE") in tool_dates else None),
                "TIME": s.get("time") or (p.get("TIME") if model_hour and re.search(rf"(?<!\d){int(model_hour.group(1))}(?!\d)", said) else None),
                "GUESTS": s.get("guests") or (p.get("GUESTS") if re.search(rf"(?<!\d){p.get('GUESTS')}(?!\d)", said) else None),
                "NAME": s.get("name") or (self._name_from_user(sess, p["NAME"]) if p.get("NAME") else None)}
        if vals["NAME"] and vals["NAME"].lower()[:4] not in said:
            vals["NAME"] = None
        for k, v in vals.items():
            if v and k in p and str(v) != p[k]:
                reply.raw = reply.raw.replace(f"{k}: {p[k]}", f"{k}: {v}")
                p[k] = str(v)
        return [k for k in NEED[reply.action] if not vals[k]]

    @staticmethod
    def _ack_ok(sess: Session, text: str) -> bool:
        """«Понял: дата — …, время — …, гостей — …» модели совпадает с тем, что сказал гость (и что вернул календарь)?"""
        tool_dates = re.findall(r"Tool: RESULT: (\d\d\.\d\d\.\d{4})", " ".join(sess.history.done + [sess.history._open or ""]))
        for lab, val in re.findall(r"(дата|время|гостей|имя) — ([^,.?]+(?:\.\d\d\.\d{4})?)", text):
            key = {"дата": "date", "время": "time", "гостей": "guests", "имя": "name"}[lab]
            mine = str(sess.slots.get(key, ""))
            if val.strip() != mine and not (key == "date" and val.strip() in tool_dates):
                return False
        said = {k for k in ("date", "time", "guests", "name") if k in sess.slots}
        named = {{"дата": "date", "время": "time", "гостей": "guests", "имя": "name"}[lab]
                 for lab in re.findall(r"(дата|время|гостей|имя) — ", text)}
        return said <= named                                             # модель не «забыла» то, что гость уже сказал

    @staticmethod
    def _ask_missing(sess: Session, missing: list[str]) -> Optional[str]:
        """«Записал: дата — 21.10.2026, время — 19:00. Подскажите, пожалуйста, сколько человек будет и на чьё имя оформить бронь?»"""
        if not missing:
            return None
        known = [f"{lab} — {sess.slots[key]}" for k, key, _, lab in FIELDS if key in sess.slots]
        asks = [q for k, _, q, _ in FIELDS if k in missing]
        ask = ", ".join(asks[:-1]) + " и " + asks[-1] if len(asks) > 1 else asks[0]
        return (f"Записал: {', '.join(known)}. " if known else "") + f"Подскажите, пожалуйста, {ask}?"

    @staticmethod
    def _expecting(sess: Session) -> Optional[str]:
        """Что бот только что спросил (одно поле) — помогает понять короткий ответ «19» или «Рита»."""
        if not sess.history.done:
            return None
        q = sess.history.done[-1].split("-> Bot:")[-1].lower()
        asked = [key for key, ws in (("date", ("какую дату", "какой день")), ("time", ("какое время", "какое точное время", "во сколько")),
                                     ("guests", ("сколько человек", "сколько вас", "сколько гостей")),
                                     ("name", ("чьё имя", "чье имя", "какое имя", "как вас зовут"))) if any(w in q for w in ws)]
        return asked[0] if len(asked) == 1 else None

    @staticmethod
    def _name_from_user(sess: Session, name: str) -> str:
        """Модель написала «Марат», а гость — «Маргарита»: берём слово гостя с тем же началом."""
        said = " ".join(sess.user_texts)
        if name.lower()[:4] in said.lower():
            return name
        for w in reversed(re.findall(r"[А-ЯЁA-Z][а-яёa-z]{2,}", said)):
            if w[:2].lower() == name[:2].lower() and w.lower() not in YES:
                return w
        return name

    def _dish_on_menu(self, user_text: str) -> Optional[str]:
        """Модель сочла вопрос «не по теме», а гость назвал блюдо из меню — отвечаем по menu.json."""
        for w in re.findall(r"[а-яёa-z]{4,}", user_text.lower()):
            items = find_items(w, self._menu())
            if len(items) == 1:
                it = items[0]
                return f"Да, у нас есть «{it.name}» — {fmt_price(it.price)}" + (f": {it.description.lower()}." if it.description else ".")
        return None

    @staticmethod
    def _lunch_day(user_text: str, text: str) -> str:
        """«А во вторник есть бизнес-ланч?» — день недели проверяет код (модель путает дни)."""
        t = user_text.lower()
        if "ланч" not in t:
            return text
        asked = [k for k in WEEKDAYS if k in t]
        if not asked or re.search(r"\d", t):
            return text
        low = text.lower()
        said_no = not low.startswith("да") and bool(re.search(r"(?<![а-яё])нет(?![а-яё])", low))
        other_day = any(k in text.lower() and k not in asked for k in ("вторник", "сред", "четверг"))
        if said_no != bool(WEEKDAYS[asked[0]]) and not other_day:
            return text                                                  # модель ответила верно
        return (f"Да, бизнес-ланчи у нас {LUNCH_SCHED}." if WEEKDAYS[asked[0]] else
                f"Нет, бизнес-ланчи у нас только по будням: {LUNCH_SCHED}.")

    async def _check_facts(self, chat_id: int, sess: Session, res: TurnResult) -> None:
        """После записи брони или показа брони модель пересказывает её своими словами — сверяем дату, время,
        гостей и имя с базой; при расхождении говорим данные из базы."""
        kinds = [str(st.get("result") or "") for st in res.steps]
        if not any(k in ("booked", "cancelled") or k.startswith("active_booking") for k in kinds):
            return
        gone = "cancelled" in kinds and "booked" not in kinds
        a = sess.cancelled if gone else await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
        if not a:
            return
        for i, m in enumerate(res.messages):
            t = m.text
            if m.html or not re.search(r"\d\d\.\d\d\.\d{4}|[Ии]мя", t):
                continue
            bad = (any(d != a.date for d in re.findall(r"\d\d\.\d\d\.\d{4}", t))
                   or any(x != a.time for x in re.findall(r"(?<!\d)\d\d:\d\d", t))
                   or any(int(g) != a.guests for g in re.findall(r"(\d+) (?:человек|гост)|человек: (\d+)", t) for g in g if g)
                   or any(n != a.name for n in re.findall(r"[Ии]мя:? ([А-ЯЁA-Z][а-яёa-z\-]+)", t)))
            lead = re.match(r"([А-ЯЁ][а-яё]+), ", t)
            if lead and lead.group(1) != a.name and lead.group(1).lower() not in OK_LEAD:
                t = t[lead.end():]
                t = t[0].upper() + t[1:]                                 # «Влад, бронь подтверждена» -> имя гостя модель не выдумывает
            if bad:
                line = booking_line(a.date, a.time, a.guests, a.name)
                t = (f"Готово, бронь подтверждена: {line}. Ждём вас в YUQI!" if "booked" in kinds else
                     f"Бронь отменена: {line}." if gone else
                     f"У вас есть бронь: {line}." + (" Отменить её?" if "отмен" in m.text.lower() else ""))
                log.warning("chat=%s: модель исказила данные брони: %r", chat_id, m.text)
                res.source = "код: проверка брони"
            res.messages[i] = Message(t)

    async def _check_claims(self, chat_id: int, res: TurnResult) -> None:
        """Любые слова модели о брони сверяются с фактами: «оформлена» — только если база ответила booked,
        «отменена» — только если cancelled, «у вас есть бронь» — только если она есть в базе и с теми же датой/временем."""
        results = " ".join(str(st.get("result") or "") for st in res.steps)
        for i, m in enumerate(res.messages):
            t = m.text.lower()
            if m.html or not (CLAIM_BOOKED.search(t) or CLAIM_CANCELLED.search(t) or CLAIM_EXISTS.search(t)):
                continue
            if CLAIM_BOOKED.search(t) and "booked" in results.split():
                continue
            if CLAIM_CANCELLED.search(t) and "cancelled" in results:
                continue
            a = await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
            if CLAIM_EXISTS.search(t) and not CLAIM_BOOKED.search(t) and not CLAIM_CANCELLED.search(t) and a \
                    and a.date in m.text and a.time in m.text:
                continue                                                 # модель верно пересказала бронь из базы
            log.warning("chat=%s: слова модели о брони не совпали с базой: %r", chat_id, m.text)
            res.messages[i] = Message(f"Сверился с базой: у вас есть бронь — {booking_line(a.date, a.time, a.guests, a.name)}." if a else
                                      "Сверился с базой: активной брони у вас сейчас нет.")
            res.source = "код: проверка брони"

    # ------------------------------------------------------------ проверка формы ответа
    def _problem(self, gen: GenResult, reply: Reply) -> Optional[str]:
        if gen.stop == "max_tokens":
            return "ответ оборван (max_tokens)"
        if not reply.raw.strip():
            return "пустой ответ"
        if reply.has_action:
            errs = validate_action(reply)
            if errs:
                return "некорректное действие: " + "; ".join(errs)
        elif not reply.text:
            return "нет ни текста, ни действия"
        if self.min_logprob is not None and gen.mean_logprob < self.min_logprob:
            return f"низкая уверенность: {gen.mean_logprob:.2f}"
        return None

    # ------------------------------------------------------------ инструменты с результатом
    async def _run_tool(self, chat_id: int, sess: Session, reply: Reply) -> tuple[Optional[str], Optional[str]]:
        """-> (результат для модели, None) или (None, текст для гостя вместо продолжения цепочки)."""
        a, p, now = reply.action, reply.params, self.now()

        if a == "call_date_agent":
            d = resolve_date(p["EXPRESSION"], now.date())
            if d is None:
                return None, BAD_DATE_TEXT
            err = validate_booking_date(d, now.date())
            return (None, err) if err else (d, None)

        if a == "call_booking_time_agent":
            problem = check_slot(p["DATE"], p["TIME"], now)
            if problem is None:
                return "time_valid", None
            if problem.code in ("outside_hours", "past_time"):
                return f"invalid_time | REASON: {problem.code}", None
            return None, problem.text

        if a == "call_free_tables_agent":
            problem = check_slot(p["DATE"], p["TIME"], now)
            if problem:
                return None, problem.text
            old = await asyncio.to_thread(self.db.active_booking, chat_id, now)
            if old:                                                      # у гостя уже есть бронь
                sess.cancel_armed = True
                return old.as_exists_result(), None
            ok = await asyncio.to_thread(self.db.has_free_table, int(p["GUESTS"]), p["DATE"], p["TIME"])
            return ("available" if ok else "unavailable"), None

        if a == "book_table":                                            # база заново проверяет время, гостей, занятость
            r = await asyncio.to_thread(self.db.book_table, int(p["GUESTS"]), p["DATE"], p["TIME"], p["NAME"], chat_id, now)
            if r.status in ("booked", "table_busy"):
                return r.status, None
            if r.status == "active_booking_exists":
                sess.cancel_armed = True
                return r.active.as_exists_result(), None
            return None, r.reason

        if a == "check_active_booking":
            old = await asyncio.to_thread(self.db.active_booking, chat_id, now)
            sess.cancel_armed = old is not None
            return (old.as_result() if old else "no_active_booking"), None

        if a == "cancel_booking":
            if not sess.cancel_armed:                                    # модель не показывала бронь гостю — не отменяем
                log.warning("chat=%s: cancel_booking без показанной брони", chat_id)
                return None, "Сначала давайте проверим вашу бронь. Напишите, пожалуйста, что хотите отменить бронь."
            old = await asyncio.to_thread(self.db.cancel_active, chat_id, now)
            sess.cancel_armed, sess.cancelled = False, old
            return ("cancelled" if old else "no_active_booking"), None

        raise ValueError(a)

    # ------------------------------------------------------------ конечные действия
    def _run_terminal(self, chat_id: int, user_name: Optional[str], user_text: str, reply: Reply, out: TurnResult) -> None:
        a, p, m = reply.action, reply.params, self._menu()
        if a == "show_menu":
            out.messages += [Message(t, True) for t in tool_show_menu(p["CATEGORY"], m)]
        elif a == "show_business_lunch":
            out.messages += [Message(t, True) for t in tool_show_business_lunch(m)]
        elif a == "call_menu_filter_agent":
            out.messages += [Message(t, True) for t in tool_filter(p["FILTER"], m)]
        elif a == "call_price_agent":
            out.messages.append(Message(tool_price(p["ITEM"], m), True))
        elif a == "notify_manager":                                      # жалоба: гость уже получил ответ модели
            try:
                self.db.save_review(chat_id, user_name, user_text, True, self.now())
            except Exception:
                log.exception("не удалось сохранить отзыв")
            out.notices.append(ManagerNotice(chat_id, user_name, user_text))
