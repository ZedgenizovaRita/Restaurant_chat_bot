# -*- coding: utf-8 -*-
"""
orchestrator.py - Цикл «модель -> действие -> инструмент -> модель».

Как идёт один ход:
  1. реплика пользователя очищается (sanitize_user_text) и добавляется в историю;
  2. модель генерирует ответ; он проверяется: не оборван, действие корректно (validate_action),
     модель уверена (mean_logprob >= min_logprob). Иначе — запасной ответ и сброс диалога:
     лучше честно «не понял», чем выполнить выдуманное;
  3. без действия — обычный текст. Если бот в нём что-то спросил, диалог остаётся открытым;
  4. «инструменты с результатом» (TOOLS_WITH_RESULT) код выполняет сам и отдаёт модели
     "Tool: RESULT: ..." — модель продолжает. Служебные реплики модели между шагами
     («Проверяю время.») пользователю не показываются, он видит только итог;
  5. остальные действия конечные: меню, бизнес-ланч, фильтры, цена, notify_manager.

Цепочка брони (каждый шаг — результат кода, а не выдумка модели):
  call_date_agent -> дата ДД.ММ.ГГГГ
  call_booking_time_agent -> time_valid | invalid_time | REASON: outside_hours / past_time
  call_free_tables_agent -> available | unavailable | active_booking_exists | ...
  book_table -> booked | table_busy
Отмена: check_active_booking -> active_booking | ... / no_active_booking; cancel_booking -> cancelled.
Перед book_table и cancel_booking код проверяет всё заново — выдуманной моделью брони не будет.
"""
from __future__ import annotations

import asyncio
import logging
import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

from dates import moscow_now, resolve_date, validate_booking_date
from db import BookingDB, check_slot
from llm import GenResult, PromptTooLong
from menu import Menu, get_menu, tool_filter, tool_price, tool_show_business_lunch, tool_show_menu
import random

import menu_router
import slots as slotlib
from db import MAX_GUESTS_PER_BOOKING
from intents import (FILTER_WORDS_RE, has_booking_words, is_abort, is_booking_request, is_cancel_intent, is_no, is_yes,
                     looks_like_business_lunch, looks_like_menu, norm, ungrounded)
from protocol import (ARROW, DialogHistory, Reply, TOOLS_WITH_RESULT, expects_answer, parse_reply,
                      sanitize_user_text, validate_action)

log = logging.getLogger("yuqi.orchestrator")

RECENT_EXCHANGES = 4       
FALLBACK_TEXT = ("Извините, я вас не понял. Я могу рассказать про меню, цены, бизнес-ланчи, адрес и часы работы, "
                 "а также забронировать столик. Попробуйте, пожалуйста, переформулировать.")
RESET_TEXT = "Давайте начнём сначала. Чем могу помочь?"
EMPTY_TEXT = "Напишите, пожалуйста, ваш вопрос текстом."
BAD_DATE_TEXT = "Не удалось понять дату. Напишите её, пожалуйста, например: 25 октября."
BUSY_TEXT = "Бронь сейчас оформить не получилось. Напишите, пожалуйста, ещё раз."
TOOL_FAIL_TEXT = "Что-то пошло не так. Напишите, пожалуйста, ещё раз."
ASK = {"date": "на какую дату", "time": "на какое точное время", "guests": "сколько человек будет", "name": "на чьё имя оформить бронь"}
NO_ACTIVE_TEXT = "У вас нет активных броней, отменять нечего. Если хотите, помогу забронировать столик."
KEEP_TEXT = "Хорошо, бронь остаётся без изменений."
DECLINE_BOOK_TEXT = ("Хорошо, ничего не бронирую. Если хотите изменить данные, напишите дату, время, "
                     "количество человек и имя заново.")
# модель просит данные брони (а не просто упоминает, что умеет бронировать)
BOOKING_ASK_RE = re.compile(r"на какую дату|на какое (точное )?время|сколько (человек|вас)|на чьё имя|имя для брони|дату, (точное )?время")
SCHED = "с понедельника по пятницу с 12:00 до 16:00"
BOOKING_ACTIONS = {"call_date_agent", "call_booking_time_agent", "call_free_tables_agent", "book_table",
                   "check_active_booking", "cancel_booking"}
BOOKING_PARAM_ACTIONS = {"call_booking_time_agent", "call_free_tables_agent", "book_table"}
EARLY_RE = re.compile(r"дата — (\d{2}\.\d{2}\.\d{4}), время — (\d{2}:\d{2})")


def people_word(n: int) -> str:
    return "человека" if (n % 10 in (1, 2, 3, 4) and n % 100 not in (11, 12, 13, 14)) else "человек"


def booking_line(d: str, t: str, g: int, n: str) -> str:
    return f"{d}, {t}, {g} {people_word(g)}, имя {n}"


# ---------------------------------------------------------------- что возвращаем Telegram-слою
@dataclass
class Message:
    text: str
    html: bool = False            # True -> отправлять с parse_mode="HTML"


@dataclass
class ManagerNotice:
    """Тихое уведомление администратору (жалоба в отзыве). Гость его не видит."""
    chat_id: int
    user_name: Optional[str]
    text: str                     # текст отзыва


@dataclass
class TurnResult:
    messages: list[Message] = field(default_factory=list)
    notices: list[ManagerNotice] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)   # отладка: что генерировала модель


@dataclass
class Session:
    history: DialogHistory = field(default_factory=DialogHistory)
    last_active: Optional[datetime] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_armed: bool = False     # бот уже показал гостю его бронь -> можно отменять после «да»
    user_texts: list = field(default_factory=list)       # реплики гостя в текущем диалоге
    known_dates: set = field(default_factory=set)        # даты, которые вернул call_date_agent
    booking: Optional[dict] = None                       # проверенные параметры брони (DATE, TIME, GUESTS, NAME)
    pending_book: Optional[dict] = None                  # ждём «да» перед бронированием
    pending_cancel: Optional[dict] = None                # ждём «да» перед отменой {"booking", "then_book", "resume_text"}
    in_booking: bool = False                             # диалог о брони уже идёт
    active_checked: bool = False                         # наличие активной брони уже проверяли в этом диалоге
    bk: Optional[dict] = None                            # бронь, которую ведёт код: {"slots": {...}, "stage": collect|confirm}
    faq_only: bool = False                               # временная сессия для ответа на вопрос посреди брони
    offered_booking: bool = False                        # последний ответ модели предлагал забронировать
    recent: list = field(default_factory=list)           # последние RECENT_EXCHANGES завершённых обменов «User: … -> Bot: …» — контекст для модели
    menu_ctx: dict = field(default_factory=dict)         # последний показанный список блюд («посоветуйте оттуда»); /reset стирает вместе с сессией

    def reset(self) -> None:
        self.history.reset()
        self.cancel_armed = False
        self.user_texts, self.known_dates = [], set()
        self.booking = self.pending_book = self.pending_cancel = None
        self.in_booking = self.active_checked = False
        self.bk, self.offered_booking = None, False


class Orchestrator:
    def __init__(self, llm, booking_db: BookingDB, *, menu: Optional[Menu] = None,
                 min_logprob: Optional[float] = -1.5, session_ttl_min: int = 30,
                 max_tool_steps: int = 6, confirm_booking: bool = True, code_cancel: bool = True,
                 grounding: bool = True, show_progress: bool = True, code_booking: bool = True,
                 code_menu: bool = True, rng: Optional[random.Random] = None,
                 now_fn: Callable[[], datetime] = moscow_now) -> None:
        """
        llm          — объект с async agenerate(prompt) -> GenResult (llm.LLM)
        confirm_booking — перед бронью код сам просит гостя написать «да» (модель при этом не нужна)
        code_cancel  — отмена брони и «у вас уже есть бронь» ведёт код, а не модель
        grounding    — имя, число гостей и время в брони должны быть в словах гостя (иначе код переспрашивает)
        code_menu    — вопросы про меню (фильтры, рекомендации, «какие есть удоны», состав, цена) отвечает код по menu.json
        code_booking — сбор брони (дата, время, гости, имя) ведёт код, а не модель: исправления гостя всегда перезаписывают старое
        show_progress — показывать служебные фразы модели («Проверяю время.»)
        min_logprob  — порог уверенности модели (средняя log-вероятность токенов ответа);
                       ниже порога -> запасной ответ. None отключает проверку. Подберите на своих данных.
        """
        self.llm, self.db, self.menu = llm, booking_db, menu
        self.min_logprob, self.ttl_min, self.max_tool_steps, self.now = min_logprob, session_ttl_min, max_tool_steps, now_fn
        self.confirm_booking, self.code_cancel, self.grounding, self.show_progress = confirm_booking, code_cancel, grounding, show_progress
        self.code_booking = code_booking
        self.code_menu, self.rng = code_menu, rng or random.Random()
        self._sessions: dict[int, Session] = {}
        self._progress = None

    # ------------------------------------------------------------ сессии
    def _session(self, chat_id: int) -> Session:
        now = self.now()
        for cid in [c for c, s in self._sessions.items()                # чистим давно ушедших
                    if s.last_active and (now - s.last_active).total_seconds() > self.ttl_min * 120]:
            if not self._sessions[cid].lock.locked():
                del self._sessions[cid]
        return self._sessions.setdefault(chat_id, Session())

    def reset(self, chat_id: int) -> None:
        """Сбросить диалог (например, по команде /start)."""
        if chat_id in self._sessions:
            self._sessions[chat_id].reset()
            self._sessions[chat_id].recent.clear()

    def wipe(self, chat_id: int) -> tuple[int, int]:
        """/reset для тестов: забыть диалог и стереть брони и отзывы этого чата. -> (броней, отзывов)."""
        self._sessions.pop(chat_id, None)
        return self.db.clear_chat(chat_id)

    def _menu(self) -> Menu:
        return self.menu or get_menu()

    # ------------------------------------------------------------ проверка ответа модели
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

    # ------------------------------------------------------------ главный вход
    async def handle_text(self, chat_id: int, text: str, user_name: Optional[str] = None,
                          on_progress=None) -> TurnResult:
        """on_progress — async-функция(Message): служебные фразы («Проверяю время.») уходят гостю сразу, а не в конце."""
        sess = self._session(chat_id)
        async with sess.lock:                                            # сообщения одного чата идут по очереди
            now = self.now()
            if sess.last_active and (now - sess.last_active).total_seconds() > self.ttl_min * 60:
                sess.reset()                                             # диалог «остыл»
                sess.recent.clear()
            sess.last_active = now

            clean = sanitize_user_text(text)
            if not clean:
                return TurnResult([Message(EMPTY_TEXT)])
            self._progress = on_progress
            return await self._dispatch(chat_id, sess, clean, user_name)

    async def _dispatch(self, chat_id: int, sess: Session, text: str, user_name: Optional[str]) -> TurnResult:
        """Сначала то, что ведёт код (ответ на вопрос «подтверждаете?», отмена, бронь), потом модель."""
        offered, sess.offered_booking = sess.offered_booking, False
        if sess.pending_book:
            pb, sess.pending_book = sess.pending_book, None
            if is_yes(text):
                return await self._do_booking(chat_id, sess, pb)
            sess.reset()
            if is_no(text):
                return TurnResult([Message(DECLINE_BOOK_TEXT)])
        elif sess.pending_cancel:
            pc, sess.pending_cancel = sess.pending_cancel, None
            if is_yes(text, cancel=True):
                return await self._do_cancel(chat_id, sess, pc)
            sess.reset()
            if is_no(text):
                return TurnResult([Message(KEEP_TEXT + (" Новую бронь не оформляю." if pc.get("resume_text") or pc.get("then_book") else ""))])
        if self.code_booking and sess.bk:                                # бронь уже идёт — ведёт код
            return await self._booking_turn(chat_id, sess, text, user_name)
        if self.code_cancel and is_cancel_intent(text):
            sess.reset()                                                 # смена темы: старый диалог не тянем
            return await self._start_cancel(chat_id, sess)
        if self.code_booking and (is_booking_request(text) or (offered and is_yes(text))):
            return await self._start_booking(chat_id, sess, "" if offered and is_yes(text) and not is_booking_request(text) else text, user_name)
        if self.code_cancel and not sess.active_checked and is_booking_request(text):
            sess.active_checked = True                                   # есть ли уже бронь — смотрим СРАЗУ
            ab = await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
            if ab:
                sess.pending_cancel = {"booking": ab, "then_book": None, "resume_text": text}
                return TurnResult([Message(self._active_text(ab))])
        sess.user_texts.append(text)
        return await self._run_turn(chat_id, sess, text, user_name)

    # ------------------------------------------------------------ отмена и бронь, которые ведёт код
    async def _start_cancel(self, chat_id: int, sess: Session) -> TurnResult:
        ab = await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
        if not ab:
            return TurnResult([Message(NO_ACTIVE_TEXT)])
        sess.pending_cancel = {"booking": ab, "then_book": None}
        return TurnResult([Message(f"У вас есть бронь: {booking_line(ab.date, ab.time, ab.guests, ab.name)}. Отменить её?")])

    async def _do_cancel(self, chat_id: int, sess: Session, pc: dict) -> TurnResult:
        now = self.now()
        old = await asyncio.to_thread(self.db.cancel_active, chat_id, now)
        if not old:
            sess.reset()
            return TurnResult([Message(NO_ACTIVE_TEXT)])
        out = TurnResult()
        then = pc.get("then_book")
        resume = pc.get("resume_text")
        if resume:                                                       # гость хотел бронь — продолжаем с его сообщения
            sess.reset()
            sess.in_booking = sess.active_checked = True
            sess.user_texts.append(resume)
            if self.code_booking:
                res = await self._start_booking(chat_id, sess, resume, None)
            else:
                res = await self._run_turn(chat_id, sess, resume, None)
            res.messages.insert(0, Message("Старую бронь отменил."))
            return res
        if not then:
            sess.reset()
            out.messages.append(Message(f"Готово, бронь отменена: {booking_line(old.date, old.time, old.guests, old.name)}. "
                                        "Если захотите, можно забронировать снова."))
            return out
        out.messages.append(Message("Старую бронь отменил. Теперь проверяю новую."))
        if self.code_booking:
            sess.reset()
            sess.in_booking = sess.active_checked = True
            sess.bk = {"slots": {"date": then["DATE"], "time": then["TIME"], "guests": int(then["GUESTS"]), "name": then["NAME"]},
                       "stage": "collect"}
            return await self._advance(chat_id, sess, out)
        ok = await asyncio.to_thread(self.db.has_free_table, int(then["GUESTS"]), then["DATE"], then["TIME"])
        if not ok:
            sess.reset()
            out.messages.append(Message(f"К сожалению, на {then['DATE']} в {then['TIME']} столик на {then['GUESTS']} "
                                        f"{people_word(int(then['GUESTS']))} занят. Напишите, пожалуйста, новую бронь: дату, "
                                        "время, количество человек и имя."))
            return out
        return self._ask_confirm(sess, then, out)

    def _ask_confirm(self, sess: Session, p: dict, out: TurnResult, keep_bk: bool = False) -> TurnResult:
        if not keep_bk:
            sess.pending_book = dict(p)
        out.messages.append(Message(
            "Свободный столик есть. Проверьте, пожалуйста, данные брони:\n"
            f"Время: {p['DATE']}, {p['TIME']}\nКоличество человек: {p['GUESTS']}\nИмя: {p['NAME']}\n"
            "Подтверждаете бронь? Напишите «да» или «нет»."))
        return out

    async def _do_booking(self, chat_id: int, sess: Session, p: dict) -> TurnResult:
        r = await asyncio.to_thread(self.db.book_table, int(p["GUESTS"]), p["DATE"], p["TIME"], p["NAME"], chat_id, self.now())
        out = TurnResult()
        sess.reset()
        if r.status == "booked":
            out.messages.append(Message(f"Готово, бронь подтверждена! Время: {p['DATE']}, {p['TIME']}. "
                                        f"Количество человек: {p['GUESTS']}. Имя: {p['NAME']}. Ждём вас в YUQI!"))
        elif r.status == "table_busy":
            out.messages.append(Message("К сожалению, этот столик только что заняли. Напишите, пожалуйста, другое время."))
        elif r.status == "active_booking_exists":
            a = r.active
            sess.pending_cancel = {"booking": a, "then_book": dict(p)}
            out.messages.append(Message(self._active_text(a)))
        else:
            out.messages.append(Message(r.reason))
        return out


    # ------------------------------------------------------------ бронь, которую ведёт код
    async def _start_booking(self, chat_id: int, sess: Session, text: str, user_name: Optional[str] = None) -> TurnResult:
        if not sess.active_checked:
            sess.active_checked = True                                   # есть ли уже бронь — смотрим СРАЗУ
            ab = await asyncio.to_thread(self.db.active_booking, chat_id, self.now())
            if ab:
                sess.pending_cancel = {"booking": ab, "then_book": None, "resume_text": text}
                return TurnResult([Message(self._active_text(ab))])
        sess.in_booking = True
        sess.bk = {"slots": {}, "stage": "collect", "started": True}
        return await self._booking_turn(chat_id, sess, text, user_name)

    async def _faq(self, chat_id: int, text: str, user_name: Optional[str]) -> TurnResult:
        """Вопрос посреди брони (адрес, Wi-Fi, меню): отвечает модель в чистой истории; бронь не трогаем."""
        tmp = Session(faq_only=True)
        tmp.user_texts.append(text)
        res = await self._run_turn(chat_id, tmp, text, user_name)
        if any(m.text == FALLBACK_TEXT or m.text == RESET_TEXT for m in res.messages):
            res.messages = []
        return res

    @staticmethod
    def _params(s: dict) -> dict:
        return {"DATE": s["date"], "TIME": s["time"], "GUESTS": str(s["guests"]), "NAME": s["name"]}

    async def _booking_turn(self, chat_id: int, sess: Session, text: str, user_name: Optional[str] = None) -> TurnResult:
        bk, out, now = sess.bk, TurnResult(), self.now()
        s = bk["slots"]
        started = bk.pop("started", False)
        if not started and is_abort(text):
            sess.reset()
            return TurnResult([Message("Хорошо, ничего не бронирую. Если понадобится столик, напишите.")])
        missing = [k for k in ("date", "time", "guests", "name") if k not in s]
        ex = slotlib.extract(text, now.date(), missing[0] if len(missing) == 1 else None)
        changed = {k: v for k, v in ex.items() if s.get(k) != v}

        if bk["stage"] == "confirm":
            if changed and set(changed) == {"name"} and is_yes(text):
                changed = {}
            if changed:
                s.update(changed)
                bk["stage"] = "collect"
                return await self._advance(chat_id, sess, out)
            if is_yes(text):
                return await self._do_booking(chat_id, sess, self._params(s))
            if is_no(text):
                sess.reset()
                return TurnResult([Message(DECLINE_BOOK_TEXT)])
            res = await self._faq(chat_id, text, user_name)
            res.messages.append(Message("Подтверждаете бронь? Напишите «да» или «нет»."))
            return res

        if not ex and not started:
            if is_no(text) or is_yes(text):
                return await self._advance(chat_id, sess, out)           # просто повторяем вопрос
            res = await self._faq(chat_id, text, user_name)
            nxt = await self._advance(chat_id, sess, TurnResult())
            res.messages += nxt.messages
            res.steps += nxt.steps
            return res
        s.update(ex)
        return await self._advance(chat_id, sess, out)

    async def _advance(self, chat_id: int, sess: Session, out: TurnResult) -> TurnResult:
        """Проверить собранное, переспросить недостающее или показать бронь на подтверждение."""
        bk, now = sess.bk, self.now()
        s = bk["slots"]
        errs: list[str] = []
        if "date" in s:
            err = validate_booking_date(s["date"], now.date())
            if err:
                errs.append(err)
                del s["date"]
        if "guests" in s and not (1 <= s["guests"] <= MAX_GUESTS_PER_BOOKING):
            errs.append(f"Забронировать столик можно максимум на {MAX_GUESTS_PER_BOOKING} человек.")
            del s["guests"]
        if "date" in s and "time" in s:
            if bk.get("time_ok") != (s["date"], s["time"]):              # то же время уже проверяли — не повторяем
                await self._say(out, Message("Проверяю время."))
                problem = check_slot(s["date"], s["time"], now)
                if problem:
                    errs.append(problem.text)
                    del s["time"]
                else:
                    bk["time_ok"] = (s["date"], s["time"])
        missing = [k for k in ("date", "time", "guests", "name") if k not in s]
        if missing:
            bk["stage"] = "collect"
            known = {"date": "дата", "time": "время", "guests": "гостей", "name": "имя"}
            have = ", ".join(f"{known[k]} — {s[k]}" for k in ("date", "time", "guests", "name") if k in s)
            ask = [ASK[k] for k in missing]
            q = "Подскажите, пожалуйста, " + (ask[0] if len(ask) == 1 else ", ".join(ask[:-1]) + " и " + ask[-1]) + "?"
            head = " ".join(errs) if errs else (f"Хорошо, {have}." if have else "")
            if errs and have:
                head += f" Пока записано: {have}."
            out.messages.append(Message((head + " " + q).strip()))
            return out
        await self._say(out, Message("Проверяю свободные столики."))
        p = self._params(s)
        ok = await asyncio.to_thread(self.db.has_free_table, s["guests"], s["date"], s["time"])
        if not ok:
            out.messages.append(Message(f"К сожалению, на {s['date']} в {s['time']} столик на {s['guests']} "
                                        f"{people_word(s['guests'])} занят. Напишите, пожалуйста, другое время."))
            del s["time"]
            bk["stage"] = "collect"
            return out
        bk["stage"] = "confirm"
        return self._ask_confirm(sess, p, out, keep_bk=True)

    @staticmethod
    def _active_text(a) -> str:
        return (f"У вас уже есть бронь: {booking_line(a.date, a.time, a.guests, a.name)}. Нельзя бронировать несколько "
                "столиков, пока не наступило время текущей брони. Хотите отменить старую бронь?")

    async def _run_turn(self, chat_id: int, sess: Session, user_text: str,
                        user_name: Optional[str] = None) -> TurnResult:
        """Обёртка: модель получает последние RECENT_EXCHANGES обменов как контекст («а в выходные?»), и обмен запоминается."""
        if sess.recent and not sess.history.done and not sess.faq_only:
            sess.history.done.extend(sess.recent)
        out = await self._run_turn_inner(chat_id, sess, user_text, user_name)
        if out.messages and not sess.faq_only and out.messages[0].text != FALLBACK_TEXT:
            sess.recent.append(self._segment(user_text, out.messages[0].text))
            del sess.recent[:-RECENT_EXCHANGES]
        return out

    @staticmethod
    def _segment(user_text: str, bot_text: str) -> str:
        """Короткий сегмент для контекста: у списков берём заголовок (первую строку), без разметки и служебных символов."""
        line = re.sub(r"<[^>]+>", "", bot_text).strip().splitlines()[0] if bot_text.strip() else ""
        line = html.unescape(line).replace("->", " ").replace("||", " ")
        line = re.sub(r"\s+", " ", line).strip()[:140]
        return f"User: {sanitize_user_text(user_text)}{ARROW} {line}".rstrip()

    async def _run_turn_inner(self, chat_id: int, sess: Session, user_text: str,
                              user_name: Optional[str] = None) -> TurnResult:
        h, out = sess.history, TurnResult()
        if self.code_menu:
            msgs = menu_router.route(user_text, self._menu(), self.rng, sess.menu_ctx)
            if msgs:
                out.messages += [Message(t, True) for t in msgs]
                sess.reset()
                return out
        prompt = h.prompt_for_user(user_text)

        for _ in range(self.max_tool_steps + 1):
            try:
                gen: GenResult = await self.llm.agenerate(prompt)
            except PromptTooLong:
                sess.reset()
                out.messages.append(Message(RESET_TEXT))
                return out
            reply = parse_reply(gen.text)
            out.steps.append({"prompt": prompt, "raw": reply.raw, "logprob": round(gen.mean_logprob, 3),
                              "action": reply.action})
            problem = self._problem(gen, reply)
            if problem:
                log.warning("chat=%s: %s | prompt=...%s | raw=%r", chat_id, problem, prompt[-120:], reply.raw)
                return self._reject(sess, out, user_text)

            # --- бронь ведёт код: модель в ней не участвует
            if self.code_booking and ((not reply.has_action and BOOKING_ASK_RE.search(reply.text.lower()))
                                      or reply.action in BOOKING_ACTIONS):
                if sess.faq_only or not has_booking_words(user_text):
                    return self._reject(sess, out, user_text)
                return await self._start_booking(chat_id, sess, user_text, user_name)

            # --- модель вдруг заговорила про бронь, хотя гость о ней не просил
            if self.grounding and not self._booking_context(sess) and (
                    (not reply.has_action and BOOKING_ASK_RE.search(reply.text.lower()))
                    or reply.action in BOOKING_ACTIONS):
                log.warning("chat=%s: бронь без запроса гостя: %r", chat_id, reply.raw)
                return self._reject(sess, out, user_text)

            # --- фильтр меню, о котором гость не спрашивал (опечатки сбивают модель)
            if (self.grounding and reply.action == "call_menu_filter_agent"
                    and not FILTER_WORDS_RE.search(norm(user_text))):
                log.warning("chat=%s: фильтр без запроса гостя: %r", chat_id, reply.raw)
                return self._reject(sess, out, user_text)

            # --- обычный текстовый ответ
            if not reply.has_action:
                early = self._early_time_check(reply.text) if self.grounding else None
                if early:                                                # время уже прошло / вне часов — говорим сразу
                    out.messages.append(Message(early))
                    h.commit(early)
                    return out
                out.messages.append(Message(reply.text))
                h.commit(reply.raw)
                if not expects_answer(reply):
                    sess.reset()                                         # вопрос закрыт — диалог закончен
                if self.code_booking and re.search(r"забронир|столик", reply.text.lower()) and reply.text.rstrip().endswith("?"):
                    sess.offered_booking = True                          # «Хотите забронировать?» -> «да» запускает бронь
                return out

            # --- инструменты с результатом: код выполняет и возвращает результат модели
            if reply.action in TOOLS_WITH_RESULT:
                if self.grounding and reply.action in BOOKING_PARAM_ACTIONS:
                    ask = self._check_params(sess, reply)
                    if ask:                                              # модель выдумала имя/число гостей/время
                        out.messages.append(Message(ask))
                        h.commit(ask)
                        return out
                if self.confirm_booking and reply.action == "book_table":
                    h.commit(reply.raw)                                  # финал и «да» ведёт код
                    self._ask_confirm(sess, reply.params, out)
                    return out
                if self.show_progress and reply.text:                    # «Проверяю время.» — сразу, до выполнения
                    await self._say(out, Message(reply.text))
                try:
                    result, direct_text = await self._run_tool(chat_id, sess, reply)
                except Exception:
                    log.exception("chat=%s: инструмент %s упал", chat_id, reply.action)
                    sess.reset()
                    out.messages.append(Message(TOOL_FAIL_TEXT))
                    return out
                if direct_text:                                          # код сам объясняет гостю (дата не понята и т.п.)
                    out.messages.append(Message(direct_text))
                    h.commit(direct_text)                                # в историю — текст вместо ACTION
                    if not expects_answer(Reply(raw=direct_text, text=direct_text)):
                        sess.reset()
                    return out
                h.commit(reply.raw)
                prompt = h.prompt_for_tool(result)
                continue

            # --- конечные действия
            if reply.text:
                out.messages.append(Message(reply.text))
            self._run_terminal(chat_id, user_name, user_text, reply, out)
            h.commit(reply.raw)
            sess.reset()
            return out

        log.warning("chat=%s: слишком много шагов инструментов", chat_id)
        sess.reset()
        out.messages.append(Message(FALLBACK_TEXT))
        return out

    def _reject(self, sess: Session, out: TurnResult, user_text: str) -> TurnResult:
        """Ответ модели не годится: пробуем понять гостя по ключевым словам (опечатки), иначе — «не понял»."""
        sess.reset()
        if self.grounding and looks_like_business_lunch(user_text):
            out.messages.append(Message(f"Бизнес-ланчи у нас {SCHED}. Присылаю меню бизнес-ланчей."))
            out.messages += [Message(t, True) for t in tool_show_business_lunch(self._menu())]
        elif self.grounding and (flt := menu_router.detect_filter(user_text)):          # модель исказила название фильтра
            out.messages += [Message(t, True) for t in tool_filter(flt, self._menu())]
        elif self.grounding and looks_like_menu(user_text):
            out.messages += [Message(t, True) for t in tool_show_menu("all", self._menu())]
        else:
            out.messages.append(Message(FALLBACK_TEXT))
        return out

    async def _say(self, out: TurnResult, m: Message) -> None:
        if self._progress:
            await self._progress(m)
        else:
            out.messages.append(m)

    # ------------------------------------------------------------ защита от выдумок модели
    @staticmethod
    def _booking_context(sess: Session) -> bool:
        return sess.in_booking or bool(sess.known_dates) or any(has_booking_words(t) for t in sess.user_texts)

    def _early_time_check(self, text: str) -> Optional[str]:
        """Модель переспрашивает и повторяет «дата — …, время — …»: если время уже прошло или вне часов, говорим сразу."""
        m = EARLY_RE.search(text)
        if not m:
            return None
        problem = check_slot(m.group(1), m.group(2), self.now())
        if problem is None:
            return None
        if problem.code == "outside_hours":
            return (f"Мы работаем с 10:00 до 23:00, поэтому на {m.group(2)} забронировать столик не получится. "
                    "Напишите, пожалуйста, другое время.")
        if problem.code == "past_time":
            return "Это время уже прошло. Напишите, пожалуйста, другое время."
        return None

    def _check_params(self, sess: Session, reply: Reply) -> Optional[str]:
        """None — параметры можно выполнять; иначе — вопрос гостю вместо выдуманного значения."""
        p = reply.params
        if reply.action == "call_booking_time_agent":
            bad = ungrounded(p, " ".join(sess.user_texts), sess.known_dates)
            if bad:
                log.warning("в брони выдуманы поля %s: %r", bad, reply.raw)
                items = [ASK[k] for k in ("date", "time", "guests", "name") if k in bad]
                return "Подскажите, пожалуйста, " + (items[0] if len(items) == 1 else ", ".join(items[:-1]) + " и " + items[-1]) + "?"
            sess.booking = dict(p)
            return None
        if sess.booking is None or any(p[k] != sess.booking[k] for k in ("DATE", "TIME", "GUESTS", "NAME")):
            log.warning("параметры брони изменились между шагами: %r", reply.raw)
            sess.booking = None
            return "Давайте уточним: напишите, пожалуйста, дату, время, количество человек и имя для брони."
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
            if err:
                return None, err
            sess.known_dates.add(d)
            return d, None

        if a == "call_booking_time_agent":
            problem = check_slot(p["DATE"], p["TIME"], now)
            if problem is None:
                return "time_valid", None
            if problem.code in ("outside_hours", "past_time"):
                return f"invalid_time | REASON: {problem.code}", None
            return None, problem.text                                    # кривая дата/время — модель такое не видела

        if a == "call_free_tables_agent":
            problem = check_slot(p["DATE"], p["TIME"], now)
            if problem:
                return None, problem.text
            old = await asyncio.to_thread(self.db.active_booking, chat_id, now)
            if old:                                                      # у гостя уже есть бронь
                if self.code_cancel:
                    sess.pending_cancel = {"booking": old, "then_book": dict(sess.booking or p)}
                    return None, self._active_text(old)
                sess.cancel_armed = True
                return old.as_exists_result(), None
            ok = await asyncio.to_thread(self.db.has_free_table, int(p["GUESTS"]), p["DATE"], p["TIME"])
            return ("available" if ok else "unavailable"), None

        if a == "book_table":
            r = await asyncio.to_thread(self.db.book_table, int(p["GUESTS"]), p["DATE"], p["TIME"],
                                        p["NAME"], chat_id, now)
            if r.status == "booked":
                return "booked", None
            if r.status == "table_busy":
                return "table_busy", None
            if r.status == "active_booking_exists":                      # гость успел забронировать параллельно
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
            sess.cancel_armed = False
            return ("cancelled" if old else "no_active_booking"), None

        raise ValueError(a)

    # ------------------------------------------------------------ конечные действия
    def _run_terminal(self, chat_id: int, user_name: Optional[str], user_text: str, reply: Reply, out: TurnResult) -> None:
        a, p = reply.action, reply.params
        if a == "show_menu":
            out.messages += [Message(t, True) for t in tool_show_menu(p["CATEGORY"], self._menu())]
        elif a == "show_business_lunch":
            out.messages += [Message(t, True) for t in tool_show_business_lunch(self._menu())]
        elif a == "call_menu_filter_agent":
            out.messages += [Message(t, True) for t in tool_filter(p["FILTER"], self._menu())]
        elif a == "call_price_agent":
            out.messages.append(Message(tool_price(p["ITEM"], self._menu()), True))
        elif a == "notify_manager":                                      # жалоба: гость уже получил ответ модели
            try:
                self.db.save_review(chat_id, user_name, user_text, True, self.now())
            except Exception:
                log.exception("не удалось сохранить отзыв")
            out.notices.append(ManagerNotice(chat_id, user_name, user_text))
