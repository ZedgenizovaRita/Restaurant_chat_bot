# -*- coding: utf-8 -*-
"""
db.py — столики и брони

Инструменты, которые вызывает модель (ACTION в датасете v5):
  check_slot(...)            call_booking_time_agent  -> None (время ок) | SlotProblem(code, text)
                             code: 'outside_hours' | 'past_time' | 'bad_date' | 'bad_time'
  has_free_table(...)        call_free_tables_agent   -> True / False 
  active_booking(chat_id)    check_active_booking     -> ActiveBooking | None
  book_table(...)            book_table               -> BookResult('booked' | 'table_busy' | 'active_booking_exists' | 'invalid')
  cancel_active(chat_id)     cancel_booking           -> ActiveBooking (отменённая) | None
  save_review(...)           журнал отзывов (для notify_manager)

«Активная» бронь — бронь гостя (chat_id), время начала которой ещё не наступило.
У одного гостя одновременно может быть только одна такая бронь.

Бронь занимает столик на BOOKING_DURATION_MIN минут. Если одного стола мало, подбираются
несколько (минимум столов, затем минимум лишних мест) — столы считаются сдвигаемыми.
Запись защищена транзакцией BEGIN IMMEDIATE: два человека не займут один столик одновременно,
а второй активной брони у одного гостя не появится даже при двух параллельных сообщениях.
"""
from __future__ import annotations

import itertools
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dates import moscow_now, validate_booking_date

TABLES = [(1, 2), (2, 2), (3, 4), (4, 4), (5, 6), (6, 6)]   # (номер стола, мест)
OPEN_MIN = 10 * 60               # открытие 10:00
CLOSE_MIN = 23 * 60              # закрытие 23:00
LAST_START_MIN = 22 * 60 + 30    # позже 22:30 бронь не делаем
BOOKING_DURATION_MIN = 120       # сколько стол занят после начала брони
MAX_GUESTS_PER_BOOKING = 6


def _to_min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _iso(date_str: str) -> str:
    return datetime.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")


def _ru(iso: str) -> str:
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d.%m.%Y")


@dataclass
class SlotProblem:
    code: str          # 'outside_hours' | 'past_time' | 'bad_date' | 'bad_time'
    text: str          # понятная причина для пользователя


@dataclass
class ActiveBooking:
    id: int
    date: str          # ДД.ММ.ГГГГ
    time: str          # ЧЧ:ММ
    guests: int
    name: str

    def as_result(self) -> str:
        """Строка для 'Tool: RESULT: ...' (формат из датасета)."""
        return f"active_booking | DATE: {self.date} | TIME: {self.time} | GUESTS: {self.guests} | NAME: {self.name}"

    def as_exists_result(self) -> str:
        return "active_booking_exists" + self.as_result()[len("active_booking"):]


@dataclass
class BookResult:
    status: str                                   # 'booked' | 'table_busy' | 'active_booking_exists' | 'invalid'
    booking_id: Optional[int] = None
    tables: list[int] = field(default_factory=list)
    active: Optional[ActiveBooking] = None
    reason: str = ""


def check_slot(date_str: str, time_str: str, now: Optional[datetime] = None) -> Optional[SlotProblem]:
    """None, если на это время можно оформлять бронь."""
    now = now or moscow_now()
    err = validate_booking_date(date_str, now.date())
    if err:
        return SlotProblem("bad_date", err)
    try:
        t = _to_min(time_str)
    except ValueError:
        return SlotProblem("bad_time", "Не удалось разобрать время. Напишите его, пожалуйста, например: 19:00.")
    if t < OPEN_MIN or t > LAST_START_MIN:
        return SlotProblem("outside_hours", "Мы работаем с 10:00 до 23:00. Напишите, пожалуйста, другое время.")
    if _iso(date_str) == now.strftime("%Y-%m-%d") and t <= now.hour * 60 + now.minute:
        return SlotProblem("past_time", "Это время уже прошло. Напишите, пожалуйста, другое время.")
    return None


class BookingDB:
    def __init__(self, path: str = "yuqi.db", tables: Optional[list[tuple[int, int]]] = None) -> None:
        self.path = path
        self.tables = list(tables or TABLES)
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS bookings(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER, name TEXT, guests INTEGER NOT NULL,
                    date TEXT NOT NULL,                -- YYYY-MM-DD
                    start_min INTEGER NOT NULL, end_min INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',     -- active | cancelled
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS booking_tables(
                    booking_id INTEGER NOT NULL REFERENCES bookings(id),
                    table_id INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER, user_name TEXT, text TEXT NOT NULL,
                    flagged INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS ix_bookings_date ON bookings(date, status);
                CREATE INDEX IF NOT EXISTS ix_bookings_chat ON bookings(chat_id, status);
            """)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=15, isolation_level=None)   # ручное управление транзакциями
        c.row_factory = sqlite3.Row
        return c

    # ------------------------------------------------------------ внутренние помощники
    def _free_tables(self, c: sqlite3.Connection, iso: str, start: int) -> list[tuple[int, int]]:
        end = start + BOOKING_DURATION_MIN
        busy = {r[0] for r in c.execute(
            """SELECT bt.table_id FROM bookings b JOIN booking_tables bt ON bt.booking_id = b.id
               WHERE b.status = 'active' AND b.date = ? AND b.start_min < ? AND b.end_min > ?""",
            (iso, end, start))}
        return [(tid, cap) for tid, cap in self.tables if tid not in busy]

    @staticmethod
    def _pick(free: list[tuple[int, int]], guests: int) -> Optional[list[int]]:
        """Минимум столов, затем минимум лишних мест."""
        for n in range(1, len(free) + 1):
            best, best_waste = None, None
            for combo in itertools.combinations(free, n):
                cap = sum(c for _, c in combo)
                if cap >= guests and (best is None or cap - guests < best_waste):
                    best, best_waste = combo, cap - guests
            if best:
                return [tid for tid, _ in best]
        return None

    @staticmethod
    def _active_row(c: sqlite3.Connection, chat_id: Optional[int], now: datetime):
        if chat_id is None:
            return None
        today, now_min = now.strftime("%Y-%m-%d"), now.hour * 60 + now.minute
        return c.execute(
            """SELECT * FROM bookings WHERE chat_id = ? AND status = 'active'
               AND (date > ? OR (date = ? AND start_min > ?)) ORDER BY date, start_min LIMIT 1""",
            (chat_id, today, today, now_min)).fetchone()

    @staticmethod
    def _to_active(r: sqlite3.Row) -> ActiveBooking:
        return ActiveBooking(r["id"], _ru(r["date"]), f"{r['start_min'] // 60:02d}:{r['start_min'] % 60:02d}",
                             r["guests"], r["name"] or "")

    # ------------------------------------------------------------ инструменты для модели
    def has_free_table(self, guests: int, date_str: str, time_str: str) -> bool:
        """call_free_tables_agent: есть ли стол (или столы) на это время и это число гостей."""
        if not (1 <= guests <= MAX_GUESTS_PER_BOOKING):
            return False
        c = self._conn()
        try:
            return self._pick(self._free_tables(c, _iso(date_str), _to_min(time_str)), guests) is not None
        finally:
            c.close()

    def active_booking(self, chat_id: Optional[int], now: Optional[datetime] = None) -> Optional[ActiveBooking]:
        """check_active_booking: ближайшая будущая активная бронь этого гостя."""
        c = self._conn()
        try:
            r = self._active_row(c, chat_id, now or moscow_now())
            return self._to_active(r) if r else None
        finally:
            c.close()

    def cancel_active(self, chat_id: Optional[int], now: Optional[datetime] = None) -> Optional[ActiveBooking]:
        """cancel_booking: отменяет активную бронь гостя. None — отменять нечего."""
        c = self._conn()
        try:
            c.execute("BEGIN IMMEDIATE")
            r = self._active_row(c, chat_id, now or moscow_now())
            if not r:
                c.execute("ROLLBACK")
                return None
            c.execute("UPDATE bookings SET status = 'cancelled' WHERE id = ?", (r["id"],))
            c.execute("COMMIT")
            return self._to_active(r)
        except Exception:
            try:
                c.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            c.close()

    def book_table(self, guests: int, date_str: str, time_str: str, name: str,
                   chat_id: Optional[int] = None, now: Optional[datetime] = None) -> BookResult:
        """book_table. Заново проверяет всё (модели не доверяем): время, лимит гостей, чужую активную
        бронь этого гостя и свободные столы — одной транзакцией."""
        now = now or moscow_now()
        if not (1 <= guests <= MAX_GUESTS_PER_BOOKING):
            return BookResult("invalid", reason=f"Забронировать столик можно максимум на {MAX_GUESTS_PER_BOOKING} человек.")
        problem = check_slot(date_str, time_str, now)
        if problem:
            return BookResult("invalid", reason=problem.text)

        iso, start = _iso(date_str), _to_min(time_str)
        name = (name or "").replace("|", " ").strip()[:60] or "Гость"
        c = self._conn()
        try:
            c.execute("BEGIN IMMEDIATE")                     # блокируем запись: проверка + вставка атомарны
            old = self._active_row(c, chat_id, now)
            if old:
                c.execute("ROLLBACK")
                return BookResult("active_booking_exists", active=self._to_active(old))
            picked = self._pick(self._free_tables(c, iso, start), guests)
            if not picked:
                c.execute("ROLLBACK")
                return BookResult("table_busy")
            cur = c.execute(
                "INSERT INTO bookings(chat_id, name, guests, date, start_min, end_min, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (chat_id, name, guests, iso, start, start + BOOKING_DURATION_MIN, now.strftime("%Y-%m-%d %H:%M:%S")))
            bid = cur.lastrowid
            c.executemany("INSERT INTO booking_tables(booking_id, table_id) VALUES (?,?)", [(bid, t) for t in picked])
            c.execute("COMMIT")
            return BookResult("booked", booking_id=bid, tables=picked)
        except Exception:
            try:
                c.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            c.close()

    def get_booking(self, booking_id: int) -> Optional[dict]:
        c = self._conn()
        try:
            r = c.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,)).fetchone()
            if not r:
                return None
            tabs = [x[0] for x in c.execute("SELECT table_id FROM booking_tables WHERE booking_id = ?", (booking_id,))]
            return {**dict(r), "tables": tabs}
        finally:
            c.close()

    # ------------------------------------------------------------ отзывы
    def save_review(self, chat_id: Optional[int], user_name: Optional[str], text: str, flagged: bool,
                    now: Optional[datetime] = None) -> int:
        c = self._conn()
        try:
            cur = c.execute("INSERT INTO reviews(chat_id, user_name, text, flagged, created_at) VALUES (?,?,?,?,?)",
                            (chat_id, user_name, text, int(flagged), (now or moscow_now()).strftime("%Y-%m-%d %H:%M:%S")))
            return cur.lastrowid
        finally:
            c.close()
