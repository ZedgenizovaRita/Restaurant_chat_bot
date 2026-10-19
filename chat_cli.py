# -*- coding: utf-8 -*-
"""Консольная версия бота YUQI — тот же бот, что в Telegram, только в терминале.

Самый простой запуск: run_console.bat (Windows) или run_console.sh (macOS/Linux) — они всё установят сами.
Вручную:  python chat_cli.py [--model yuqi_model] [--debug]
Если папки модели нет, программа предложит скачать её из GitHub Releases.
Команды в чате: /reset — стереть диалог и брони, /help — подсказка, /exit — выход."""
from __future__ import annotations

import argparse
import asyncio
import html
import os
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
from model_fetch import DEFAULT_URL
MODEL_URL = os.environ.get("MODEL_URL", DEFAULT_URL)
GREETING = ("Здравствуйте! Я помощник ресторана YUQI. Расскажу про меню, цены и бизнес-ланчи, адрес и время работы, "
            "а также помогу забронировать или отменить столик. Просто напишите, что вас интересует.")
HELP = ("Примеры вопросов:\n"
        "  • Покажи меню / Что посоветуешь без мяса? / Том Ям острый?\n"
        "  • Сколько стоит бизнес-ланч? / До скольки вы работаете? / Где вы находитесь?\n"
        "  • Хочу столик на завтра в 19:00 на двоих / Отмени мою бронь\n"
        "  • Отзыв: «Было очень вкусно, спасибо!»\n"
        "Команды: /reset — начать заново, /exit — выход.")

# ---------------------------------------------------------------- оформление вывода
if os.name == "nt":
    os.system("")                                    # включает цвета в консоли Windows 10+
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
B, DIM, CYAN, GREEN, RESET = ("\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[0m") if COLOR else ("",) * 5


def render(text: str) -> str:
    """Telegram-HTML -> текст для консоли: <b> — жирным, остальные теги убираем."""
    text = re.sub(r"</?b>", lambda m: RESET if m.group(0) == "</b>" else B, text)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text)


def say_bot(text: str) -> None:
    body = render(text).replace("\n", "\n     ")
    print(f"\n{CYAN}{B}Бот:{RESET} {body}")


# ---------------------------------------------------------------- модель
def ensure_model(path: str) -> str:
    from model_fetch import download_model, has_model
    if has_model(path):
        return path
    print(f"Папка модели «{path}» не найдена.")
    ans = input("Скачать модель (~100 МБ) из GitHub Releases? [Y/n]: ").strip().lower()
    if ans not in ("", "y", "yes", "д", "да"):
        sys.exit("Положите папку yuqi_model рядом с chat_cli.py и запустите снова.")
    try:
        download_model(MODEL_URL, path)
    except Exception as e:
        sys.exit(f"\nНе удалось скачать модель: {e}\nСкачайте yuqi_model.zip вручную: {MODEL_URL}")
    print(f"  модель готова: {path}")
    return path


# ---------------------------------------------------------------- чат
async def main(a) -> None:
    from db import BookingDB
    from llm import LLM
    from orchestrator import Orchestrator

    model = ensure_model(a.model if os.path.isabs(a.model) else os.path.join(HERE, a.model))
    print(f"{DIM}Загружаю модель, это займёт 10–30 секунд...{RESET}")
    llm = LLM.from_hf(model)
    orch = Orchestrator(llm, BookingDB(a.db))
    print(f"{DIM}Готово. /help — примеры вопросов, /reset — начать заново, /exit — выход.{RESET}")
    say_bot(GREETING)
    while True:
        try:
            text = (await asyncio.to_thread(input, f"\n{GREEN}{B}Вы:{RESET} ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        cmd = text.lower()
        if cmd in ("/exit", "/quit", "выход"):
            break
        if cmd in ("/help", "помощь"):
            print(HELP)
            continue
        if cmd in ("/reset", "/new", "/start"):
            b, _ = orch.wipe(1)
            print(f"{DIM}— диалог сброшен, броней удалено: {b} —{RESET}")
            say_bot(GREETING)
            continue

        async def progress(m):
            say_bot(m.text)
        print(f"{DIM}  …печатает{RESET}", end="\r", flush=True)
        r = await orch.handle_text(1, text, "Гость", progress)
        print(" " * 20, end="\r")
        if a.debug:
            for s in r.steps:
                print(f"{DIM}  [модель, logprob {s['logprob']}] {s['raw']}{RESET}")
        for m in r.messages:
            say_bot(m.text)
        for n in r.notices:
            print(f"{DIM}  [тихое уведомление администратору] {n.text}{RESET}")
    print("\nДо встречи в YUQI!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Бот YUQI в консоли")
    ap.add_argument("--model", default="yuqi_model", help="папка модели (по умолчанию yuqi_model рядом с программой)")
    ap.add_argument("--debug", action="store_true", help="показывать, что генерирует модель")
    ap.add_argument("--db", default=os.path.join(tempfile.gettempdir(), "yuqi_cli.db"), help="файл базы броней")
    try:
        asyncio.run(main(ap.parse_args()))
    except KeyboardInterrupt:
        print("\nДо встречи в YUQI!")
