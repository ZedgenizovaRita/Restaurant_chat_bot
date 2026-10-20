# -*- coding: utf-8 -*-
"""
main.py — запуск Telegram-бота YUQI (python-telegram-bot v20+).

    pip install -r requirements.txt
    export BOT_TOKEN=...            # токен от @BotFather
    export MODEL_DIR=yuqi_model     # папка модели (результат training/colab_minitron.ipynb)
    export MANAGER_CHAT_ID=123456789      # куда слать жалобы (узнать id: @userinfobot); можно не задавать
    export ALLOW_RESET=1                  # команда /reset (стирает диалог и брони чата) — для тестов; в бою поставьте 0
    python main.py

Бот работает только текстом: ни кнопок, ни форм. Всё, что нужно, пользователь пишет в чат.
"""
import asyncio
import logging
import os
import re

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, NetworkError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from db import BookingDB
from llm import LLM
from orchestrator import Message, Orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
for _noisy in ("httpx", "httpcore", "telegram.ext.Updater"):          # не засорять консоль служебными сообщениями
    logging.getLogger(_noisy).setLevel(logging.WARNING)
log = logging.getLogger("yuqi.main")

GREETING = ("Здравствуйте! Я помощник ресторана YUQI. Расскажу про меню, цены и бизнес-ланчи, адрес и время работы, "
            "а также помогу забронировать или отменить столик. Просто напишите, что вас интересует.")
NON_TEXT = "Я понимаю только текстовые сообщения. Напишите, пожалуйста, ваш вопрос текстом."


async def send(update: Update, m: Message) -> None:
    """Отправка одного сообщения; если Telegram не принял HTML — отправляем простым текстом."""
    try:
        await update.effective_chat.send_message(m.text, parse_mode=ParseMode.HTML if m.html else None)
    except BadRequest:
        await update.effective_chat.send_message(re.sub(r"<[^>]+>", "", m.text))


async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    ctx.application.bot_data["orch"].reset(update.effective_chat.id)
    await update.effective_chat.send_message(GREETING)


async def on_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """/reset — начать с чистого листа: диалог, брони и отзывы ЭТОГО чата стираются (чужие не трогаются)."""
    if not ctx.application.bot_data.get("allow_reset"):
        await update.effective_chat.send_message("Команда недоступна.")
        return
    n_book, n_rev = await asyncio.to_thread(ctx.application.bot_data["orch"].wipe, update.effective_chat.id)
    await update.effective_chat.send_message(f"Готово: диалог сброшен, броней удалено: {n_book}. Можно начинать заново.")


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    orch: Orchestrator = ctx.application.bot_data["orch"]
    chat = update.effective_chat
    user = update.effective_user
    await chat.send_action(ChatAction.TYPING)
    async def progress(msg: Message) -> None:                   # «Проверяю время.» — сразу, не дожидаясь конца цепочки
        await send(update, msg)
        await chat.send_action(ChatAction.TYPING)

    res = await orch.handle_text(chat.id, update.message.text, user.first_name if user else None, progress)
    for m in res.messages:
        await send(update, m)
    manager = ctx.application.bot_data.get("manager_chat_id")
    if manager:
        for n in res.notices:                              # тихое уведомление о жалобе; гость его не видит
            who = n.user_name or "гость"
            link = f' (<a href="tg://user?id={n.chat_id}">профиль</a>)'
            text = f"⚠️ Жалоба в отзыве от {who}{link}:\n\n{re.sub('<', '&lt;', n.text)}"
            try:
                await ctx.bot.send_message(manager, text, parse_mode=ParseMode.HTML)
            except Exception:
                log.exception("не удалось отправить уведомление администратору")


async def on_other(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_chat.send_message(NON_TEXT)


async def on_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """/stats — сколько ответов дала модель (текстом или через инструмент), а сколько — запасной ответ."""
    if not ctx.application.bot_data.get("allow_reset"):
        return
    st = ctx.application.bot_data["orch"].stats
    total = sum(st.values()) or 1
    lines = [f"{k}: {v} ({v * 100 // total}%)" for k, v in st.most_common()]
    await update.effective_chat.send_message("Кто отвечал с момента запуска:\n" + ("\n".join(lines) or "пока никто"))


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if isinstance(ctx.error, NetworkError):                       # обрыв связи/таймаут/прокси: само переподключится, трейс не нужен
        log.warning("проблема с сетью (%s), пробую снова", type(ctx.error).__name__)
        return
    log.exception("необработанная ошибка", exc_info=ctx.error)


def main() -> None:
    default_dir = "yuqi_model" if os.path.isdir("yuqi_model") or not os.path.isdir("out-rugpt") else "out-rugpt"
    model_dir = os.environ.get("MODEL_DIR", default_dir)               # папка модели (yuqi_model — mini-tron-50 после дообучения)
    from model_fetch import download_model, has_model
    if not has_model(model_dir) and os.environ.get("MODEL_URL"):         # хостинг: модели нет в репозитории — качаем из Releases
        log.info("модели нет в %s, скачиваю %s", model_dir, os.environ["MODEL_URL"])
        download_model(os.environ["MODEL_URL"], model_dir, progress=False)
    llm = LLM.from_hf(model_dir)
    log.info("модель загружена: device=%s block_size=%s", llm.device, llm.block_size)
    orch = Orchestrator(llm, BookingDB(os.environ.get("DB_PATH", "yuqi.db")))

    async def post_init(a: Application) -> None:                  # подсказка команд в меню Telegram
        from telegram import BotCommand
        cmds = [BotCommand("start", "Начать сначала")] + ([BotCommand("reset", "Стереть диалог и брони (тест)")] if a.bot_data.get("allow_reset") else [])
        await a.bot.set_my_commands(cmds)

    builder = Application.builder().token(os.environ["BOT_TOKEN"]).concurrent_updates(True).post_init(post_init)
    proxy = os.environ.get("TELEGRAM_PROXY", "").strip()                # необязательно: socks5://user:pass@host:port или http://host:port
    if proxy:
        builder = builder.proxy(proxy).get_updates_proxy(proxy)
        log.info("Telegram через прокси: %s", proxy.split("@")[-1])      # пароль в лог не пишем
    app = builder.build()
    app.bot_data["orch"] = orch
    app.bot_data["manager_chat_id"] = int(os.environ["MANAGER_CHAT_ID"]) if os.environ.get("MANAGER_CHAT_ID") else None
    app.bot_data["allow_reset"] = os.environ.get("ALLOW_RESET", "1") != "0"
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("reset", on_reset))
    app.add_handler(CommandHandler("stats", on_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(~filters.COMMAND, on_other))
    app.add_error_handler(on_error)
    try:
        app.run_polling(timeout=30, bootstrap_retries=-1)        # Ctrl+C — штатная остановка
    except KeyboardInterrupt:
        pass
    log.info("бот остановлен")


if __name__ == "__main__":
    main()
