# -*- coding: utf-8 -*-
"""Чат с ботом в консоли:
    python chat_cli.py --model out-rugpt [--debug]
    (для nanoGPT: --ckpt ckpt.pt --nanogpt nanogpt [--meta meta.pkl])
Выход: Ctrl+C. /new — начать диалог заново. --debug показывает вызовы инструментов."""
import argparse, asyncio, tempfile, os
from db import BookingDB
from llm import LLM
from orchestrator import Orchestrator

ap = argparse.ArgumentParser()
ap.add_argument("--model"); ap.add_argument("--ckpt"); ap.add_argument("--nanogpt")
ap.add_argument("--meta"); ap.add_argument("--debug", action="store_true")
ap.add_argument("--db", default=os.path.join(tempfile.gettempdir(), "yuqi_cli.db"))
a = ap.parse_args()


async def main():
    llm = LLM.from_hf(a.model) if a.model else LLM.from_nanogpt(a.ckpt, a.nanogpt, meta_path=a.meta)
    orch = Orchestrator(llm, BookingDB(a.db))
    print("Готово. Пишите сообщения (Ctrl+C — выход, /new — сначала).")
    while True:
        text = (await asyncio.to_thread(input, "\nВы: ")).strip()
        if text == "/new":
            orch.reset(1); print("— диалог сброшен —"); continue
        async def progress(m):
            print("\nБот:", m.text)
        r = await orch.handle_text(1, text, "Консоль", progress)
        if a.debug:
            for s in r.steps: print(f"  [{s['logprob']}] {s['raw']}")
        for m in r.messages: print("\nБот:", m.text)
        for n in r.notices: print("  [уведомление администратору]", n.text)

try:
    asyncio.run(main())
except (KeyboardInterrupt, EOFError):
    pass
