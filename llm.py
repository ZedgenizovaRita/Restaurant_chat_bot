# -*- coding: utf-8 -*-
"""
llm.py — загрузка обученной модели (папка yuqi_model: mini-tron-50/nanoGPT, или out-rugpt) и генерация ответа бота.

Что делает:
  - LLM.from_hf(папка) загружает модель и токенайзер;
  - generate(prompt) дописывает текст после "-> Bot:" и ОСТАНАВЛИВАЕТСЯ на конце ответа
    (перевод строки, "||", <|endoftext|>) — в отличие от model.generate из nanoGPT,
    который всегда генерирует ровно max_new_tokens токенов;
  - по умолчанию жадное декодирование (temperature=0): для бота с командами нужна
    предсказуемость, а не разнообразие;
  - возвращает среднюю лог-вероятность ответа — по ней оркестратор сможет отсечь
    неуверенные ответы («не понял, переформулируйте»);
  - fit_prompt режет историю диалога по границам сегментов, чтобы не вылезти за block_size.

Зависимости: torch, transformers, sentencepiece.
"""
from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from protocol import SEG_SEP, clean_generation


class PromptTooLong(Exception):
    """Даже после обрезки истории промпт не помещается в контекст модели."""


@dataclass
class GenResult:
    text: str             # очищенный ответ бота (обрезан по '\n' / '||')
    mean_logprob: float   # средняя log-вероятность токенов ответа; ближе к 0 = увереннее
    n_tokens: int
    stop: str             # 'newline' | 'sep' | 'eot' | 'max_tokens'


# ---------------------------------------------------------------- чистая логика (без torch)
def detect_stop(generated: str) -> Optional[str]:
    """Причина остановки по уже сгенерированному тексту или None."""
    if "\n" in generated:
        return "newline"
    if "||" in generated:
        return "sep"
    if "<|endoftext|>" in generated:
        return "eot"
    return None


def fit_prompt(prompt: str, count_tokens: Callable[[str], int], max_tokens: int) -> str:
    """Выкидывает самые старые сегменты диалога, пока промпт не влезет в max_tokens.
    Первым остаётся сегмент 'User:' (как в обучающих диалогах). Если не получилось —
    PromptTooLong (оркестратор в этом случае сбрасывает диалог)."""
    segs = prompt.split(SEG_SEP)
    while len(segs) > 1 and count_tokens(SEG_SEP.join(segs)) > max_tokens:
        segs.pop(0)
        while len(segs) > 1 and not segs[0].startswith("User:"):
            segs.pop(0)
    out = SEG_SEP.join(segs)
    if not segs[0].startswith("User:") or count_tokens(out) > max_tokens:
        raise PromptTooLong(f"не удалось уместить промпт: {count_tokens(out)} ток. при лимите {max_tokens}, "
                            f"начало сегмента: {segs[0][:12]!r}")
    return out


def _last_logits(out):
    """Достаёт логиты последней позиции из ответа модели любого типа:
    nanoGPT -> (logits, loss); HuggingFace -> объект с .logits; просто тензор."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits
    return out[:, -1, :]


# ---------------------------------------------------------------- модель
class LLM:
    def __init__(self, model, encode: Callable[[str], list], decode: Callable[[list], str],
                 block_size: int, *, device: Optional[str] = None,
                 eot_id: Optional[int] = None, valid_vocab: Optional[int] = None) -> None:
        """
        Ядро для любой модели.
          model       — torch.nn.Module: model(x) с x формы [1, T] (LongTensor) возвращает
                        логиты [1, T, V] (или [1, 1, V]), либо кортеж (logits, ...), либо объект с .logits
          encode      — str -> list[int];  decode — list[int] -> str
          block_size  — максимальная длина контекста модели (в токенах)
          eot_id      — id токена конца текста, если есть
          valid_vocab — сколько первых id словаря реальные (у nanoGPT vocab дополнен до 50304)

        Пример со своей моделью:
            model = MyGPT(cfg); model.load_state_dict(torch.load("my.pt")); 
            llm = LLM(model, encode, decode, block_size=cfg.block_size)
        """
        import torch
        self.torch = torch
        if device:
            model = model.to(device)
        self.device = str(next(model.parameters()).device)
        self.model = model.eval()
        self.encode, self.decode = encode, decode
        self.block_size = block_size
        self.eot_id = eot_id
        self.valid_vocab = valid_vocab
        self.hf = False
        self.stop_ids: set = set()      # доп. токены-стоп (роли чат-шаблона)
        self._lock = threading.Lock()   # одна генерация за раз (бот может получать сообщения параллельно)

    # -------- фабрика: модель HuggingFace (ruGPT и т.п.), папка после save_pretrained
    @classmethod
    def from_hf(cls, model_dir: str, device: Optional[str] = None) -> "LLM":
        """model_dir — папка с config.json, весами и токенизатором (yuqi_model из training/colab_minitron.ipynb или out-rugpt)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.isdir(model_dir):
            raise SystemExit(f"Папка модели не найдена: {os.path.abspath(model_dir)}\n"
                             "Укажите путь к папке модели, например: set MODEL_DIR=yuqi_model")
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        cfg = model.config
        block = getattr(cfg, "n_positions", None) or cfg.max_position_embeddings
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            model = model.float()                       # веса могут быть сохранены в float16; на процессоре считаем во float32
        sp_meta = os.path.join(model_dir, "yuqi_tokenizer.json")
        if os.path.exists(sp_meta):                     # mini-tron-50 (nanoGPT): токенайзер SentencePiece
            import json
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "tokenizer.model"))
            m = json.load(open(sp_meta, encoding="utf-8"))
            if m.get("format") == "chat":               # родной чат-шаблон модели (см. chatfmt.py)
                from chatfmt import ChatIds
                ch = ChatIds(sp, m["eot_id"], m["user_id"], m["assistant_id"], m.get("system_id"), m.get("system", ""))
                llm = cls(model, ch.encode, ch.decode, block, device=device, eot_id=m["eot_id"])
                llm.stop_ids = ch.stop_ids
            else:
                llm = cls(model, lambda s: sp.encode(s), lambda ids: sp.decode(ids), block, device=device, eot_id=m["eot_id"])
        else:                                           # ruGPT и другие модели с токенайзером HuggingFace
            tok = AutoTokenizer.from_pretrained(model_dir)
            llm = cls(model, lambda s: tok.encode(s, add_special_tokens=False), lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False),
                      block, device=device, eot_id=tok.eos_token_id)
        llm.hf = True                                  # включает кэш ключей/значений (быстрая генерация)
        return llm

    def count_tokens(self, s: str) -> int:
        return len(self.encode(s))

    # -------- генерация
    def generate(self, prompt: str, max_new_tokens: int = 320,
                 temperature: float = 0.0, top_k: Optional[int] = None,
                 seed: Optional[int] = None) -> GenResult:
        """Дописывает ответ после последнего '-> Bot:' и останавливается на конце ответа.
        temperature=0 — жадное декодирование (рекомендуется для бота)."""
        torch = self.torch
        reserve = min(max_new_tokens, self.block_size // 2)
        prompt = fit_prompt(prompt, self.count_tokens, self.block_size - reserve)

        if seed is not None:
            torch.manual_seed(seed)
        x = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        new_ids: list[int] = []
        logps: list[float] = []
        stop = "max_tokens"

        past = None
        with self._lock, torch.inference_mode():
            for _ in range(max_new_tokens):
                if self.hf:                                      # HuggingFace: подаём только новый токен + кэш
                    o = self.model(x if past is None else x[:, -1:], past_key_values=past, use_cache=True)
                    past, logits = o.past_key_values, o.logits[:, -1, :].float()
                else:
                    logits = _last_logits(self.model(x[:, -self.block_size:])).float()
                if self.valid_vocab is not None and self.valid_vocab < logits.size(-1):
                    logits[:, self.valid_vocab:] = float("-inf")
                logp = torch.log_softmax(logits, dim=-1)

                if temperature <= 0:
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    scaled = logits / temperature
                    if top_k:
                        kth = torch.topk(scaled, min(top_k, scaled.size(-1))).values[:, [-1]]
                        scaled[scaled < kth] = float("-inf")
                    nxt = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)

                tid = int(nxt.item())
                logps.append(float(logp[0, tid]))
                if (self.eot_id is not None and tid == self.eot_id) or tid in self.stop_ids:
                    stop = "eot"
                    break
                new_ids.append(tid)
                x = torch.cat([x, nxt], dim=1)

                reason = detect_stop(self.decode(new_ids))
                if reason:
                    stop = reason
                    break

        text = clean_generation(self.decode(new_ids))
        mean_lp = sum(logps) / len(logps) if logps else 0.0
        return GenResult(text=text, mean_logprob=mean_lp, n_tokens=len(new_ids), stop=stop)

    async def agenerate(self, *args, **kwargs) -> GenResult:
        """Неблокирующая версия для асинхронного бота (aiogram / python-telegram-bot v20+)."""
        return await asyncio.to_thread(self.generate, *args, **kwargs)


# ---------------------------------------------------------------- дымовой тест
if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Проверка загрузки модели и генерации")
    ap.add_argument("--model", required=True, help="папка модели (yuqi_model)")
    args = ap.parse_args()

    t0 = time.time()
    llm = LLM.from_hf(args.model)
    print(f"загружено за {time.time() - t0:.1f} c, device={llm.device}, block_size={llm.block_size}")

    tests = [
        "User: Есть парковка рядом? -> Bot:",
        "User: Хочу забронировать стол -> Bot: Конечно! Напишите, пожалуйста, дату, точное время, количество человек и имя для брони. "
        "|| User: Завтра в 19:00, нас трое, Даша -> Bot:",
        "User: Покажи меню -> Bot:",
        "User: Сколько стоит Том Ям? -> Bot:",
    ]
    for p in tests:
        t0 = time.time()
        r = llm.generate(p)
        print(f"\n{p[-70:]}\n  -> {r.text}\n  [{r.stop}, {r.n_tokens} ток., mean_logprob={r.mean_logprob:.2f}, {time.time() - t0:.1f} c]")
