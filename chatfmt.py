# -*- coding: utf-8 -*-
"""Перевод нашего формата диалога в родной чат-шаблон mini-tron-50.

Наш формат (protocol.py):   User: вопрос -> Bot: ответ || Tool: RESULT: x -> Bot: ответ || User: вопрос -> Bot:
Шаблон модели:              <|system|>…<|endoftext|><|user|>вопрос<|endoftext|><|assistant|>ответ<|endoftext|>…<|assistant|>

Модель обучалась отвечать в роли <|assistant|> на реплику <|user|> — в этом шаблоне она гораздо лучше
связывает ответ с вопросом, чем в незнакомом «User: … -> Bot:». Один и тот же код используется
при подготовке датасета (training/minitron/prepare_yuqi_sp.py) и в боте (llm.py)."""
from __future__ import annotations

import re

SEG_SEP = " || "
SEG_RE = re.compile(r"^(User|Tool): (.*?) -> Bot:(?: (.*))?$", re.DOTALL)
SYSTEM = "Ты помощник ресторана YUQI. Отвечай вежливо и по делу."


class ChatIds:
    def __init__(self, sp, eot: int, user: int, assistant: int, system: int | None = None, system_text: str = SYSTEM):
        self.sp, self.eot, self.user, self.assistant, self.system, self.system_text = sp, eot, user, assistant, system, system_text

    @classmethod
    def from_sp(cls, sp, meta: dict | None = None, system_text: str = SYSTEM) -> "ChatIds":
        st = (meta or {}).get("special_tokens") or {}

        def tid(name, piece):
            v = st.get(name)
            if v is None:
                v = sp.piece_to_id(piece)
                if v == sp.unk_id():
                    raise ValueError(f"в токенайзере нет {piece}")
            return int(v)
        sys_id = None
        try:
            sys_id = tid("system", "<|system|>")
        except ValueError:
            pass
        return cls(sp, tid("endoftext", "<|endoftext|>"), tid("user", "<|user|>"), tid("assistant", "<|assistant|>"), sys_id, system_text)

    def encode(self, text: str) -> list[int]:
        """Строка в нашем формате (целый диалог или промпт, кончающийся '-> Bot:') -> id в чат-шаблоне."""
        ids: list[int] = []
        if self.system is not None and self.system_text:
            ids += [self.system] + self.sp.encode(self.system_text) + [self.eot]
        for seg in text.split(SEG_SEP):
            m = SEG_RE.match(seg.strip())
            if not m:                                    # обрывок без формата — просто текст реплики
                ids += [self.user] + self.sp.encode(seg) + [self.eot]
                continue
            kind, head, reply = m.groups()
            ids += [self.user] + self.sp.encode(head if kind == "User" else f"Tool: {head}") + [self.eot] + [self.assistant]
            if reply is not None and reply.strip():
                ids += self.sp.encode(reply.strip()) + [self.eot]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i for i in ids if i not in (self.eot, self.user, self.assistant, self.system)])

    @property
    def stop_ids(self) -> set:
        return {i for i in (self.eot, self.user, self.assistant, self.system) if i is not None}
