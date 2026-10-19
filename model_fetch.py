# -*- coding: utf-8 -*-
"""Скачивание папки модели (yuqi_model.zip из GitHub Releases), если её ещё нет.
Используется консольной версией (chat_cli.py) и ботом на хостинге (main.py, переменная MODEL_URL)."""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile

DEFAULT_URL = "https://github.com/ZedgenizovaRita/Restaurant_chat_bot/releases/download/v2.0/yuqi_model.zip"


def has_model(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def download_model(url: str, dest: str, progress: bool = True) -> str:
    """Скачивает zip по ссылке и кладёт папку модели (ту, где лежит config.json) в dest."""
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "model.zip")
    last = [-1]

    def hook(n, bs, total):
        if not progress:
            return
        done = n * bs if total <= 0 else min(n * bs, total)
        mb = int(done / 2**20)
        if mb != last[0]:                                # не чаще раза на мегабайт
            last[0] = mb
            print(f"\r  скачано {mb} МБ" + (f" из {total / 2**20:.0f}" if total > 0 else ""), end="", flush=True)
    try:
        urllib.request.urlretrieve(url, zpath, hook)
        if progress:
            print("\n  распаковываю...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(tmp, "m"))
        src = next((r for r, _, files in os.walk(os.path.join(tmp, "m")) if "config.json" in files), None)
        if not src:
            raise RuntimeError("в архиве не найдена модель (нет config.json)")
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        shutil.move(src, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest
