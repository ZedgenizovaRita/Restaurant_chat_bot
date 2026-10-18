# Запуск бота

```bash
pip install -r requirements.txt

# 1) проверка в консоли (без Telegram)
python chat_cli.py --model out-rugpt

# 2) Telegram
set BOT_TOKEN=...  # от @BotFather (без кавычек)
set MODEL_DIR=out-rugpt
set ALLOW_RESET=1 # с поддержкой функции /reset - очищает историю чата
python main.py
```