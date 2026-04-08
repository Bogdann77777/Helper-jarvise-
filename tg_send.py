"""
tg_send.py — инструмент для отправки файлов и сообщений пользователю через Telegram.

Использование из кода:
    from tg_send import tg_msg, tg_file, tg_photo

Использование из командной строки:
    python tg_send.py "текст сообщения"
    python tg_send.py path/to/file.pdf
    python tg_send.py path/to/image.png
"""

import sys
import os
import mimetypes
import requests
from pathlib import Path

# Конфигурация
def _load_config():
    try:
        from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        return TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    except Exception:
        pass
    # Fallback: read .env directly (no dotenv dependency)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))
    env_file = Path(__file__).parent / ".env"
    if env_file.exists() and (not token or not chat_id):
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("TELEGRAM_BOT_TOKEN=") and not token:
                token = line.split("=", 1)[1].strip()
            elif line.startswith("TELEGRAM_CHAT_ID=") and not chat_id:
                try:
                    chat_id = int(line.split("=", 1)[1].strip())
                except ValueError:
                    pass
    return token, chat_id

TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID = _load_config()

API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def tg_msg(text: str, silent: bool = False) -> bool:
    """Отправить текстовое сообщение."""
    r = requests.post(f"{API}/sendMessage", json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_notification": silent,
    }, timeout=10)
    return r.ok


def tg_file(path: str | Path, caption: str = "") -> bool:
    """Отправить любой файл (PDF, TXT, ZIP, …)."""
    path = Path(path)
    if not path.exists():
        print(f"[tg_send] Файл не найден: {path}")
        return False
    with open(path, "rb") as f:
        r = requests.post(f"{API}/sendDocument", data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption or path.name,
            "parse_mode": "HTML",
        }, files={"document": (path.name, f)}, timeout=60)
    return r.ok


def tg_photo(path: str | Path, caption: str = "") -> bool:
    """Отправить изображение (JPG, PNG, …)."""
    path = Path(path)
    if not path.exists():
        print(f"[tg_send] Файл не найден: {path}")
        return False
    with open(path, "rb") as f:
        r = requests.post(f"{API}/sendPhoto", data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption or path.name,
            "parse_mode": "HTML",
        }, files={"photo": (path.name, f)}, timeout=60)
    return r.ok


def tg_video(path: str | Path, caption: str = "") -> bool:
    """Отправить видео (MP4, …)."""
    path = Path(path)
    if not path.exists():
        print(f"[tg_send] Файл не найден: {path}")
        return False
    with open(path, "rb") as f:
        r = requests.post(f"{API}/sendVideo", data={
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": caption or path.name,
            "parse_mode": "HTML",
        }, files={"video": (path.name, f)}, timeout=120)
    return r.ok


def tg_auto(path: str | Path, caption: str = "") -> bool:
    """Автоопределение типа и отправка."""
    path = Path(path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        if mime.startswith("image/"):
            return tg_photo(path, caption)
        if mime.startswith("video/"):
            return tg_video(path, caption)
    return tg_file(path, caption)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tg_send.py <text_or_filepath> [caption]")
        sys.exit(1)

    arg = sys.argv[1]
    caption = sys.argv[2] if len(sys.argv) > 2 else ""

    p = Path(arg)
    if p.exists():
        ok = tg_auto(p, caption)
        print("✅ Отправлено" if ok else "❌ Ошибка отправки")
    else:
        ok = tg_msg(arg)
        print("✅ Отправлено" if ok else "❌ Ошибка отправки")
