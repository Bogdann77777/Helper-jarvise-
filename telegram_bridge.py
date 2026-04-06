"""
telegram_bridge.py — Telegram → CLI Helper file bridge

Принимает файлы (фото, видео, PDF, аудио, документы) от пользователя в Telegram,
сохраняет в static/uploads/ и уведомляет сервер через POST /telegram-file,
который транслирует событие по WebSocket в браузерный UI.

Запуск: python telegram_bridge.py
Или через: start_telegram.bat
"""

import asyncio
import json
import logging
import mimetypes
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# ── Конфигурация ───────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Допустимые chat_id — только владелец бота
try:
    from config import TELEGRAM_CHAT_ID as _OWNER_CHAT_ID
    ALLOWED_CHAT_IDS: list[int] = [_OWNER_CHAT_ID]
except ImportError:
    ALLOWED_CHAT_IDS: list[int] = []

SERVER_URL = os.environ.get("CLI_SERVER_URL", "http://localhost:8000")

UPLOADS_DIR = Path(__file__).parent / "static" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Очередь входящих текстовых сообщений от пользователя
TG_INBOX = Path(__file__).parent / "tg_inbox.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("telegram_bridge")


# ── Вспомогательные функции ───────────────────────────────────────────────────

def is_allowed(chat_id: int) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True  # открытый режим
    return chat_id in ALLOWED_CHAT_IDS


def make_filename(prefix: str, original: str | None) -> str:
    """Генерирует уникальное имя файла с временной меткой."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if original:
        # Оставляем только безопасные символы
        safe = "".join(c for c in original if c.isalnum() or c in "._-")
        return f"tg_{ts}_{safe}"
    ext = mimetypes.guess_extension(prefix) or ""
    return f"tg_{ts}{ext}"


def file_type_emoji(file_type: str) -> str:
    return {"photo": "🖼️", "video": "🎬", "audio": "🎵", "voice": "🎤", "document": "📄"}.get(file_type, "📎")


def inbox_append(text: str):
    """Сохраняет входящее текстовое сообщение в очередь tg_inbox.json."""
    msgs = []
    if TG_INBOX.exists():
        try:
            msgs = json.loads(TG_INBOX.read_text(encoding="utf-8"))
        except Exception:
            msgs = []
    msgs.append({"ts": datetime.now().isoformat(), "text": text})
    TG_INBOX.write_text(json.dumps(msgs, ensure_ascii=False, indent=2), encoding="utf-8")


async def notify_server(filename: str, file_type: str, size: int):
    """Уведомляет CLI-сервер о новом файле (non-blocking)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{SERVER_URL}/telegram-file",
                json={"filename": filename, "file_type": file_type, "size": size},
            )
    except Exception as e:
        logger.warning(f"Не удалось уведомить сервер: {e}")


async def notify_server_text(text: str):
    """Уведомляет CLI-сервер о входящем текстовом сообщении."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                f"{SERVER_URL}/telegram-message",
                json={"text": text},
            )
    except Exception as e:
        logger.warning(f"Не удалось отправить текст на сервер: {e}")


async def download_and_save(tg_file, filename: str) -> int:
    """Скачивает файл из Telegram и сохраняет на диск. Возвращает размер в байтах."""
    dest = UPLOADS_DIR / filename
    content = await tg_file.download_as_bytearray()
    dest.write_bytes(content)
    return len(content)


# ── Обработчики ───────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    await update.message.reply_text(
        "Привет! Отправь мне фото, видео, PDF или любой файл — "
        "я сохраню его и уведомлю твой CLI Helper.\n\n"
        "Поддерживаемые типы: фото, видео, аудио, голосовые, документы (PDF, TXT, …)"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    files = sorted(UPLOADS_DIR.glob("tg_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        await update.message.reply_text("Папка uploads пуста.")
        return
    lines = ["Последние файлы из Telegram:"]
    for f in files[:10]:
        size_kb = f.stat().st_size // 1024
        lines.append(f"• {f.name}  ({size_kb} KB)")
    await update.message.reply_text("\n".join(lines))


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    photo = update.message.photo[-1]  # наибольшее разрешение
    tg_file = await photo.get_file()
    filename = make_filename("photo", f"{photo.file_unique_id}.jpg")
    size = await download_and_save(tg_file, filename)
    await notify_server(filename, "photo", size)
    await update.message.reply_text(
        f"🖼️ Фото сохранено\n`{filename}`\n\nРазмер: {size // 1024} KB",
        parse_mode="Markdown",
    )
    logger.info(f"Photo saved: {filename} ({size} bytes)")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    video = update.message.video
    tg_file = await video.get_file()
    original = video.file_name or f"{video.file_unique_id}.mp4"
    filename = make_filename("video", original)
    size = await download_and_save(tg_file, filename)
    await notify_server(filename, "video", size)
    await update.message.reply_text(
        f"🎬 Видео сохранено\n`{filename}`\n\nРазмер: {size // 1024} KB",
        parse_mode="Markdown",
    )
    logger.info(f"Video saved: {filename} ({size} bytes)")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    doc = update.message.document
    tg_file = await doc.get_file()
    filename = make_filename("document", doc.file_name or doc.file_unique_id)
    size = await download_and_save(tg_file, filename)

    # Определяем тип по mime
    file_type = "document"
    if doc.mime_type:
        if "pdf" in doc.mime_type:
            file_type = "pdf"
        elif "image" in doc.mime_type:
            file_type = "photo"

    await notify_server(filename, file_type, size)
    emoji = "📄" if file_type == "document" else ("📕" if file_type == "pdf" else "🖼️")
    await update.message.reply_text(
        f"{emoji} Документ сохранён\n`{filename}`\n\nРазмер: {size // 1024} KB",
        parse_mode="Markdown",
    )
    logger.info(f"Document saved: {filename} ({size} bytes)")


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    audio = update.message.audio
    tg_file = await audio.get_file()
    original = audio.file_name or f"{audio.file_unique_id}.mp3"
    filename = make_filename("audio", original)
    size = await download_and_save(tg_file, filename)
    await notify_server(filename, "audio", size)
    await update.message.reply_text(
        f"🎵 Аудио сохранено\n`{filename}`\n\nРазмер: {size // 1024} KB",
        parse_mode="Markdown",
    )
    logger.info(f"Audio saved: {filename} ({size} bytes)")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    voice = update.message.voice
    tg_file = await voice.get_file()
    filename = make_filename("voice", f"{voice.file_unique_id}.ogg")
    size = await download_and_save(tg_file, filename)
    await notify_server(filename, "voice", size)
    await update.message.reply_text(
        f"🎤 Голосовое сохранено\n`{filename}`\n\nРазмер: {size // 1024} KB",
        parse_mode="Markdown",
    )
    logger.info(f"Voice saved: {filename} ({size} bytes)")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_chat.id):
        return
    text = update.message.text or ""
    if not text:
        return
    # Сохраняем в локальную очередь
    inbox_append(text)
    # Уведомляем сервер → WebSocket → UI
    await notify_server_text(text)
    await update.message.reply_text("✅ Получено, добавлено в очередь.", parse_mode="Markdown")
    logger.info(f"Text queued: {text[:80]}")


# ── Запуск ────────────────────────────────────────────────────────────────────

def main():
    token = BOT_TOKEN
    if not token:
        print("ERROR: BOT_TOKEN не задан.")
        print("Установи переменную окружения TELEGRAM_BOT_TOKEN или пропиши токен в telegram_bridge.py")
        sys.exit(1)

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info(f"Telegram Bridge запущен. Uploads → {UPLOADS_DIR}")
    logger.info(f"Server notify URL: {SERVER_URL}/telegram-file")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
