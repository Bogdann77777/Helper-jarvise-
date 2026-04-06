"""
bot.py — Telegram bot for personal task manager.

Commands (send in Telegram):
  Произвольный текст    → Claude Haiku парсит задачи и сохраняет
  список                → все задачи по всем проектам
  по 76                 → задачи проекта 76
  сегодня               → сегодняшний отчёт
  сделано 42            → закрыть задачу #42
  удали 42              → удалить задачу #42
  отпуск 1-15 мая       → vacation mode
  отпуск список         → показать отпуска
  помощь / /help        → список команд

Run: python -m task_manager.bot
"""
import asyncio
import logging
import os
import re
import sys
import json
from datetime import date
from pathlib import Path

# Add cli helper root to path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from openai import AsyncOpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from telegram.constants import ParseMode

from task_manager.db import (
    init_db, add_task, get_tasks, get_todays_report_tasks,
    mark_done, delete_task, set_vacation, list_vacations,
    cancel_vacation, is_vacation_today, get_task,
)
from task_manager.date_parser import parse_date, is_urgent, date_to_iso
from task_manager.formatter import (
    morning_report, project_report, full_list_report,
    task_added_confirm,
)

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID   = int(os.environ.get("TELEGRAM_CHAT_ID", "0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "logs" / "task_bot.log",
                            encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("task_bot")

# ── OpenRouter client for task parsing ────────────────────────────────────────
try:
    from config import OPENROUTER_API_KEY
except ImportError:
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

_llm = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)
_HAIKU_MODEL = "anthropic/claude-haiku-4-5"

_EXTRACT_TOOL = {
    "name": "save_tasks",
    "description": "Сохранить задачи из сообщения пользователя",
    "input_schema": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "project":  {"type": "string",
                                     "description": "Номер проекта, например '76' или '331'"},
                        "text":     {"type": "string",
                                     "description": "Текст задачи (оригинальный язык)"},
                        "date_str": {"type": ["string", "null"],
                                     "description": "Дата как сказал пользователь: 'до 15 апреля', 'следующей неделе'. null если нет даты"},
                        "priority": {"type": "string", "enum": ["high", "normal"],
                                     "description": "'high' если срочно/urgent, иначе 'normal'"},
                    },
                    "required": ["project", "text", "date_str", "priority"],
                }
            }
        },
        "required": ["tasks"],
    },
}

_SYSTEM_PROMPT = """\
Ты помощник строительного прораба. Из сообщения пользователя извлекай задачи и вызывай save_tasks.
Каждая отдельная задача — отдельный объект. Проект = числовой код ('76', '331' и т.д.).
Если проект не упомянут — используй последний упомянутый в сообщении.
date_str — сохраняй точно как сказал пользователь, не переводи в дату.
Вызывай ТОЛЬКО инструмент, без пояснений."""


async def _parse_tasks_from_text(text: str) -> list[dict]:
    """Use Claude Haiku via OpenRouter to extract structured tasks."""
    try:
        response = await _llm.chat.completions.create(
            model=_HAIKU_MODEL,
            max_tokens=1024,
            tools=[{"type": "function", "function": {
                "name": _EXTRACT_TOOL["name"],
                "description": _EXTRACT_TOOL["description"],
                "parameters": _EXTRACT_TOOL["input_schema"],
            }}],
            tool_choice={"type": "function", "function": {"name": "save_tasks"}},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
        )
        for choice in response.choices:
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    if tc.function.name == "save_tasks":
                        data = json.loads(tc.function.arguments)
                        return data.get("tasks", [])
    except Exception as e:
        logger.error(f"LLM parse error: {e}")
    return []


# ── Regex patterns ─────────────────────────────────────────────────────────────
_RE_DONE   = re.compile(r"^(сделано|выполнено|done|готово)\s+(\d+)$", re.I)
_RE_DELETE = re.compile(r"^(удали|удалить|delete|убери)\s+(\d+)$", re.I)
_RE_PROJECT = re.compile(r"^(по|проект)\s+(\S+)$", re.I)
_RE_VACATION_SET = re.compile(
    r"^отпуск\s+с?\s*(\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?|\d{1,2}\s+\S+)\s*"
    r"(?:по|до|-|—)\s*(\d{1,2}[.\-/]\d{1,2}(?:[.\-/]\d{2,4})?|\d{1,2}\s+\S+)",
    re.I | re.UNICODE,
)
_RE_VACATION_CANCEL = re.compile(r"^отпуск\s+(отмена|отменить|cancel)\s*(\d+)?$", re.I)
_RE_VACATION_LIST   = re.compile(r"^отпуск\s+(список|list)$", re.I)
_TODAY_WORDS = {"сегодня", "today", "report", "отчёт", "отчет"}
_LIST_WORDS  = {"список", "все", "list", "all", "задачи"}
_HELP_WORDS  = {"помощь", "help", "/help", "команды"}


def _is_authorized(update: Update) -> bool:
    return update.effective_chat.id == TELEGRAM_CHAT_ID


async def _reply(update: Update, text: str):
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


# ── Handlers ──────────────────────────────────────────────────────────────────

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return

    text = (update.message.text or "").strip()
    low  = text.lower().strip()

    # ── /help ─────────────────────────────────────────────────────────────
    if low in _HELP_WORDS or low.startswith("/help"):
        await _reply(update, HELP_TEXT)
        return

    # ── сегодня ───────────────────────────────────────────────────────────
    if low in _TODAY_WORDS:
        tasks = get_todays_report_tasks()
        await _reply(update, morning_report(tasks))
        return

    # ── список / все ──────────────────────────────────────────────────────
    if low in _LIST_WORDS:
        tasks = get_tasks()
        await _reply(update, full_list_report(tasks))
        return

    # ── по 76 ─────────────────────────────────────────────────────────────
    m = _RE_PROJECT.match(low)
    if m:
        proj = m.group(2).upper()
        tasks = get_tasks(project=proj)
        await _reply(update, project_report(proj, tasks))
        return

    # ── сделано 42 ────────────────────────────────────────────────────────
    m = _RE_DONE.match(low)
    if m:
        tid = int(m.group(2))
        task = get_task(tid)
        if not task:
            await _reply(update, f"❌ Задача #{tid} не найдена")
            return
        mark_done(tid)
        await _reply(update, f"✅ Закрыто: <b>[{task['project']}]</b> {task['text']} <code>#{tid}</code>")
        return

    # ── удали 42 ──────────────────────────────────────────────────────────
    m = _RE_DELETE.match(low)
    if m:
        tid = int(m.group(2))
        task = get_task(tid)
        if not task:
            await _reply(update, f"❌ Задача #{tid} не найдена")
            return
        delete_task(tid)
        await _reply(update, f"🗑 Удалено: <b>[{task['project']}]</b> {task['text']}")
        return

    # ── отпуск список ─────────────────────────────────────────────────────
    if _RE_VACATION_LIST.match(low):
        vacs = list_vacations()
        if not vacs:
            await _reply(update, "📅 Нет запланированных отпусков")
        else:
            lines = ["📅 <b>Отпуска:</b>"]
            for v in vacs:
                lines.append(f"  #{v['id']} {v['start_date']} — {v['end_date']}  {v['note']}")
            await _reply(update, "\n".join(lines))
        return

    # ── отпуск отмена ─────────────────────────────────────────────────────
    m = _RE_VACATION_CANCEL.match(low)
    if m:
        vid_str = m.group(2)
        if vid_str:
            cancel_vacation(int(vid_str))
            await _reply(update, f"✅ Отпуск #{vid_str} отменён")
        else:
            vacs = list_vacations()
            if vacs:
                cancel_vacation(vacs[0]["id"])
                await _reply(update, "✅ Ближайший отпуск отменён")
            else:
                await _reply(update, "Нет отпусков для отмены")
        return

    # ── отпуск с X по Y ───────────────────────────────────────────────────
    if low.startswith("отпуск"):
        m = _RE_VACATION_SET.match(text)
        if m:
            d1 = parse_date(m.group(1))
            d2 = parse_date(m.group(2))
            if d1 and d2:
                vid = set_vacation(date_to_iso(d1), date_to_iso(d2))
                await _reply(update,
                    f"🏖 Отпуск #{vid} записан: {d1.strftime('%d.%m')} — {d2.strftime('%d.%m.%Y')}\n"
                    f"В эти дни уведомления приходить не будут.")
                return
        await _reply(update, "Не понял даты отпуска. Пример:\n<code>отпуск с 1 мая по 10 мая</code>")
        return

    # ── Всё остальное → парсим как задачи через Claude ────────────────────
    await update.message.reply_text("⏳ Записываю...")
    raw_tasks = await _parse_tasks_from_text(text)

    if not raw_tasks:
        await _reply(update,
            "🤔 Не понял. Попробуй:\n"
            "<code>по 76: договор с Ивановым до 20 апреля, срочно</code>\n"
            "или напиши <code>помощь</code>")
        return

    added = []
    for t in raw_tasks:
        due_date = None
        if t.get("date_str"):
            d = parse_date(t["date_str"])
            if d:
                due_date = date_to_iso(d)
        priority = t.get("priority", "normal")
        if is_urgent(t.get("text", "")):
            priority = "high"
        tid = add_task(t["project"], t["text"], due_date, priority)
        added.append({**t, "id": tid, "due_date": due_date, "priority": priority})

    await _reply(update, task_added_confirm(added))


async def handle_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _is_authorized(update):
        return
    await _reply(update, "👋 Привет! Я твой личный планировщик.\n\nНапиши <code>помощь</code> для списка команд.")


HELP_TEXT = """\
📋 <b>Команды планировщика</b>

<b>Добавить задачи</b> — просто напиши:
<code>по 76: договор с Ивановым до 20 апреля, срочно заказать материалы</code>

<b>Посмотреть</b>
<code>сегодня</code> — отчёт на сегодня
<code>список</code> — все открытые задачи
<code>по 76</code> — задачи проекта 76

<b>Закрыть / удалить</b>
<code>сделано 42</code> — закрыть задачу #42
<code>удали 42</code> — удалить задачу #42

<b>Отпуск</b>
<code>отпуск с 1 мая по 10 мая</code> — нет уведомлений в эти дни
<code>отпуск список</code> — показать отпуска
<code>отпуск отмена</code> — отменить ближайший

<i>ID задачи указан в конце строки как #42</i>"""


def main():
    init_db()

    # Ensure logs dir exists
    Path(__file__).parent.parent.joinpath("logs").mkdir(exist_ok=True)

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("help",  handle_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Task bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
