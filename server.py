# server.py — FastAPI сервер, WebSocket, три режима (Live / Step / Board)

import asyncio
import json
import logging
import logging.handlers
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from typing import List, Optional
from config import SERVER_HOST, SERVER_PORT, AUDIO_DIR, UPLOADS_DIR, LOG_LEVEL, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, OPENROUTER_API_KEY, MEMORY_ENABLED, AGENT_MODE_ENABLED, AGENT_SYSTEM_PROMPT
from qwen3tts_client import get_qwen3tts_client, QWEN_SPEAKERS
from xtts_manager import get_xtts_manager
import edge_tts_ru
from stt_manager import get_stt_manager
from cli_executor import run_claude_streaming, new_session as cli_new_session, run_claude_agent_streaming, new_agent_session, cancel_agent, inject_user_message, remove_client, restore_agent_session
from board.orchestrator import run_board_session, run_followup, get_session
from board.data_gate import validate_gate_input, DataGateError
from board.memory_store import get_memory_store

# --- Логирование ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)


# --- WebSocket менеджер ---
class ConnectionManager:
    BUFFER_MAX = 300  # сообщений на сессию
    BUFFER_FILE = Path("buffer_store.json")
    BUFFER_MAX_AGE_DAYS = 7  # удалять буферы сессий старше N дней

    def __init__(self):
        self.active: dict[str, WebSocket] = {}
        self.buffer: dict[str, deque] = {}   # ws_id → последние N сообщений
        self.known_ids: set = set()           # ws_ids которые хотя бы раз подключались
        self._last_seen: dict[str, float] = {}  # ws_id → timestamp последнего подключения
        self._save_pending = False
        self._load_buffer()

    def _load_buffer(self):
        """Загрузить буфер с диска при старте сервера. Удаляет старые сессии."""
        try:
            if self.BUFFER_FILE.exists():
                data = json.loads(self.BUFFER_FILE.read_text(encoding="utf-8"))
                import time as _time
                cutoff = _time.time() - self.BUFFER_MAX_AGE_DAYS * 86400
                for ws_id, entry in data.items():
                    # Поддержка нового формата {msgs, last_seen} и старого [...]
                    if isinstance(entry, dict):
                        msgs = entry.get("msgs", [])
                        last_seen = entry.get("last_seen", 0)
                    else:
                        msgs = entry
                        last_seen = 0
                    if last_seen < cutoff:
                        continue  # пропускаем старые сессии
                    self.buffer[ws_id] = deque(msgs[-self.BUFFER_MAX:], maxlen=self.BUFFER_MAX)
                    self.known_ids.add(ws_id)
                    self._last_seen[ws_id] = last_seen
                logger.info(f"Буфер загружен с диска: {len(self.buffer)} сессий")
        except Exception as e:
            logger.warning(f"Ошибка загрузки буфера: {e}")

    def _save_buffer(self):
        """Сохранить буфер на диск (с debounce через asyncio.call_later)."""
        if self._save_pending:
            return  # уже запланировано сохранение
        self._save_pending = True
        try:
            loop = asyncio.get_running_loop()
            loop.call_later(1.0, self._do_save_buffer)
        except RuntimeError:
            # Нет активного event loop (напр. при загрузке) — сохраняем сразу
            self._do_save_buffer()

    def _do_save_buffer(self):
        """Реальная запись буфера на диск."""
        self._save_pending = False
        import time as _time
        try:
            data = {
                ws_id: {"msgs": list(msgs), "last_seen": self._last_seen.get(ws_id, 0)}
                for ws_id, msgs in self.buffer.items()
            }
            self.BUFFER_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Ошибка сохранения буфера: {e}")

    async def connect(self, ws_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active[ws_id] = websocket
        is_reconnect = ws_id in self.known_ids
        self.known_ids.add(ws_id)
        import time as _time
        self._last_seen[ws_id] = _time.time()
        logger.info(f"WebSocket {'переподключён' if is_reconnect else 'подключён'}: {ws_id[:8]}")

        # При переподключении — отдаём буфер пропущенных сообщений
        if is_reconnect:
            buffered = list(self.buffer.get(ws_id, []))
            if buffered:
                logger.info(f"Replay {len(buffered)} сообщений → {ws_id[:8]}")
                try:
                    await websocket.send_json({"type": "reconnect_replay_start", "count": len(buffered)})
                    for msg in buffered:
                        await websocket.send_json(msg)
                except Exception as e:
                    logger.warning(f"Replay ошибка {ws_id[:8]}: {e}")
                finally:
                    try:
                        await websocket.send_json({"type": "reconnect_replay_end"})
                    except Exception:
                        pass
                # After replay, always send full chat history so client can restore context
                history = chat_log.recent(300)
                if history:
                    try:
                        await websocket.send_json({"type": "chat_history", "msgs": history})
                    except Exception:
                        pass
            else:
                await websocket.send_json({"type": "reconnect_no_buffer"})
                history = chat_log.recent(300)
                if history:
                    await websocket.send_json({"type": "chat_history", "msgs": history})
        else:
            # Brand new connection — send chat history if available
            history = chat_log.recent(300)
            if history:
                await websocket.send_json({"type": "chat_history", "msgs": history})

    def disconnect(self, ws_id: str):
        self.active.pop(ws_id, None)
        # Буфер и known_ids НЕ чистим — нужны для replay при следующем подключении
        logger.info(f"WebSocket отключён: {ws_id[:8]}")

    # Типы сообщений которые НЕ нужно буферизировать
    # chat_history нельзя буферизировать — при replay старая история перетирает DOM
    _NO_BUFFER_TYPES = frozenset({
        "tts_chunk", "tts_done", "status_update",
        "chat_history", "reconnect_replay_start", "reconnect_replay_end", "reconnect_no_buffer",
    })

    def _buffer(self, ws_id: str, data: dict):
        if data.get("type") in self._NO_BUFFER_TYPES:
            return  # TTS аудио и статусы не буферизируем — не должны играть при реконнекте
        if ws_id not in self.buffer:
            self.buffer[ws_id] = deque(maxlen=self.BUFFER_MAX)
        self.buffer[ws_id].append(data)
        self._save_buffer()

    async def send(self, ws_id: str, data: dict):
        self._buffer(ws_id, data)  # буферизируем (с фильтром)
        ws = self.active.get(ws_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as e:
                logger.warning(f"Ошибка отправки WebSocket {ws_id[:8]}: {e}")

    async def broadcast(self, data: dict):
        for ws in list(self.active.values()):
            try:
                await ws.send_json(data)
            except Exception:
                pass


manager = ConnectionManager()


# --- Chat Log (persistent chat history across browser restarts) ---
class ChatLog:
    LOG_FILE = Path("chat_log.json")
    MAX_MESSAGES = 300

    def __init__(self):
        self._msgs: list = []
        self._load()

    def _load(self):
        try:
            if self.LOG_FILE.exists():
                data = json.loads(self.LOG_FILE.read_text(encoding="utf-8"))
                self._msgs = data.get("msgs", [])[-self.MAX_MESSAGES:]
        except Exception:
            self._msgs = []

    def add(self, role: str, text: str):
        if not text or not text.strip():
            return
        self._msgs.append({"role": role, "text": text.strip()})
        if len(self._msgs) > self.MAX_MESSAGES:
            self._msgs = self._msgs[-self.MAX_MESSAGES:]
        try:
            self.LOG_FILE.write_text(
                json.dumps({"msgs": self._msgs}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"ChatLog save error: {e}")

    def recent(self, n: int = 50) -> list:
        return self._msgs[-n:]


chat_log = ChatLog()


# --- Startup: загрузка моделей ---
def _restore_provider_from_settings():
    """Restore CLAUDE_PROVIDER and OPENROUTER_MODEL_ID from settings.json on startup.

    Without this, after a server restart config.CLAUDE_PROVIDER resets to
    "anthropic" and cli_executor passes --model claude-sonnet-4-6, which
    overrides ANTHROPIC_MODEL env var → selected OpenRouter model is ignored.
    """
    import json as _json
    import pathlib as _pathlib
    settings_path = _pathlib.Path.home() / ".claude" / "settings.json"
    try:
        if not settings_path.exists():
            return
        settings = _json.loads(settings_path.read_text(encoding="utf-8"))
        env = settings.get("env", {})
        base_url = env.get("ANTHROPIC_BASE_URL", "")
        model_id  = env.get("ANTHROPIC_MODEL", "")
        if ("openrouter.ai" in base_url or "/proxy/or" in base_url) and model_id:
            config.CLAUDE_PROVIDER    = "openrouter"
            config.OPENROUTER_MODEL_ID = model_id
            # Migrate old direct OpenRouter URL to local proxy (fixes context-management-2025-06-27 error)
            proxy_base = f"http://127.0.0.1:{SERVER_PORT}/proxy/or"
            if "openrouter.ai" in base_url and base_url != proxy_base:
                env["ANTHROPIC_BASE_URL"] = proxy_base
                settings_path.write_text(_json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"Migrated OpenRouter URL from direct to proxy: {proxy_base}")
            logger.info(f"Restored OpenRouter provider from settings.json: {model_id}")
    except Exception as e:
        logger.warning(f"Could not restore provider from settings.json: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск сервера, загрузка моделей...")
    _restore_provider_from_settings()
    if getattr(config, "USE_REMOTE_AI", False):
        logger.info(f"Remote AI mode: STT={config.STT_REMOTE_URL} TTS={config.TTS_REMOTE_URL}")
        # No local model loading — STT/TTS served by remote 192.168.1.243
    else:
        get_stt_manager()
        # Запускаем Qwen3-TTS subprocess в фоне (модель грузится 30-120с)
        asyncio.create_task(get_qwen3tts_client().start())

    # Initialize memory store
    if MEMORY_ENABLED:
        try:
            await get_memory_store().initialize()
            logger.info("Board memory store initialized")
        except Exception as e:
            logger.warning(f"Memory store init failed (non-fatal): {e}")

    local_ip = _get_local_ip()
    logger.info(f"Сервер готов. Открыть на телефоне/ноутбуке: http://{local_ip}:{SERVER_PORT}")
    print(f"\n{'='*50}")
    print(f"  FastAPI server started!")
    print(f"  Local IP:  http://{local_ip}:{SERVER_PORT}")
    print(f"  Localhost: http://127.0.0.1:{SERVER_PORT}")
    print(f"{'='*50}\n")

    yield

    logger.info("Сервер останавливается")
    await get_qwen3tts_client().stop()


app = FastAPI(title="Claude Voice Interface", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # локальная сеть, без жёстких ограничений
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")


# --- Утилиты ---

_TEXT_EXTS = {'.txt', '.md', '.csv', '.json', '.py', '.js', '.ts', '.html', '.xml', '.yaml', '.yml', '.log', '.sql'}
_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.heic', '.heif'}


async def _save_attachment(file: UploadFile) -> Optional[str]:
    """Save uploaded file to UPLOADS_DIR. Return absolute path or None."""
    if file is None or not file.filename:
        return None
    content = await file.read()
    if not content:
        return None
    safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
    filepath = os.path.join(UPLOADS_DIR, safe_name)
    with open(filepath, "wb") as f:
        f.write(content)
    return os.path.abspath(filepath)


def _build_file_context(abs_path: str, filename: str, mime: str) -> str:
    """Build the text context block that tells Claude about the attached file."""
    ext = os.path.splitext(filename)[1].lower()
    lines = [
        f"\n\n[Пользователь приложил файл: {filename}]",
        f"[Абсолютный путь: {abs_path}]",
    ]
    if ext in _IMAGE_EXTS:
        lines.append("[Это изображение. Используй инструмент Read чтобы открыть и проанализировать файл.]")
    elif ext in _TEXT_EXTS:
        lines.append("[Текстовый файл. Используй инструмент Read для чтения содержимого.]")
    else:
        lines.append("[Используй инструмент Read или Bash для работы с этим файлом если нужно.]")
    return "\n".join(lines) + "\n"


# --- Эндпоинты ---

@app.get("/", response_class=HTMLResponse)
async def root():
    """Отдаём главную страницу."""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/speakers")
async def get_speakers():
    """Возвращает список доступных голосов: только Qwen3-TTS (XTTS отключён)."""
    client = get_qwen3tts_client()
    current = client.current_speaker
    return {"speakers": list(client.get_speakers()), "current": current}


@app.post("/api/speaker")
async def set_speaker(speaker: str = Form(...)):
    """Устанавливает активный голос (только Qwen3-TTS, XTTS отключён)."""
    global _active_tts_engine
    client = get_qwen3tts_client()
    _active_tts_engine = "qwen"
    if speaker in QWEN_SPEAKERS:
        client.set_speaker(speaker)
    else:
        # Неизвестный голос (например, старый XTTS из localStorage) — используем Ryan
        client.set_speaker("Ryan")
    return {"ok": True, "speaker": client.current_speaker, "engine": "qwen"}


@app.post("/api/set-tts-engine")
async def set_tts_engine(engine: str = Form(...)):
    """Переключает TTS движок: 'qwen', 'edge', 'xtts'."""
    global _active_tts_engine
    allowed = {"qwen", "edge", "xtts"}
    if engine not in allowed:
        return {"ok": False, "error": f"Unknown engine '{engine}'. Allowed: {allowed}"}
    _active_tts_engine = engine
    return {"ok": True, "engine": engine}


@app.post("/api/voice")
async def process_voice(
    audio: UploadFile = File(...),
    ws_id: str = Form(default=""),
    mode: str = Form(default="step"),  # "live" или "step"
    attachments: List[UploadFile] = File(default=[]),
):
    """
    Принимает аудио файл, запускает полный пайплайн:
    STT → Claude CLI → TTS (XTTS).
    Статус отправляется через WebSocket (если ws_id передан).
    """
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Новый голосовой запрос, режим={mode}, ws_id={ws_id}")

    audio_bytes = await audio.read()
    mime_type = audio.content_type or "audio/webm"

    async def status(msg: str, stage: str = "", progress: int = 0):
        logger.info(f"[{request_id}] {msg}")
        if ws_id:
            await manager.send(ws_id, {
                "type": "status_update",
                "stage": stage,
                "message": msg,
                "progress": progress,
                "request_id": request_id,
            })

    try:
        # Шаг 1: STT
        await status("Распознавание речи...", "stt", 20)
        stt = get_stt_manager()
        transcript = stt.transcribe(audio_bytes, mime_type)

        if not transcript:
            await status("Не удалось распознать речь", "error", 0)
            return {"ok": False, "error": "Речь не распознана"}

        await status(f"Распознано: {transcript[:80]}{'...' if len(transcript) > 80 else ''}", "stt_done", 35)

        # Отправляем транскрипт клиенту
        if ws_id:
            await manager.send(ws_id, {
                "type": "transcript",
                "text": transcript,
                "request_id": request_id,
            })

        # Шаг 1.5: Файл-вложения — сохраняем и добавляем контекст к запросу
        prompt = transcript
        for attachment in (attachments or []):
            if attachment and attachment.filename:
                abs_path = await _save_attachment(attachment)
                if abs_path:
                    prompt += _build_file_context(abs_path, attachment.filename, attachment.content_type or "")
                    await status(f"Файл получен: {attachment.filename}", "stt_done", 38)

        # Сохраняем user-сообщение сразу — до обработки, чтобы не потерять при краше сервера
        chat_log.add("user", transcript)

        # Шаг 2: Claude CLI (agent или простой режим)
        await status("Отправка в Claude CLI...", "cli_execution", 40)

        if AGENT_MODE_ENABLED:
            # Heartbeat task: send periodic status while agent runs so browser knows server is alive
            heartbeat_task = None
            if ws_id:
                async def _heartbeat():
                    n = 0
                    while True:
                        await asyncio.sleep(30)
                        n += 1
                        await manager.send(ws_id, {
                            "type": "status_update",
                            "stage": "working",
                            "message": f"⏳ Агент работает... ({n * 30}с)",
                            "progress": 50,
                            "request_id": request_id,
                        })
                heartbeat_task = asyncio.create_task(_heartbeat())
            try:
                response_text = await _run_agent_request(prompt, ws_id, request_id, status)
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
        else:
            response_text = await _run_simple_request(prompt, ws_id, request_id, status)

        # TTS is fired sentence-by-sentence inside _run_*_request via _TtsPipeline
        await status("Готово!", "complete", 100)

        # Сохраняем claude-ответ (user уже сохранён выше)
        if response_text:
            chat_log.add("claude", response_text)

        return {
            "ok": True,
            "transcript": transcript,
            "response_text": response_text,
            "audio_url": "",
            "request_id": request_id,
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Ошибка обработки: {error_msg}", exc_info=True)
        await status(f"Ошибка: {error_msg}", "error", 0)
        return {"ok": False, "error": error_msg}


@app.post("/api/text")
async def process_text(
    text: str = Form(...),
    ws_id: str = Form(default=""),
    mode: str = Form(default="step"),
    attachments: List[UploadFile] = File(default=[]),
):
    """
    Принимает текст напрямую (Step-режим, без STT).
    Запускает: Claude CLI → TTS (XTTS).
    """
    request_id = uuid.uuid4().hex[:8]
    logger.info(f"[{request_id}] Текстовый запрос, длина={len(text)}")

    async def status(msg: str, stage: str = "", progress: int = 0):
        logger.info(f"[{request_id}] {msg}")
        if ws_id:
            await manager.send(ws_id, {
                "type": "status_update",
                "stage": stage,
                "message": msg,
                "progress": progress,
                "request_id": request_id,
            })

    try:
        # Файл-вложения — сохраняем и добавляем контекст к запросу
        prompt = text
        for attachment in (attachments or []):
            if attachment and attachment.filename:
                abs_path = await _save_attachment(attachment)
                if abs_path:
                    prompt += _build_file_context(abs_path, attachment.filename, attachment.content_type or "")
                    await status(f"Файл получен: {attachment.filename}", "stt_done", 20)

        # Сохраняем user-сообщение сразу — до обработки, чтобы не потерять при краше сервера
        chat_log.add("user", text)

        if AGENT_MODE_ENABLED:
            heartbeat_task = None
            if ws_id:
                async def _heartbeat_text():
                    n = 0
                    while True:
                        await asyncio.sleep(30)
                        n += 1
                        await manager.send(ws_id, {
                            "type": "status_update",
                            "stage": "working",
                            "message": f"⏳ Агент работает... ({n * 30}с)",
                            "progress": 50,
                            "request_id": request_id,
                        })
                heartbeat_task = asyncio.create_task(_heartbeat_text())
            try:
                response_text = await _run_agent_request(prompt, ws_id, request_id, status)
            finally:
                if heartbeat_task:
                    heartbeat_task.cancel()
        else:
            response_text = await _run_simple_request(prompt, ws_id, request_id, status)

        # TTS is fired sentence-by-sentence inside _run_*_request via _TtsPipeline
        await status("Готово!", "complete", 100)

        # Сохраняем claude-ответ (user уже сохранён выше)
        if response_text:
            chat_log.add("claude", response_text)

        # Отправляем полный текст через WS — клиент использует его если HTTP fetch упал
        # (response_complete буферизируется и воспроизводится при реконнекте)
        if ws_id:
            await manager.send(ws_id, {
                "type": "response_complete",
                "response_text": response_text,
                "request_id": request_id,
            })

        return {
            "ok": True,
            "transcript": text,
            "response_text": response_text,
            "audio_url": "",
            "request_id": request_id,
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{request_id}] Ошибка: {error_msg}", exc_info=True)
        await status(f"Ошибка: {error_msg}", "error", 0)
        return {"ok": False, "error": error_msg}


@app.post("/api/cancel")
async def cancel_current_agent(ws_id: str = Form(...)):
    """Отменить текущую задачу агента для данного клиента."""
    was_running = await cancel_agent(ws_id)
    if was_running:
        logger.info(f"[{ws_id[:8]}] Агент отменён пользователем")
        if ws_id:
            await manager.send(ws_id, {"type": "agent_cancelled"})
    return {"ok": True, "was_running": was_running}


@app.post("/api/inject")
async def inject_message(ws_id: str = Form(...), text: str = Form(...)):
    """Inject a user message between tool calls of the running agent."""
    from cli_executor import _sessions
    state = _sessions.get(ws_id)
    is_running = state is not None and state.active_process is not None
    inject_user_message(ws_id, text)
    logger.info(f"[{ws_id[:8]}] Injection queued ({'agent running' if is_running else 'queued for next run'}): {text[:60]}")
    if ws_id:
        await manager.send(ws_id, {
            "type": "status_update",
            "stage": "inject_queued",
            "message": f"💬 Сообщение в очереди — доставлю между тулами",
            "progress": 50,
        })
    return {"ok": True, "queued": True, "agent_running": is_running}


@app.post("/api/stt")
async def stt_only(
    audio: UploadFile = File(...),
):
    """
    Только STT: принимает аудио, возвращает транскрипт.
    Используется в Step-режиме для голосового ввода в textarea.
    """
    audio_bytes = await audio.read()
    mime_type = audio.content_type or "audio/webm"
    try:
        stt = get_stt_manager()
        transcript = stt.transcribe(audio_bytes, mime_type)
        return {"ok": True, "transcript": transcript}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/tts")
async def synthesize_tts_fallback(text: str = Form(...)):
    """
    Fallback TTS — синхронная генерация для мобильных устройств.
    Используется когда WS-стриминг чанков не доходит до клиента.
    """
    try:
        clean = _strip_markdown(text)
        if getattr(config, "USE_REMOTE_AI", False):
            from remote_ai_client import remote_synthesize
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: remote_synthesize(clean, speaker="Ryan", language="Russian")
            )
            audio_url = result["url"]
        elif _active_tts_engine == "xtts":
            loop = asyncio.get_running_loop()
            xtts_mgr = get_xtts_manager()
            voice = _active_xtts_voice
            audio_url = await loop.run_in_executor(
                None, lambda: xtts_mgr.synthesize(clean, role=voice)
            )
        else:
            client = get_qwen3tts_client()
            result = await client.synthesize(clean)
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "TTS failed"))
            audio_url = result["url"]
        logger.info(f"[TTS fallback] generated: {audio_url}")
        return {"ok": True, "audio_url": audio_url}
    except Exception as e:
        logger.error(f"[TTS fallback] error: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/avatar/generate")
async def generate_avatar(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(default=None),
    num_steps: int = Form(default=20),
    guidance_scale: float = Form(default=4.5),
    audio_scale: float = Form(default=5.0),
):
    """
    Генерирует видео говорящего аватара (OmniAvatar).
    image: фото лица (jpg/png)
    audio: аудио речи (wav/mp3)
    Работает на GPU 1, не блокирует CLI helper на GPU 0.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from omni_avatar_tool import generate_avatar_video

    # Сохраняем загруженные файлы (Path + basename против path traversal)
    img_path = Path(UPLOADS_DIR) / f"avatar_img_{os.path.basename(image.filename)}"
    aud_path = Path(UPLOADS_DIR) / f"avatar_aud_{os.path.basename(audio.filename)}"
    with open(img_path, "wb") as f:
        f.write(await image.read())
    with open(aud_path, "wb") as f:
        f.write(await audio.read())

    result = generate_avatar_video(
        image_path=str(img_path),
        audio_path=str(aud_path),
        prompt=prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        audio_scale=audio_scale,
    )

    if not result["success"]:
        return JSONResponse(status_code=500, content=result)

    video_path = Path(result["video_path"])
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
    )


@app.get("/api/avatar/status")
async def avatar_status():
    """GPU статус и готовность OmniAvatar."""
    import subprocess as _sp
    smi = _sp.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    gpus = []
    for line in smi.stdout.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append({
                "index": parts[0],
                "memory_used": parts[1],
                "memory_free": parts[2],
                "utilization": parts[3],
            })
    return {"ok": True, "gpus": gpus, "omni_gpu": "1"}


@app.post("/api/flux/generate")
async def flux_generate(
    prompt: str = Form(...),
    steps: int = Form(default=8),
    resolution: str = Form(default="1024x1024"),
    seed: int = Form(default=-1),
):
    """
    Генерирует фото через FLUX.2 Dev 32B на двух GPU.
    ~2м50с на 1024x1024, 8 шагов. Результат отправляется в Telegram.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from flux_image_tool import generate_image

    task = generate_image(
        prompt=prompt,
        model="flux2_dev",
        steps=steps,
        resolution=resolution,
        seed=seed,
    )
    return {
        "ok": True,
        "pid": task["pid"],
        "log": task["log"],
        "message": f"FLUX.2 Dev запущен (PID {task['pid']}). Результат придёт в Telegram (~2м50с).",
    }


@app.get("/api/flux/status")
async def flux_status():
    """Последний лог FLUX генерации."""
    import glob as _glob
    logs = sorted(_glob.glob("F:/project/Wan2GP/logs/flux_*.log"), key=os.path.getmtime, reverse=True)
    if not logs:
        return {"ok": False, "message": "Логов нет"}
    with open(logs[0], "r", encoding="utf-8", errors="replace") as f:
        tail = f.read()[-800:]
    return {"ok": True, "log_file": logs[0], "tail": tail}


@app.post("/api/fluxklein/generate")
async def fluxklein9b_generate(
    prompt: str = Form(...),
    lora: str = Form(default="cheek_blowjob_3500"),
    lora_strength: float = Form(default=1.0),
    image_path: str = Form(default=None),
    denoising_strength: float = Form(default=0.75),
    steps: int = Form(default=4),
    resolution: str = Form(default="1024x1024"),
    seed: int = Form(default=-1),
):
    """
    Генерирует фото через FLUX.2 Klein 9B + LoRA.
    image_path — путь к референс фото для img2img (опционально).
    denoising_strength — 0.0=копия оригинала, 1.0=полная свобода (default: 0.75).
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from fluxklein9b_tool import generate_image

    task = generate_image(
        prompt=prompt,
        lora=lora if lora else None,
        lora_strength=lora_strength,
        image_path=image_path or None,
        denoising_strength=denoising_strength,
        steps=steps,
        resolution=resolution,
        seed=seed,
    )
    return {
        "ok": True,
        "pid": task["pid"],
        "log": task["log"],
        "message": f"Klein 9B запущен (PID {task['pid']}). Результат в Telegram (~4 шага).",
    }


@app.get("/api/fluxklein/loras")
async def fluxklein9b_loras():
    """Список доступных LoRA для Klein 9B."""
    lora_dir = Path("F:/project/Wan2GP/loras/flux2_klein_9b")
    if not lora_dir.exists():
        return {"ok": True, "loras": []}
    loras = [f.stem for f in lora_dir.glob("*.safetensors")]
    return {"ok": True, "loras": loras}


@app.get("/api/fluxklein/status")
async def fluxklein9b_status():
    """Последний лог Klein 9B генерации."""
    import glob as _glob
    logs = sorted(_glob.glob("F:/project/Wan2GP/logs/klein9b_*.log"), key=os.path.getmtime, reverse=True)
    if not logs:
        return {"ok": False, "message": "Логов нет"}
    with open(logs[0], "r", encoding="utf-8", errors="replace") as f:
        tail = f.read()[-800:]
    return {"ok": True, "log_file": logs[0], "tail": tail}


@app.post("/api/multitalk/generate")
async def multitalk_generate(
    image_path: str = Form(default=None),      # путь к готовому PNG (если есть)
    audio_path: str = Form(default=None),      # путь к готовому WAV (если есть)
    tts_text: str = Form(default=None),        # текст для TTS (если audio_path не передан)
    tts_speaker: str = Form(default="Vivian"), # голос: Vivian/Ryan/Aiden/Serena
    prompt: str = Form(default=""),            # промпт для MultiTalk
    image: UploadFile = File(default=None),    # или загрузить файл напрямую
    audio: UploadFile = File(default=None),
):
    """
    Генерирует говорящее видео через MultiTalk 2GPU.
    Настройки: FusionX 4 шага + audio_guide=1.0 + TeaCache=0.2 → ~20 мин.
    Результат отправляется в Telegram.
    """
    import sys as _sys, shutil as _shutil, json as _json, datetime as _dt, subprocess as _sp
    _sys.path.insert(0, str(Path(__file__).parent))

    MULTITALK_DIR = Path("F:/project/MultiTalk")
    PYTHON        = "F:/project/multitalk_env/Scripts/python.exe"
    OUTPUT_DIR    = Path(__file__).parent / "outputs" / "multitalk"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    custom_dir = MULTITALK_DIR / "examples" / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)

    # ── Сохраняем изображение ───────────────────────────────────────────────
    if image:
        img_dst = custom_dir / f"mt_img_{image.filename}"
        with open(img_dst, "wb") as f:
            f.write(await image.read())
    elif image_path:
        img_dst = custom_dir / Path(image_path).name
        _shutil.copy2(image_path, img_dst)
    else:
        # По умолчанию — последний FLUX кадр
        latest = max(
            list((Path(__file__).parent / "outputs" / "flux_images").glob("*.mp4")) +
            list((Path(__file__).parent / "outputs" / "flux_images").glob("*.png")),
            key=lambda f: f.stat().st_mtime, default=None
        )
        if latest is None:
            return {"ok": False, "error": "Нет изображения. Передай image или image_path."}
        if latest.suffix == ".mp4":
            img_dst = custom_dir / "mt_latest_frame.png"
            _sp.run(["ffmpeg", "-i", str(latest), "-frames:v", "1", "-update", "1",
                     str(img_dst), "-y"], capture_output=True)
        else:
            img_dst = latest

    # ── Аудио: загруженный файл или TTS ─────────────────────────────────────
    if audio:
        aud_dst = custom_dir / f"mt_audio_{audio.filename}"
        with open(aud_dst, "wb") as f:
            f.write(await audio.read())
    elif audio_path:
        aud_dst = custom_dir / Path(audio_path).name
        _shutil.copy2(audio_path, aud_dst)
    elif tts_text:
        from remote_ai_client import remote_synthesize
        aud_dst = custom_dir / "mt_tts.wav"
        remote_synthesize(text=tts_text, speaker=tts_speaker,
                          language="English", save_to=str(aud_dst))
    else:
        return {"ok": False, "error": "Нужен audio, audio_path или tts_text."}

    # ── Input JSON ───────────────────────────────────────────────────────────
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_data = {
        "prompt": prompt or (
            "A person speaking naturally to camera, realistic facial expressions, "
            "subtle head movement, photorealistic, warm lighting, medium shot."
        ),
        "cond_image": f"examples/custom/{img_dst.name}",
        "cond_audio": {"person1": f"examples/custom/{aud_dst.name}"},
    }
    json_path = MULTITALK_DIR / f"examples/mt_input_{ts}.json"
    json_path.write_text(_json.dumps(input_data, indent=2))

    save_file = str(OUTPUT_DIR / f"multitalk_{ts}")
    log_file  = OUTPUT_DIR / f"multitalk_{ts}.log"

    # ── Запуск generate_multitalk_2gpu.py (лучшие настройки) ────────────────
    fusionx = str(MULTITALK_DIR / "weights/MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors")
    cmd = [
        PYTHON, "generate_multitalk_2gpu.py",
        "--task", "multitalk-14B", "--size", "multitalk-480",
        "--ckpt_dir", "weights/Wan2.1-I2V-14B-480P",
        "--wav2vec_dir", "weights/chinese-wav2vec2-base",
        "--quant", "int8", "--quant_dir", "weights/MeiGen-MultiTalk",
        "--num_persistent_param_in_dit", "0",
        "--use_teacache", "--teacache_thresh", "0.2",
        "--mode", "streaming", "--sample_shift", "7",
        "--lora_dir", fusionx, "--lora_scale", "1.0",
        "--sample_steps", "4",
        "--sample_text_guide_scale", "1.0",
        "--sample_audio_guide_scale", "1.0",   # 1 forward pass → ~55s/step
        "--save_file", save_file,
        "--input_json", str(json_path),
    ]

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1",
        "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1",
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29510",
    })

    log_f = open(log_file, "w", encoding="utf-8")
    proc = _sp.Popen(
        cmd, cwd=str(MULTITALK_DIR),
        stdout=log_f, stderr=_sp.STDOUT, stdin=_sp.DEVNULL, env=env,
    )

    return {
        "ok": True,
        "pid": proc.pid,
        "log": str(log_file),
        "message": f"MultiTalk запущен PID {proc.pid}. ~20 мин. Результат в Telegram.",
    }


@app.get("/api/multitalk/status")
async def multitalk_status():
    """Последний лог MultiTalk генерации."""
    import glob as _glob
    logs = sorted(
        _glob.glob(str(Path(__file__).parent / "outputs/multitalk/multitalk_*.log")),
        key=os.path.getmtime, reverse=True
    )
    if not logs:
        return {"ok": False, "message": "Логов нет"}
    with open(logs[0], "r", encoding="utf-8", errors="replace") as f:
        tail = f.read()[-800:]
    return {"ok": True, "log_file": logs[0], "tail": tail}


@app.post("/api/heartmula/generate")
async def heartmula_generate(
    tags: str = Form(default="hip-hop,instrumental,boom bap,dark,hard beats,piano melody,808 bass,no vocals"),
    lyrics: str = Form(default=None),
    max_seconds: int = Form(default=180),
    cfg_scale: float = Form(default=1.5),
    output_name: str = Form(default=None),
):
    """
    Генерирует музыку через HeartMuLa 3B.
    ~4 мин на 3 мин трека. Результат отправляется в Telegram.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from heartmula_tool import generate_music

    task = generate_music(
        tags=tags,
        lyrics=lyrics,
        max_seconds=max_seconds,
        cfg_scale=cfg_scale,
        output_name=output_name,
    )
    return {
        "ok": True,
        "pid": task["pid"],
        "log": task["log"],
        "output_wav": task["output_wav"],
        "message": task["message"],
    }


@app.get("/api/heartmula/status")
async def heartmula_status():
    """Последний лог HeartMuLa генерации."""
    import glob as _glob
    logs = sorted(
        _glob.glob(str(Path(__file__).parent / "outputs/heartmula/*.log")),
        key=os.path.getmtime, reverse=True
    )
    if not logs:
        return {"ok": False, "message": "Логов нет"}
    with open(logs[0], "r", encoding="utf-8", errors="replace") as f:
        tail = f.read()[-800:]
    return {"ok": True, "log_file": logs[0], "tail": tail}


# ── Task Manager API ──────────────────────────────────────────────────────────

@app.get("/api/tasks")
async def api_tasks_list(project: str = None, include_done: bool = False):
    """List tasks. Optional ?project=Офис filter."""
    import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
    from task_manager.db import init_db as _idb, get_tasks as _gt
    _idb()
    tasks = _gt(project=project, include_done=include_done)
    return {"ok": True, "tasks": tasks}


@app.post("/api/tasks")
async def api_tasks_add(request: Request):
    """Add task. Body: {project, text, due_date?, priority?}"""
    from task_manager.db import init_db as _idb, add_task as _at
    _idb()
    body = await request.json()
    tid = _at(
        body["project"], body["text"],
        body.get("due_date"), body.get("priority", "normal")
    )
    return {"ok": True, "id": tid}


@app.post("/api/tasks/{task_id}/done")
async def api_tasks_done(task_id: int):
    """Mark task as done."""
    from task_manager.db import init_db as _idb, mark_done as _md
    _idb()
    ok = _md(task_id)
    return {"ok": ok}


@app.delete("/api/tasks/{task_id}")
async def api_tasks_delete(task_id: int):
    """Delete task."""
    from task_manager.db import init_db as _idb, delete_task as _dt
    _idb()
    ok = _dt(task_id)
    return {"ok": ok}


@app.patch("/api/tasks/{task_id}")
async def api_tasks_update(task_id: int, request: Request):
    """Update task fields: text, project, due_date, priority, today."""
    from task_manager.db import init_db as _idb, update_task as _ut, set_today as _st
    _idb()
    body = await request.json()
    # handle today flag separately
    if "today" in body:
        _st(task_id, bool(body.pop("today")))
    if body:
        _ut(task_id, **body)
    return {"ok": True}


@app.get("/api/tasks/today")
async def api_tasks_today():
    """Get today's plan."""
    from task_manager.db import init_db as _idb, get_today_plan as _gtp
    _idb()
    tasks = _gtp()
    return {"ok": True, "tasks": tasks}


@app.post("/api/tasks/today")
async def api_tasks_set_today(request: Request):
    """Set today's plan. Body: {ids: [1,2,3], replace: true}"""
    from task_manager.db import init_db as _idb, set_today as _st, clear_today_plan as _ctp
    _idb()
    body = await request.json()
    if body.get("replace", True):
        _ctp()
    ok_ids, fail_ids = [], []
    for tid in body.get("ids", []):
        if _st(int(tid), True):
            ok_ids.append(tid)
        else:
            fail_ids.append(tid)
    return {"ok": True, "set": ok_ids, "not_found": fail_ids}


OPENROUTER_MODELS = {
    "1": "deepseek/deepseek-r1-0528:free",          # DeepSeek R1 (free, reasoning)
    "2": "meta-llama/llama-3.3-70b-instruct:free",  # Llama 3.3 70B (free, chat)
    "3": "qwen/qwen3-coder:free",                   # Qwen3 Coder (free, code)
    "4": "minimax/minimax-m2.5:free",               # MiniMax M2.5 (free)
    "5": "deepseek/deepseek-v3.2-exp",              # DeepSeek v3.2 Exp (paid, code)
    "6": "google/gemini-2.5-flash-lite",            # Gemini 2.5 Flash Lite (paid, fast)
    "7": "x-ai/grok-4.1-fast",                     # Grok 4.1 Fast (paid, fast)
}

# Anthropic beta headers that OpenRouter doesn't support — strip these before forwarding
_OR_UNSUPPORTED_BETAS = {"context-management-2025-06-27"}

# Proxy base URL: Claude CLI → this server → OpenRouter (strips unsupported beta headers)
_OR_PROXY_BASE = f"http://127.0.0.1:{SERVER_PORT}/proxy/or"
_OR_TARGET_BASE = "https://openrouter.ai/api"


@app.api_route("/proxy/or/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def openrouter_proxy(path: str, request: Request):
    """
    Local proxy: strips Anthropic beta headers unsupported by OpenRouter
    (context-management-2025-06-27), then forwards to https://openrouter.ai/api.

    Claude CLI sends anthropic-beta: context-management-2025-06-27 automatically
    (no way to disable it via flag/env). OpenRouter returns 400 for non-Anthropic
    models. This proxy is the only reliable fix.
    """
    import httpx

    target_url = f"{_OR_TARGET_BASE}/{path}"
    if request.url.query:
        # Strip ?beta=true — Anthropic-specific param; OpenRouter ignores or rejects it
        filtered_query = "&".join(
            p for p in request.url.query.split("&") if not p.startswith("beta=")
        )
        if filtered_query:
            target_url += f"?{filtered_query}"

    # Build headers — strip unsupported betas, remove hop-by-hop headers
    headers = {}
    for name, value in request.headers.items():
        name_lower = name.lower()
        if name_lower == "anthropic-beta":
            betas = [b.strip() for b in value.split(",")]
            filtered = [b for b in betas if b not in _OR_UNSUPPORTED_BETAS]
            if filtered:
                headers["anthropic-beta"] = ", ".join(filtered)
            # else: skip header entirely (all betas were unsupported)
        elif name_lower in ("host", "content-length", "transfer-encoding", "connection"):
            pass  # skip — httpx sets these correctly
        else:
            headers[name] = value

    import time as _time
    import json as _json_proxy
    logger.info(f"OR proxy ENTRY → {request.method} {path}")
    body = await request.body()
    logger.info(f"OR proxy BODY → {len(body)}b")

    # Strip thinking/redacted_thinking blocks — non-Anthropic models reject them with 400.
    # Claude Code embeds cryptographic thinking signatures in resumed sessions; forwarding
    # these to OpenRouter non-Anthropic models causes silent failures or 400 errors.
    # See: github.com/anthropics/claude-code/issues/21726
    if body and request.method == "POST":
        try:
            _bj = _json_proxy.loads(body)
            _model_or = _bj.get("model", "")
            if not _model_or.startswith("anthropic/"):
                _body_modified = False

                # 1. Strip anthropic_beta from JSON body — Claude Code sends it in both
                # HTTP header AND body; OpenRouter non-Anthropic endpoints reject it with 400.
                if "anthropic_beta" in _bj:
                    _removed_betas = _bj.pop("anthropic_beta")
                    _body_modified = True
                    logger.info(f"OR proxy stripped anthropic_beta from body: {_removed_betas}")

                # 2. Strip context_management — top-level field added by context-management-2025-06-27 beta.
                # OpenRouter does not know this field and returns 400.
                if "context_management" in _bj:
                    _bj.pop("context_management")
                    _body_modified = True
                    logger.info("OR proxy stripped context_management from body")

                # 5. Strip thinking/redacted_thinking blocks from assistant messages.
                _thinking_stripped = 0
                for _msg in _bj.get("messages", []):
                    if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), list):
                        _orig = len(_msg["content"])
                        _msg["content"] = [
                            _b for _b in _msg["content"]
                            if _b.get("type") not in ("thinking", "redacted_thinking")
                        ]
                        _thinking_stripped += _orig - len(_msg["content"])

                if _thinking_stripped or _body_modified:
                    body = _json_proxy.dumps(_bj).encode("utf-8")
                    logger.info(f"OR proxy body rewritten: thinking={_thinking_stripped} modified={_body_modified} size={len(body)}b")
        except Exception as _pe:
            logger.debug(f"OR proxy body parse skip: {_pe}")

    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
    try:
        req = client.build_request(request.method, target_url, headers=headers, content=body)
        _t0 = _time.monotonic()
        resp = await client.send(req, stream=True)
        _headers_ms = int((_time.monotonic() - _t0) * 1000)
        logger.info(f"OR proxy ← status={resp.status_code} headers_in={_headers_ms}ms x-ratelimit-remaining={resp.headers.get('x-ratelimit-requests-remaining', '?')}")
        if resp.status_code >= 400:
            error_body = await resp.aread()
            logger.error(f"OR proxy ERROR {resp.status_code}: {error_body[:500].decode('utf-8', errors='replace')}")
            await client.aclose()
            return JSONResponse({"error": error_body.decode("utf-8", errors="replace")}, status_code=resp.status_code)
    except Exception as e:
        await client.aclose()
        logger.error(f"OpenRouter proxy error: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)

    # Forward response with streaming — log when first chunk arrives (TTFT)
    async def _stream_with_ttft():
        first = True
        async for chunk in resp.aiter_bytes():
            if first:
                logger.info(f"OR proxy first-chunk TTFT={int((_time.monotonic() - _t0)*1000)}ms size={len(chunk)}b")
                first = False
            yield chunk

    resp_headers = {
        k: v for k, v in resp.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "connection")
    }
    return StreamingResponse(
        _stream_with_ttft(),
        status_code=resp.status_code,
        headers=resp_headers,
        background=BackgroundTask(client.aclose),
    )


@app.post("/api/openrouter")
async def set_openrouter(model: str = Form(...)):
    """Переключить Claude Code на OpenRouter с выбранной моделью."""
    import json, pathlib
    model_id = OPENROUTER_MODELS.get(model.strip(), model.strip())
    settings_path = pathlib.Path.home() / ".claude" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        env = settings.setdefault("env", {})
        # Route through local proxy to strip unsupported anthropic-beta headers
        env["ANTHROPIC_BASE_URL"]   = _OR_PROXY_BASE
        env["ANTHROPIC_AUTH_TOKEN"] = OPENROUTER_API_KEY or "sk-or-YOUR_KEY_HERE"
        env["ANTHROPIC_API_KEY"]    = ""   # обязательно пусто — иначе конфликт auth + beta-заголовки
        env["ANTHROPIC_MODEL"]      = model_id
        settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        config.CLAUDE_PROVIDER = "openrouter"
        config.OPENROUTER_MODEL_ID = model_id
        # Сбрасываем все сессии — resume-токены Anthropic не работают в OpenRouter
        from cli_executor import _sessions
        _sessions.clear()
        logger.info(f"OpenRouter enabled via proxy, model: {model_id}, all sessions cleared")
        return {"ok": True, "model": model_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/api/anthropic")
async def set_anthropic():
    """Вернуть Claude Code на прямой Anthropic API."""
    import json, pathlib
    settings_path = pathlib.Path.home() / ".claude" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
        env = settings.setdefault("env", {})
        for key in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"]:
            env.pop(key, None)
        env["ANTHROPIC_MODEL"] = config.AGENT_MODEL
        settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
        config.CLAUDE_PROVIDER = "anthropic"
        config.OPENROUTER_MODEL_ID = ""
        # Сбрасываем все сессии — resume-токены OpenRouter не работают в Anthropic
        from cli_executor import _sessions
        _sessions.clear()
        logger.info(f"Switched back to Anthropic, model: {config.AGENT_MODEL}, all sessions cleared")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/restart")
async def restart_server():
    """
    Перезапустить Python-сервер. ngrok-тоннель не трогается — URL остаётся тем же.
    Сервер запускает новый процесс через 3 секунды и завершает текущий.
    """
    logger.info("Restart requested via /restart command")

    async def _do_restart():
        await asyncio.sleep(0.8)  # Даём HTTP-ответу уйти к клиенту
        logger.info("Restarting server process (ngrok stays running)...")
        server_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "server.py"))
        cwd = os.path.dirname(server_py)
        python_exe = sys.executable

        # Пишем временный bat-файл — надёжнее чем cmd /c с кавычками и пробелами в путях
        bat_path = os.path.join(cwd, "_restart_tmp.bat")
        with open(bat_path, "w", encoding="ascii") as f:
            f.write("@echo off\n")
            f.write("timeout /t 4 /nobreak > nul\n")
            f.write(f'"{python_exe}" "{server_py}"\n')

        # CREATE_NEW_CONSOLE — открывает новое окно, не привязано к текущему процессу
        subprocess.Popen(
            bat_path,
            cwd=cwd,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            close_fds=True,
        )
        os._exit(0)

    asyncio.create_task(_do_restart())
    return {"ok": True, "message": "Перезапуск через ~4 секунды. Страница обновится автоматически."}


@app.post("/api/history/clear")
async def clear_history(ws_id: str = Form(default="")):
    """Начать новый диалог — следующий запрос создаст новую сессию CLI."""
    await cli_new_session(client_id=ws_id)
    await new_agent_session(client_id=ws_id)
    logger.info(f"Сессия CLI [{ws_id[:8]}] сброшена по запросу пользователя")
    return {"ok": True}


@app.post("/api/session/restore")
async def session_restore(ws_id: str = Form(...), agent_session_id: str = Form(...)):
    """Восстановить Claude-сессию после перезагрузки страницы."""
    if config.CLAUDE_PROVIDER == "openrouter":
        logger.info(f"Session restore BLOCKED [{ws_id[:8]}] — OR mode requires fresh context")
        return {"ok": False, "reason": "openrouter_mode"}
    restore_agent_session(client_id=ws_id, agent_session_id=agent_session_id)
    logger.info(f"Сессия [{ws_id[:8]}] восстановлена из браузера: {agent_session_id[:12]}")
    return {"ok": True}


@app.get("/api/screen/latest")
async def get_latest_screen():
    """Возвращает последний скриншот экрана (для Claude)."""
    screen_dir = os.path.join(os.path.dirname(__file__), "static", "screen")
    latest = os.path.join(screen_dir, "latest.jpg")
    if not os.path.exists(latest):
        return JSONResponse({"ok": False, "error": "No screenshots yet. Start screen_watcher.py first."})
    mtime = os.path.getmtime(latest)
    age_sec = int(time.time() - mtime)
    return JSONResponse({"ok": True, "url": "/static/screen/latest.jpg", "age_sec": age_sec})


@app.get("/api/status")
async def server_status():
    """Статус сервера и загруженных моделей."""
    client = get_qwen3tts_client()
    current_speaker = client.current_speaker if _active_tts_engine == "qwen" else _active_xtts_voice
    return {
        "status": "ok",
        "stt_model": "faster-whisper",
        "tts_model": _active_tts_engine,
        "tts_ready": client.is_ready if _active_tts_engine == "qwen" else True,
        "tts_speaker": current_speaker,
        "speakers": client.get_speakers(),
        "agent_model": config.AGENT_MODEL,
    }


# --- Agent model switching ---

ALLOWED_MODELS = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
}


@app.get("/api/model")
async def get_model():
    """Текущая модель агента."""
    model_id = config.AGENT_MODEL
    short = next((k for k, v in ALLOWED_MODELS.items() if v == model_id), model_id)
    return {"ok": True, "model": model_id, "short": short, "available": ALLOWED_MODELS}


@app.post("/api/model")
async def set_model(model: str = Form(...)):
    """Переключить модель агента. Принимает 'sonnet', 'opus' или полный ID."""
    model_id = ALLOWED_MODELS.get(model.lower().strip(), model.strip())
    config.AGENT_MODEL = model_id
    # Сбрасываем все сессии — новая модель не должна наследовать контекст старой
    from cli_executor import _sessions
    _sessions.clear()
    short = next((k for k, v in ALLOWED_MODELS.items() if v == model_id), model_id)
    logger.info(f"Agent model changed to: {model_id} ({short}), all sessions cleared")
    return {"ok": True, "model": model_id, "short": short}


# --- Markdown stripping for TTS ---

_MD_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
_MD_INLINE_CODE = re.compile(r'`[^`\n]+`')
_MD_HEADER = re.compile(r'^#{1,6}\s+', re.MULTILINE)
_MD_BOLD_ITALIC = re.compile(r'\*{1,3}([^*\n]+)\*{1,3}')
_MD_UNDERSCORE = re.compile(r'_{1,3}([^_\n]+)_{1,3}')
_MD_STRIKETHROUGH = re.compile(r'~~([^~\n]+)~~')
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_IMAGE = re.compile(r'!\[[^\]]*\]\([^)]+\)')
_MD_BULLET = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
_MD_NUMBERED = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
_MD_HR = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
_MD_TABLE_SEP = re.compile(r'\|')
_MD_BACKSLASH = re.compile(r'\\([^\s])')
_MD_EXTRA_NL = re.compile(r'\n{3,}')


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting symbols before passing text to TTS."""
    text = _MD_CODE_BLOCK.sub('', text)
    text = _MD_INLINE_CODE.sub('', text)
    text = _MD_HEADER.sub('', text)
    text = _MD_BOLD_ITALIC.sub(r'\1', text)
    text = _MD_UNDERSCORE.sub(r'\1', text)
    text = _MD_STRIKETHROUGH.sub(r'\1', text)
    text = _MD_IMAGE.sub('', text)
    text = _MD_LINK.sub(r'\1', text)
    text = _MD_BULLET.sub('', text)
    text = _MD_NUMBERED.sub('', text)
    text = _MD_HR.sub('', text)
    text = _MD_TABLE_SEP.sub(' ', text)
    text = _MD_BACKSLASH.sub(r'\1', text)
    text = _MD_EXTRA_NL.sub('\n\n', text)
    text = _normalize_symbols(text)
    return text.strip()


# ── Symbol normalization for TTS ──────────────────────────────────────────────
# Allowed: letters, digits, spaces, basic sentence punctuation (. , ! ? ; :)
# Replaced: % $ + × = ÷  and context-aware -
# Banned (removed): everything else

def _normalize_symbols(text: str) -> str:
    """Replace math/currency symbols with spoken words, remove all others."""
    # 1. Number + % → "N процентов"
    text = re.sub(r'(\d+)\s*%', r'\1 процентов', text)
    # 2. $ + number → "N долларов"
    text = re.sub(r'\$\s*(\d+)', r'\1 долларов', text)
    # 3. number + $ → "N долларов"
    text = re.sub(r'(\d+)\s*\$', r'\1 долларов', text)
    # 4. × → "умножить на"
    text = text.replace('×', ' умножить на ')
    text = text.replace('*', ' умножить на ')  # math asterisk
    # 5. ÷ → "разделить на"
    text = text.replace('÷', ' разделить на ')
    # 6. = → "равно"  (but not inside words like "!=")
    text = re.sub(r'(?<![!<>])\s*=\s*(?!=)', ' равно ', text)
    # 7. + → "плюс" (math context: digit or space around it)
    text = re.sub(r'(?<=\d)\s*\+\s*(?=\d)', ' плюс ', text)
    text = re.sub(r'^\s*\+\s*', 'плюс ', text, flags=re.MULTILINE)  # line-starting +
    # 8. - → "минус" only between digits (e.g. "5 - 3"), keep hyphens in words
    text = re.sub(r'(?<=\d)\s+-\s+(?=\d)', ' минус ', text)
    # 9. Remove all remaining special characters not needed for speech
    #    Keep: letters (any unicode), digits, space, newline, . , ! ? ; : - (hyphen in words)
    text = re.sub(r'[^\w\s\.,!?;:\-\n]', '', text, flags=re.UNICODE)
    # 10. Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text


# --- Background TTS (fire-and-forget) ---

# Active TTS engine routing: "qwen" uses Qwen3-TTS subprocess, "xtts" uses XTTS v2
_active_tts_engine: str = "qwen"
_active_xtts_voice: str = "olena"

# Keep strong references to TTS tasks to prevent GC cancellation mid-execution
_active_tts_tasks: set = set()


async def _background_tts(text: str, ws_id: str, request_id: str, role: str = None):
    """Fire-and-forget TTS generation. Streams audio URL via WebSocket."""
    text = _strip_markdown(text).strip()
    if not text:
        return
    try:
        await manager.send(ws_id, {
            "type": "status_update",
            "stage": "tts",
            "message": "Generating voice...",
            "progress": 80,
            "request_id": request_id,
        })
        await _stream_tts(text, ws_id, request_id, role)
    except Exception as e:
        logger.warning(f"[{request_id}] Background TTS failed: {e}")
    finally:
        # Always notify client so auto-live can restart recording (even on TTS error)
        await manager.send(ws_id, {"type": "tts_done", "request_id": request_id})


# --- TTS via Qwen3-TTS subprocess or XTTS ---

async def _stream_tts(text: str, ws_id: str, request_id: str, role: str = None) -> str:
    """
    Генерирует TTS через Qwen3-TTS subprocess (qwen) или XTTS v2 (xtts).
    Если USE_REMOTE_AI=True — роутит на remote TTS server (192.168.1.243:8002).
    Возвращает URL аудио файла.
    """
    if getattr(config, "USE_REMOTE_AI", False):
        from remote_ai_client import remote_synthesize
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: remote_synthesize(text, speaker="Ryan", language="Russian")
        )
        url = result["url"]
        if ws_id:
            await manager.send(ws_id, {
                "type": "tts_chunk",
                "audio_url": url,
                "chunk_index": 0,
                "total_chunks": 1,
                "request_id": request_id,
            })
        return url

    if _active_tts_engine == "xtts":
        loop = asyncio.get_running_loop()
        xtts_mgr = get_xtts_manager()
        voice = _active_xtts_voice
        url = await loop.run_in_executor(
            None,
            lambda: xtts_mgr.synthesize(text, role=voice)
        )
    elif _active_tts_engine == "edge":
        result = await edge_tts_ru.synthesize(text)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "edge-tts failed"))
        url = result["url"]
    else:
        client = get_qwen3tts_client()
        result = await client.synthesize(text)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "Qwen3-TTS synthesis failed"))
        url = result["url"]
    if ws_id:
        await manager.send(ws_id, {
            "type": "tts_chunk",
            "audio_url": url,
            "chunk_index": 0,
            "total_chunks": 1,
            "request_id": request_id,
        })
    return url


# --- Pipeline TTS: fire per sentence as Claude streams ---

# Split on sentence endings only when followed by uppercase (Cyrillic or Latin) or end of string.
# Avoids false splits on: "v1.2.3 ", "т.е. ", "$15.5 million", "Mr. Smith"
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[А-ЯA-Z\u0400-\u04FF])')


class _TtsPipeline:
    """
    Buffers streaming text, accumulates 2 sentences, then fires _background_tts.
    Batching 2 sentences per TTS call gives natural prosody while keeping latency low.
    """

    SENTENCES_PER_CHUNK = 4

    def __init__(self, ws_id: str, request_id: str):
        self._ws_id = ws_id
        self._request_id = request_id
        self._buf = ""
        self._pending: list[str] = []

    def feed(self, chunk: str):
        self._buf += chunk
        while True:
            m = _SENTENCE_END_RE.search(self._buf)
            if not m:
                break
            sentence = self._buf[:m.start() + 1].strip()
            self._buf = self._buf[m.end():]
            if sentence:
                self._pending.append(sentence)
            if len(self._pending) >= self.SENTENCES_PER_CHUNK:
                self._fire()

    def _fire(self):
        if not self._pending:
            return
        text = " ".join(self._pending)
        self._pending = []
        task = asyncio.create_task(
            _background_tts(text, self._ws_id, self._request_id)
        )
        _active_tts_tasks.add(task)
        task.add_done_callback(_active_tts_tasks.discard)
        logger.debug(f"[TTS pipeline] Fired chunk ({len(text)} chars): {text[:60]}")

    def flush(self):
        remaining = self._buf.strip()
        if remaining:
            self._pending.append(remaining)
        self._buf = ""
        self._fire()


# --- Step mode helpers (agent vs simple) ---

async def _run_simple_request(text: str, ws_id: str, request_id: str, status_fn) -> str:
    """Простой режим: claude --print, потоковый вывод текста."""
    full_response = []
    tts_pipe = _TtsPipeline(ws_id, request_id) if ws_id else None

    async def on_chunk(line: str):
        full_response.append(line)
        if ws_id:
            await manager.send(ws_id, {
                "type": "cli_output",
                "data": line,
                "request_id": request_id,
            })
        if tts_pipe:
            tts_pipe.feed(line)

    async def on_cli_status(msg: str):
        await status_fn(msg, "cli_execution", 60)

    result = await run_claude_streaming(
        text,
        on_chunk=on_chunk,
        on_status=on_cli_status,
        client_id=ws_id,
    )
    if tts_pipe:
        tts_pipe.flush()
    return result


async def _run_agent_request(text: str, ws_id: str, request_id: str, status_fn) -> str:
    """Agent-режим: claude -p --dangerously-skip-permissions --output-format stream-json."""
    import time as _time
    agent_start = _time.monotonic()
    tool_count = 0
    tts_pipe = _TtsPipeline(ws_id, request_id) if ws_id else None

    await status_fn("Claude Agent запущен...", "cli_execution", 40)

    async def on_tool_use(tool_name: str, tool_input: dict):
        nonlocal tool_count
        tool_count += 1
        elapsed = _time.monotonic() - agent_start
        if ws_id:
            await manager.send(ws_id, {
                "type": "agent_tool_use",
                "tool": tool_name,
                "input": tool_input,
                "tool_index": tool_count,
                "elapsed_sec": round(elapsed, 1),
                "request_id": request_id,
            })
            # Also send human-readable status
            await manager.send(ws_id, {
                "type": "status_update",
                "stage": "cli_execution",
                "message": f"[{elapsed:.0f}s] Tool #{tool_count}: {tool_name}",
                "progress": min(40 + tool_count * 5, 90),
                "request_id": request_id,
            })

    async def on_tool_result(tool_name: str, summary: str):
        elapsed = _time.monotonic() - agent_start
        if ws_id:
            await manager.send(ws_id, {
                "type": "agent_tool_result",
                "tool": tool_name,
                "summary": summary,
                "tool_index": tool_count,
                "elapsed_sec": round(elapsed, 1),
                "request_id": request_id,
            })

    async def on_text(text_content: str):
        if ws_id:
            await manager.send(ws_id, {
                "type": "cli_output",
                "data": text_content,
                "request_id": request_id,
            })
            await manager.send(ws_id, {
                "type": "agent_text",
                "data": text_content,
                "request_id": request_id,
            })
        if tts_pipe:
            tts_pipe.feed(text_content)

    async def on_result(result_info: dict):
        elapsed = _time.monotonic() - agent_start
        if ws_id:
            await manager.send(ws_id, {
                "type": "agent_result",
                "cost_usd": result_info.get("cost_usd", 0),
                "duration_ms": result_info.get("duration_ms", 0),
                "turns": result_info.get("turns", 0),
                "num_events": result_info.get("num_events", 0),
                "total_elapsed_sec": round(elapsed, 1),
                "request_id": request_id,
            })

    async def on_status(msg: str):
        await status_fn(msg, "cli_execution", 60)

    async def on_session_id(session_id: str):
        if ws_id:
            await manager.send(ws_id, {
                "type": "agent_session_id",
                "session_id": session_id,
            })

    system_prompt = AGENT_SYSTEM_PROMPT if AGENT_SYSTEM_PROMPT else None

    try:
        result = await run_claude_agent_streaming(
            text,
            on_tool_use=on_tool_use,
            on_tool_result=on_tool_result,
            on_text=on_text,
            on_result=on_result,
            on_status=on_status,
            on_session_id=on_session_id,
            system_prompt=system_prompt,
            client_id=ws_id,
        )
    except RuntimeError as e:
        if "No conversation found" in str(e):
            # Stale session cleared by cli_executor — retry as new session
            logger.warning(f"[{request_id}] Stale agent session, retrying as new session...")
            result = await run_claude_agent_streaming(
                text,
                on_tool_use=on_tool_use,
                on_tool_result=on_tool_result,
                on_text=on_text,
                on_result=on_result,
                on_status=on_status,
                on_session_id=on_session_id,
                system_prompt=system_prompt,
                client_id=ws_id,
                new_session=True,
            )
        else:
            raise
    if tts_pipe:
        tts_pipe.flush()
    return result


# --- Board of Directors endpoints ---

@app.post("/api/board/validate")
async def board_validate(request: Request):
    """Валидация данных без запуска сессии."""
    try:
        data = await request.json()
        gate_input = validate_gate_input(data)
        return {"ok": True, "validated": gate_input.model_dump()}
    except DataGateError as e:
        return JSONResponse(
            status_code=422,
            content={"ok": False, "errors": e.errors},
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "errors": [str(e)]},
        )


@app.post("/api/board/session")
async def board_session(request: Request):
    """Запуск board-сессии. Отправляет прогресс через WebSocket."""
    try:
        body = await request.json()
        data = body.get("data", body)
        ws_id = body.get("ws_id", "")

        async def on_status(msg: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "board_status",
                    "message": msg,
                })

        async def on_director_complete(director_response):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "director_complete",
                    "director": director_response.model_dump(),
                })

        async def on_director_audio(role: str, audio_url: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "director_audio",
                    "role": role,
                    "audio_url": audio_url,
                })

        async def on_round2_complete(r2_response):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "round2_complete",
                    "response": r2_response.model_dump(),
                })

        async def on_round2_audio(role: str, audio_url: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "round2_audio",
                    "role": role,
                    "audio_url": audio_url,
                })

        async def on_round2_status(msg: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "round2_status",
                    "message": msg,
                })

        async def on_ceo_chunk(line: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "ceo_output",
                    "data": line,
                })

        async def on_ceo_audio(audio_url: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "ceo_audio",
                    "audio_url": audio_url,
                })

        session = await run_board_session(
            data=data,
            on_status=on_status,
            on_director_complete=on_director_complete,
            on_ceo_chunk=on_ceo_chunk,
            on_director_audio=on_director_audio,
            on_ceo_audio=on_ceo_audio,
            on_round2_complete=on_round2_complete,
            on_round2_audio=on_round2_audio,
            on_round2_status=on_round2_status,
        )

        return {"ok": True, "session": session.model_dump()}

    except DataGateError as e:
        return JSONResponse(
            status_code=422,
            content={"ok": False, "errors": e.errors},
        )
    except Exception as e:
        logger.error(f"Board session error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )


@app.get("/api/board/session/{session_id}")
async def board_session_get(session_id: str):
    """Получить результат board-сессии по ID."""
    session = await get_session(session_id)
    if not session:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "Session not found"},
        )
    return {"ok": True, "session": session.model_dump()}


@app.post("/api/board/followup")
async def board_followup(request: Request):
    """Follow-up вопрос к CEO (продолжение сессии)."""
    try:
        body = await request.json()
        session_id = body.get("session_id", "")
        question = body.get("question", "")
        ws_id = body.get("ws_id", "")

        if not session_id or not question:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "session_id and question are required"},
            )

        async def on_ceo_chunk(line: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "ceo_output",
                    "data": line,
                })

        async def on_ceo_audio(audio_url: str):
            if ws_id:
                await manager.send(ws_id, {
                    "type": "ceo_audio",
                    "audio_url": audio_url,
                })

        decision = await run_followup(
            session_id=session_id,
            question=question,
            on_ceo_chunk=on_ceo_chunk,
            on_ceo_audio=on_ceo_audio,
        )

        return {"ok": True, "decision": decision.model_dump()}

    except ValueError as e:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": str(e)},
        )
    except Exception as e:
        logger.error(f"Board followup error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )


@app.post("/api/board/session/{session_id}/outcome")
async def board_session_outcome(session_id: str, request: Request):
    """Record user's decision and outcome notes for a board session."""
    try:
        body = await request.json()
        user_choice = body.get("user_choice", "")
        outcome_notes = body.get("outcome_notes", "")

        if not user_choice:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "user_choice is required"},
            )

        if not MEMORY_ENABLED:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "Memory system is disabled"},
            )

        memory_store = get_memory_store()
        await memory_store.record_outcome(session_id, user_choice, outcome_notes)
        return {"ok": True, "session_id": session_id}

    except Exception as e:
        logger.error(f"Board outcome error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )


@app.get("/api/board/sessions")
async def board_sessions_list(company_name: str = "", limit: int = 10):
    """List past board sessions, optionally filtered by company."""
    if not MEMORY_ENABLED:
        return {"ok": True, "sessions": []}

    try:
        memory_store = get_memory_store()
        sessions = await memory_store.list_sessions(
            company_name=company_name or None, limit=limit
        )
        return {"ok": True, "sessions": sessions}
    except Exception as e:
        logger.error(f"Board sessions list error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)},
        )


@app.get("/api/board/companies")
async def board_companies_list():
    """List all known companies with profile summaries for autocomplete."""
    if not MEMORY_ENABLED:
        return {"ok": True, "companies": []}
    try:
        memory_store = get_memory_store()
        companies = await memory_store.list_companies()
        return {"ok": True, "companies": companies}
    except Exception as e:
        logger.error(f"Board companies list error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/board/company/{company_name}/profile")
async def board_company_profile(company_name: str):
    """Get full company profile (for auto-filling the board form)."""
    if not MEMORY_ENABLED:
        return {"ok": False, "profile": None}
    try:
        memory_store = get_memory_store()
        profile = await memory_store.get_company_profile(company_name)
        return {"ok": True, "profile": profile}
    except Exception as e:
        logger.error(f"Board company profile error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/api/board/projects")
async def board_projects_list(company_name: str = "", status: str = "active"):
    """List projects, optionally filtered by company."""
    if not MEMORY_ENABLED:
        return {"ok": True, "projects": []}
    try:
        memory_store = get_memory_store()
        projects = await memory_store.list_projects(
            company_name=company_name or None, status=status
        )
        return {"ok": True, "projects": projects}
    except Exception as e:
        logger.error(f"Board projects list error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/api/board/projects")
async def board_projects_create(request: Request):
    """Create a new project."""
    if not MEMORY_ENABLED:
        return JSONResponse(status_code=503, content={"ok": False, "error": "Memory disabled"})
    try:
        data = await request.json()
        memory_store = get_memory_store()
        project = await memory_store.create_project(
            name=data["name"],
            company_name=data["company_name"],
            description=data.get("description", ""),
            phase=data.get("phase", ""),
        )
        return {"ok": True, "project": project}
    except Exception as e:
        logger.error(f"Board project create error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.patch("/api/board/projects/{project_id}")
async def board_projects_update(project_id: str, request: Request):
    """Update project fields (context_notes, phase, status)."""
    try:
        data = await request.json()
        memory_store = get_memory_store()
        project = await memory_store.update_project(project_id, **data)
        return {"ok": True, "project": project}
    except Exception as e:
        logger.error(f"Board project update error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


# ---------------------------------------------------------------------------
# OmniAvatar — говорящий аватар из фото + аудио (GPU 1)
# ---------------------------------------------------------------------------
_avatar_executor = ThreadPoolExecutor(max_workers=1)
_avatar_jobs: dict = {}


def _run_avatar_sync(job_id: str, image_path: str, audio_path: str,
                     prompt: str, num_steps: int, guidance_scale: float, audio_scale: float):
    sys.path.insert(0, str(Path(__file__).parent))
    from omni_avatar_tool import generate_avatar_video
    _avatar_jobs[job_id]["status"] = "running"
    try:
        result = generate_avatar_video(
            image_path=image_path, audio_path=audio_path,
            prompt=prompt, output_name=job_id,
            num_steps=num_steps, guidance_scale=guidance_scale, audio_scale=audio_scale,
        )
        if result["success"]:
            _avatar_jobs[job_id]["status"] = "done"
            _avatar_jobs[job_id]["video_path"] = result["video_path"]
        else:
            _avatar_jobs[job_id]["status"] = "error"
            _avatar_jobs[job_id]["error"] = result["error"]
    except Exception as e:
        _avatar_jobs[job_id]["status"] = "error"
        _avatar_jobs[job_id]["error"] = str(e)
        logger.error(f"[Avatar] {job_id} failed: {e}", exc_info=True)


@app.post("/api/avatar/generate")
async def avatar_generate(request: Request):
    """Запускает генерацию аватар-видео асинхронно. Возвращает job_id."""
    body = await request.json()
    image_path = body.get("image_path", "")
    audio_path = body.get("audio_path", "")
    if not image_path or not audio_path:
        return JSONResponse(status_code=400,
                            content={"ok": False, "error": "image_path and audio_path required"})

    job_id = f"avatar_{uuid.uuid4().hex[:8]}"
    _avatar_jobs[job_id] = {
        "status": "queued", "video_path": None, "error": None,
        "started_at": datetime.now().isoformat(),
    }
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _avatar_executor, _run_avatar_sync, job_id,
        image_path, audio_path,
        body.get("prompt"), int(body.get("num_steps", 20)),
        float(body.get("guidance_scale", 4.5)), float(body.get("audio_scale", 5.0)),
    )
    logger.info(f"[Avatar] queued {job_id}")
    return {"ok": True, "job_id": job_id}


@app.get("/api/avatar/status/{job_id}")
async def avatar_status(job_id: str):
    """Статус job: queued | running | done | error"""
    job = _avatar_jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not found"})
    return {"ok": True, "job_id": job_id, **job}


@app.get("/api/avatar/video/{job_id}")
async def avatar_video(job_id: str):
    """Отдаёт готовый MP4."""
    job = _avatar_jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not found"})
    if job["status"] != "done":
        return JSONResponse(status_code=202, content={"ok": False, "status": job["status"]})
    return FileResponse(job["video_path"], media_type="video/mp4", filename=f"{job_id}.mp4")


@app.post("/telegram-message")
async def telegram_message_notify(request: Request):
    """Входящий текст от пользователя через Telegram → broadcast WebSocket → UI."""
    try:
        data = await request.json()
        text = data.get("text", "")
        await manager.broadcast({"type": "telegram_message", "text": text})
        logger.info(f"[TG] Incoming text: {text[:80]}")
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/telegram-file")
async def telegram_file_notify(request: Request):
    """Вызывается Telegram Bridge — рассылает событие всем WebSocket-клиентам."""
    try:
        data = await request.json()
        filename = data.get("filename", "")
        file_type = data.get("file_type", "document")
        size = data.get("size", 0)
        await manager.broadcast({
            "type": "telegram_file",
            "filename": filename,
            "path": f"/static/uploads/{filename}",
            "file_type": file_type,
            "size": size,
        })
        logger.info(f"[TG] File broadcast: {filename} ({file_type}, {size} bytes)")
        return JSONResponse({"ok": True})
    except Exception as e:
        logger.error(f"[TG] telegram-file error: {e}")
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.post("/api/gpu/offload")
async def gpu_offload():
    """Выгрузить STT + TTS из VRAM — освобождает память GPU."""
    import torch
    results = {}

    stt = get_stt_manager()
    await asyncio.get_running_loop().run_in_executor(None, stt.offload_to_cpu)
    results["stt"] = "offloaded"

    await get_qwen3tts_client().stop()
    results["qwen3tts"] = "stopped"

    free_mb = torch.cuda.mem_get_info(0)[0] // 1024**2 if torch.cuda.is_available() else 0
    logger.info(f"[GPU] Models offloaded. GPU0 free: {free_mb} MB")
    return JSONResponse({"ok": True, "free_mb": free_mb, "results": results})


@app.post("/api/gpu/full_unload")
async def gpu_full_unload():
    """Полностью выгрузить STT+TTS из RAM+VRAM."""
    import torch, gc
    results = {}

    stt = get_stt_manager()
    await asyncio.get_running_loop().run_in_executor(None, stt.full_unload)
    results["stt"] = "unloaded"

    await get_qwen3tts_client().stop()
    results["qwen3tts"] = "stopped"

    gc.collect()
    free_mb = torch.cuda.mem_get_info(0)[0] // 1024**2 if torch.cuda.is_available() else 0
    logger.info(f"[GPU] Models fully unloaded. GPU0 free: {free_mb} MB")
    return JSONResponse({"ok": True, "free_mb": free_mb, "results": results})


@app.post("/api/gpu/reload")
async def gpu_reload():
    """Вернуть STT + TTS на GPU."""
    import torch
    results = {}

    stt = get_stt_manager()
    await asyncio.get_running_loop().run_in_executor(None, stt.reload_to_gpu)
    results["stt"] = "reloaded"

    asyncio.create_task(get_qwen3tts_client().start())
    results["qwen3tts"] = "starting"

    free_mb = torch.cuda.mem_get_info(0)[0] // 1024**2 if torch.cuda.is_available() else 0
    logger.info(f"[GPU] Models reloading. GPU0 free: {free_mb} MB")
    return JSONResponse({"ok": True, "free_mb": free_mb, "results": results})


@app.websocket("/ws/{ws_id}")
async def websocket_endpoint(websocket: WebSocket, ws_id: str):
    """WebSocket для real-time статуса."""
    await manager.connect(ws_id, websocket)
    try:
        while True:
            # Держим соединение живым, ожидаем ping от клиента
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "debug_log":
                        logger.info(f"[CLIENT:{ws_id[:8]}] {msg.get('message', '')}")
                except Exception:
                    pass
    except WebSocketDisconnect:
        manager.disconnect(ws_id)


# --- Утилиты ---
def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── Copywriting Agent ─────────────────────────────────────────────────────────

class CopywritingRequest(BaseModel):
    voc_data: str
    product_brief: str
    platform: str = "TikTok"
    num_variants: int = 5
    framework: str = "auto"
    audience_temps: list | None = None  # ["cold","warm","retarget"] или подмножество

@app.post("/api/copywriting")
async def run_copywriting(req: CopywritingRequest):
    """
    AI Copywriting Agent v2.
    Pipeline: VOC+JTBD → ELM strategy ×3 temps (parallel) → Critic loop → Fogg MAP scoring.
    Возвращает варианты Cold/Warm/Retarget + overall top_recommendation.
    """
    try:
        from copywriting_agent_tool import run_copywriting_agent
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_copywriting_agent(
                voc_data=req.voc_data,
                product_brief=req.product_brief,
                platform=req.platform,
                num_variants=req.num_variants,
                framework=req.framework,
                audience_temps=req.audience_temps,
                save_output=True,
            )
        )
        return JSONResponse({"ok": True, "result": result})
    except Exception as e:
        logger.error(f"[COPYWRITING] Error: {e}", exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Creative Brief Agent ───────────────────────────────────────────────────────

class CreativeRequest(BaseModel):
    approved_copy: dict | str          # dict с hook/headline/body/cta или путь к JSON
    brand_kit: dict | None = None
    platform: str = "TikTok"
    duration_sec: int = 15
    style: str = "ugc"

@app.post("/api/creative")
async def run_creative(req: CreativeRequest):
    """
    Запустить AI Creative Brief Agent.
    Шаги: концепция → сторибоард → дизайн-спека + промпты для Runway/Kling/MJ.
    Принимает dict с копирайтингом или путь к JSON из /api/copywriting.
    """
    try:
        from creative_agent_tool import run_creative_agent
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: run_creative_agent(
                approved_copy=req.approved_copy,
                brand_kit=req.brand_kit,
                platform=req.platform,
                duration_sec=req.duration_sec,
                style=req.style,
                save_output=True,
            )
        )
        return JSONResponse({"ok": True, "result": result})
    except Exception as e:
        logger.error(f"[CREATIVE] Error: {e}", exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Marketing Factory ──────────────────────────────────────────────────────────

class FactoryRequest(BaseModel):
    product_category: str       # "luxury hair salon services"
    product_description: str    # "Premium keratin treatments, $200-400, Asheville NC"
    market: str                 # "Asheville, North Carolina"
    platform: str = "TikTok"   # TikTok | Meta | Google
    duration_sec: int = 30      # video duration for creative brief
    send_telegram: bool = True


@app.post("/api/factory/run")
async def factory_run(req: FactoryRequest):
    """
    Marketing Factory v2: product+market → 7 PDF reports → Telegram.
    Stages: Market Research → VOC+CI (parallel) → Customer Journey Map → Copywriting → Creative Brief → Measurement Framework → 7 PDFs.
    Long-running (~2-3 min). Returns PDF paths on completion.
    """
    try:
        from marketing_factory.pipeline import run_factory, FactoryInput
        inp = FactoryInput(
            product_category=req.product_category,
            product_description=req.product_description,
            market=req.market,
            platform=req.platform,
            duration_sec=req.duration_sec,
        )
        pdf_paths = await run_factory(inp, send_telegram=req.send_telegram)
        return JSONResponse({
            "ok": True,
            "pdfs": [str(p) for p in pdf_paths],
            "count": len(pdf_paths),
        })
    except Exception as e:
        logger.error(f"[FACTORY] Error: {e}", exc_info=True)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        log_level=LOG_LEVEL.lower(),
    )
