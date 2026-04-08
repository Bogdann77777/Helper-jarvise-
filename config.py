# config.py — настройки всей системы

import os
from dotenv import load_dotenv

load_dotenv()  # загружает .env в переменные окружения

# --- Сервер ---
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# --- Remote AI Server (192.168.1.243 RTX 3070Ti) ---
# Set to "" to use local models instead
STT_REMOTE_URL = os.environ.get("STT_REMOTE_URL", "http://192.168.1.243:8001")
TTS_REMOTE_URL = os.environ.get("TTS_REMOTE_URL", "http://192.168.1.243:8002")
USE_REMOTE_AI  = True   # True = route STT/TTS to remote server, False = local models

# --- STT (Whisper) — used only when USE_REMOTE_AI=False ---
WHISPER_MODEL = "large-v3"       # large-v3 / medium / small
WHISPER_DEVICE = "cpu"           # cpu — VRAM now free for image/video generation
WHISPER_COMPUTE_TYPE = "int8"    # int8 for CPU

# --- TTS (XTTS v2 — единый движок для всех режимов) ---
XTTS_DEVICE = "cuda"            # cuda / cpu

# Папка с общими голосовыми образцами (WAV файлы = "спикеры")
VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

# Голос по умолчанию для Step/Live режимов (имя файла без .wav из voices/)
XTTS_DEFAULT_VOICE = "olena"

# Board-специфичные голоса (voices/board/)
XTTS_VOICES_DIR = os.path.join(os.path.dirname(__file__), "voices", "board")
os.makedirs(XTTS_VOICES_DIR, exist_ok=True)

XTTS_VOICE_CSO = os.path.join(XTTS_VOICES_DIR, "cso.wav")
XTTS_VOICE_CFO = os.path.join(XTTS_VOICES_DIR, "cfo.wav")
XTTS_VOICE_CTO = os.path.join(XTTS_VOICES_DIR, "cto.wav")
XTTS_VOICE_CEO = os.path.join(XTTS_VOICES_DIR, "ceo.wav")

XTTS_LANGUAGE = "auto"  # "auto" = detect from text, or force "en"/"ru"/etc.

# --- Claude CLI ---
CLAUDE_CLI_PATH = "claude"       # или полный путь, например: C:/Users/bogdan/AppData/Roaming/npm/claude.cmd
CLAUDE_CLI_TIMEOUT = 300         # секунд (5 минут) — на большие запросы

# --- Provider switching (runtime) ---
# "anthropic" = direct Anthropic API (default)
# "openrouter" = OpenRouter proxy (set by /api/openrouter endpoint)
CLAUDE_PROVIDER = "anthropic"
OPENROUTER_MODEL_ID = ""  # e.g. "anthropic/claude-sonnet-4-5"

# --- Режим работы ---
# Время тишины до автоотправки в Live-режиме (секунд)
SILENCE_TIMEOUT = 5.0

# --- Аудио файлы ---
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- Загрузка файлов от пользователя ---
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# --- OpenRouter (Board of Directors) ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Модели директоров (OpenRouter)
MODEL_CSO = "x-ai/grok-3-fast"                      # Chief Strategy Officer (Viktor)
MODEL_CFO = "google/gemini-2.5-flash-lite-preview-09-2025"  # Chief Financial Officer (Margaret)
MODEL_CTO = "deepseek/deepseek-v3.2-exp"            # Chief Technology Officer (James)
MODEL_CONFLICT_DETECTOR = "google/gemini-2.5-flash-lite-preview-09-2025"  # Детектор конфликтов

# Board таймауты
BOARD_CEO_TIMEOUT = 600   # секунд для CEO (Claude CLI) в board-сессии
BOARD_DIRECTOR_TIMEOUT = 30  # секунд на одного директора (OpenRouter)

# --- Memory (Board of Directors persistence) ---
MEMORY_ENABLED = True
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "memory")
MEMORY_QDRANT_DIR = os.path.join(MEMORY_DIR, "qdrant_data")
MEMORY_ARCHIVE_DIR = os.path.join(MEMORY_DIR, "archives")
MEMORY_DB_PATH = os.path.join(MEMORY_DIR, "board_memory.db")
MEMORY_MAX_CONTEXT_CHARS = 6000   # Production budget: compressed facts, not raw text
                                   # Mem0 benchmark: ~2K tokens extracted facts beats 24K raw

# Memory tier budgets (chars) — production-aligned (Mem0 paper: arxiv.org/abs/2504.19413)
MEMORY_TIER_FACTS    = 2000   # Extracted facts per role (Mem0: ~2K tokens optimal)
MEMORY_TIER_COMPANY  = 800    # T1: company/project profile (always loaded)
MEMORY_TIER_RECENT   = 2000   # T2: last 2 sessions summary (recency)
MEMORY_TIER_DECISIONS = 1200  # T4: decision ledger with outcomes (feedback loop)
# T3 (cross-company similar) removed from director context — CEO synthesis handles it

# --- Agent Mode (Step mode with tools + auto-approve) ---
AGENT_MODE_ENABLED = True
AGENT_MODEL = "claude-sonnet-4-6"  # "claude-sonnet-4-6" или "claude-opus-4-6"
AGENT_SYSTEM_PROMPT = """For simple conversational messages (greetings, thanks, short questions), respond with plain text only — do NOT use any tools. Only use tools when the user explicitly asks to perform a task, edit files, run code, or search for information."""  # Optional: append to system prompt (empty = use default)
AGENT_INACTIVITY_TIMEOUT = 60   # секунд (1 мин) — если процесс не выдаёт вывод 60с → таймаут → UI разблокируется
AGENT_MAX_TIMEOUT = 1800        # секунд (30 мин) — абсолютный потолок, аварийный стоп
AGENT_BUFFER_LIMIT = 50 * 1024 * 1024  # 50MB — JSON-строки с Read огромных файлов

# --- Telegram Sender (отправка файлов/сообщений пользователю) ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = 1873549647  # Bogdan_Lysetskyi

# --- Task Manager Bot (отдельный бот только для задач/расписания) ---
TASK_BOT_TOKEN = os.environ.get("TASK_BOT_TOKEN", "")
TASK_CHAT_ID   = 1873549647  # тот же пользователь

# --- Vast.ai Serverless (MultiTalk) ---
VASTAI_API_KEY     = os.environ.get("VASTAI_API_KEY", "")
VASTAI_ENDPOINT_ID = os.environ.get("VASTAI_ENDPOINT_ID", "")
VASTAI_SSH_KEY     = os.path.expanduser("~/.ssh/vast_ai")  # приватный ключ (работает на Linux и Windows)

# --- Perplexity Search API ---
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_MAX_RESULTS = 5       # результатов на запрос
PERPLEXITY_MAX_TOKENS_PAGE = 512 # токенов на страницу (меньше = дешевле)

# --- Логирование ---
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(os.path.dirname(__file__), "server.log")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB до ротации
LOG_BACKUP_COUNT = 3               # хранить 3 архивных лога

# --- Аудио хранение ---
AUDIO_MAX_FILES = 20  # максимум TTS файлов в static/audio/ (Board генерирует 8+)
