# Claude Voice Interface

Голосовой интерфейс для Claude CLI через локальную сеть.

## Быстрый старт

### 1. Установить зависимости

```bash
# Сначала torch с CUDA (для RTX 5060 Ti — CUDA 12.x)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Затем всё остальное
pip install -r requirements.txt
```

### 2. Проверить config.py

```python
# Убедись что Claude CLI доступен:
CLAUDE_CLI_PATH = "claude"   # должен работать в терминале

# Модель Whisper (large-v3 точнее, medium быстрее):
WHISPER_MODEL = "large-v3"

# Голос (изменяется через UI, но дефолт можно поставить здесь):
SILERO_SPEAKER = "xenia"
```

### 3. Запустить сервер

```bash
python server.py
```

При первом запуске автоматически скачает:
- Whisper large-v3 (~3 GB)
- Silero TTS v5_1_ru (~50 MB)

### 4. Открыть в браузере

- На ПК: `http://127.0.0.1:8000`
- На телефоне/ноутбуке в той же Wi-Fi сети: `http://<IP_ПК>:8000`

IP адрес ПК будет показан в консоли при запуске.

## Режимы работы

| Режим | Описание |
|-------|----------|
| **Step** | Надиктовать или написать текст → нажать Отправить |
| **Live** | Говорить, 5 секунд тишины → автоматическая отправка |

## Выбор голоса

Выпадающий список в шапке — все доступные русские голоса Silero:
`aidar`, `baya`, `kseniya`, `xenia`, `eugene` (и другие если есть в модели).

## Структура проекта

```
cli helper/
├── server.py          # FastAPI сервер (главный)
├── config.py          # Все настройки
├── tts_manager.py     # Silero TTS
├── stt_manager.py     # faster-whisper STT
├── cli_executor.py    # Claude CLI через subprocess
├── requirements.txt
├── static/
│   ├── index.html     # Веб-интерфейс
│   └── audio/         # Сгенерированные аудио файлы
└── server.log         # Лог
```

## Требования

- Python 3.10+
- NVIDIA GPU с CUDA 12.x
- Claude CLI установлен и доступен в PATH
- PowerShell 7+ (опционально, для fallback)
