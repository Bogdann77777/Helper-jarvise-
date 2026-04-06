# Изменения: Pipeline TTS (2026-03-25)

## Цель
Уменьшить задержку между появлением текста и стартом озвучки с ~9-10 сек до ~2-3 сек.
Вместо запуска TTS ПОСЛЕ полного ответа Claude — теперь TTS стартует на каждом
завершённом предложении по мере стриминга.

## Файл резервной копии
`server.py.bak_tts_pipeline_20260325_123824`

## Откат (если что-то сломалось)
```bash
cp "server.py.bak_tts_pipeline_20260325_123824" server.py
```

---

## Изменения в server.py

### 1. Импорт `re` (строка ~5)
**Добавлено:**
```python
import re
```

### 2. Класс `_TtsPipeline` (после функции `_stream_tts`, перед `# --- Step mode helpers`)
**Добавлено:**
```python
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

class _TtsPipeline:
    def __init__(self, ws_id, request_id): ...
    def feed(self, chunk): ...   # накапливает, при завершении предложения → create_task(_background_tts)
    def flush(self): ...         # сбрасывает остаток в конце
```

### 3. `_run_simple_request` (был: ~строка 418)
**Добавлено:**
- `tts_pipe = _TtsPipeline(ws_id, request_id) if ws_id else None` — в начале функции
- `if tts_pipe: tts_pipe.feed(line)` — в конце `on_chunk`
- `if tts_pipe: tts_pipe.flush()` — после `run_claude_streaming`
- `return result` вместо `return await ...`

### 4. `_run_agent_request` (был: ~строка 442)
**Добавлено:**
- `tts_pipe = _TtsPipeline(ws_id, request_id) if ws_id else None` — в начале функции
- `if tts_pipe: tts_pipe.feed(text_content)` — в конце `on_text`
- `result = await run_claude_agent_streaming(...)` + `tts_pipe.flush()` + `return result` вместо `return await ...`

### 5. `/api/voice` endpoint (был: ~строка 198)
**Удалено:**
```python
# Fire TTS in background — don't block the HTTP response
if ws_id:
    asyncio.create_task(_background_tts(response_text, ws_id, request_id))
```
**Заменено на:**
```python
# TTS is fired sentence-by-sentence inside _run_*_request via _TtsPipeline
```

### 6. `/api/text` endpoint (был: ~строка 249)
**То же самое — удалён `create_task(_background_tts(...))`, добавлен комментарий.**

---

## Как это работает теперь

```
Claude стримит:  "Хорошо, засекай."  → feed() → XTTS стартует немедленно
                 " Это тестовое..."  → feed() → буферизуем
                 "...сообщение."     → feed() → XTTS в очередь (через _tts_lock)
                 " Жду результат."   → flush() → XTTS в очередь
```

Первый звук появляется через ~2-3 секунды после первой фразы Claude,
а не через 9-10 секунд после полного завершения ответа.
