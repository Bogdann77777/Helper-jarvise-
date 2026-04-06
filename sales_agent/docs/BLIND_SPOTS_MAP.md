# AI Voice Sales Agent — Blind Spots & Weak Spots Map
**Date:** 2026-04-03
**Status:** Pre-implementation audit

---

## EXECUTIVE SUMMARY

Blueprint покрывает ~40% реального production. Остальные 60% — это слепые пятна, которые определяют разницу между "работает на демо" и "приносит продажи".

**Что уже ЕСТЬ в кодовой базе:** STT (Whisper + Kyutai), TTS (XTTS v2 + Qwen3), FastAPI сервер, WebSocket инфраструктура, VAD (Silero), шумоподавление (DeepFilterNet3), голоса (7 клонированных).

**Что ОТСУТСТВУЕТ:** Conversation FSM, sales prompts, sentence-level streaming, call analytics, campaign manager, barge-in, telephony.

---

## КАРТА СЛЕПЫХ ПЯТ (BLIND SPOTS)

### 🔴 КРИТИЧЕСКИЕ (без них продажи невозможны)

#### BS-01: Conversation State Machine (FSM)
- **Проблема:** Blueprint говорит "GPT-4o handles objections gracefully" — но КАК? Без FSM агент не знает в какой фазе звонка он находится. Он будет питчить когда надо слушать, и слушать когда надо закрывать.
- **Реальные фазы звонка:**
  1. `INTRO` — представление, проверка что это нужный человек
  2. `HOOK` — причина звонка (≤20 секунд)
  3. `QUALIFY` — квалификация (нужен ли продукт)
  4. `PITCH` — ценностное предложение
  5. `OBJECTION` — работа с возражениями (3-5 итераций)
  6. `CLOSE` — запрос на следующий шаг (встреча / демо / покупка)
  7. `FOLLOWUP` — договорённости после согласия
  8. `REJECTED` — graceful exit
  9. `VOICEMAIL` — специальный скрипт для автоответчика
  10. `ENDED` — звонок завершён
- **Без этого:** GPT будет "блуждать" в разговоре → потеря конверсии
- **Решение:** Python dataclass FSM с переходами по триггерам LLM

#### BS-02: Sales System Prompt Architecture
- **Проблема:** Blueprint называет это "самой важной инженерной работой" — но не даёт ни одного примера. Это 0% готово.
- **Что нужно:**
  - Persona (кто агент, откуда звонит, зачем)
  - Opening script (точные первые 3 фразы)
  - Qualification questions (5-7 вопросов)
  - Value proposition (3 ключевых аргумента)
  - Objection library (15-20 типичных возражений + ответы)
  - Closing techniques (3-5 вариантов)
  - Transition phrases (как переходить между фазами естественно)
- **Без этого:** Технически идеальный агент = плохой продавец
- **Решение:** Structured prompt template + JSON objection library

#### BS-03: Sentence-Level Streaming LLM→TTS
- **Проблема:** Текущий переводчик ждёт ПОЛНЫЙ ответ LLM → потом TTS. Для продаж это +1-2 сек задержки. Человек услышит "мёртвую тишину".
- **Цель:** Первое предложение воспроизводится пока LLM ещё генерирует второе.
- **Как:** LLM stream → sentence splitter → TTS queue → audio queue → WebSocket
- **Задержка без этого:** 3-4 сек. **С этим:** 0.8-1.2 сек.

#### BS-04: Barge-In (Interruption Handling)
- **Проблема:** Когда человек перебивает агента — агент должен НЕМЕДЛЕННО замолчать. Это триггер "realness".
- **Blueprint:** Упоминает LiveKit barge-in. НО для работы без LiveKit нужна своя реализация.
- **Что нужно:** VAD детектирует речь → посылает STOP сигнал → TTS/audio очередь очищается → FSM переходит в LISTENING.
- **Без этого:** Агент "не слушает" — самое раздражающее поведение у роботов

#### BS-05: AMD — Answering Machine Detection
- **Проблема:** 40-60% outbound попадает на voicemail. Без AMD агент начинает питчить автоответчику.
- **Варианты:**
  1. Telnyx native AMD (Premium plan) — $0.004/call, задержка 2-3 сек
  2. Пауза 3-4 сек → анализ ответа ("This is John" vs "Please leave a message")
  3. ML: SpeechBrain silence/energy pattern → vmail detection
- **Рекомендация:** Telnyx AMD в Phase 3 + backup NLP classifier в Phase 2

---

### 🟡 ВАЖНЫЕ (снижают конверсию и надёжность)

#### BS-06: Voice Persona & SSML
- **Проблема:** Blueprint рекомендует SSML паузы для naturalness. ElevenLabs поддерживает только `<break>` и `<emphasis>`. XTTS v2 не поддерживает SSML вообще.
- **Обходной путь:** Добавлять "..." и "—" в текст → XTTS делает натуральные паузы. Добавить normalization числе ("23 million dollars" → "twenty-three million dollars").
- **Что нужно:** Text normalizer перед TTS (числа, аббревиатуры, currency, даты)

#### BS-07: Call Recording & Transcript Storage
- **Проблема:** Без записей нет QA, нет обучения, нет legal coverage.
- **Что нужно:** WAV запись каждого звонка + JSON транскрипт с временными метками + metadata (кто, когда, outcome)
- **Storage:** `outputs/calls/{date}/{call_id}/`

#### BS-08: Post-Call Analytics & Scoring
- **Проблема:** Без аналитики невозможно улучшать скрипты. Blind optimization.
- **Метрики:**
  - Conversion rate по фазам FSM
  - Talk ratio (агент:prospect должен быть 40:60)
  - Objection frequency distribution
  - Average call duration по outcome
  - Sentiment trajectory (начало vs конец)
- **LLM scoring:** После звонка → Haiku анализирует транскрипт → score 0-10 + feedback
- **Без этого:** Нет данных для A/B тестирования скриптов

#### BS-09: LLM Fallback & Circuit Breaker
- **Проблема:** OpenRouter down → агент молчит. Нужна degraded mode.
- **Решение:**
  - Primary: Gemini 2.5 Flash (быстрый, дешёвый)
  - Fallback: Claude Haiku (разные серверы)
  - Emergency: Pre-scripted responses для топ-20 ситуаций (ответы без LLM)
- **Latency budget:** LLM должен отвечать в ≤2 сек. Если >3 сек → filler phrase ("Let me think about that...")

#### BS-10: Context Window Management
- **Проблема:** Длинный звонок (10+ мин) = большой контекст = медленный и дорогой LLM.
- **Решение:**
  - Conversation window: последние 6-8 обменов + ключевые факты
  - Extracted facts: имя, компания, боль, интерес — хранить отдельно в structured state
  - Rolling summary: каждые 5 обменов суммировать старые

---

### 🔵 PRODUCTION INFRASTRUCTURE (без них нельзя масштабировать)

#### BS-11: Phone Audio Normalization
- **Проблема:** Telnyx поставляет аудио в μ-law 8kHz (G.711) или OPUS 16kHz. Текущий STT ожидает 16kHz PCM.
- **Решение:** Реальтайм декодинг μ-law → PCM → upsample 8k→16k. Это должно быть в phone adapter.
- **Без этого:** STT accuracy падает на 15-20% на телефонном аудио

#### BS-12: Concurrent Call Management
- **Проблема:** Blueprint говорит "5 concurrent calls". Текущая архитектура — 1 WebSocket = 1 session. Нужен session pool.
- **Решение:** Sessions dict + asyncio, отдельная очередь на каждый звонок
- **GPU вопрос:** 2x RTX 5060 Ti = можно держать XTTS + STT параллельно, но при 5 звонках нужен TTS queue

#### BS-13: Error Recovery & Call State Persistence
- **Проблема:** Что если сервер упал в середине звонка? Или потерялось WebSocket соединение?
- **Решение:** Redis/SQLite call state → при reconnect восстановить FSM state

#### BS-14: Monitoring & Alerting
- **Проблема:** В продакшне нужно знать в реальном времени: сколько активных звонков, ошибки, latency p95.
- **Что есть:** Telegram бот (tg_send.py), metrics (monitoring/)
- **Что добавить:** Real-time dashboard, Telegram алерты при error_rate > 5%

---

### 🟣 NON-CODE BLIND SPOTS (влияют на продажи, не на код)

#### NC-01: Sales Script Quality (ВАЖНЕЙШЕЕ)
- **Правило:** Технически идеальный агент с плохим скриптом = провал.
- **Что нужно:**
  - Конкретный продукт/услуга и ценностное предложение
  - 3 главных pain point целевой аудитории
  - 15-20 типичных возражений + проверенные ответы
  - Social proof: кейсы, цифры, компании
  - Clear CTA: что агент должен "закрыть" за звонок

#### NC-02: Legal Disclosure (FCC 2024 — ОБЯЗАТЕЛЬНО)
- **Требование:** С февраля 2024 ВСЕ AI-generated voice calls обязаны раскрываться.
- **Реализация:** Первые слова агента: "...I'm an AI assistant calling on behalf of [Company]..."
- **Без этого:** Штраф $10,000-$50,000 за звонок (FCC)

#### NC-03: Caller ID & Local Presence
- **Проблема:** Звонки с "Unknown" или 800-номеров → pickup rate 5-10%. С местным номером → 30-40%.
- **Решение:** Telnyx local numbers — покупать локальные номера для каждого regional market

#### NC-04: A/B Testing Framework
- **Без этого:** Нет данных для улучшения. Нужно тестировать разные:
  - Opening lines
  - Objection responses
  - Closing techniques
  - Voice personas (мужской vs женский голос)

#### NC-05: DNC & TCPA Compliance
- **FTC DNC Registry:** API для проверки перед каждым звонком
- **TCPA:** Cell phones требуют prior written consent или EBR
- **Calling hours:** 8am-9pm LOCAL time получателя
- **Opt-out:** Немедленное удаление из списка при "stop calling"

---

## РЕЮЗАЕМЫЕ КОМПОНЕНТЫ (уже готово)

| Компонент | Файл | Готовность | Нужно адаптировать |
|-----------|------|------------|-------------------|
| FastAPI server | server.py | ✅ 100% | Добавить /sales endpoints |
| WebSocket manager | server.py | ✅ 100% | Уже production-ready |
| STT Whisper | stt_manager.py | ✅ 100% | Добавить phone audio adapter |
| STT Kyutai | translator/kyutai_stt.py | ✅ 100% | Лучше для реального времени |
| TTS XTTS v2 | xtts_manager.py | ✅ 100% | Добавить text normalizer |
| VAD Silero | translator/vad.py | ✅ 95% | Добавить barge-in signal |
| Noise filter | translator/noise_filter.py | ✅ 100% | Готово |
| Hallucination filter | translator/hallucination_filter.py | ✅ 100% | Готово |
| Orchestrator pattern | translator/orchestrator.py | ✅ 80% | Заменить LLM на sales brain |
| Telegram alerts | tg_send.py | ✅ 100% | Готово |
| Metrics | translator/monitoring/ | ✅ 90% | Добавить sales метрики |
| Voices (7 клонов) | voices/*.wav | ✅ 100% | Выбрать sales persona |
| Board system | board/ | ✅ 60% | Адаптировать для QA analysis |

---

## ИТОГ: ЧТО НЕ ПОКРЫТО BLUEPRINT

Blueprint покрывает только "счастливый путь" (happy path технической архитектуры). Реальный production требует:

1. **Conversation FSM** — без неё агент слеп
2. **Sales prompt engineering** — самая критичная часть
3. **Sentence streaming** — без неё задержка убивает naturalness
4. **Barge-in** — без неё агент звучит как робот
5. **AMD** — без неё 50% звонков теряются на voicemail
6. **Call analytics** — без неё нет данных для улучшения
7. **Legal compliance** — без неё риск штрафов
8. **Local presence numbers** — без неё pickup rate 5%
9. **LLM fallback** — без неё downtime = потерянные лиды
10. **Phone audio normalization** — без неё STT на 15% хуже
