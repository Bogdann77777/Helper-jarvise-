# AI Voice Sales Agent — Architecture & Phased Plan
**Date:** 2026-04-03
**Hardware:** 2x RTX 5060 Ti 16GB | Windows 11
**Language:** English (outbound) | Stack: Python 3.12 + FastAPI

---

## SYSTEM OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI VOICE SALES AGENT                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 3 (later)          PHASE 1+2 (now)                       │
│  ┌──────────┐             ┌──────────────────────────────────┐  │
│  │  Telnyx  │─WebSocket──▶│         FastAPI Server           │  │
│  │  (SIP)   │             │  /ws/call  /api/campaign         │  │
│  └──────────┘             └───────────────┬──────────────────┘  │
│                                           │                      │
│                           ┌───────────────▼──────────────────┐  │
│                           │         CALL SESSION              │  │
│                           │  ┌─────────────────────────────┐ │  │
│                           │  │     ConversationFSM          │ │  │
│                           │  │  INTRO→QUALIFY→PITCH→CLOSE   │ │  │
│                           │  └──────────┬──────────────────┘ │  │
│                           │             │                     │  │
│  AUDIO IN ───────────────▶│  ┌──────────▼──────────────────┐ │  │
│                           │  │  VAD → NoiseFilter → STT    │ │  │
│                           │  │  (Silero + DeepFilter + Whisper)│ │
│                           │  └──────────┬──────────────────┘ │  │
│                           │             │                     │  │
│                           │  ┌──────────▼──────────────────┐ │  │
│                           │  │    Sales Brain (LLM)        │ │  │
│                           │  │  Gemini 2.5 Flash streaming  │ │  │
│                           │  │  + FSM state injection      │ │  │
│                           │  │  + Function calling         │ │  │
│                           │  └──────────┬──────────────────┘ │  │
│                           │             │ sentence stream      │  │
│                           │  ┌──────────▼──────────────────┐ │  │
│                           │  │   Text Normalizer + XTTS    │ │  │
│                           │  │   (local GPU, zero cost)    │ │  │
│                           │  └──────────┬──────────────────┘ │  │
│                           │             │                     │  │
│  AUDIO OUT ◀──────────────│  ┌──────────▼──────────────────┐ │  │
│                           │  │   Barge-In Controller       │ │  │
│                           │  │   Audio Queue Manager       │ │  │
│                           │  └─────────────────────────────┘ │  │
│                           └──────────────────────────────────┘  │
│                                                                  │
│                           ┌──────────────────────────────────┐  │
│                           │    Post-Call Analytics           │  │
│                           │  Transcript → LLM Scoring →      │  │
│                           │  Telegram Report                 │  │
│                           └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## PHASE 1 — CORE ENGINE (Текущая фаза)
**Цель:** Агент проводит реальный разговор в браузере. Без телефонии.
**Результат:** Demo-able product. Можно показать инвесторам.

### Компоненты Phase 1:

#### 1.1 ConversationFSM (`sales_agent/core/fsm.py`)
```
States: INTRO → HOOK → QUALIFY → PITCH → OBJECTION → CLOSE → FOLLOWUP → REJECTED → VOICEMAIL → ENDED
Transitions: triggered by LLM function call `transition_state(new_state, reason)`
State carries: extracted facts (name, company, pain, interest level)
```

#### 1.2 Sales Brain (`sales_agent/core/sales_brain.py`)
- LLM: Gemini 2.5 Flash via OpenRouter (streaming)
- System prompt: динамический (меняется по FSM state)
- Functions: `transition_state()`, `extract_fact()`, `end_call(outcome)`
- Sentence splitter: `re.split(r'(?<=[.!?])\s+')` → sentence queue → TTS
- Max response: 30-50 words (voice = short sentences)
- Fallback: pre-scripted responses если LLM >2.5 сек

#### 1.3 Sales Prompt Engine (`sales_agent/core/prompts.py`)
- Persona template (configurable: name, company, product)
- State-specific instructions per FSM phase
- Objection library (JSON): 20+ objections + responses
- Dynamic context injection: extracted facts + conversation history

#### 1.4 Barge-In Controller (`sales_agent/core/barge_in.py`)
- VAD detects speech while agent is speaking
- Sends `INTERRUPT` signal → clears TTS queue → LLM context gets "interrupted"
- Agent acknowledges: "Sure, go ahead..." → back to LISTENING

#### 1.5 Text Normalizer (`sales_agent/core/text_normalizer.py`)
- Numbers → words: "23 million" → "twenty-three million"
- Abbreviations: "CEO" → "C-E-O", "ROI" → "R-O-I"
- Currency: "$500" → "five hundred dollars"
- Dates: "Q2 2026" → "second quarter twenty twenty-six"
- Adds natural pauses via commas and em-dashes

#### 1.6 Call Recorder (`sales_agent/core/call_recorder.py`)
- WAV audio: все сегменты агента + transcripts
- JSON transcript: `{timestamp, speaker, text, confidence, state}`
- Storage: `outputs/calls/{YYYY-MM-DD}/{call_id}/`

#### 1.7 Browser Demo Interface (`sales_agent/static/index.html`)
- Simulates incoming call
- Shows FSM state in real-time
- Live transcript + audio waveform
- Post-call summary card

#### 1.8 Sales Config (`sales_agent/config.yaml`)
```yaml
agent:
  name: "Sarah"
  company: "YourCompany"
  product: "YourProduct"
  value_prop: "..."
llm:
  model: "google/gemini-2.5-flash"
  max_tokens: 80
  temperature: 0.7
  timeout: 2.5
  fallback_model: "anthropic/claude-haiku-4-5"
tts:
  engine: "xtts"  # xtts | elevenlabs
  voice: "anastasia"  # из voices/
  device: "cuda:0"
stt:
  engine: "whisper"
  model: "large-v3"
  device: "cpu"
```

### Phase 1 файловая структура:
```
sales_agent/
├── __init__.py
├── config.yaml          ← настройки агента
├── server.py            ← FastAPI app (порт 8001)
├── core/
│   ├── __init__.py
│   ├── fsm.py           ← ConversationFSM
│   ├── sales_brain.py   ← LLM с streaming + function calling
│   ├── prompts.py       ← Prompt engine (state-aware)
│   ├── barge_in.py      ← Interruption controller
│   ├── text_normalizer.py ← TTS text prep
│   ├── call_recorder.py ← WAV + JSON storage
│   └── session.py       ← CallSession (всё вместе)
├── pipeline/
│   ├── __init__.py
│   ├── audio_in.py      ← VAD + noise + STT chain
│   └── audio_out.py     ← TTS + queue + barge-in
├── data/
│   ├── objections.json  ← Библиотека возражений
│   ├── scripts/
│   │   ├── intro.txt
│   │   ├── pitch.txt
│   │   └── closing.txt
│   └── personas/
│       └── sarah.yaml   ← Persona definition
├── static/
│   └── index.html       ← Demo UI
└── docs/
    ├── BLIND_SPOTS_MAP.md  ← этот файл
    └── ARCHITECTURE_PHASES.md
```

---

## PHASE 2 — INTELLIGENCE LAYER
**Цель:** Агент умнеет — учится на данных, A/B тестирует скрипты.

### Компоненты Phase 2:

#### 2.1 Post-Call Analytics (`sales_agent/analytics/`)
- LLM scoring: транскрипт → Haiku → JSON: `{score, conversion_probability, objections, talk_ratio, sentiment_arc}`
- SQLite storage: все звонки с метриками
- Telegram daily report: конверсия, топ-возражения, avg duration

#### 2.2 A/B Testing Framework
- Версии скриптов: A/B/C по opening, pitch, close
- Random assignment per call
- Tracking: какая версия → какой outcome
- Auto-suggest: "Script B показал +15% conversion на этой неделе"

#### 2.3 AMD (Answering Machine Detection) — software version
- Анализ первого ответа: длина паузы, паттерн речи
- NLP classifier: "Please leave a message" / "press 1" → VOICEMAIL state
- Без Telnyx AMD (тот приходит в Phase 3)

#### 2.4 Sentiment Detector (real-time)
- Анализ каждого utterance prospect: позитив/негатив/нейтрал
- Адаптация агента: если негатив растёт → переключить тактику или завершить

#### 2.5 Campaign Manager (basic)
- SQLite: contacts list (phone, name, company, status, next_retry)
- Call scheduler: соблюдение временных окон (9am-8pm)
- Retry logic: no-answer → retry after 4hr, max 3 attempts
- Web UI для управления контактами

#### 2.6 ElevenLabs Integration (optional upgrade)
- Если нужно ещё лучшее качество голоса для English
- API wrapper поверх text_normalizer + TTS pipeline
- A/B: XTTS v2 vs ElevenLabs Flash v2.5

---

## PHASE 3 — TELEPHONY & SCALE
**Цель:** Реальные звонки на реальные номера. Масштаб до 50+ concurrent.

### Компоненты Phase 3:

#### 3.1 Telnyx Integration
- TeXML app + WebSocket streaming
- OPUS 16kHz codec configuration
- Call events: call.initiated, call.answered, call.machine.detection.ended
- μ-law audio adapter (phone → 16kHz PCM для STT)
- Outbound dialing via REST API

#### 3.2 LiveKit Agents (optional, evaluate)
- Semantic VAD (better than Silero for English phone)
- Native barge-in
- Built-in recording

#### 3.3 Telnyx AMD (Premium)
- Native AMD: $0.004/call
- Event: `call.machine.detection.ended` → `result: machine|human`
- If machine: switch to VOICEMAIL script, leave message, hang up

#### 3.4 Concurrent Call Manager
- Session pool: 5-50 concurrent calls
- GPU scheduling: XTTS queue with priority
- Circuit breaker: если GPU memory > 90% → reject new calls

#### 3.5 DNC & TCPA Compliance
- FTC DNC API integration
- Per-state calling hour validation
- Opt-out registry (internal SQLite + webhook)
- Consent logging

#### 3.6 CRM Integration
- Webhook: call outcome → update contact status
- Airtable / HubSpot / custom webhook
- Function calling: mid-call lookup (pricing, availability)

---

## ТЕХНИЧЕСКИЙ СТЕК

| Компонент | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Telephony | Browser WebSocket | Browser + basic VoIP | Telnyx SIP |
| STT | Whisper large-v3 (CPU) | + Deepgram fallback | Deepgram Nova-3 primary |
| LLM | Gemini 2.5 Flash | + A/B testing | + function calling CRM |
| TTS | XTTS v2 (GPU) | + ElevenLabs option | same |
| Orchestration | Custom FastAPI | + campaign manager | + LiveKit evaluate |
| Analytics | File logs | SQLite + LLM scoring | Full dashboard |
| Compliance | Script disclosure | + DNC check | + TCPA full |

---

## LATENCY BUDGET (Phase 1 цели)

```
Phone audio in → first audio out:
  Audio chunk (0.5s chunks):  500ms  [network/VAD]
  STT (Whisper CPU):          300ms  [existing: ~280ms]
  LLM TTFT (Gemini Flash):    400ms  [TTFT ~350-450ms]
  TTS first chunk (XTTS):     200ms  [streaming]
  ─────────────────────────────────
  TOTAL:                    ~1.4 sec ← TARGET: < 1.5 sec ✅

vs. Human phone response:  1.0-1.5 sec (natural pause)
```

---

## COST ESTIMATE (Phase 1 — per 5-min call)

```
LLM (Gemini 2.5 Flash, ~30 exchanges, ~3000 tokens):
  Input:  2000 tok × $0.30/1M = $0.0006
  Output:  1000 tok × $2.50/1M = $0.0025
  TOTAL LLM: ~$0.003/call

STT (Whisper local): $0.00 (GPU)
TTS (XTTS local):    $0.00 (GPU)
Telephony (Phase 1 = browser): $0.00

TOTAL Phase 1: ~$0.003/call ← практически бесплатно
TOTAL Phase 3 с Telnyx+Deepgram: ~$0.06-0.08/call
```

---

## PHASE 1 IMPLEMENTATION ORDER

```
1. sales_agent/core/fsm.py                 ← State machine
2. sales_agent/data/objections.json        ← Objection library
3. sales_agent/core/prompts.py             ← Prompt engine
4. sales_agent/core/text_normalizer.py     ← TTS prep
5. sales_agent/core/sales_brain.py         ← LLM streaming + functions
6. sales_agent/core/barge_in.py            ← Interruption
7. sales_agent/core/call_recorder.py       ← Recording
8. sales_agent/core/session.py             ← Ties everything together
9. sales_agent/pipeline/audio_in.py        ← Audio input chain
10. sales_agent/pipeline/audio_out.py      ← Audio output chain
11. sales_agent/server.py                  ← FastAPI + WebSocket
12. sales_agent/static/index.html          ← Demo UI
```

---

## NON-CODE REQUIREMENTS (нужны от тебя)

До начала реализации промпта нужно ответить на:

1. **Продукт:** Что именно продаёт агент? (название, описание, цена)
2. **Целевая аудитория:** Кто получатель звонка? (должность, индустрия, размер компании)
3. **Главные боли:** Какие проблемы решает продукт?
4. **Persona агента:** Имя, пол, от чьего лица звонит?
5. **Цель звонка:** Что считается "закрытием"? (встреча, демо, покупка, форма)
6. **Топ-5 возражений:** Самые частые "нет" и как сейчас на них отвечают живые продавцы?
