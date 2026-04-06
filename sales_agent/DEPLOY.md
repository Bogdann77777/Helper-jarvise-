# Sales Agent — Deployment Guide
**From "code ready" to "making real calls"**

---

## Шаг 1: Тест на ПК (прямо сейчас, без телефонии)

```bash
cd "E:/project/ceo 2.0/cli helper"
cd sales_agent
python server.py
# → http://localhost:8001
# Открой браузер, нажми Start Call — говоришь с агентом
```

Это Phase 1 — проверяешь разговор, FSM, голос, latency.

---

## Шаг 2: Тест с реальными звонками (ngrok, БЕЗ VPS)

Telnyx нужен публичный HTTPS URL. Решение — ngrok (уже есть start_ngrok.bat):

```bash
# 1. Запусти сервер
python server.py

# 2. В другом терминале — открой туннель
cd "E:/project/ceo 2.0/cli helper"
start_ngrok.bat
# Получишь URL типа: https://abc123.ngrok.io

# 3. Скопируй URL — это твой webhook в Telnyx
```

### В Telnyx Dashboard (когда купишь):
1. **Connections** → Create TeXML Application
2. **Streaming URL:** `wss://abc123.ngrok.io/telnyx/stream`
3. **Webhooks URL:** `https://abc123.ngrok.io/telnyx/events`
4. Codec: **PCMU** (G.711 μ-law, 8kHz — универсально)
5. **Enable AMD Premium** (бесплатно, определяет автоответчик)
6. Enable **Record calls** (для QA)
7. Купи US local number → привяжи к этому приложению

### В .env добавь:
```
TELNYX_API_KEY=KEY_...
TELNYX_FROM_NUMBER=+12125551234
TELNYX_CONNECTION_ID=1234567890   # ID твоего TeXML App
SERVER_URL=https://abc123.ngrok.io
```

**Перезапусти сервер** — теперь он будет принимать звонки от Telnyx.

### Первый тест:
```bash
# Позвони другу или самому себе на другой телефон
python -c "
import asyncio
from telephony.call_manager import get_call_manager
async def main():
    m = get_call_manager()
    call_id = await m.dial('+380991234567', 1, 1, 'https://abc123.ngrok.io')
    print('Dialing:', call_id)
asyncio.run(main())
"
```

---

## Шаг 3: Production (VPS для масштаба и минимального latency)

Для серьёзных продаж нужен VPS рядом с Telnyx (US датацентры):

**Рекомендация:** Hetzner US (Ashburn, VA) — $5-15/мес, 1-4 cores
- Telnyx US media servers в Ashburn → latency к ним: **<10ms**
- Vs твой ПК в Украине → latency: **150-200ms** (добавляет к E2E)

**Но!** STT+TTS остаются на твоём ПК (GPU) — это не проблема:
```
Telnyx → VPS (relay) → твой ПК (STT+TTS+LLM) → VPS → Telnyx
```

**Или:** запусти всё на ПК + ngrok для тестов с друзьями (latency ≈1.5-2с — приемлемо для теста).

### VPS Setup (Ubuntu 22.04):
```bash
# На VPS:
sudo apt install python3.12 python3.12-venv
git clone <repo> или scp папку
pip install fastapi uvicorn httpx pyyaml python-dotenv

# Запуск (без GPU — только relay):
uvicorn server:app --host 0.0.0.0 --port 8001 --workers 2

# Тут STT/TTS не нужны — звонки маршрутизируются на твой ПК
```

---

## Шаг 4: Запуск Campaign Manager

```bash
# Создай кампанию и добавь контакты
python -c "
from campaign.db import init_db, create_campaign, add_contact
init_db()
camp_id = create_campaign('Test Campaign', 'template.yaml')
add_contact(camp_id, '+12125551234', 'John', 'Smith', 'ACME Corp', 'Manager')
print('Campaign ID:', camp_id)
"

# Dry run — покажет кого бы позвонил, без реальных звонков
python campaign/manager.py --campaign 1 --dry-run

# Реальный запуск
python campaign/manager.py --campaign 1 --server-url https://your-url.com --concurrency 3
```

---

## Latency Budget (твой ПК + ngrok)

```
Phone audio → Telnyx (US) → ngrok → твой ПК:  ~200ms  [network]
STT (Whisper, CPU):                            ~280ms
LLM TTFT (Gemini Flash, API):                  ~450ms
TTS first chunk (XTTS, GPU):                   ~150ms
ПК → ngrok → Telnyx → phone:                  ~200ms
─────────────────────────────────────────────────────
TOTAL E2E:                                    ~1.28s  ← OK для теста

Human natural pause:                          1.0-1.5s
```

Для продакшена (VPS в US) → latency ~200ms меньше = **~1.0s** — профессионально.

---

## Что нужно заплатить

| Сервис | Цена | Когда |
|--------|------|-------|
| Telnyx номер | $1-2/мес | При старте |
| Telnyx звонки | $0.005/мин (outbound US) | По факту |
| Telnyx AMD | Бесплатно | — |
| Telnyx запись | $0.002/мин | По факту |
| LLM (Gemini Flash) | ~$0.003/звонок | По факту |
| ngrok (paid) | $8/мес | Если >5 соед. |
| VPS Hetzner | $5/мес | Для продакшена |
| **Итого тест** | **~$3-5** | Первые 100 звонков |

---

## Checklist перед первым реальным звонком

- [ ] Заполнен `data/personas/template.yaml` (продукт, боли, возражения)
- [ ] Проверен скрипт в браузерном демо (Start Call)
- [ ] Установлен TELNYX_API_KEY в .env
- [ ] Куплен US local number в Telnyx
- [ ] ngrok запущен и URL в Telnyx dashboard
- [ ] Тестовый звонок на свой номер прошёл
- [ ] AMD корректно определяет человека vs автоответчик
