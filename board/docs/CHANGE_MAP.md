# Board of Directors — Change Map
*Аудит слабых мест + план улучшений. Апрель 2026.*

---

## АУДИТ: ЧТО СЛОМАНО

### 1. Анкета при старте (КРИТИЧНО — UX)
**Проблема:** Каждую сессию пользователь заполняет 14+ полей (company, financials, market, team...).
Если компания уже есть в памяти — это лишняя работа. Убивает поток.

**Что должно быть:** Ввести только "какая компания" + "в чём проблема".
Система сама загружает профиль из памяти. Ничего лишнего.

---

### 2. Туннельное зрение директоров (КРИТИЧНО — качество анализа)
**Проблема:** Каждый директор видит ТОЛЬКО свои данные (data minimization) и анализирует только своё.
В R1 нет требования:
- Перечислить все известные факты по всей ситуации
- Назвать, что другие директора будут анализировать отдельно
- Явно обозначить свои зависимости от чужих доменов ("мои рекомендации предполагают, что у CFO runway > 12 мес")
- Флагнуть, если рекомендация рушится при других предположениях

**Что должно быть:** Systems Thinking Protocol — обязательный шаг перед доменным анализом.
Каждый директор держит ВСЮ картину в голове, но отвечает за своё.

---

### 3. Память — 2000 символов (КРИТИЧНО — долгосрочность)
**Проблема:** `MEMORY_MAX_CONTEXT_CHARS = 2000` ≈ 500 токенов.
Это ничто. Для годового ведения проектов нужно ~30k токенов рабочей памяти.
Текущая память плоская (один блок текста), без приоритизации.

**Что должно быть:**
- Бюджет: 24000 символов (≈30k токенов)
- Тиерная загрузка с relevance scoring:
  - T1 (всегда): профиль компании / проекта (2k)
  - T2 (недавние): последние 3 сессии (8k)
  - T3 (релевантные): семантически похожие прошлые (10k)
  - T4 (решения): принятые решения + что вышло (4k)
- Активный буфер: обновляется внутри сессии

---

### 4. Нет проектов (АРХИТЕКТУРНЫЙ ПРОБЕЛ)
**Проблема:** Память индексируется по company_name. Нет концепции "проекта".
Нельзя переключиться между "Проект A (реструктуризация)" и "Проект B (новый рынок)" для одной компании.
Нельзя вести несколько долгосрочных направлений параллельно.

**Что должно быть:**
- Проект = {name, description, company, phase, context_notes}
- Memory привязана к проекту, не только к компании
- Быстрое переключение: dropdown в интерфейсе

---

### 5. Нет корпоративного уровня (ЗРЕЛОСТЬ)
**Проблема:**
- Нет дашборда принятых решений (что решили → что вышло)
- Нет cross-project learning ("в похожей ситуации с другим проектом мы сделали X и получили Y")
- Нет истории сессий в sidebar с быстрым доступом

---

## КАРТА ИЗМЕНЕНИЙ

### Изменение 1: Убрать анкету → минимальный ввод
**Файлы:** `static/index.html`, `server.py`, `board/memory_store.py`

```
ДО: 5 секций, 14+ полей, обязательные финансы каждый раз
ПОСЛЕ:
  - [Company name] с автокомплитом из истории
  - [Что за проблема?] textarea + mic
  - Если компания в памяти → показать: "Loaded: Stage, Revenue, Team..." с кнопкой Edit
  - Если новая компания → появляется блок финансов (минимум 1 поле)
```

New API: `GET /api/board/companies` → список с профилями для autocomplete
New: `GET /api/board/company/{name}/profile` → полный профиль
`memory_store.py`: добавить `list_companies()`, `get_company_profile_summary()`
`data_gate.py`: принимать memory_override (загруженный профиль как дефолты)

---

### Изменение 2: Systems Thinking Protocol (анти-туннельное зрение)
**Файлы:** `board/director_base.py`, `board/models.py`, `board/directors.py`

В `OUTPUT_FORMAT_INSTRUCTIONS` добавить обязательный раздел ПЕРЕД доменным анализом:
```json
"systems_scan": {
  "all_known_facts": ["FACT: Revenue is $X", "FACT: Runway is Y months", ...],
  "other_domains": {
    "CSO": "Will analyze: market positioning, competitive response, growth strategy",
    "CFO": "Will analyze: unit economics, burn rate, fundraising",
    "CTO": "Will analyze: tech feasibility, timeline, tech debt"
  },
  "my_dependencies": [
    "My recommendation ASSUMES CFO confirms runway > 12 months",
    "My recommendation ASSUMES CTO estimates delivery < 6 months"
  ],
  "breaks_if": [
    "Strategy collapses if burn rate doubles (ask CFO)",
    "Timeline breaks if team < 5 engineers (ask CTO)"
  ]
}
```

`models.py`: добавить `SystemsScan` + `cross_domain_risks` в `DirectorResponse`
`director_base.py`: parse `systems_scan` из JSON output
`directors.py`: добавить instruction во все 3 директора

---

### Изменение 3: Smart Memory Buffer (30k)
**Файлы:** `config.py`, `board/memory_store.py`

`config.py`:
- `MEMORY_MAX_CONTEXT_CHARS = 2000` → `24000`
- `MEMORY_TIER_COMPANY = 2000` (профиль компании/проекта)
- `MEMORY_TIER_RECENT = 8000` (последние 3 сессии)
- `MEMORY_TIER_SIMILAR = 10000` (семантически похожие)
- `MEMORY_TIER_DECISIONS = 4000` (решения + исходы)

`memory_store.py`:
- Новый метод `get_smart_context(company, project, problem)` → tiered loading
- Relevance score = recency_weight * 0.3 + semantic_similarity * 0.5 + outcome_available * 0.2
- Добавить `projects` table
- Добавить `get_project_context(project_id)`

---

### Изменение 4: Проекты
**Файлы:** `board/memory_store.py`, `server.py`, `static/index.html`

SQLite table:
```sql
CREATE TABLE projects (
  project_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT DEFAULT '',
  company_name TEXT NOT NULL,
  status TEXT DEFAULT 'active',  -- active/archived
  phase TEXT DEFAULT '',         -- e.g. "Q2 2026 - Expansion"
  context_notes TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)
```

API:
- `GET /api/board/projects` → list
- `POST /api/board/projects` → create
- `PATCH /api/board/projects/{id}` → update context_notes, phase
- `POST /api/board/session` принимает `project_id` (опционально)

Frontend:
- Project selector в sidebar (active project badge)
- Быстрое создание проекта из board form

---

### Изменение 5: Corporate dashboard (в рамках board view)
**Файлы:** `static/index.html`, `server.py`

- Session history: последние 10 сессий в sidebar с company + problem preview
- Decision ledger: список принятых решений (user_choice) + outcomes
- Кнопка "Resume" → загружает старую сессию для follow-up

---

## ПОРЯДОК РЕАЛИЗАЦИИ

1. `config.py` — увеличить memory budget (2 мин)
2. `board/models.py` — добавить SystemsScan + cross_domain_risks (15 мин)
3. `board/director_base.py` — Systems Thinking Protocol prompt (20 мин)
4. `board/memory_store.py` — projects table + smart context (45 мин)
5. `server.py` — companies API + projects API (20 мин)
6. `static/index.html` — minimal form + autocomplete + project switcher (60 мин)

---

## ОЦЕНКА КАК ПОЛЬЗОВАТЕЛЬ

**Что я ожидаю от системы Board of Directors уровня большой корпорации:**

1. Я описываю проблему одним абзацем — система всё знает о компании
2. Директора не противоречат друг другу по базовым фактам (видят одну реальность)
3. CEO синтезирует с учётом ВСЕЙ истории, а не только сегодняшней сессии
4. Могу переключиться между проектами одной компании в 1 клик
5. Через год пользования — система помнит ВСЁ и учится на прошлых решениях
6. Не нужно повторять одно и то же каждый раз

**Текущая система закрывает:** пп. 2 (частично), 3 (частично)
**После изменений закроет:** все 6 пунктов
