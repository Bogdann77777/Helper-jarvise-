"""
copywriting_agent_tool.py — AI Copywriting Agent для CLI helper.

Pipeline (enterprise-grade, v2):

  1. VOC + JTBD Forces Analysis
     — боли/желания/страхи + Push/Pull/Anxiety/Habit из Джобс-ту-би-Дан

  2. Strategy Selection (ELM-aware)
     — фреймворк (PAS/AIDA/BAB/DIC) + ELM persuasion route по температуре аудитории
     — Cold → peripheral route (эмоция, соц. доказательство)
     — Warm → central route (конкретные выгоды, работа с возражениями)
     — Retarget → loss aversion + urgency (Kahneman prospect theory)

  3. Parallel Copy Generation ×3 (Cold / Warm / Retarget)
     — ThreadPoolExecutor: все три температуры одновременно
     — каждый вариант тегируется эмоцией (8-class taxonomy)

  4. Critic Loop
     — если top_score < 6.5 → Critic LLM даёт конкретный фидбек
     — regeneration с feedback в контексте
     — max 1 итерация (2 попытки)

  5. Multi-judge Scoring (расширенный)
     — hook_strength, voc_alignment, clarity, cta_strength (как раньше)
     — + Fogg B=MAP: motivation, ability, prompt (Fogg Behavior Model)
     — Judge 1 (Claude) + Judge 2 (Gemini) параллельно
     — weighted total: MAP dims × 1.2 (более предсказуют conversion)

Источники паттернов:
  - Persado: emotion taxonomy + pre-generation scoring
  - WPP Open: Critic LLM loop
  - ELM (Petty & Cacioppo): central vs peripheral route by audience temp
  - Fogg Behavior Model: B = Motivation × Ability × Prompt
  - JTBD (Christensen): 4 forces diagram
  - Kahneman: loss aversion 2x gain for retarget
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx

# ── Пути ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
MEMORY_FILE = BASE_DIR / "memory" / "copywriting_memory.json"
OUTPUTS_DIR = BASE_DIR / "outputs" / "copywriting"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Models ───────────────────────────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_COPY     = "anthropic/claude-sonnet-4-6"
MODEL_JUDGE2   = "google/gemini-2.5-flash-lite-preview-09-2025"
MODEL_CRITIC   = "anthropic/claude-sonnet-4-6"   # critic = same model, different role

MAX_RETRIES     = 3
REQUEST_TIMEOUT = 90.0
CRITIC_THRESHOLD = 6.5   # если top score < этого → запускаем critic loop

# ── Emotion taxonomy (8-class, based on Persado's 15-class compressed) ───────
EMOTIONS_8 = [
    "urgency",        # FOMO, scarcity, time pressure
    "curiosity",      # intrigue, mystery, open loops
    "trust",          # safety, reliability, authority
    "aspiration",     # identity, achievement, status
    "empathy",        # understanding, validation, "I get you"
    "excitement",     # enthusiasm, energy, hype
    "anxiety_relief", # problem solved, fear removed, reassurance
    "social_proof",   # belonging, others like me, community
]

# ── ELM Persuasion Routes ────────────────────────────────────────────────────
ELM_ROUTES = {
    "cold": {
        "route": "peripheral",
        "description": "Audience has no prior brand contact. Use emotional hooks, bold claims, "
                       "social proof. Make it visceral. Central argument is secondary.",
        "primary_emotions": ["curiosity", "aspiration", "urgency"],
        "avoid": ["detailed specs", "price comparison", "brand references"],
        "fogg_focus": "Motivation first — establish why they should care before asking anything",
    },
    "warm": {
        "route": "central",
        "description": "Audience visited site or engaged with prior ad. They are considering. "
                       "Use specific benefits, address objections, reduce cognitive load.",
        "primary_emotions": ["trust", "anxiety_relief", "social_proof"],
        "avoid": ["generic claims", "vague CTAs", "re-introducing the problem they already know"],
        "fogg_focus": "Ability focus — make action feel easy, fast, low-risk",
    },
    "retarget": {
        "route": "loss_aversion",
        "description": "Audience showed purchase intent (cart, checkout, high-intent page). "
                       "Apply Kahneman: losses feel 2x more painful than gains. "
                       "Use specific urgency (not generic), objection removal, social proof for exact product.",
        "primary_emotions": ["urgency", "anxiety_relief", "trust"],
        "avoid": ["re-explaining the product", "awareness-style hooks", "soft CTAs"],
        "fogg_focus": "Prompt focus — CTA must be specific, immediate, low-friction",
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── LLM Client ───────────────────────────────────────────────────────────────
def _call_llm(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """Синхронный вызов OpenRouter с exponential backoff."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        try:
            sys.path.insert(0, str(BASE_DIR))
            from config import OPENROUTER_API_KEY as cfg_key
            api_key = cfg_key
        except (ImportError, AttributeError):
            pass
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY не задан.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "CLI Helper / Copywriting Agent v2",
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                r = client.post(OPENROUTER_URL, json=payload, headers=headers)
                if r.status_code == 200:
                    content = r.json()["choices"][0]["message"]["content"]
                    logger.info(f"[{model.split('/')[-1]}] {len(content)} chars (attempt {attempt + 1})")
                    return content
                if r.status_code in (429, 500, 503):
                    wait = 2 ** attempt
                    logger.warning(f"[{model}] HTTP {r.status_code}, retry in {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text[:300]}")
        except httpx.TimeoutException:
            last_error = f"Timeout attempt {attempt + 1}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
        except httpx.HTTPError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Все {MAX_RETRIES} попытки провалились [{model}]: {last_error}")


def _parse_json(text: str) -> dict:
    """Парсинг JSON из LLM — убирает markdown-обёртки."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


# ── Step 1: VOC + JTBD Forces Analysis ──────────────────────────────────────
def _voc_analysis(raw_data: str) -> dict:
    """
    Извлекает Voice of Customer insights + JTBD 4-Forces diagram.
    JTBD Forces: Push (недовольство текущим), Pull (желание нового),
                 Anxiety (страх перехода), Habit (инерция).
    """
    logger.info("Step 1/4: VOC + JTBD Forces Analysis...")

    # Если VOC > 12k символов — сначала сжимаем через LLM
    if len(raw_data) > 12000:
        logger.info(f"VOC data {len(raw_data)} chars — summarizing first...")
        raw_data = _call_llm(
            MODEL_COPY,
            [{
                "role": "user",
                "content": (
                    "Summarize the following customer data, preserving ALL specific pain points, "
                    "exact quotes, desires, fears, and unique language customers use. "
                    "Do not paraphrase quotes. Keep emotional language verbatim. "
                    f"Max 6000 chars.\n\n{raw_data[:20000]}"
                ),
            }],
            temperature=0.2,
            max_tokens=2000,
        )

    system = (
        "You are a Voice of Customer analyst + JTBD researcher with 10+ years experience. "
        "Extract EXACT customer language — never paraphrase, always quote verbatim. "
        "JTBD 4-Forces: what pushes people away from current solution, "
        "what pulls them toward new, what creates anxiety about switching, what keeps them stuck. "
        "Output ONLY valid JSON."
    )
    user = f"""Analyze this customer data. Extract VOC Map + JTBD Forces.

DATA:
{raw_data[:10000]}

Output:
{{
  "pain_points": [
    {{"theme": "...", "quote": "exact words", "frequency": "high|med|low"}}
  ],
  "desires": [
    {{"theme": "...", "quote": "exact words"}}
  ],
  "fears": [
    {{"theme": "...", "quote": "exact words"}}
  ],
  "emotional_triggers": ["urgency", "trust", "aspiration", "FOMO"],
  "key_vocabulary": ["exact", "words", "customers", "use"],
  "top_insight": "Single most powerful insight for copywriting (1-2 sentences)",
  "jtbd_forces": {{
    "push": "What makes people dissatisfied with their current solution (specific)",
    "pull": "What makes the new solution desirable (specific promise)",
    "anxiety": "What fears stop them from switching (specific objection)",
    "habit": "What keeps them stuck with current approach (inertia reason)"
  }}
}}

Extract 3-5 items per category. Prioritize specificity. Generic = no value."""

    raw = _call_llm(
        MODEL_COPY,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=2000,
    )
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"VOC JSON parse failed: {e}. Fallback.")
        return {
            "pain_points": [], "desires": [], "fears": [],
            "emotional_triggers": [], "key_vocabulary": [],
            "top_insight": raw[:500],
            "jtbd_forces": {"push": "", "pull": "", "anxiety": "", "habit": ""},
            "_parse_error": str(e),
        }


# ── Step 2: Strategy Selection (ELM-aware) ───────────────────────────────────
def _strategy_selection(
    voc_map: dict,
    product_brief: str,
    platform: str,
    framework: str = "auto",
    audience_temp: str = "cold",
) -> dict:
    """
    Выбирает фреймворк, тон, угол атаки.
    Учитывает ELM persuasion route по температуре аудитории.
    """
    elm = ELM_ROUTES.get(audience_temp, ELM_ROUTES["cold"])

    platform_context = {
        "TikTok": "15-30s, Gen Z/Millennial, authentic > polished, trend-aware, no logo first 3s",
        "Meta":   "Static/video, broad audience, benefit-driven, value prop in 3s, captions needed",
        "YouTube": "15-60s pre-roll, intent-based, problem/solution narrative, skip at 5s",
        "Instagram": "Visual-first, aspirational, Reels/Stories, trending audio",
    }.get(platform, "Social media ad, mobile-first, scroll-stopper needed")

    jtbd = voc_map.get("jtbd_forces", {})

    system = (
        "You are a direct response strategist using ELM (Elaboration Likelihood Model) and JTBD. "
        "Select angle based ONLY on VOC evidence + audience temperature. "
        "Output ONLY valid JSON."
    )
    user = f"""Select optimal copywriting strategy.

PLATFORM: {platform} — {platform_context}
AUDIENCE TEMPERATURE: {audience_temp.upper()}
ELM ROUTE: {elm['route']} — {elm['description']}
PRIMARY EMOTIONS TO TARGET: {elm['primary_emotions']}
FOGG FOCUS: {elm['fogg_focus']}

PRODUCT: {product_brief[:1500]}
VOC MAP: {json.dumps(voc_map, ensure_ascii=False)[:2000]}
FRAMEWORK: {framework} (auto = pick best)

JTBD Forces:
- Push (why they leave current): {jtbd.get('push', 'unknown')}
- Pull (why they want new): {jtbd.get('pull', 'unknown')}
- Anxiety (what stops them): {jtbd.get('anxiety', 'unknown')}
- Habit (what keeps them stuck): {jtbd.get('habit', 'unknown')}

Frameworks: PAS (strong pain), AIDA (awareness), BAB (transformation), DIC (curiosity)

Output:
{{
  "framework": "PAS|AIDA|BAB|DIC",
  "framework_reason": "why this fits VOC + ELM route (1 sentence)",
  "primary_message": "ONE core message — max 15 words",
  "tone": "aggressive|curious|empathetic|urgent|inspirational|humorous|direct",
  "hook_angle": "fear|desire|curiosity|shock|social_proof|urgency|identity|loss_aversion",
  "jtbd_lever": "which JTBD force to amplify: push|pull|anxiety_removal|habit_breaking",
  "avoid": ["clichés to avoid based on VOC"],
  "winning_patterns": ["specific tactics for this audience+platform combo"]
}}"""

    raw = _call_llm(
        MODEL_COPY,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.5,
        max_tokens=900,
    )
    try:
        result = _parse_json(raw)
        result["audience_temp"] = audience_temp
        result["elm_route"] = elm["route"]
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Strategy JSON parse failed: {e}")
        return {
            "framework": framework if framework != "auto" else "PAS",
            "primary_message": product_brief[:100],
            "tone": "direct",
            "hook_angle": "desire",
            "audience_temp": audience_temp,
            "elm_route": elm["route"],
            "_parse_error": str(e),
        }


# ── Step 3: Copy Generation (single temperature) ────────────────────────────
def _copy_generation(
    voc_map: dict,
    strategy: dict,
    product_brief: str,
    platform: str,
    num_variants: int = 5,
    audience_temp: str = "cold",
    critic_feedback: str = "",
) -> list[dict]:
    """
    Генерирует N вариантов для одной температуры аудитории.
    Если есть critic_feedback — включает его в контекст.
    """
    elm = ELM_ROUTES.get(audience_temp, ELM_ROUTES["cold"])

    platform_specs = {
        "TikTok":    "Hook: max 7 words. CTA: 3-5 words. Body: 30-50 words.",
        "Meta":      "Hook: max 8 words. CTA: 5-7 words. Body: 50-80 words.",
        "YouTube":   "Hook: 10-15 words (complete thought). CTA: direct action. Body: 60-100 words.",
        "Instagram": "Hook: max 7 words. CTA: 4-6 words. Body: 30-60 words.",
    }.get(platform, "Hook: max 8 words. CTA: 5 words. Body: 50-80 words.")

    memory     = _load_memory()
    win_note   = ""
    if memory.get("winning_hooks"):
        winners = [
            h for h in memory["winning_hooks"]
            if h.get("winner") and h.get("platform") == platform
            and h.get("audience_temp", "cold") == audience_temp
        ]
        if winners:
            win_note = "\nPROVEN WINNERS FOR THIS PLATFORM + AUDIENCE TEMP (mirror style):\n" + "\n".join(
                f'- "{w["hook"]}" (CTR: {w.get("ctr", "N/A")})' for w in winners[-3:]
            )

    critic_note = ""
    if critic_feedback:
        critic_note = f"\n\nCRITIC FEEDBACK ON PREVIOUS ATTEMPT (fix these issues):\n{critic_feedback}"

    system = (
        "You are a direct response copywriter with 15 years of paid ads experience. "
        "You write using the customer's EXACT language from VOC data. "
        "BANNED words: revolutionary, innovative, unique, game-changer, transform your life, unlock, unleash. "
        "Write like the audience talks — specific, visceral, human. "
        "Each variant must have a primary emotion tag from: "
        f"{', '.join(EMOTIONS_8)}. "
        "Output ONLY valid JSON."
    )
    user = f"""Generate {num_variants} distinct ad copy variants for {platform}.

AUDIENCE: {audience_temp.upper()} — {elm['description']}
ELM PERSUASION ROUTE: {elm['route']}
FOGG PRINCIPLE: {elm['fogg_focus']}

PRODUCT: {product_brief[:1500]}
FRAMEWORK: {strategy.get('framework', 'PAS')} | TONE: {strategy.get('tone', 'direct')}
HOOK ANGLE: {strategy.get('hook_angle', 'desire')}
PRIMARY MESSAGE: {strategy.get('primary_message', '')}
JTBD LEVER: {strategy.get('jtbd_lever', 'pull')}
PLATFORM SPECS: {platform_specs}
VOC VOCABULARY (use naturally): {json.dumps(voc_map.get('key_vocabulary', []))}
TOP INSIGHT: {voc_map.get('top_insight', '')}
JTBD FORCES: {json.dumps(voc_map.get('jtbd_forces', {}), ensure_ascii=False)}{win_note}{critic_note}

RULES:
- Each variant MUST use a completely different angle — zero repetition
- Mirror VOC vocabulary where natural
- Hook must stop scrolling on first read
- No passive voice, no "we" language
- For {audience_temp}: {', '.join(elm['primary_emotions'])} emotions should dominate
- Short slogan for text overlay: 2-6 words

Return ONLY this JSON:
{{
  "variants": [
    {{
      "id": "v1",
      "hook": "...",
      "headline": "...",
      "body": "...",
      "cta": "...",
      "short_slogan": "2-6 words",
      "angle": "what makes this variant unique (5-10 words)",
      "primary_emotion": "one from: urgency|curiosity|trust|aspiration|empathy|excitement|anxiety_relief|social_proof",
      "fogg_map": {{
        "motivation_hook": "how this creates motivation",
        "ability_ease": "what makes action feel easy",
        "prompt_clarity": "what makes CTA irresistible"
      }}
    }}
  ]
}}"""

    raw = _call_llm(
        MODEL_COPY,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.85,
        max_tokens=3500,
    )
    try:
        data = _parse_json(raw)
        variants = data.get("variants", [])
        # Tag all with audience temp
        for v in variants:
            v["audience_temp"] = audience_temp
        return variants
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Copy generation JSON parse failed ({audience_temp}): {e}")
        return []


# ── Step 4: Critic Loop ──────────────────────────────────────────────────────
def _critic_pass(
    variants: list[dict],
    voc_map: dict,
    platform: str,
    audience_temp: str,
    top_score: float,
) -> str | None:
    """
    Critic LLM анализирует варианты и возвращает конкретный фидбек для регенерации.
    Запускается только если top_score < CRITIC_THRESHOLD.
    Возвращает строку с feedback или None если всё ок.
    """
    if top_score >= CRITIC_THRESHOLD or not variants:
        return None

    logger.info(f"[CRITIC] top score {top_score:.1f} < {CRITIC_THRESHOLD} → critic pass for {audience_temp}")

    variants_summary = json.dumps(
        [{"id": v["id"], "hook": v["hook"], "cta": v["cta"], "total_score": v.get("total_score")} for v in variants[:3]],
        ensure_ascii=False, indent=2,
    )

    system = (
        "You are a ruthless direct response copywriting critic. "
        "Your job: identify SPECIFIC reasons why these ad variants underperform. "
        "Be concrete. No vague feedback like 'make it more engaging'. "
        "Every critique must name what specifically to fix and why."
    )
    user = f"""Critique these {audience_temp.upper()} audience ad variants for {platform}. Top score: {top_score:.1f}/10.

VOC KEY VOCABULARY: {voc_map.get('key_vocabulary', [])}
TOP INSIGHT: {voc_map.get('top_insight', '')}
JTBD FORCES: {json.dumps(voc_map.get('jtbd_forces', {}), ensure_ascii=False)}

VARIANTS:
{variants_summary}

Identify the 3 most critical problems. For each:
1. What specifically is wrong
2. What to do instead (concrete, actionable)
3. Which ELM/JTBD principle is being violated

Format as plain text, max 200 words total. Be direct and specific."""

    try:
        feedback = _call_llm(
            MODEL_CRITIC,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=400,
        )
        logger.info(f"[CRITIC] Feedback: {feedback[:100]}...")
        return feedback
    except Exception as e:
        logger.warning(f"[CRITIC] Failed: {e}")
        return None


# ── Step 5: Multi-judge Scoring (расширенный) ────────────────────────────────
def _multi_judge_scoring(
    variants: list[dict],
    voc_map: dict,
    platform: str,
    audience_temp: str = "cold",
) -> list[dict]:
    """
    Dual-judge scoring (Claude + Gemini) параллельно.
    7 dimensions: 4 стандартных + Fogg MAP (motivation, ability, prompt).
    MAP dimensions weighted ×1.2 (лучше предсказывают conversion).
    """
    if not variants:
        return []

    logger.info(f"Scoring {len(variants)} variants ({audience_temp}) — 2 judges in parallel...")

    variants_json = json.dumps(variants, ensure_ascii=False, indent=2)
    voc_vocab = voc_map.get("key_vocabulary", [])
    elm = ELM_ROUTES.get(audience_temp, ELM_ROUTES["cold"])

    scoring_prompt = f"""Rate these {platform} ad copy variants for {audience_temp.upper()} audience.
ELM route: {elm['route']}. Score 1-10 on each dimension.

VOC VOCABULARY (higher if copy uses these): {voc_vocab[:8]}

VARIANTS:
{variants_json}

Scoring dimensions:
1. hook_strength: Stops scrolling? Irresistible opening? (1=weak, 10=impossible to ignore)
2. voc_alignment: Uses customer's own language? Sounds like audience talks? (1=generic, 10=perfect mirror)
3. clarity: Clear message in 3 seconds? No confusion? (1=confusing, 10=crystal clear)
4. cta_strength: Compelling, specific CTA? (1=weak/vague, 10=must click now)
5. motivation: Does copy create sufficient desire/urgency for this audience temp? (Fogg M)
6. ability: Does it minimize perceived effort/risk/complexity? (Fogg A)
7. prompt: Is CTA fired at right moment with right phrasing for {audience_temp}? (Fogg P)

Return ONLY this JSON:
{{
  "scores": {{
    "v1": {{"hook_strength": 0, "voc_alignment": 0, "clarity": 0, "cta_strength": 0, "motivation": 0, "ability": 0, "prompt": 0}},
    "v2": {{"hook_strength": 0, "voc_alignment": 0, "clarity": 0, "cta_strength": 0, "motivation": 0, "ability": 0, "prompt": 0}}
  }}
}}"""

    judge_system = (
        "You are a direct response copywriting expert and judge. "
        "Rate OBJECTIVELY. No bias toward longer or fancier copy. "
        "Fogg MAP dimensions: M=motivation created, A=ease of action, P=quality of call-to-action prompt. "
        "Output ONLY valid JSON."
    )

    # Параллельный запуск обоих судей
    judge1_scores: dict = {}
    judge2_scores: dict = {}

    def _judge1():
        try:
            raw = _call_llm(
                MODEL_COPY,
                [{"role": "system", "content": judge_system}, {"role": "user", "content": scoring_prompt}],
                temperature=0.1, max_tokens=800,
            )
            return _parse_json(raw).get("scores", {})
        except Exception as e:
            logger.warning(f"Judge 1 failed: {e}")
            return {}

    def _judge2():
        try:
            raw = _call_llm(
                MODEL_JUDGE2,
                [{"role": "system", "content": judge_system}, {"role": "user", "content": scoring_prompt}],
                temperature=0.1, max_tokens=800,
            )
            return _parse_json(raw).get("scores", {})
        except Exception as e:
            logger.warning(f"Judge 2 (Gemini) failed: {e}")
            return {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(_judge1)
        f2 = executor.submit(_judge2)
        judge1_scores = f1.result()
        judge2_scores = f2.result()

    # Standard dims (weight 1.0) + MAP dims (weight 1.2)
    std_dims = ["hook_strength", "voc_alignment", "clarity", "cta_strength"]
    map_dims = ["motivation", "ability", "prompt"]

    for v in variants:
        vid = v["id"]
        s1 = judge1_scores.get(vid, {})
        s2 = judge2_scores.get(vid, {})

        scored: dict = {}
        for dim in std_dims + map_dims:
            vals = []
            for s in [s1, s2]:
                if s.get(dim) is not None:
                    try:
                        vals.append(float(s[dim]))
                    except (TypeError, ValueError):
                        pass
            scored[dim] = round(sum(vals) / len(vals), 1) if vals else 0.0

        # Weighted total: MAP dims × 1.2
        std_sum = sum(scored[d] for d in std_dims)
        map_sum = sum(scored[d] for d in map_dims) * 1.2
        total_dims = len(std_dims) + len(map_dims) * 1.2
        v["scores"] = scored
        v["total_score"] = round((std_sum + map_sum) / total_dims, 1)
        v["dual_judge"] = bool(judge2_scores)
        v["fogg_map_avg"] = round(sum(scored[d] for d in map_dims) / 3, 1)

    variants.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    if variants:
        variants[0]["recommended"] = True

    return variants


# ── Full Pipeline for one temperature ────────────────────────────────────────
def _run_for_temperature(
    audience_temp: str,
    voc_map: dict,
    product_brief: str,
    platform: str,
    num_variants: int,
    framework: str,
) -> dict:
    """
    Полный sub-pipeline для одной температуры аудитории.
    Запускается в ThreadPoolExecutor.
    """
    logger.info(f"[{audience_temp.upper()}] Starting pipeline...")

    strategy = _strategy_selection(voc_map, product_brief, platform, framework, audience_temp)
    variants = _copy_generation(voc_map, strategy, product_brief, platform, num_variants, audience_temp)
    scored = _multi_judge_scoring(variants, voc_map, platform, audience_temp)

    # Critic loop — если топ-скор низкий
    if scored:
        top_score = scored[0].get("total_score", 0)
        feedback = _critic_pass(scored, voc_map, platform, audience_temp, top_score)
        if feedback:
            logger.info(f"[{audience_temp.upper()}] Critic feedback → regenerating...")
            improved = _copy_generation(
                voc_map, strategy, product_brief, platform, num_variants,
                audience_temp, critic_feedback=feedback,
            )
            if improved:
                improved_scored = _multi_judge_scoring(improved, voc_map, platform, audience_temp)
                # Берём лучших из обоих запусков
                combined = scored + improved_scored
                combined.sort(key=lambda x: x.get("total_score", 0), reverse=True)
                combined = combined[:num_variants]
                if combined:
                    combined[0]["recommended"] = True
                scored = combined
                logger.info(
                    f"[{audience_temp.upper()}] After critic: "
                    f"top {scored[0].get('total_score', 0):.1f}/10"
                )

    top_score = scored[0].get("total_score", 0) if scored else 0
    logger.info(f"[{audience_temp.upper()}] Done. Top score: {top_score:.1f}/10")

    return {
        "audience_temp": audience_temp,
        "elm_route": ELM_ROUTES[audience_temp]["route"],
        "strategy": strategy,
        "variants": scored,
        "top_recommendation": scored[0] if scored else None,
    }


# ── Memory ───────────────────────────────────────────────────────────────────
def _load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"winning_hooks": [], "losing_patterns": []}


def _save_to_memory(result: dict, platform: str) -> None:
    memory = _load_memory()
    for temp_data in result.get("by_temperature", {}).values():
        top = temp_data.get("top_recommendation")
        if top:
            entry = {
                "date": datetime.now().isoformat(),
                "platform": platform,
                "audience_temp": top.get("audience_temp", "cold"),
                "hook": top.get("hook", ""),
                "score": top.get("total_score", 0),
                "fogg_map_avg": top.get("fogg_map_avg", 0),
                "primary_emotion": top.get("primary_emotion", ""),
                "ctr": None,
                "winner": False,
            }
            memory.setdefault("winning_hooks", []).append(entry)
    memory["winning_hooks"] = memory["winning_hooks"][-100:]
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_winner(hook_text: str, ctr: float, platform: str = "", audience_temp: str = "") -> None:
    """Отметить хук как победителя с реальным CTR после кампании."""
    memory = _load_memory()
    for entry in memory.get("winning_hooks", []):
        if entry.get("hook") == hook_text:
            if platform and entry.get("platform") != platform:
                continue
            if audience_temp and entry.get("audience_temp") != audience_temp:
                continue
            entry["ctr"] = ctr
            entry["winner"] = True
            break
    MEMORY_FILE.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Marked winner: '{hook_text[:50]}' CTR={ctr:.1%}")


# ── Main Function ─────────────────────────────────────────────────────────────
def run_copywriting_agent(
    voc_data: str,
    product_brief: str,
    platform: str = "TikTok",
    num_variants: int = 5,
    framework: str = "auto",
    audience_temps: list[str] | None = None,
    save_output: bool = True,
) -> dict:
    """
    Запускает полный copywriting pipeline v2.

    Args:
        voc_data:       Сырые данные аудитории (отзывы, форумы, интервью)
        product_brief:  Описание продукта, УТП, цена, целевая аудитория
        platform:       TikTok | Meta | YouTube | Instagram
        num_variants:   Вариантов на каждую температуру (3-7 рекомендуется)
        framework:      auto | PAS | AIDA | BAB | DIC
        audience_temps: ["cold", "warm", "retarget"] или любое подмножество
                       (default: все три параллельно)
        save_output:    Сохранить JSON в outputs/copywriting/

    Returns:
        dict: {
          metadata,
          voc_map,         # VOC + JTBD forces
          by_temperature: {
            "cold":    {strategy, variants, top_recommendation, elm_route},
            "warm":    {strategy, variants, top_recommendation, elm_route},
            "retarget":{strategy, variants, top_recommendation, elm_route},
          },
          top_recommendation,  # общий победитель по всем температурам
        }
    """
    start_time = time.time()
    if audience_temps is None:
        audience_temps = ["cold", "warm", "retarget"]

    logger.info(
        f"=== Copywriting Agent v2 START | {platform} | "
        f"temps={audience_temps} | variants={num_variants} ===",
    )

    if not voc_data.strip():
        raise ValueError("voc_data пустой.")
    if not product_brief.strip():
        raise ValueError("product_brief пустой.")

    # Step 1: VOC + JTBD (один раз для всех температур)
    voc_map = _voc_analysis(voc_data)

    # Steps 2-5: Параллельно для каждой температуры аудитории
    by_temperature: dict = {}
    with ThreadPoolExecutor(max_workers=len(audience_temps)) as executor:
        futures = {
            executor.submit(
                _run_for_temperature,
                temp, voc_map, product_brief, platform, num_variants, framework,
            ): temp
            for temp in audience_temps
        }
        for future in as_completed(futures):
            temp = futures[future]
            try:
                by_temperature[temp] = future.result()
            except Exception as e:
                logger.error(f"[{temp}] Pipeline failed: {e}", exc_info=True)
                by_temperature[temp] = {"error": str(e), "audience_temp": temp}

    # Общий топ-победитель по всем температурам
    all_tops = [
        by_temperature[t]["top_recommendation"]
        for t in audience_temps
        if by_temperature.get(t, {}).get("top_recommendation")
    ]
    overall_top = max(all_tops, key=lambda v: v.get("total_score", 0)) if all_tops else None

    result = {
        "metadata": {
            "platform": platform,
            "framework": "auto",
            "audience_temps": audience_temps,
            "num_variants_per_temp": num_variants,
            "critic_threshold": CRITIC_THRESHOLD,
            "generated_at": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - start_time, 1),
        },
        "voc_map": voc_map,
        "by_temperature": by_temperature,
        "top_recommendation": overall_top,
    }

    if save_output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUTS_DIR / f"copy_{platform.lower()}_{ts}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_file"] = str(out_path)
        logger.info(f"Saved → {out_path}")
        _save_to_memory(result, platform)

    logger.info(
        f"=== DONE in {result['metadata']['elapsed_sec']}s | "
        f"overall top: {overall_top.get('total_score', 'N/A') if overall_top else 'N/A'}/10 ===",
    )
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI Copywriting Agent v2 — VOC + JTBD + ELM + Fogg MAP + Critic Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python copywriting_agent_tool.py --voc reviews.txt --brief "Protein bar, $3.99, athletes" --platform TikTok
  python copywriting_agent_tool.py --voc voc.txt --brief "SaaS tool" --platform Meta --temps cold warm
  python copywriting_agent_tool.py --voc "Customers hate slow delivery..." --brief "Fast shipping" --temps retarget
        """,
    )
    parser.add_argument("--voc", required=True, help="Путь к файлу с VOC данными ИЛИ сырой текст")
    parser.add_argument("--brief", required=True, help="Описание продукта (USP, цена, аудитория)")
    parser.add_argument("--platform", default="TikTok", choices=["TikTok", "Meta", "YouTube", "Instagram"])
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--framework", default="auto", choices=["auto", "PAS", "AIDA", "BAB", "DIC"])
    parser.add_argument(
        "--temps", nargs="+", default=["cold", "warm", "retarget"],
        choices=["cold", "warm", "retarget"],
        help="Температуры аудитории (default: все три)",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    voc_path = Path(args.voc)
    voc_data = voc_path.read_text(encoding="utf-8") if voc_path.exists() else args.voc

    result = run_copywriting_agent(
        voc_data=voc_data,
        product_brief=args.brief,
        platform=args.platform,
        num_variants=args.variants,
        framework=args.framework,
        audience_temps=args.temps,
        save_output=not args.no_save,
    )

    print("\n" + "=" * 70)
    print(f"  COPYWRITING v2  |  {result['metadata']['platform']}  |  {result['metadata']['elapsed_sec']}s")
    print("=" * 70)

    for temp in result["metadata"]["audience_temps"]:
        temp_data = result["by_temperature"].get(temp, {})
        if temp_data.get("error"):
            print(f"\n[{temp.upper()}] ERROR: {temp_data['error']}")
            continue
        strat = temp_data.get("strategy", {})
        variants = temp_data.get("variants", [])
        print(f"\n[{temp.upper()}] ELM:{temp_data.get('elm_route','?')} | "
              f"Framework:{strat.get('framework','?')} | Tone:{strat.get('tone','?')}")

        for v in variants[:2]:
            marker = " ★" if v.get("recommended") else ""
            s = v.get("scores", {})
            print(f"  [{v['id']}] Score:{v.get('total_score','?')}/10{marker} "
                  f"| emotion:{v.get('primary_emotion','?')} "
                  f"| MAP:{v.get('fogg_map_avg','?')}")
            print(f"    hook_str={s.get('hook_strength','?')} "
                  f"voc={s.get('voc_alignment','?')} "
                  f"clarity={s.get('clarity','?')} "
                  f"cta={s.get('cta_strength','?')} "
                  f"M={s.get('motivation','?')} "
                  f"A={s.get('ability','?')} "
                  f"P={s.get('prompt','?')}")
            print(f"    Hook:     {v.get('hook', '')}")
            print(f"    CTA:      {v.get('cta', '')}")

    if result.get("top_recommendation"):
        top = result["top_recommendation"]
        print(f"\n★ OVERALL TOP [{top.get('audience_temp','').upper()}] "
              f"Score:{top.get('total_score','?')}/10")
        print(f"  Hook: {top.get('hook', '')}")
        print(f"  CTA:  {top.get('cta', '')}")

    if result.get("output_file"):
        print(f"\nSaved → {result['output_file']}")


if __name__ == "__main__":
    main()
