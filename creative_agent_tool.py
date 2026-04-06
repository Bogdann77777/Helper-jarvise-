"""
creative_agent_tool.py — AI Creative Brief Agent для CLI helper.

Принимает одобренный копирайтинг → создаёт полный creative brief для видео-рекламы.

Pipeline (3 шага):
  1. Creative Strategy  — тип видео, visual hook (0-3с), emotional arc
  2. Storyboard         — покадровое описание: камера, движение, текст оверлей, VO
  3. Design + Prompts   — цвета, шрифты, готовые промпты для Runway Gen-4 / Kling 3.0 / Midjourney

Blind spots из оригинального blueprint — что исправлено:
  - claude-3-7-sonnet → claude-sonnet-4-6
  - Flux Video не существует → используем Runway Gen-4 и Kling 3.0
  - Veo3 (Vertex AI, дорого) → не в MVP
  - RunwayML Gen-3 deprecated → Gen-4
  - Агент НЕ генерирует видео сам — создаёт готовые промпты для пользователя
  - Platform-specific constraints встроены в каждый шаг
  - Error handling на каждом шаге

Использование:
    from creative_agent_tool import run_creative_agent

    result = run_creative_agent(
        approved_copy={"hook": "...", "headline": "...", "body": "...", "cta": "..."},
        brand_kit={"colors": ["#FF4500", "#FFFFFF"], "fonts": ["Inter Bold"], "tone": "energetic"},
        platform="TikTok",
        duration_sec=15,
        style="ugc",
    )

    # Из файла copywriting_agent_tool (автоматически берёт top_recommendation):
    result = run_creative_agent(approved_copy="outputs/copywriting/copy_tiktok_20260403.json")

    # CLI:
    python creative_agent_tool.py --copy outputs/copywriting/copy_tiktok_123.json --platform TikTok --duration 15
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# ── Пути ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs" / "creative_briefs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── OpenRouter ───────────────────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_CREATIVE = "anthropic/claude-sonnet-4-6"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── LLM Client ───────────────────────────────────────────────────────────
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
        raise RuntimeError("OPENROUTER_API_KEY не задан. Добавь в .env или переменную окружения.")

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
        "X-Title": "CLI Helper / Creative Agent",
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
    """Парсинг JSON из ответа LLM — убирает markdown-обёртки."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


# ── Step 1: Creative Strategy ─────────────────────────────────────────────
def _creative_strategy(
    approved_copy: dict,
    brand_kit: dict,
    platform: str,
    duration_sec: int,
    style: str,
) -> dict:
    """
    Выбирает концепцию видео, тип visual hook и emotional arc.
    Первые 3 секунды — самое важное решение.
    """
    logger.info("Step 1/3: Creative Strategy...")

    platform_specs = {
        "TikTok": "9:16 vertical, 15-30s, authentic UGC wins, NO logo/brand in first 3s, captions on",
        "Meta": "1:1 or 4:5, 15-30s, clear benefit in first 3s, captions essential (85% watch muted)",
        "YouTube": "16:9, 15-60s, can build narrative, skip button at 5s — hook must hit before that",
        "Instagram": "9:16 Reels or 1:1 Feed, 15-30s, visual-first, trending audio boosts reach",
    }.get(platform, "9:16, 15-30s, mobile-first, muted-first design")

    system = (
        "You are a Senior Art Director specializing in direct response video ads. "
        "You've made 500+ ads. You know exactly what stops scrolling. "
        "You think in VISUALS first, copy second. "
        "Output ONLY valid JSON, no explanations."
    )
    user = f"""Design the creative strategy for this {duration_sec}s {platform} ad.

APPROVED COPY:
{json.dumps(approved_copy, ensure_ascii=False)}

BRAND KIT:
{json.dumps(brand_kit, ensure_ascii=False)}

PLATFORM SPECS: {platform_specs}
PREFERRED STYLE: {style}

Video style options:
- ugc: Shot-on-phone, organic, person-to-camera, slightly unpolished = trust
- demo: Product in action, close-ups, real usage, shows vs tells
- testimonial: Real person sharing result (scripted but authentic-looking)
- animation: Motion graphics, text-driven, works without footage
- problem_solution: Opens with relatable problem moment, product solves it

CRITICAL RULE: The hook (first 3 seconds) must be UNEXPECTED.
Never start with: logo, brand name, generic "Are you tired of...", or slow pan.

Output:
{{
  "concept_name": "Short memorable name for this concept",
  "video_style": "ugc|demo|testimonial|animation|problem_solution",
  "hook_type": "shocking_statement|question|visual_surprise|relatable_moment|bold_claim|sound_hook",
  "hook_description": "EXACT description of first 3 seconds — what camera sees + what's heard",
  "emotional_arc": {{
    "start_emotion": "what viewer feels at second 0",
    "peak_emotion": "what viewer feels at peak moment",
    "end_emotion": "what viewer feels at CTA"
  }},
  "key_visual": "The single most memorable visual moment in the entire ad",
  "music_direction": {{
    "genre": "...",
    "energy": "low|medium|high",
    "bpm_range": "80-100|100-120|120-140",
    "reference_style": "sounds like [known artist/track style]",
    "note": "any special instruction (e.g. drop at CTA, silence for impact)"
  }},
  "pacing": {{
    "cuts_per_minute": 0,
    "rhythm": "slow_build|steady|fast_cuts|beat_synced"
  }},
  "why_it_works": "1-2 sentences: the psychological mechanism that makes this effective"
}}"""

    raw = _call_llm(
        MODEL_CREATIVE,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.75,
        max_tokens=1200,
    )
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Creative strategy JSON parse failed: {e}")
        return {
            "concept_name": "Default Concept",
            "video_style": style,
            "hook_type": "visual_surprise",
            "hook_description": approved_copy.get("hook", ""),
            "_parse_error": str(e),
        }


# ── Step 2: Storyboard ────────────────────────────────────────────────────
def _storyboard(
    concept: dict,
    approved_copy: dict,
    duration_sec: int,
    platform: str,
) -> list[dict]:
    """
    Создаёт покадровое описание видео.
    Каждая сцена — точное описание: что на экране, текст, голос, камера.
    """
    logger.info("Step 2/3: Storyboard...")

    if duration_sec <= 15:
        scene_guide = (
            "4 scenes: "
            "Scene 1 (0-3s) = HOOK — unexpected visual that stops scroll. "
            "Scene 2 (3-8s) = PROBLEM or DESIRE — connect with audience. "
            "Scene 3 (8-12s) = PRODUCT/SOLUTION — show, don't just tell. "
            "Scene 4 (12-15s) = CTA — specific action, urgency if relevant."
        )
    elif duration_sec <= 30:
        scene_guide = (
            "5-6 scenes: "
            "Hook(0-3s), Context(3-8s), Problem(8-14s), Solution+Product(14-22s), "
            "Social proof(22-27s), CTA(27-30s)."
        )
    else:
        scene_guide = (
            "7-8 scenes with clear narrative arc. "
            "Hook(0-3s), Setup(3-10s), Conflict(10-20s), Solution(20-35s), "
            "Demonstration(35-45s), Proof(45-52s), CTA(52-60s)."
        )

    # Platform-specific text overlay rules
    overlay_rules = {
        "TikTok": "Avoid bottom 20% (progress bar + nav). Top safe area: top 5-15%.",
        "Meta": "Keep overlays in center 80%. Avoid all 4 edges (platform chrome).",
        "Instagram": "Avoid bottom 25% for Reels (UI). Center-top area safest.",
        "YouTube": "Avoid bottom-right 15% (YouTube controls). Top-left for text.",
    }.get(platform, "Keep text in center 80% safe zone.")

    system = (
        "You are a storyboard director for short-form video ads. "
        "You describe EXACTLY what is on screen — every detail matters. "
        "A great storyboard leaves no ambiguity for the production team. "
        "Output ONLY valid JSON."
    )
    user = f"""Create a shot-by-shot storyboard for this {duration_sec}s {platform} ad.

CONCEPT: {json.dumps(concept, ensure_ascii=False)}

COPY TO PLACE IN SCENES:
- Hook line:     {approved_copy.get("hook", "")}
- Headline:      {approved_copy.get("headline", "")}
- Body:          {approved_copy.get("body", "")}
- CTA:           {approved_copy.get("cta", "")}
- Short slogan:  {approved_copy.get("short_slogan", "")}

SCENE STRUCTURE: {scene_guide}
OVERLAY SAFE ZONE: {overlay_rules}

Rules:
1. First frame = visual hook. NO logo, NO brand name, NO slow intro.
2. Text overlays: max 5 words on screen at once
3. Each scene must "earn" the next second of viewer attention
4. CTA scene: show the actual button/link/action, not just "buy now"
5. If UGC style: camera should feel handheld, unpolished, real

Output:
{{
  "scenes": [
    {{
      "id": "s1",
      "time_range": "0-3s",
      "purpose": "HOOK|PROBLEM|DESIRE|SOLUTION|PRODUCT|PROOF|CTA",
      "visual": "Detailed description: subject, angle, background, lighting, movement",
      "camera_shot": "extreme_close_up|close_up|medium|wide|overhead|pov|handheld",
      "camera_movement": "static|pan_right|pan_left|zoom_in|zoom_out|handheld_shake|none",
      "text_overlay": "exact text shown (null if none)",
      "text_position": "top_center|center|bottom_center|null",
      "text_style": "large_bold|caption_small|animated_pop|typewriter|none",
      "voiceover": "exact words spoken (null if no VO)",
      "sfx": "sound effect (null if none)",
      "music_note": "music instruction for this scene (null if no change)",
      "transition_to_next": "hard_cut|fade|swipe_left|match_cut|none"
    }}
  ]
}}"""

    raw = _call_llm(
        MODEL_CREATIVE,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.7,
        max_tokens=2500,
    )
    try:
        data = _parse_json(raw)
        return data.get("scenes", [])
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Storyboard JSON parse failed: {e}")
        return []


# ── Step 3: Design Spec + Generation Prompts ──────────────────────────────
def _design_and_prompts(
    scenes: list[dict],
    concept: dict,
    brand_kit: dict,
    platform: str,
) -> dict:
    """
    Создаёт design spec + готовые промпты для AI видео/изображений.

    Поддерживаемые инструменты (2026):
    - Runway Gen-4   — text-to-video, лучший баланс цена/качество
    - Kling 3.0      — text-to-video, 4K, лучший в нише UGC
    - Midjourney v7  — key frames / thumbnail
    - DALL-E 3       — статичные ad creatives

    НЕ включаем:
    - Flux Video     — не существует (только images)
    - Veo3           — Vertex AI только, $0.35-0.50/sec, не для MVP
    - Runway Gen-3   — deprecated, Gen-4 вышел
    """
    logger.info("Step 3/3: Design spec + generation prompts...")

    # Берём первые 3 ключевые сцены для генерации промптов
    key_scenes = scenes[:3] if scenes else []

    system = (
        "You are a visual designer and AI prompt engineer. "
        "You create both design specs AND production-ready prompts for AI video/image tools. "
        "Prompts must be SPECIFIC: describe lighting, mood, camera, subject, style in detail. "
        "Vague prompts = bad results. "
        "Output ONLY valid JSON."
    )
    user = f"""Create complete design spec and AI generation prompts for this ad.

CONCEPT: {json.dumps(concept, ensure_ascii=False)}
BRAND KIT: {json.dumps(brand_kit, ensure_ascii=False)}
PLATFORM: {platform}

KEY SCENES TO GENERATE:
{json.dumps(key_scenes, ensure_ascii=False, indent=2)}

Output:
{{
  "design_spec": {{
    "color_palette": {{
      "primary": "#HEX",
      "secondary": "#HEX",
      "accent": "#HEX",
      "text_on_dark": "#HEX",
      "text_on_light": "#HEX",
      "rationale": "why these colors work for this ad"
    }},
    "typography": {{
      "headline_font": "font name + weight (e.g. Inter Bold 700)",
      "body_font": "font name + weight",
      "headline_size_mobile_pt": "36-48pt",
      "body_size_mobile_pt": "18-24pt",
      "text_animation": "fade_in|slide_up|pop|typewriter|bounce|none",
      "letter_spacing": "tight|normal|wide"
    }},
    "visual_style_guide": "e.g. High contrast, desaturated background with vibrant product, vertical format",
    "do_not": ["specific visual elements to avoid for this brand/product"]
  }},
  "generation_prompts": {{
    "runway_gen4": {{
      "description": "Scene to generate with Runway Gen-4 (API: runwayml Python SDK)",
      "text_prompt": "Detailed prompt: subject + action + environment + lighting + camera + style + mood",
      "negative_prompt": "What to avoid: e.g. blurry, watermark, text, low quality, overexposed",
      "duration_sec": 5,
      "resolution": "1080p",
      "motion_amount": "low|medium|high",
      "api_note": "pip install runwayml | model: gen4_turbo"
    }},
    "kling_3": {{
      "description": "Scene to generate with Kling 3.0 (best for UGC/authentic style, 4K)",
      "text_prompt": "Detailed prompt: same level of detail as Runway",
      "negative_prompt": "What to avoid",
      "duration_sec": 5,
      "camera_control": "static|pan_left|pan_right|zoom_in|tilt_up|none",
      "cfg_scale": 0.5,
      "api_note": "POST https://api.klingai.com/v1/videos/text2video | auth: Bearer token"
    }},
    "midjourney_v7": {{
      "description": "Key still frame or thumbnail for the ad",
      "prompt": "Detailed Midjourney prompt with all visual details",
      "parameters": "--ar 9:16 --style raw --v 7 --q 2"
    }},
    "dall_e_3": {{
      "description": "Static ad creative (fallback for quick generation)",
      "prompt": "Detailed DALL-E 3 prompt",
      "size": "1024x1792",
      "quality": "hd",
      "style": "natural|vivid"
    }}
  }},
  "production_notes": [
    "Specific technical note for video editor",
    "Specific note about transitions or effects",
    "Music sync note",
    "Caption/subtitle note"
  ]
}}"""

    raw = _call_llm(
        MODEL_CREATIVE,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.6,
        max_tokens=3000,
    )
    try:
        return _parse_json(raw)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Design/prompts JSON parse failed: {e}")
        return {"_parse_error": str(e), "_raw_truncated": raw[:500]}


# ── Main Function ─────────────────────────────────────────────────────────
def run_creative_agent(
    approved_copy: "dict | str",
    brand_kit: "dict | None" = None,
    platform: str = "TikTok",
    duration_sec: int = 15,
    style: str = "ugc",
    save_output: bool = True,
) -> dict:
    """
    Запускает полный creative brief pipeline.

    Args:
        approved_copy:  dict с ключами hook/headline/body/cta/short_slogan
                       ИЛИ путь к JSON файлу из copywriting_agent_tool
                       (автоматически берёт top_recommendation)
        brand_kit:      dict с ключами colors (list), fonts (list), tone (str)
                       Дефолт: нейтральный современный стиль
        platform:       TikTok | Meta | YouTube | Instagram
        duration_sec:   15 | 30 | 60
        style:          ugc | demo | testimonial | animation | problem_solution
        save_output:    Сохранить JSON в outputs/creative_briefs/

    Returns:
        dict: {metadata, concept, storyboard, design_spec, generation_prompts,
               production_notes, approved_copy}
    """
    start_time = time.time()
    logger.info(f"=== Creative Agent START | platform={platform} | {duration_sec}s | style={style} ===")

    # Загрузка копирайтинга
    if isinstance(approved_copy, str):
        p = Path(approved_copy)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            # Автоматически берём лучший вариант из copywriting_agent_tool output
            if "top_recommendation" in data and data["top_recommendation"]:
                approved_copy = data["top_recommendation"]
                logger.info(f"Loaded top_recommendation from {p.name}")
            elif "variants" in data and data["variants"]:
                approved_copy = data["variants"][0]
                logger.info(f"Loaded variants[0] from {p.name}")
            else:
                approved_copy = data
        else:
            try:
                approved_copy = json.loads(approved_copy)
            except json.JSONDecodeError:
                # Строку интерпретируем как текст хука
                approved_copy = {
                    "hook": approved_copy,
                    "headline": "",
                    "body": "",
                    "cta": "Learn More",
                    "short_slogan": "",
                }

    if not isinstance(approved_copy, dict):
        raise ValueError("approved_copy должен быть dict или путь к JSON файлу.")

    # Дефолтный brand kit
    if brand_kit is None:
        brand_kit = {
            "colors": ["#1A1A1A", "#FFFFFF", "#FF4500"],
            "fonts": ["Inter Bold", "Inter Regular"],
            "tone": "direct and authentic",
        }

    # Pipeline
    concept = _creative_strategy(approved_copy, brand_kit, platform, duration_sec, style)
    scenes = _storyboard(concept, approved_copy, duration_sec, platform)
    design = _design_and_prompts(scenes, concept, brand_kit, platform)

    result = {
        "metadata": {
            "platform": platform,
            "duration_sec": duration_sec,
            "style": style,
            "concept_name": concept.get("concept_name", ""),
            "video_style": concept.get("video_style", style),
            "generated_at": datetime.now().isoformat(),
            "elapsed_sec": round(time.time() - start_time, 1),
        },
        "approved_copy": approved_copy,
        "concept": concept,
        "storyboard": scenes,
        "design_spec": design.get("design_spec", {}),
        "generation_prompts": design.get("generation_prompts", {}),
        "production_notes": design.get("production_notes", []),
    }

    if save_output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUTS_DIR / f"brief_{platform.lower()}_{ts}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output_file"] = str(out_path)
        logger.info(f"Saved → {out_path}")

    logger.info(f"=== DONE in {result['metadata']['elapsed_sec']}s | concept: {concept.get('concept_name', '')} ===")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI Creative Brief Agent — storyboard + промпты для видео-рекламы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Из файла copywriting_agent_tool (автоматически берёт лучший вариант):
  python creative_agent_tool.py --copy outputs/copywriting/copy_tiktok_123.json --platform TikTok

  # Inline JSON копирайтинга:
  python creative_agent_tool.py --copy '{"hook":"Stop wasting money on..","headline":"..","cta":"Try Free"}' --platform Meta --duration 30

  # С брендбуком:
  python creative_agent_tool.py --copy copy.json --brand brand.json --platform YouTube --duration 60 --style demo
        """,
    )
    parser.add_argument(
        "--copy", required=True,
        help="Путь к JSON файлу с копирайтингом (из copywriting_agent_tool) ИЛИ inline JSON string",
    )
    parser.add_argument("--brand", default=None, help="Путь к JSON с brand kit (опционально)")
    parser.add_argument("--platform", default="TikTok", choices=["TikTok", "Meta", "YouTube", "Instagram"])
    parser.add_argument("--duration", type=int, default=15, choices=[15, 30, 60])
    parser.add_argument(
        "--style", default="ugc",
        choices=["ugc", "demo", "testimonial", "animation", "problem_solution"],
    )
    parser.add_argument("--no-save", action="store_true", help="Не сохранять результат в файл")
    args = parser.parse_args()

    brand_kit = None
    if args.brand:
        brand_path = Path(args.brand)
        if brand_path.exists():
            brand_kit = json.loads(brand_path.read_text(encoding="utf-8"))

    result = run_creative_agent(
        approved_copy=args.copy,
        brand_kit=brand_kit,
        platform=args.platform,
        duration_sec=args.duration,
        style=args.style,
        save_output=not args.no_save,
    )

    # Вывод результатов
    print("\n" + "=" * 65)
    print(f"  CREATIVE BRIEF  |  {result['metadata']['platform']}  |  {result['metadata']['duration_sec']}s  |  {result['metadata']['video_style']}")
    print("=" * 65)
    c = result["concept"]
    print(f"Concept:    {c.get('concept_name', '')}")
    print(f"Hook type:  {c.get('hook_type', '')} — {c.get('hook_description', '')[:80]}")
    print(f"Emotional:  {c.get('emotional_arc', {}).get('start_emotion', '')} → "
          f"{c.get('emotional_arc', {}).get('end_emotion', '')}")
    print(f"Music:      {c.get('music_direction', {}).get('genre', '')} | "
          f"{c.get('music_direction', {}).get('bpm_range', '')} BPM")
    print(f"Pacing:     {c.get('pacing', {}).get('rhythm', '')} | "
          f"{c.get('pacing', {}).get('cuts_per_minute', '')} cuts/min")
    print()

    print("STORYBOARD:")
    for scene in result["storyboard"]:
        print(f"  [{scene.get('time_range', '?')}] {scene.get('purpose', '')}")
        print(f"    Visual: {scene.get('visual', '')[:90]}")
        if scene.get("text_overlay"):
            print(f"    TEXT: \"{scene['text_overlay']}\"  [{scene.get('text_position', '')}]")
        if scene.get("voiceover"):
            print(f"    VO: \"{scene['voiceover']}\"")

    print("\nGENERATION PROMPTS:")
    for tool_name, prompt_data in result.get("generation_prompts", {}).items():
        if isinstance(prompt_data, dict) and prompt_data.get("text_prompt") or prompt_data.get("prompt"):
            p_text = prompt_data.get("text_prompt") or prompt_data.get("prompt") or ""
            print(f"\n  [{tool_name.upper()}]")
            print(f"  {p_text[:150]}...")
            if prompt_data.get("api_note"):
                print(f"  API: {prompt_data['api_note']}")

    if result.get("production_notes"):
        print("\nPRODUCTION NOTES:")
        for note in result["production_notes"]:
            print(f"  • {note}")

    if result.get("output_file"):
        print(f"\nSaved → {result['output_file']}")


if __name__ == "__main__":
    main()
