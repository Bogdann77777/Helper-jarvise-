"""
marketing_factory/creative_agent.py — Agentic Creative Brief Stage

TRUE AGENT: Claude (OpenRouter) with tool_use agentic loop.
Tools backed by Perplexity for real-time web research.

Agent searches for:
  - Trending UGC/video formats on the specific platform right now
  - Visual references and aesthetic styles that convert in this category
  - Platform-specific creative benchmarks (hook retention, completion rates)
  - AI video generation specs (Runway, Kling prompts that actually work)

Generates production-ready creative briefs grounded in REAL creative intelligence.

Replaces the simple run_creative_brief() single-shot call.
"""

from __future__ import annotations

import json
import time
import requests
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PERPLEXITY_API_KEY, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from marketing_factory.prompts import CreativeBriefOutput, CopyVariant

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
CLAUDE_MODEL    = "anthropic/claude-sonnet-4-6"
MAX_TOKENS      = 8000
MAX_TOOL_ROUNDS = 4
PERPLEXITY_MAX  = 2000
SONAR_PRO       = "sonar-pro"

# ─────────────────────────────────────────────────────────────────────────────
#  Tool definitions
# ─────────────────────────────────────────────────────────────────────────────
CREATIVE_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_trending_formats",
            "description": (
                "Search for currently trending video/creative formats on the platform for this category. "
                "Returns: what formats are going viral, what styles are getting highest engagement, "
                "average video length performing best, text overlay styles, music trends."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "Platform: TikTok, Instagram Reels, Facebook, YouTube Shorts"
                    },
                    "product_category": {
                        "type": "string",
                        "description": "Product/service category"
                    }
                },
                "required": ["platform", "product_category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_ugc_references",
            "description": (
                "Find successful UGC (User Generated Content) examples and authentic ad styles "
                "in this product category. What does authentic/native content look like? "
                "What visual storytelling approaches work? POV formats, testimonial styles, "
                "before/after formats that perform."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_category": {
                        "type": "string",
                        "description": "Product/service category"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform"
                    },
                    "audience_type": {
                        "type": "string",
                        "description": "cold traffic | warm | retarget"
                    }
                },
                "required": ["product_category", "platform"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_creative_benchmarks",
            "description": (
                "Get platform-specific creative performance benchmarks: hook retention rates, "
                "completion rates, ideal video duration, best caption lengths, thumbnail CTR, "
                "and what separates top 10% performing ads from average."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "platform": {
                        "type": "string",
                        "description": "Ad platform"
                    },
                    "product_category": {
                        "type": "string",
                        "description": "Product category for industry-specific data"
                    },
                    "video_duration_sec": {
                        "type": "integer",
                        "description": "Target video length in seconds"
                    }
                },
                "required": ["platform", "product_category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_ai_generation_specs",
            "description": (
                "Find best practices and effective prompt techniques for AI video/image generation "
                "tools (Runway Gen-4, Kling 3.0, Midjourney, Stable Diffusion) for this type of ad creative. "
                "Returns prompting strategies, style modifiers, negative prompts that work."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "creative_style": {
                        "type": "string",
                        "description": "e.g. 'UGC authentic beauty', 'luxury product showcase', 'before/after transformation'"
                    },
                    "tool": {
                        "type": "string",
                        "description": "runway | kling | midjourney | all"
                    }
                },
                "required": ["creative_style"]
            }
        }
    }
]

# ─────────────────────────────────────────────────────────────────────────────
#  Tool implementations — Perplexity-backed
# ─────────────────────────────────────────────────────────────────────────────

def _perplexity_search(query: str, max_tokens: int = PERPLEXITY_MAX) -> str:
    payload = {
        "model": SONAR_PRO,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a creative strategy researcher. Return specific, actionable information "
                    "with real examples. Be concise — max 400 words. Focus on what is working RIGHT NOW."
                )
            },
            {"role": "user", "content": query}
        ],
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if "<think>" in content:
            content = content.split("</think>")[-1].strip()
        return content
    except Exception as e:
        return f"Search failed: {e}"


def _tool_search_trending_formats(platform: str, product_category: str) -> str:
    query = (
        f"Trending {platform} video ad formats for {product_category} 2024-2025. "
        f"What styles are going viral? What video formats get highest engagement? "
        f"Best performing video lengths? Text overlay styles that convert? "
        f"What does the top 1% of {platform} ads for {product_category} look like? "
        f"Give specific creative format examples."
    )
    return _perplexity_search(query)


def _tool_search_ugc_references(
    product_category: str, platform: str, audience_type: str = ""
) -> str:
    audience_filter = f"for {audience_type} audience" if audience_type else ""
    query = (
        f"Best performing UGC and authentic ad styles for {product_category} on {platform} "
        f"{audience_filter} 2024-2025. "
        f"What does high-converting authentic content look like? "
        f"POV formats, testimonial styles, before/after structures that perform. "
        f"Real examples of UGC approaches working in this category."
    )
    return _perplexity_search(query)


def _tool_search_creative_benchmarks(
    platform: str, product_category: str, video_duration_sec: int = 30
) -> str:
    query = (
        f"{platform} video ad creative benchmarks for {product_category} {video_duration_sec}s videos 2024-2025. "
        f"Average hook retention (% watching past 3 seconds)? "
        f"Average completion rate? Industry average CTR? "
        f"What separates top 10% performing ads from average? "
        f"Ideal hook duration, caption length, thumbnail style."
    )
    return _perplexity_search(query)


def _tool_search_ai_generation_specs(creative_style: str, tool: str = "all") -> str:
    tool_filter = f"for {tool}" if tool != "all" else "for Runway Gen-4 and Kling 3.0"
    query = (
        f"Best AI video generation prompt techniques {tool_filter} for {creative_style} style ads. "
        f"What prompt structure works? Key style modifiers? What to include/avoid? "
        f"Effective prompting for authentic-looking, conversion-focused video ads. "
        f"Real examples of prompts that produce good results."
    )
    return _perplexity_search(query)


def _dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "search_trending_formats":
        return _tool_search_trending_formats(
            tool_input["platform"],
            tool_input["product_category"],
        )
    elif tool_name == "search_ugc_references":
        return _tool_search_ugc_references(
            tool_input["product_category"],
            tool_input["platform"],
            tool_input.get("audience_type", ""),
        )
    elif tool_name == "search_creative_benchmarks":
        return _tool_search_creative_benchmarks(
            tool_input["platform"],
            tool_input["product_category"],
            tool_input.get("video_duration_sec", 30),
        )
    elif tool_name == "search_ai_generation_specs":
        return _tool_search_ai_generation_specs(
            tool_input["creative_style"],
            tool_input.get("tool", "all"),
        )
    else:
        return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────────────────────────────────────
#  Agent system prompt
# ─────────────────────────────────────────────────────────────────────────────

CREATIVE_AGENT_SYSTEM = """\
You are a Creative Director at a top-tier performance marketing agency.
You write video ad briefs that production teams execute in 1 day.
Every creative decision is data-driven: real trending formats, proven UGC styles, actual benchmarks.

RESEARCH PHASE (use your tools — MANDATORY):
Before writing any brief, you MUST:
1. Call search_trending_formats — find what's actually trending on this platform/category NOW
2. Call search_ugc_references — find successful authentic content styles in this category
3. Call search_creative_benchmarks — get real performance benchmarks for the platform
4. Call search_ai_generation_specs — find effective prompting for Runway/Kling

Use this REAL data to make creative decisions. Not generic advice — actual market intelligence.

CREATIVE PRINCIPLES:
- FORMAT: Native-first (no polished corporate look), authentic, platform-native
- RULE 1: Hook creates pattern interrupt in first 2 seconds
- RULE 2: Generate 2-3 DISTINCT concept variants (different hook types, different styles)
- RULE 3: Benchmarks must match actual platform data you researched
- RULE 4: AI generation prompts must be specific and production-ready

FINAL OUTPUT — return ONLY JSON, no prose, matching this schema:
{
  "metadata": {"platform": "...", "duration_sec": <int>},
  "concepts": [
    {
      "concept_id": "A",
      "hook_type": "...",
      "video_style": "...",
      "emotional_arc": {"start": "...", "middle": "...", "end": "..."},
      "creative_concept": "...",
      "best_for_audience": "cold traffic|warm|retarget",
      "storyboard": [
        {
          "scene_number": 1,
          "visual_description": "...",
          "voiceover": "...",
          "text_overlay": "...",
          "sfx": "..."
        }
      ]
    }
  ],
  "top_concept_id": "A",
  "design_spec": {
    "color_palette": ["#hex1", "#hex2", "#hex3"],
    "typography": {"heading": "...", "body": "...", "accent": "..."},
    "visual_style_guide": "..."
  },
  "generation_prompts": {
    "midjourney": "...",
    "stable_diffusion": "...",
    "runway": "...",
    "kling": "..."
  },
  "aspect_ratio_specs": [
    {"ratio": "9:16", "platform_placement": "...", "crop_notes": "..."},
    {"ratio": "1:1", "platform_placement": "...", "crop_notes": "..."}
  ],
  "performance_benchmarks": {
    "ctr_target_pct": <float>,
    "hook_retention_3s_pct": <float>,
    "completion_rate_pct": <float>
  }
}

Minimum: 2 concepts, 4 storyboard scenes per concept, 2 aspect ratio specs.
Return ONLY the JSON object as your final message.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Agentic loop
# ─────────────────────────────────────────────────────────────────────────────

def run_creative_agent(
    product_description: str,
    platform: str,
    voc_brief: str,
    top_copy: CopyVariant,
    duration_sec: int = 30,
    journey_context: str = "",
    product_category: str = "",
) -> CreativeBriefOutput:
    """
    Agentic creative brief: Claude researches trending formats, UGC styles, and benchmarks
    via Perplexity tools, then generates production-ready creative briefs.
    Returns same CreativeBriefOutput schema as before.
    """
    print("[creative-agent] Starting agentic creative brief loop...", flush=True)

    user_prompt = f"""\
Write a {platform} video creative brief for: {product_description}
Category: {product_category} | Duration: {duration_sec}s

VOC INTELLIGENCE (speak this language, address these pains):
{voc_brief}

TOP PERFORMING COPY (build primary concept around this):
Hook: {top_copy.hook}
Body: {top_copy.body}
CTA: {top_copy.cta}
Hook type: {top_copy.hook_type}
Ad angle: {top_copy.ad_angle}

CUSTOMER JOURNEY CONTEXT (which stage each concept targets):
{journey_context}

START by using your research tools to find:
1. Trending creative formats on {platform} for this category
2. Successful UGC styles that convert
3. Real performance benchmarks
4. AI generation techniques for authentic-looking content

Then generate 2-3 distinct creative concepts grounded in actual creative intelligence.
Return the final JSON output after your research.\
"""

    messages: list[dict] = [
        {"role": "user", "content": user_prompt}
    ]

    tool_round = 0
    final_json_str: Optional[str] = None

    while tool_round <= MAX_TOOL_ROUNDS:
        print(f"[creative-agent] API call (round {tool_round})...", flush=True)

        payload = {
            "model": CLAUDE_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "tools": CREATIVE_AGENT_TOOLS,
        }
        if tool_round >= MAX_TOOL_ROUNDS:
            payload.pop("tools", None)
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(
            OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        response_data = resp.json()
        message = response_data["choices"][0]["message"]
        stop_reason = response_data["choices"][0].get("finish_reason", "stop")

        messages.append({"role": "assistant", "content": message.get("content") or ""})

        tool_calls = message.get("tool_calls", [])

        if not tool_calls or stop_reason == "stop":
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            final_json_str = content
            print(f"[creative-agent] Agent finished after {tool_round} tool rounds", flush=True)
            break

        print(f"[creative-agent] Tool calls: {[tc['function']['name'] for tc in tool_calls]}", flush=True)
        tool_results = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_input = json.loads(tc["function"]["arguments"])
            print(f"[creative-agent]   → {tool_name}({list(tool_input.values())})", flush=True)
            result_text = _dispatch_tool(tool_name, tool_input)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_text,
            })

        messages.extend(tool_results)
        tool_round += 1

        if tool_round <= MAX_TOOL_ROUNDS:
            time.sleep(1)

    return _parse_creative_output(final_json_str, platform, duration_sec)


def _parse_creative_output(
    raw: Optional[str], platform: str, duration_sec: int
) -> CreativeBriefOutput:
    from pydantic import ValidationError

    if not raw:
        raise RuntimeError("Creative agent returned empty response")

    content = raw.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        content = content[start:end]

    try:
        data = json.loads(content)
        result = CreativeBriefOutput.model_validate(data)
        print("[creative-agent] ✓ Output validated", flush=True)
        return result
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[creative-agent] Parse error: {e} — falling back", flush=True)
        return _fallback_creative_generation(raw, platform, duration_sec)


def _fallback_creative_generation(
    context: str, platform: str, duration_sec: int
) -> CreativeBriefOutput:
    from marketing_factory.prompts import _claude_with_validation, CREATIVE_SYSTEM
    print("[creative-agent] Falling back to direct Claude call...", flush=True)
    return _claude_with_validation(
        system_prompt=CREATIVE_SYSTEM,
        user_prompt=(
            f"Based on this research context, generate the creative brief JSON "
            f"for {platform} {duration_sec}s video:\n\n{context[:3000]}"
        ),
        schema_name="creative_fallback",
        pydantic_model=CreativeBriefOutput,
    )
