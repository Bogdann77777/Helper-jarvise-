"""
marketing_factory/copywriting_agent.py — Agentic Copywriting Stage

TRUE AGENT: Claude (OpenRouter) with tool_use agentic loop.
Tools backed by Perplexity for real-time web research.

Agent searches for:
  - Winning ad hooks currently performing in this category/platform
  - Competitor copy angles running right now
  - Platform-specific hook benchmarks (CTR, hook rates)

Then generates copy grounded in REAL data, not just context.

Replaces the simple run_copywriting() single-shot call.
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
from marketing_factory.prompts import CopywritingOutput, CopyVariant

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
CLAUDE_MODEL    = "anthropic/claude-sonnet-4-6"
MAX_TOKENS      = 8000
MAX_TOOL_ROUNDS = 4          # agent can call tools up to 4 rounds
PERPLEXITY_MAX  = 2000       # shorter for tool responses
SONAR_PRO       = "sonar-pro"

# ─────────────────────────────────────────────────────────────────────────────
#  Tool definitions — declared to Claude
# ─────────────────────────────────────────────────────────────────────────────
COPY_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_winning_hooks",
            "description": (
                "Search for winning, high-performing ad hooks currently used on the platform "
                "in this product category. Returns real examples of hooks with high CTR/engagement, "
                "trending formats, and what the top performing ads open with."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_category": {
                        "type": "string",
                        "description": "The product/service category to search for"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Ad platform: TikTok, Instagram, Facebook, YouTube"
                    },
                    "hook_angle": {
                        "type": "string",
                        "description": "Optional: specific angle to explore (pain, transformation, social_proof)"
                    }
                },
                "required": ["product_category", "platform"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_competitor_copy",
            "description": (
                "Find actual competitor ad copy and messaging running right now in this market. "
                "Returns competitor headlines, CTAs, main angles, and what's working for them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_category": {
                        "type": "string",
                        "description": "Product/service category"
                    },
                    "market": {
                        "type": "string",
                        "description": "Geographic market or audience"
                    },
                    "platform": {
                        "type": "string",
                        "description": "Platform to check"
                    }
                },
                "required": ["product_category", "market", "platform"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "benchmark_copy_performance",
            "description": (
                "Get copy performance benchmarks for this platform and category: "
                "average CTR, hook retention rates, which copy formulas (AIDA/PAS/BAB) "
                "perform best, ideal copy length, best-performing CTAs."
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
                        "description": "Product category for industry-specific benchmarks"
                    }
                },
                "required": ["platform", "product_category"]
            }
        }
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  Tool implementations — Perplexity-backed
# ─────────────────────────────────────────────────────────────────────────────

def _perplexity_search(query: str, max_tokens: int = PERPLEXITY_MAX) -> str:
    """Raw Perplexity search — returns text result."""
    payload = {
        "model": SONAR_PRO,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a market research assistant. Return factual, specific information "
                    "with real examples. Be concise — max 400 words. Focus on actionable specifics."
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


def _tool_search_winning_hooks(product_category: str, platform: str, hook_angle: str = "") -> str:
    angle_filter = f"with {hook_angle} angle" if hook_angle else ""
    query = (
        f"Best performing {platform} ad hooks for {product_category} {angle_filter} 2024-2025. "
        f"Give me 5-10 specific real examples of high-CTR opening lines, hooks, and headlines "
        f"that are actually working right now. Include format: what the hook says exactly. "
        f"Include what hook type/formula they use."
    )
    return _perplexity_search(query)


def _tool_search_competitor_copy(product_category: str, market: str, platform: str) -> str:
    query = (
        f"Current {platform} ad copy and messaging for {product_category} in {market} market 2024-2025. "
        f"What are the top competitors saying? What headlines, CTAs, and angles are they using? "
        f"Include actual examples of ad copy if possible. What's working, what's being repeated."
    )
    return _perplexity_search(query)


def _tool_benchmark_copy_performance(platform: str, product_category: str) -> str:
    query = (
        f"{platform} ad copy performance benchmarks for {product_category} 2024-2025. "
        f"What is the average CTR? Which copy formulas (AIDA, PAS, BAB) work best? "
        f"Ideal hook length? Best CTA words? How long should body copy be? "
        f"What hook types have highest 3-second retention?"
    )
    return _perplexity_search(query)


def _dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool call and return result string."""
    if tool_name == "search_winning_hooks":
        return _tool_search_winning_hooks(
            tool_input["product_category"],
            tool_input["platform"],
            tool_input.get("hook_angle", ""),
        )
    elif tool_name == "search_competitor_copy":
        return _tool_search_competitor_copy(
            tool_input["product_category"],
            tool_input["market"],
            tool_input["platform"],
        )
    elif tool_name == "benchmark_copy_performance":
        return _tool_benchmark_copy_performance(
            tool_input["platform"],
            tool_input["product_category"],
        )
    else:
        return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────────────────────────────────────
#  Agent system prompt
# ─────────────────────────────────────────────────────────────────────────────

COPY_AGENT_SYSTEM = """\
You are a world-class direct-response copywriter — Gary Halbert + Eugene Schwartz level.
You write copy that converts because it speaks exact customer language and addresses real pain.

RESEARCH PHASE (use your tools):
Before writing any copy, you MUST:
1. Call search_winning_hooks — find real hooks currently performing in this category/platform
2. Call search_competitor_copy — see what competitors say so you can differentiate
3. Call benchmark_copy_performance — understand platform-specific benchmarks

Only AFTER tool research do you generate the final copy output.

COPY FRAMEWORK:
- Cold (unaware): AIDA + ELM peripheral — pattern-interrupt, emotion, identity
- Warm (problem-aware): PAS — problem agitate solution, ELM central
- Retarget (solution-aware): BAB — before/after/bridge + urgency + loss aversion

9 HOOK TYPES (label every variant):
  1. pain_agitation  2. transformation_reveal  3. social_proof
  4. curiosity_gap   5. pattern_interrupt       6. authority
  7. loss_aversion   8. number_statistic        9. question

PROOF ELEMENTS on every variant:
- Specific statistics (use real numbers from VOC/research)
- Before/After statements
- Social validation with numbers

FINAL OUTPUT — return ONLY JSON, no prose, matching this schema:
{
  "metadata": {"platform": "...", "framework_used": "AIDA+PAS+BAB+ELM", "total_variants_generated": <int>},
  "top_recommendation": {"hook": "...", "body": "...", "cta": "...", "emotion": "...",
    "score": <float>, "hook_type": "...", "proof_element": "...", "urgency_mechanic": "...",
    "ad_angle": "pain|transformation|authority|social_proof"},
  "by_temperature": {
    "cold":    {"elm_route": "peripheral", "variants": [{...}, ...]},
    "warm":    {"elm_route": "central",    "variants": [{...}, ...]},
    "retarget":{"elm_route": "peripheral", "variants": [{...}, ...]}
  },
  "by_formula": {
    "AIDA": {"formula": "AIDA", "audience_awareness": "unaware",        "variants": [{...}, ...]},
    "PAS":  {"formula": "PAS",  "audience_awareness": "problem_aware",  "variants": [{...}, ...]},
    "BAB":  {"formula": "BAB",  "audience_awareness": "solution_aware", "variants": [{...}, ...]}
  }
}

Each variant needs: hook, body, cta, emotion, score (0-10), hook_type, proof_element, urgency_mechanic, ad_angle.
Minimum: 2 variants per temperature block, 2 variants per formula block.
Return ONLY the JSON object as your final message.\
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Agentic loop
# ─────────────────────────────────────────────────────────────────────────────

def run_copywriting_agent(
    product_description: str,
    platform: str,
    voc_brief: str,
    dominant_narrative: str,
    white_space: str,
    journey_context: str = "",
    product_category: str = "",
    market: str = "",
) -> CopywritingOutput:
    """
    Agentic copywriting: Claude researches winning hooks/competitor copy via Perplexity tools,
    then generates grounded copy variants. Returns same CopywritingOutput schema as before.
    """
    print("[copy-agent] Starting agentic copywriting loop...", flush=True)

    # Initial user message
    user_prompt = f"""\
Create {platform} ad copy for: {product_description}
Category: {product_category} | Market: {market}

VOC INTELLIGENCE (speak this language, address these pains):
{voc_brief}

COMPETITOR CONTEXT:
Dominant narrative (differentiate from this): {dominant_narrative}
White space to exploit: {white_space}

CUSTOMER JOURNEY CONTEXT:
{journey_context}

START by using your research tools to find real winning hooks and competitor copy.
Then generate copy variants grounded in actual market data.
Return the final JSON output after completing your research.\
"""

    messages: list[dict] = [
        {"role": "user", "content": user_prompt}
    ]

    tool_round = 0
    final_json_str: Optional[str] = None

    while tool_round <= MAX_TOOL_ROUNDS:
        print(f"[copy-agent] API call (round {tool_round})...", flush=True)

        payload = {
            "model": CLAUDE_MODEL,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "tools": COPY_AGENT_TOOLS,
        }
        # On the last round, force text output (no more tools)
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

        # Append assistant message
        messages.append({"role": "assistant", "content": message.get("content") or ""})

        # Check for tool calls
        tool_calls = message.get("tool_calls", [])

        if not tool_calls or stop_reason == "stop":
            # Claude finished — extract JSON from content
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") for block in content if isinstance(block, dict)
                )
            final_json_str = content
            print(f"[copy-agent] Agent finished after {tool_round} tool rounds", flush=True)
            break

        # Execute tool calls
        print(f"[copy-agent] Tool calls: {[tc['function']['name'] for tc in tool_calls]}", flush=True)
        tool_results = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_input = json.loads(tc["function"]["arguments"])
            print(f"[copy-agent]   → {tool_name}({list(tool_input.values())})", flush=True)
            result_text = _dispatch_tool(tool_name, tool_input)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_text,
            })

        messages.extend(tool_results)
        tool_round += 1

        # Small pause between rounds
        if tool_round <= MAX_TOOL_ROUNDS:
            time.sleep(1)

    # Parse and validate output
    return _parse_copy_output(final_json_str, platform)


def _parse_copy_output(raw: Optional[str], platform: str) -> CopywritingOutput:
    """Parse JSON string → CopywritingOutput, with fallback on parse failure."""
    from pydantic import ValidationError

    if not raw:
        raise RuntimeError("Copy agent returned empty response")

    # Clean markdown wrappers
    content = raw.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # Find JSON object
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        content = content[start:end]

    try:
        data = json.loads(content)
        result = CopywritingOutput.model_validate(data)
        print("[copy-agent] ✓ Output validated", flush=True)
        return result
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"[copy-agent] Parse error: {e} — retrying with explicit JSON request", flush=True)
        # One more attempt with explicit JSON demand
        return _fallback_copy_generation(raw, platform)


def _fallback_copy_generation(context: str, platform: str) -> CopywritingOutput:
    """Fallback: ask Claude to produce clean JSON from the research context."""
    from marketing_factory.prompts import _claude_with_validation, COPY_SYSTEM, _build_copy_user_prompt
    print("[copy-agent] Falling back to direct Claude call...", flush=True)
    return _claude_with_validation(
        system_prompt=COPY_SYSTEM,
        user_prompt=(
            f"Based on this research context, generate the copy JSON:\n\n{context[:3000]}"
        ),
        schema_name="copywriting_fallback",
        pydantic_model=CopywritingOutput,
    )
