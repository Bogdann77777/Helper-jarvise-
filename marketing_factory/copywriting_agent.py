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
#  Research phase — runs tool loop, returns collected intelligence as text
# ─────────────────────────────────────────────────────────────────────────────

def _run_research_phase(
    product_category: str,
    market: str,
    platform: str,
) -> str:
    """
    Runs the agentic tool loop to gather live market intelligence.
    Returns a compact text summary of all tool results for use in the output phase.
    """
    research_prompt = (
        f"Research {platform} copywriting intelligence for {product_category} in {market}. "
        f"Use all 3 tools: search_winning_hooks, search_competitor_copy, benchmark_copy_performance."
    )
    messages: list[dict] = [{"role": "user", "content": research_prompt}]
    collected: list[str] = []

    for tool_round in range(MAX_TOOL_ROUNDS):
        print(f"[copy-agent] Research round {tool_round}...", flush=True)
        resp = requests.post(
            OPENROUTER_BASE_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": CLAUDE_MODEL, "messages": messages, "max_tokens": 500, "tools": COPY_AGENT_TOOLS},
            timeout=60,
        )
        if not resp.ok:
            print(f"[copy-agent] Research API error {resp.status_code}: {resp.text[:300]}", flush=True)
            break
        response_data = resp.json()
        message = response_data["choices"][0]["message"]
        stop_reason = response_data["choices"][0].get("finish_reason", "stop")
        tool_calls = message.get("tool_calls", [])

        if not tool_calls or stop_reason == "stop":
            break

        print(f"[copy-agent] Tools: {[tc['function']['name'] for tc in tool_calls]}", flush=True)
        assistant_msg: dict = {"role": "assistant", "content": message.get("content")}
        assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        tool_results = []
        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            tool_input = json.loads(tc["function"]["arguments"])
            print(f"[copy-agent]   → {tool_name}", flush=True)
            result_text = _dispatch_tool(tool_name, tool_input)
            collected.append(f"[{tool_name}]\n{result_text}")
            tool_results.append({"role": "tool", "tool_call_id": tc["id"], "content": result_text})

        messages.extend(tool_results)
        time.sleep(1)

        # If all 3 tools called, research is complete
        called_names = {tc["function"]["name"] for tc in tool_calls}
        if called_names >= {"search_winning_hooks", "search_competitor_copy", "benchmark_copy_performance"}:
            break

    return "\n\n".join(collected) if collected else ""


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
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
    Phase 1: Agent researches winning hooks + competitor copy + benchmarks via Perplexity tools.
    Phase 2: Structured Claude call generates schema-compliant CopywritingOutput using research.
    """
    print("[copy-agent] Phase 1: Research...", flush=True)
    live_research = _run_research_phase(product_category, market, platform)
    print(f"[copy-agent] Research complete: {len(live_research)} chars", flush=True)

    print("[copy-agent] Phase 2: Structured copy generation...", flush=True)
    from marketing_factory.prompts import _claude_with_validation, COPY_SYSTEM, _build_copy_user_prompt

    # Append live research to the copy prompt
    base_prompt = _build_copy_user_prompt(
        product_description, platform, voc_brief, dominant_narrative, white_space, journey_context
    )
    enhanced_prompt = (
        f"{base_prompt}\n\n"
        f"LIVE MARKET RESEARCH (use these real findings to inform your copy):\n"
        f"{live_research[:3000]}"
    ) if live_research else base_prompt

    return _claude_with_validation(
        system_prompt=COPY_SYSTEM,
        user_prompt=enhanced_prompt,
        schema_name="copywriting_agent",
        pydantic_model=CopywritingOutput,
    )


