"""
Post-Call Scorer — LLM-as-Judge for call quality analysis.

After every call: transcript → Haiku → structured score JSON.

Metrics scored:
  - Overall quality (0-10)
  - Script adherence (0-10)
  - Objection handling (0-10)
  - Talk ratio analysis (agent should talk < 40%)
  - Sentiment trajectory (beginning vs end)
  - Conversion probability estimate
  - Key moments (best/worst lines)
  - Actionable improvement suggestions

Uses Claude Haiku via OpenRouter (cheap, fast — $0.0004 per call to score).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger("sales.scorer")

_MODEL = "anthropic/claude-haiku-4-5"
_SCORE_SCHEMA = {
    "overall_score": "float 0-10",
    "script_adherence": "float 0-10",
    "objection_handling": "float 0-10",
    "closing_strength": "float 0-10",
    "talk_ratio": "float (agent portion, target < 0.40)",
    "sentiment_start": "positive|neutral|negative",
    "sentiment_end": "positive|neutral|negative",
    "conversion_probability": "float 0-1",
    "best_moment": "string: the agent's best line",
    "worst_moment": "string: the agent's worst line or missed opportunity",
    "improvements": "list of 3 specific improvements",
    "outcome_analysis": "string: why this call succeeded or failed",
}


async def score_call(
    transcript: list[dict],
    outcome: str,
    persona_context: str = "",
    api_key: str = "",
) -> dict | None:
    """
    Score a call transcript using LLM-as-Judge.

    Args:
        transcript: list of {speaker, text, state, timestamp} dicts
        outcome: "converted" | "rejected" | "voicemail" | etc.
        persona_context: brief description of product/persona for context
        api_key: OpenRouter API key

    Returns: dict with scores, or None on error
    """
    if not api_key:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.warning("[SCORER] No API key — skipping scoring")
        return None

    if not transcript:
        return None

    # Format transcript for LLM
    formatted = "\n".join(
        f"[{t.get('state', '?').upper()}] {t['speaker'].upper()}: {t['text']}"
        for t in transcript
    )

    # Count basic stats
    agent_words = sum(len(t["text"].split()) for t in transcript if t["speaker"] == "agent")
    prospect_words = sum(len(t["text"].split()) for t in transcript if t["speaker"] == "prospect")
    total = agent_words + prospect_words
    talk_ratio = round(agent_words / total, 2) if total > 0 else 0.5

    prompt = f"""You are an expert sales call analyst. Analyze this AI sales agent call and provide structured feedback.

CALL OUTCOME: {outcome}
AGENT TALK RATIO: {talk_ratio} (ideal is < 0.40 — agent should talk less than prospect)
{f"CONTEXT: {persona_context}" if persona_context else ""}

TRANSCRIPT:
{formatted}

Analyze this call and respond with ONLY a JSON object matching this schema:
{json.dumps(_SCORE_SCHEMA, indent=2)}

Be specific and actionable. Reference exact quotes from the transcript in best_moment and worst_moment."""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 800,
                    "response_format": {"type": "json_object"},
                }
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            result["talk_ratio"] = talk_ratio  # Add our pre-calculated value
            logger.info(f"[SCORER] Score: {result.get('overall_score', '?')}/10 outcome={outcome}")
            return result

    except Exception as e:
        logger.error(f"[SCORER] Scoring failed: {e}")
        return None


def format_score_for_telegram(score: dict, call_id: str, contact_name: str = "") -> str:
    """Format scoring result for Telegram notification."""
    overall = score.get("overall_score", 0)
    stars = "⭐" * round(overall / 2)
    emoji = "✅" if overall >= 7 else "⚠️" if overall >= 5 else "❌"

    return (
        f"{emoji} Call Score: {overall:.1f}/10 {stars}\n"
        f"{'👤 ' + contact_name if contact_name else ''}\n"
        f"Script: {score.get('script_adherence', 0):.1f} | "
        f"Objections: {score.get('objection_handling', 0):.1f} | "
        f"Closing: {score.get('closing_strength', 0):.1f}\n"
        f"Talk ratio: {score.get('talk_ratio', 0):.0%} (agent) {'✅' if score.get('talk_ratio', 1) < 0.4 else '❌'}\n"
        f"Sentiment: {score.get('sentiment_start', '?')} → {score.get('sentiment_end', '?')}\n"
        f"Conversion prob: {score.get('conversion_probability', 0):.0%}\n\n"
        f"💡 Top improvement: {score.get('improvements', ['?'])[0] if score.get('improvements') else '?'}\n"
        f"🏆 Best: {score.get('best_moment', '?')[:100]}\n"
        f"⚠️ Missed: {score.get('worst_moment', '?')[:100]}"
    )
