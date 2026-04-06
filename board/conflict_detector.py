# board/conflict_detector.py — Обнаружение конфликтов между позициями директоров

import json
import logging
import re

from board.models import Conflict, ConflictType, DirectorResponse, Round2Response
from board.openrouter_client import get_openrouter_client
from config import MODEL_CONFLICT_DETECTOR

logger = logging.getLogger(__name__)

CONFLICT_SYSTEM_PROMPT = """You are a conflict detection system for a Board of Directors.

You receive summaries and recommendations from 3 directors (CSO, CFO, CTO).
Your job is to find REAL conflicts between their positions.

Conflict types:
- BUDGET: Directors disagree on spending priorities or resource allocation
- TIMELINE: Directors have incompatible timelines or urgency levels
- STRATEGIC: Fundamental disagreement on direction or approach
- PRIORITY: Directors prioritize different things that can't all be done

Rules:
- Only flag REAL conflicts, not minor differences in emphasis
- Each conflict must involve at least 2 directors
- Severity: "high" = blocking (must resolve before proceeding), "medium" = significant, "low" = minor tension

Respond in valid JSON:
{
  "conflicts": [
    {
      "type": "BUDGET|TIMELINE|STRATEGIC|PRIORITY",
      "directors": ["CSO", "CFO"],
      "description": "Clear description of the conflict",
      "severity": "low|medium|high"
    }
  ]
}

If there are no real conflicts, return: {"conflicts": []}
Do NOT wrap JSON in markdown code blocks.
Respond in the same language as the input.
"""


async def detect_conflicts(
    directors: list[DirectorResponse],
    round2_responses: list[Round2Response] | None = None,
) -> list[Conflict]:
    """Обнаружить конфликты между позициями директоров (R1 + optional R2)."""
    # Собираем позиции директоров в текст
    positions = []
    for d in directors:
        if d.error:
            continue
        section = f"=== {d.role} (Round 1) ===\n"
        section += f"Summary: {d.summary}\n"
        if d.recommendations:
            section += "Recommendations:\n"
            for r in d.recommendations:
                section += f"  - {r}\n"
        if d.risks:
            section += "Risks:\n"
            for r in d.risks:
                section += f"  - {r}\n"
        positions.append(section)

    # Add Round 2 data if available (richer conflict detection)
    if round2_responses:
        for r2 in round2_responses:
            if r2.error:
                continue
            section = f"=== {r2.role} (Round 2 - Revised) ===\n"
            section += f"Revised Position: {r2.revised_position}\n"
            if r2.challenges:
                section += "Challenges raised:\n"
                for c in r2.challenges:
                    section += f"  - [{c.severity}] To {c.target_director}: {c.point}\n"
            if r2.revised_recommendations:
                section += "Revised Recommendations:\n"
                for r in r2.revised_recommendations:
                    section += f"  - {r}\n"
            positions.append(section)

    if len(positions) < 2:
        logger.info("Less than 2 valid director responses — skipping conflict detection")
        return []

    user_prompt = "Analyze these director positions for conflicts:\n\n" + "\n".join(positions)

    try:
        client = get_openrouter_client()
        raw = await client.chat_completion(
            model=MODEL_CONFLICT_DETECTOR,
            messages=[
                {"role": "system", "content": CONFLICT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Точность важнее креативности
            max_tokens=1500,
        )

        return _parse_conflicts(raw)

    except Exception as e:
        logger.error(f"Conflict detection error: {e}", exc_info=True)
        return []


def _parse_conflicts(raw: str) -> list[Conflict]:
    """Парсинг JSON ответа конфликт-детектора."""
    raw_clean = raw.strip()

    # Убираем markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_clean)
    if json_match:
        raw_clean = json_match.group(1).strip()

    try:
        data = json.loads(raw_clean)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{[\s\S]*\}", raw_clean)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Conflict detector: JSON parse failed")
                return []
        else:
            return []

    conflicts = []
    for c in data.get("conflicts", []):
        try:
            conflict_type = ConflictType(c.get("type", "STRATEGIC").upper())
        except ValueError:
            conflict_type = ConflictType.STRATEGIC

        conflicts.append(Conflict(
            type=conflict_type,
            directors=c.get("directors", []),
            description=c.get("description", ""),
            severity=c.get("severity", "medium"),
        ))

    logger.info(f"Conflict detector found {len(conflicts)} conflicts")
    return conflicts
