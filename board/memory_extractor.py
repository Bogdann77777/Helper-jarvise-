# board/memory_extractor.py — LLM-based fact extraction (Mem0 production pattern)
#
# Architecture based on: arxiv.org/abs/2504.19413 (Mem0)
# Key insight: store EXTRACTED FACTS (5-10 bullets), NOT raw session summaries.
# Result: 91% fewer tokens, 26% better accuracy vs raw text injection.
#
# Pipeline:
#   BoardSession → Haiku extracts salient facts → compare with existing facts
#   → ADD / UPDATE / DELETE / NOOP → store in memory_facts table

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import aiohttp

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

_EXTRACTOR_MODEL = "anthropic/claude-haiku-4-5"  # fast + cheap, $0.25/1M input

# Domain tags for role-filtered retrieval
DOMAIN_FINANCIAL = "financial"
DOMAIN_STRATEGIC = "strategic"
DOMAIN_TECHNICAL = "technical"
DOMAIN_TEAM      = "team"
DOMAIN_MARKET    = "market"
DOMAIN_DECISION  = "decision"   # user_choice + outcome — always high importance

# Which domains each director gets in their context
DIRECTOR_DOMAINS = {
    "CFO": {DOMAIN_FINANCIAL, DOMAIN_TEAM, DOMAIN_DECISION},
    "CSO": {DOMAIN_STRATEGIC, DOMAIN_MARKET, DOMAIN_DECISION},
    "CTO": {DOMAIN_TECHNICAL, DOMAIN_TEAM, DOMAIN_DECISION},
    "CEO": {DOMAIN_FINANCIAL, DOMAIN_STRATEGIC, DOMAIN_TECHNICAL, DOMAIN_TEAM, DOMAIN_MARKET, DOMAIN_DECISION},
}

_EXTRACT_PROMPT = """\
You are a corporate memory system. Extract salient facts from this board session.

Rules:
- Write each fact as ONE concise sentence
- Only include facts that would matter for FUTURE board decisions about this company
- Tag each fact by domain: financial / strategic / technical / team / market / decision
- Tag importance: high (critical for decisions) / medium (useful context) / low (background)
- DO NOT include generic observations, opinions without data, or duplicates
- Maximum 12 facts total

Return ONLY valid JSON, no markdown:
{
  "facts": [
    {"text": "Company has 14 months runway at current $80k/month burn", "domain": "financial", "importance": "high"},
    {"text": "Main competitor (CompA) raised $5M Series A in Q1 2026", "domain": "market", "importance": "medium"}
  ]
}

Board session to analyze:
COMPANY: {company_name}
PROBLEM DISCUSSED: {problem_statement}
DIRECTOR ANALYSES:
{director_summaries}
CEO RECOMMENDATION:
{ceo_summary}
USER DECISION: {user_choice}
OUTCOME (if known): {outcome_notes}
"""

_RESOLVE_PROMPT = """\
You are a memory conflict resolver. For each NEW FACT, decide the operation:
- ADD: no equivalent exists in memory
- UPDATE: this fact updates/supersedes an existing fact (provide existing_id)
- DELETE: this fact makes an existing fact obsolete/wrong (provide existing_id)
- NOOP: this fact is already covered by existing memory (duplicate or less specific)

Existing memory facts for {company_name}:
{existing_facts_json}

New facts to process:
{new_facts_json}

Return ONLY valid JSON array:
[
  {"action": "ADD", "new_fact_index": 0},
  {"action": "UPDATE", "new_fact_index": 1, "existing_id": "abc123"},
  {"action": "NOOP", "new_fact_index": 2},
  {"action": "DELETE", "existing_id": "def456"}
]
"""


@dataclass
class ExtractedFact:
    text: str
    domain: str
    importance: str  # high / medium / low


@dataclass
class FactOperation:
    action: str            # ADD / UPDATE / DELETE / NOOP
    new_fact_index: Optional[int] = None   # index in new_facts list
    existing_id: Optional[str] = None      # for UPDATE/DELETE


async def extract_facts_from_session(
    company_name: str,
    problem_statement: str,
    director_summaries: list[str],
    ceo_summary: str,
    user_choice: str = "",
    outcome_notes: str = "",
) -> list[ExtractedFact]:
    """
    Call Haiku to extract salient facts from a completed board session.
    Returns list of ExtractedFact objects.
    """
    if not OPENROUTER_API_KEY:
        logger.warning("No OPENROUTER_API_KEY — skipping fact extraction")
        return []

    prompt = _EXTRACT_PROMPT.format(
        company_name=company_name,
        problem_statement=problem_statement[:500],
        director_summaries="\n".join(f"- {s}" for s in director_summaries),
        ceo_summary=ceo_summary[:800],
        user_choice=user_choice or "Not recorded",
        outcome_notes=outcome_notes or "Not recorded",
    )

    try:
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _EXTRACTOR_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 1000,
                },
                timeout=aiohttp.ClientTimeout(total=20),
            )
            data = await resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        facts = []
        for f in parsed.get("facts", []):
            if f.get("text") and f.get("domain") and f.get("importance"):
                facts.append(ExtractedFact(
                    text=f["text"],
                    domain=f["domain"].lower(),
                    importance=f["importance"].lower(),
                ))
        logger.info(f"Extracted {len(facts)} facts for {company_name}")
        return facts

    except Exception as e:
        logger.warning(f"Fact extraction failed: {e}")
        return []


async def resolve_fact_conflicts(
    company_name: str,
    new_facts: list[ExtractedFact],
    existing_facts: list[dict],  # list of {fact_id, text, domain, importance}
) -> list[FactOperation]:
    """
    Compare new facts against existing memory. Return ADD/UPDATE/DELETE/NOOP operations.
    Implements Mem0's conflict resolution: "latest truth wins".
    """
    if not new_facts:
        return []
    if not existing_facts:
        # No existing facts — ADD everything
        return [FactOperation(action="ADD", new_fact_index=i) for i in range(len(new_facts))]
    if not OPENROUTER_API_KEY:
        return [FactOperation(action="ADD", new_fact_index=i) for i in range(len(new_facts))]

    existing_json = json.dumps([
        {"id": f["fact_id"], "text": f["fact_text"], "domain": f["domain"]}
        for f in existing_facts
    ], ensure_ascii=False)

    new_json = json.dumps([
        {"index": i, "text": f.text, "domain": f.domain}
        for i, f in enumerate(new_facts)
    ], ensure_ascii=False)

    prompt = _RESOLVE_PROMPT.format(
        company_name=company_name,
        existing_facts_json=existing_json,
        new_facts_json=new_json,
    )

    try:
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _EXTRACTOR_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 800,
                },
                timeout=aiohttp.ClientTimeout(total=15),
            )
            data = await resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        ops_raw = json.loads(raw)
        ops = []
        for op in ops_raw:
            ops.append(FactOperation(
                action=op.get("action", "NOOP"),
                new_fact_index=op.get("new_fact_index"),
                existing_id=op.get("existing_id"),
            ))
        return ops

    except Exception as e:
        logger.warning(f"Conflict resolution failed: {e} — defaulting to ADD all")
        return [FactOperation(action="ADD", new_fact_index=i) for i in range(len(new_facts))]


def format_facts_for_context(facts: list[dict], role: str = "CEO") -> str:
    """
    Format extracted facts into a compact context string for LLM injection.
    Role-filtered: CFO sees financial facts, CTO sees technical, etc.
    Budget: ~1500-2000 chars (Mem0's ~2K token target).
    """
    if not facts:
        return ""

    allowed_domains = DIRECTOR_DOMAINS.get(role, set(DIRECTOR_DOMAINS["CEO"]))

    # Sort: high importance first, then by domain relevance
    domain_priority = list(allowed_domains)
    filtered = [f for f in facts if f.get("domain") in allowed_domains and f.get("is_active", 1)]

    # Sort: high > medium > low
    importance_order = {"high": 0, "medium": 1, "low": 2}
    filtered.sort(key=lambda f: importance_order.get(f.get("importance", "low"), 2))

    lines = ["=== COMPANY MEMORY (extracted facts) ==="]
    char_count = len(lines[0])
    budget = 2000

    for f in filtered:
        importance_tag = f.get("importance", "").upper()
        line = f"\n[{importance_tag}] {f['fact_text']}"
        if char_count + len(line) > budget:
            break
        lines.append(line)
        char_count += len(line)

    return "\n".join(lines)
