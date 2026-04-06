# board/director_base.py — Базовый класс директора

import json
import logging
import re
import time
from typing import Optional

from board.models import (
    DirectorResponse, TaggedStatement, TagType, DataGateInput,
    Round2Response, ChallengePoint, DirectedQuestion, SystemsScan,
)
from board.openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)

# Инструкция по формату вывода, встраивается в system prompt каждого директора
OUTPUT_FORMAT_INSTRUCTIONS = """
You MUST respond in valid JSON format. Any other format will be REJECTED.

=== SYSTEMS THINKING PROTOCOL (mandatory first step) ===
Before focusing on your domain, you MUST scan the FULL situation:

Required JSON structure:
{
  "systems_scan": {
    "all_known_facts": [
      "FACT: Revenue is $X (stated in brief)",
      "FACT: Team size is Y (stated in brief)",
      "... list ALL facts from the brief, not just your domain"
    ],
    "other_domains": {
      "CSO": "Will analyze: market positioning, competitive moat, GTM strategy",
      "CFO": "Will analyze: unit economics, burn rate, runway, fundraising",
      "CTO": "Will analyze: tech feasibility, timeline, scalability, tech debt"
    },
    "my_dependencies": [
      "My recommendation ASSUMES CFO confirms runway > 12 months",
      "My recommendation ASSUMES CTO estimates delivery < 6 months"
    ],
    "breaks_if": [
      "This strategy collapses if burn rate doubles — must verify with CFO",
      "Timeline assumption breaks if engineering team < 5 — must verify with CTO"
    ]
  },
  "summary": "2-3 sentence executive summary of YOUR DOMAIN analysis",
  "statements": [
    {
      "tag": "FACT|ASSUMPTION|ESTIMATE|DISPUTED",
      "statement": "Your specific claim or observation",
      "confidence": 0.0-1.0
    }
  ],
  "recommendations": ["Specific actionable recommendation 1", "..."],
  "risks": ["Risk in YOUR domain 1", "..."],
  "cross_domain_risks": ["Risk that SPANS multiple domains — e.g. 'If CFO confirms runway < 9 months AND CTO says feature takes 4 months, we cannot execute this strategy'"]
}

Tag definitions:
- FACT: Verified from provided data (confidence >= 0.8)
- ASSUMPTION: Based on your interpretation, not directly stated (confidence 0.4-0.7)
- ESTIMATE: Numerical projection or forecast (always state the basis)
- DISPUTED: Contradicts common practice or other likely positions (flag it)

Rules:
- systems_scan is MANDATORY — skip it and your response is invalid
- Minimum 3 statements, minimum 2 recommendations, minimum 1 risk
- cross_domain_risks: at least 1 item that connects your domain to another
- Every numerical claim must have a tag
- Do NOT wrap JSON in markdown code blocks
"""


ROUND2_OUTPUT_FORMAT = """
You MUST respond in valid JSON format. Any other format will be REJECTED.

Required JSON structure:
{
  "agreements": ["Point you agree with from another director"],
  "challenges": [
    {
      "target_director": "CSO|CFO|CTO",
      "point": "What you challenge",
      "reasoning": "Why you disagree",
      "severity": "low|medium|high"
    }
  ],
  "revised_position": "Your updated position after considering others' analyses",
  "questions_for_others": [
    {
      "target_director": "CSO|CFO|CTO",
      "question": "A question you want them to address"
    }
  ],
  "revised_recommendations": ["Updated recommendation 1", "..."],
  "cross_domain_insights": ["Insight that connects multiple domains"]
}

Rules:
- Minimum 1 agreement, minimum 1 challenge
- revised_position must reflect how your thinking evolved
- Cross-domain insights should connect your expertise with others'
- Do NOT wrap JSON in markdown code blocks
"""


class DirectorBase:
    """Базовый класс для директоров Board of Directors."""

    role: str = ""          # CSO / CFO / CTO
    model: str = ""         # OpenRouter model ID
    identity: str = ""      # Кто этот директор
    responsibility: str = ""  # За что отвечает
    blind_spots: str = ""   # Что намеренно НЕ видит (data minimization)
    temperature: float = 0.7

    def _get_character_prompt(self) -> str:
        """Override in subclass: deep character description (~200 words)."""
        return ""

    def _get_expertise_prompt(self) -> str:
        """Override in subclass: deep expertise/frameworks description (~200 words)."""
        return ""

    def build_system_prompt(self) -> str:
        character = self._get_character_prompt()
        expertise = self._get_expertise_prompt()

        # If deep prompts are defined, use them; otherwise fall back to legacy
        if character and expertise:
            return f"""You are the {self.role} on a Board of Directors.

=== CHARACTER ===
{character}

=== EXPERTISE & FRAMEWORKS ===
{expertise}

BLIND SPOTS (you do NOT have access to this data and should NOT speculate about it):
{self.blind_spots}

ANCHOR: Always ground your analysis in the data provided. Do not make up numbers.

{OUTPUT_FORMAT_INSTRUCTIONS}

Respond in the same language as the user's input (Russian if input is Russian, English if English).
"""
        else:
            return f"""You are the {self.role} ({self.identity}) on a Board of Directors.

RESPONSIBILITY: {self.responsibility}

BLIND SPOTS (you do NOT have access to this data and should NOT speculate about it):
{self.blind_spots}

ANCHOR: Always ground your analysis in the data provided. Do not make up numbers.

{OUTPUT_FORMAT_INSTRUCTIONS}

Respond in the same language as the user's input (Russian if input is Russian, English if English).
"""

    def _build_debate_system_prompt(self) -> str:
        """System prompt for Round 2 debate."""
        character = self._get_character_prompt()
        expertise = self._get_expertise_prompt()

        return f"""You are the {self.role} on a Board of Directors, now in a DEBATE ROUND.

=== CHARACTER ===
{character}

=== EXPERTISE & FRAMEWORKS ===
{expertise}

=== DEBATE INSTRUCTIONS ===
You have seen the Round 1 analyses from all directors (including your own).
Now you must:
1. Identify points of AGREEMENT with other directors
2. CHALLENGE positions you disagree with — be specific and substantive
3. REVISE your own position based on new information from others
4. Ask QUESTIONS that expose blind spots in others' analyses
5. Offer CROSS-DOMAIN INSIGHTS that connect your expertise to others'

Stay in character. Be direct and professional. Defend your domain but acknowledge
when others have valid points. Your goal is better collective analysis, not winning.

{ROUND2_OUTPUT_FORMAT}

Respond in the same language as the Round 1 analyses.
"""

    def build_user_prompt(self, gate_input: DataGateInput, memory_context: Optional[str] = None) -> str:
        """Строит user prompt с данными, релевантными для этого директора.
        Переопределяется в подклассах для data minimization."""
        prompt = self._full_context(gate_input)
        if memory_context:
            prompt += "\n\n" + memory_context
        return prompt

    def _full_context(self, g: DataGateInput) -> str:
        """Полный бизнес-контекст (используется как база)."""
        parts = [
            f"Company: {g.company_name}",
            f"Problem/Question: {g.problem_statement}",
        ]
        if g.revenue:
            parts.append(f"Revenue: {g.revenue}")
        if g.expenses:
            parts.append(f"Expenses: {g.expenses}")
        if g.runway_months is not None:
            parts.append(f"Runway: {g.runway_months} months")
        if g.funding:
            parts.append(f"Funding: {g.funding}")
        if g.market_size:
            parts.append(f"Market size: {g.market_size}")
        if g.competitors:
            parts.append(f"Competitors: {g.competitors}")
        if g.target_audience:
            parts.append(f"Target audience: {g.target_audience}")
        if g.tech_stack:
            parts.append(f"Tech stack: {g.tech_stack}")
        if g.current_stage:
            parts.append(f"Stage: {g.current_stage}")
        if g.team_size is not None:
            parts.append(f"Team size: {g.team_size}")
        if g.team_description:
            parts.append(f"Team: {g.team_description}")
        if g.additional_context:
            parts.append(f"Additional context: {g.additional_context}")
        return "\n".join(parts)

    async def analyze(self, gate_input: DataGateInput, memory_context: Optional[str] = None) -> DirectorResponse:
        """Запуск анализа директором. Возвращает DirectorResponse."""
        start = time.time()
        try:
            client = get_openrouter_client()
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": self.build_user_prompt(gate_input, memory_context=memory_context)},
            ]

            raw = await client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )

            response = self._parse_response(raw)
            response.duration_seconds = round(time.time() - start, 2)
            return response

        except Exception as e:
            logger.error(f"Director {self.role} error: {e}", exc_info=True)
            return DirectorResponse(
                role=self.role,
                model=self.model,
                summary=f"Error: {e}",
                raw_response="",
                duration_seconds=round(time.time() - start, 2),
                error=str(e),
            )

    def _parse_response(self, raw: str) -> DirectorResponse:
        """Парсинг JSON ответа директора с fallback."""
        raw_clean = raw.strip()

        # Убираем markdown code blocks если есть
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_clean)
        if json_match:
            raw_clean = json_match.group(1).strip()

        try:
            data = json.loads(raw_clean)
        except json.JSONDecodeError:
            # Fallback: пытаемся найти JSON объект в тексте
            brace_match = re.search(r"\{[\s\S]*\}", raw_clean)
            if brace_match:
                try:
                    data = json.loads(brace_match.group(0))
                except json.JSONDecodeError:
                    return self._fallback_response(raw)
            else:
                return self._fallback_response(raw)

        # Parse systems_scan
        systems_scan = None
        scan_data = data.get("systems_scan")
        if scan_data and isinstance(scan_data, dict):
            systems_scan = SystemsScan(
                all_known_facts=scan_data.get("all_known_facts", []),
                other_domains=scan_data.get("other_domains", {}),
                my_dependencies=scan_data.get("my_dependencies", []),
                breaks_if=scan_data.get("breaks_if", []),
            )

        # Парсим statements
        statements = []
        for s in data.get("statements", []):
            try:
                tag = TagType(s.get("tag", "ASSUMPTION").upper())
            except ValueError:
                tag = TagType.ASSUMPTION
            statements.append(TaggedStatement(
                tag=tag,
                statement=s.get("statement", ""),
                confidence=min(1.0, max(0.0, float(s.get("confidence", 0.5)))),
            ))

        return DirectorResponse(
            role=self.role,
            model=self.model,
            summary=data.get("summary", ""),
            systems_scan=systems_scan,
            statements=statements,
            recommendations=data.get("recommendations", []),
            risks=data.get("risks", []),
            cross_domain_risks=data.get("cross_domain_risks", []),
            raw_response=raw,
        )

    def _fallback_response(self, raw: str) -> DirectorResponse:
        """Fallback когда JSON парсинг не удался — берём текст как есть."""
        logger.warning(f"Director {self.role}: JSON parse failed, using fallback")
        return DirectorResponse(
            role=self.role,
            model=self.model,
            summary=raw[:500],
            statements=[
                TaggedStatement(
                    tag=TagType.ASSUMPTION,
                    statement="Response was not in expected JSON format",
                    confidence=0.3,
                )
            ],
            recommendations=[],
            risks=["Response format was invalid — analysis may be incomplete"],
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    #  Round 2: Debate
    # ------------------------------------------------------------------

    async def debate(
        self,
        own_r1: DirectorResponse,
        all_r1: list[DirectorResponse],
        prior_r2: list[Round2Response],
        gate_input: DataGateInput,
    ) -> Round2Response:
        """Round 2 debate: respond to other directors' R1 analyses + prior R2 responses."""
        start = time.time()
        try:
            client = get_openrouter_client()

            user_prompt = self._build_debate_user_prompt(own_r1, all_r1, prior_r2, gate_input)

            messages = [
                {"role": "system", "content": self._build_debate_system_prompt()},
                {"role": "user", "content": user_prompt},
            ]

            raw = await client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature + 0.1,  # slightly more creative for debate
            )

            response = self._parse_round2_response(raw)
            response.duration_seconds = round(time.time() - start, 2)
            return response

        except Exception as e:
            logger.error(f"Director {self.role} debate error: {e}", exc_info=True)
            return Round2Response(
                role=self.role,
                model=self.model,
                revised_position=f"Error during debate: {e}",
                raw_response="",
                duration_seconds=round(time.time() - start, 2),
                error=str(e),
            )

    def _build_debate_user_prompt(
        self,
        own_r1: DirectorResponse,
        all_r1: list[DirectorResponse],
        prior_r2: list[Round2Response],
        gate_input: DataGateInput,
    ) -> str:
        """Build the user prompt for Round 2 debate."""
        parts = []

        parts.append("=== BUSINESS CONTEXT ===")
        parts.append(f"Company: {gate_input.company_name}")
        parts.append(f"Problem: {gate_input.problem_statement}")
        parts.append("")

        # Own R1 analysis
        parts.append("=== YOUR ROUND 1 ANALYSIS ===")
        parts.append(f"Summary: {own_r1.summary}")
        if own_r1.recommendations:
            parts.append("Your recommendations:")
            for r in own_r1.recommendations:
                parts.append(f"  - {r}")
        if own_r1.risks:
            parts.append("Your identified risks:")
            for r in own_r1.risks:
                parts.append(f"  - {r}")
        parts.append("")

        # Other directors' R1
        parts.append("=== OTHER DIRECTORS' ROUND 1 ANALYSES ===")
        for d in all_r1:
            if d.role == self.role:
                continue
            parts.append(f"\n--- {d.role} ---")
            parts.append(f"Summary: {d.summary}")
            if d.statements:
                parts.append("Key statements:")
                for s in d.statements:
                    parts.append(f"  [{s.tag.value}] {s.statement} (confidence: {s.confidence:.1f})")
            if d.recommendations:
                parts.append("Recommendations:")
                for r in d.recommendations:
                    parts.append(f"  - {r}")
            if d.risks:
                parts.append("Risks:")
                for r in d.risks:
                    parts.append(f"  - {r}")

        # Prior R2 responses (sequential: CSO sees none, CFO sees CSO's, CTO sees both)
        if prior_r2:
            parts.append("")
            parts.append("=== PRIOR ROUND 2 DEBATE RESPONSES ===")
            for r2 in prior_r2:
                parts.append(f"\n--- {r2.role} (Round 2) ---")
                parts.append(f"Revised position: {r2.revised_position}")
                if r2.agreements:
                    parts.append("Agreements:")
                    for a in r2.agreements:
                        parts.append(f"  + {a}")
                if r2.challenges:
                    parts.append("Challenges:")
                    for c in r2.challenges:
                        parts.append(f"  ! [{c.severity}] To {c.target_director}: {c.point}")
                        parts.append(f"    Reasoning: {c.reasoning}")
                if r2.questions_for_others:
                    parts.append("Questions:")
                    for q in r2.questions_for_others:
                        parts.append(f"  ? To {q.target_director}: {q.question}")
                if r2.cross_domain_insights:
                    parts.append("Cross-domain insights:")
                    for i in r2.cross_domain_insights:
                        parts.append(f"  * {i}")

        parts.append("")
        parts.append("Now provide your Round 2 debate response as JSON.")

        return "\n".join(parts)

    def _parse_round2_response(self, raw: str) -> Round2Response:
        """Parse Round 2 JSON response."""
        raw_clean = raw.strip()

        # Remove markdown code blocks if present
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
                    return self._fallback_round2(raw)
            else:
                return self._fallback_round2(raw)

        # Parse challenges
        challenges = []
        for c in data.get("challenges", []):
            challenges.append(ChallengePoint(
                target_director=c.get("target_director", ""),
                point=c.get("point", ""),
                reasoning=c.get("reasoning", ""),
                severity=c.get("severity", "medium"),
            ))

        # Parse questions
        questions = []
        for q in data.get("questions_for_others", []):
            questions.append(DirectedQuestion(
                target_director=q.get("target_director", ""),
                question=q.get("question", ""),
            ))

        return Round2Response(
            role=self.role,
            model=self.model,
            agreements=data.get("agreements", []),
            challenges=challenges,
            revised_position=data.get("revised_position", ""),
            questions_for_others=questions,
            revised_recommendations=data.get("revised_recommendations", []),
            cross_domain_insights=data.get("cross_domain_insights", []),
            raw_response=raw,
        )

    def _fallback_round2(self, raw: str) -> Round2Response:
        """Fallback for Round 2 when JSON parsing fails."""
        logger.warning(f"Director {self.role}: Round 2 JSON parse failed, using fallback")
        return Round2Response(
            role=self.role,
            model=self.model,
            revised_position=raw[:500],
            agreements=["Unable to parse structured response"],
            raw_response=raw,
        )
