# board/ceo_prompt.py — CEO Advisory Prompt (recommendation, not decision)

from board.models import DataGateInput, DirectorResponse, Conflict, Round2Response


def build_ceo_prompt(
    gate_input: DataGateInput,
    directors: list[DirectorResponse],
    conflicts: list[Conflict],
    round2_responses: list[Round2Response] | None = None,
    memory_context: str | None = None,
) -> str:
    """Builds unified prompt for Claude CLI (CEO advisory synthesis).
    Sees both Round 1 + Round 2 + conflicts. Presents OPTIONS, not a single decision."""

    sections = []

    # === CEO Role (advisor, not dictator) ===
    sections.append(
        "You are the CEO providing a STRATEGIC RECOMMENDATION to the founder/decision-maker. "
        "Your Board of Directors (CSO, CFO, CTO) has analyzed the situation in two rounds:\n"
        "- Round 1: Independent parallel analysis with tagged statements (FACT/ASSUMPTION/ESTIMATE/DISPUTED)\n"
        "- Round 2: Sequential debate where directors challenged each other and revised their positions\n"
        "\nYour job:\n"
        "1. Synthesize all perspectives from BOTH rounds into a coherent recommendation\n"
        "2. Present 2-3 clear OPTIONS with pros, cons, and risk level for each\n"
        "3. State which option you RECOMMEND and why\n"
        "4. Acknowledge trade-offs honestly — there is no perfect answer\n"
        "5. Explicitly state: the final decision belongs to the user\n"
    )

    # === Business context ===
    sections.append("=" * 60)
    sections.append("BUSINESS CONTEXT")
    sections.append("=" * 60)
    ctx_parts = [
        f"Company: {gate_input.company_name}",
        f"Problem: {gate_input.problem_statement}",
    ]
    if gate_input.revenue:
        ctx_parts.append(f"Revenue: {gate_input.revenue}")
    if gate_input.expenses:
        ctx_parts.append(f"Expenses: {gate_input.expenses}")
    if gate_input.runway_months is not None:
        ctx_parts.append(f"Runway: {gate_input.runway_months} months")
    if gate_input.funding:
        ctx_parts.append(f"Funding: {gate_input.funding}")
    if gate_input.market_size:
        ctx_parts.append(f"Market: {gate_input.market_size}")
    if gate_input.competitors:
        ctx_parts.append(f"Competitors: {gate_input.competitors}")
    if gate_input.target_audience:
        ctx_parts.append(f"Target audience: {gate_input.target_audience}")
    if gate_input.tech_stack:
        ctx_parts.append(f"Tech stack: {gate_input.tech_stack}")
    if gate_input.current_stage:
        ctx_parts.append(f"Stage: {gate_input.current_stage}")
    if gate_input.team_size is not None:
        ctx_parts.append(f"Team size: {gate_input.team_size}")
    if gate_input.team_description:
        ctx_parts.append(f"Team: {gate_input.team_description}")
    if gate_input.additional_context:
        ctx_parts.append(f"Context: {gate_input.additional_context}")
    sections.append("\n".join(ctx_parts))

    # === Historical context from memory ===
    if memory_context:
        sections.append("")
        sections.append("=" * 60)
        sections.append("HISTORICAL CONTEXT (from board memory)")
        sections.append("=" * 60)
        sections.append(memory_context)
        sections.append(
            "\nUse this historical context to inform your recommendation. "
            "Reference past decisions where relevant, note trends, and flag "
            "if the current problem is similar to a previously analyzed one."
        )

    # === Round 1: Director positions ===
    sections.append("")
    sections.append("=" * 60)
    sections.append("ROUND 1: INDEPENDENT ANALYSIS")
    sections.append("=" * 60)
    for d in directors:
        sections.append("")
        sections.append(f"--- {d.role} (model: {d.model}) ---")

        if d.error:
            sections.append(f"[ERROR: {d.error}]")
            continue

        sections.append(f"Summary: {d.summary}")

        if d.statements:
            sections.append("Tagged Statements:")
            for s in d.statements:
                sections.append(
                    f"  [{s.tag.value}] (confidence: {s.confidence:.1f}) {s.statement}"
                )

        if d.recommendations:
            sections.append("Recommendations:")
            for i, r in enumerate(d.recommendations, 1):
                sections.append(f"  {i}. {r}")

        if d.risks:
            sections.append("Risks identified:")
            for r in d.risks:
                sections.append(f"  - {r}")

    # === Round 2: Debate results ===
    if round2_responses:
        sections.append("")
        sections.append("=" * 60)
        sections.append("ROUND 2: DEBATE & CROSS-EXAMINATION")
        sections.append("=" * 60)

        for r2 in round2_responses:
            sections.append("")
            sections.append(f"--- {r2.role} (Round 2) ---")

            if r2.error:
                sections.append(f"[ERROR: {r2.error}]")
                continue

            sections.append(f"Revised Position: {r2.revised_position}")

            if r2.agreements:
                sections.append("Agreements:")
                for a in r2.agreements:
                    sections.append(f"  + {a}")

            if r2.challenges:
                sections.append("Challenges raised:")
                for c in r2.challenges:
                    sections.append(f"  ! [{c.severity}] To {c.target_director}: {c.point}")
                    sections.append(f"    Reasoning: {c.reasoning}")

            if r2.revised_recommendations:
                sections.append("Revised Recommendations:")
                for i, r in enumerate(r2.revised_recommendations, 1):
                    sections.append(f"  {i}. {r}")

            if r2.cross_domain_insights:
                sections.append("Cross-domain Insights:")
                for ins in r2.cross_domain_insights:
                    sections.append(f"  * {ins}")

    # === Conflicts ===
    if conflicts:
        sections.append("")
        sections.append("=" * 60)
        sections.append("CONFLICTS DETECTED")
        sections.append("=" * 60)
        for i, c in enumerate(conflicts, 1):
            sections.append(
                f"\n{i}. [{c.type.value}] (severity: {c.severity}) "
                f"Between: {', '.join(c.directors)}"
            )
            sections.append(f"   {c.description}")

    # === Advisory output instructions ===
    sections.append("")
    sections.append("=" * 60)
    sections.append("YOUR TASK AS CEO ADVISOR")
    sections.append("=" * 60)
    sections.append(
        "Based on BOTH rounds of analysis and the conflicts above, provide your "
        "CEO RECOMMENDATION:\n"
        "\n"
        "1. **Situation Assessment** (3-5 sentences): What the board analyses reveal\n"
        "\n"
        "2. **Options** (present 2-3 distinct options):\n"
        "   For each option:\n"
        "   - **Name**: Short label (e.g., 'Aggressive Growth', 'Conservative Path')\n"
        "   - **Description**: What this option entails\n"
        "   - **Pros**: Key advantages\n"
        "   - **Cons**: Key disadvantages\n"
        "   - **Risk Level**: Low / Medium / High\n"
        "   - **Which directors support this**: CSO/CFO/CTO alignment\n"
        "\n"
        "3. **Recommended Option**: Which option you recommend and WHY\n"
        "\n"
        "4. **Implementation Priorities**: If the recommended option is chosen, "
        "the top 3-5 immediate actions\n"
        "\n"
        "5. **Key Risks & Mitigations**: The 2-3 biggest risks regardless of option chosen\n"
        "\n"
        "6. **The final decision is yours.** Close with a statement acknowledging this is "
        "a recommendation and the decision-maker should weigh these options against their "
        "own judgment, risk tolerance, and context only they know.\n"
        "\nRespond in the same language as the business context above.\n"
        "Be clear and structured. A good advisor presents options, not ultimatums."
    )

    return "\n".join(sections)


def build_followup_prompt(question: str) -> str:
    """Follow-up prompt for CEO advisor (uses --continue)."""
    return (
        f"Follow-up question from the decision-maker:\n\n"
        f"{question}\n\n"
        f"Respond as CEO advisor. If the user has chosen one of the options you presented, "
        f"help them think through execution: next steps, resource allocation, timeline, "
        f"and potential pitfalls.\n"
        f"If the user disagrees with your recommendation, explore their reasoning "
        f"respectfully — they may have context you don't. Help them stress-test their "
        f"preferred approach.\n"
        f"Reference your previous recommendation and the directors' analyses."
    )
