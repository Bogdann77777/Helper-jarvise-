# board/orchestrator.py — Main flow: gate -> R1 (parallel) -> R2 (sequential debate) -> conflicts -> CEO advisor

import asyncio
import logging
import time
from typing import Callable, Optional

from board.models import BoardSession, SessionStatus, CEORecommendation, DataGateInput, Round2Response
from board.data_gate import validate_gate_input, DataGateError
from board.directors import get_all_directors
from board.conflict_detector import detect_conflicts
from board.ceo_prompt import build_ceo_prompt, build_followup_prompt
from board.memory_store import get_memory_store
from cli_executor import run_claude_streaming, new_session as cli_new_session
from config import MEMORY_ENABLED
from xtts_manager import get_xtts_manager

logger = logging.getLogger(__name__)

# In-memory session storage (MVP — no persistence)
_sessions: dict[str, BoardSession] = {}


def _resolve_dependencies(
    role: str,
    r1_by_role: dict[str, "DirectorResponse"],
) -> str:
    """
    Google ADK dependency injection pattern (output_key targeting).
    Each director declared its dependencies in systems_scan.my_dependencies during R1.
    Orchestrator resolves those dependencies from other directors' R1 outputs.
    Returns a targeted context string with ONLY what this director needs.

    This prevents context explosion: CFO doesn't need CTO's tech architecture,
    CTO doesn't need full financial P&L — only the specific numbers they flagged.
    """
    own_r1 = r1_by_role.get(role)
    if not own_r1 or not own_r1.systems_scan:
        return ""

    declared_needs = own_r1.systems_scan.my_dependencies
    if not declared_needs:
        return ""

    # Build targeted context from other directors' R1 data
    # Map keywords to source directors
    keyword_map = {
        "CFO": ["financial", "runway", "burn", "revenue", "budget", "funding", "unit economics"],
        "CTO": ["technical", "tech", "engineering", "timeline", "delivery", "architecture", "scalability"],
        "CSO": ["strategy", "market", "competitive", "positioning", "gtm", "growth"],
    }
    other_roles = [r for r in r1_by_role if r != role]

    lines = ["=== DEPENDENCY INJECTION (data from peer directors' R1 analysis) ==="]
    lines.append(f"Based on your declared dependencies, here is the specific data from peer analyses:\n")

    for other_role in other_roles:
        other_r1 = r1_by_role.get(other_role)
        if not other_r1:
            continue

        # Check if this director's declared needs reference the other role's domain
        other_keywords = keyword_map.get(other_role, [])
        needs_this_role = any(
            kw in need.lower()
            for need in declared_needs
            for kw in other_keywords
        )

        if needs_this_role or len(other_roles) == 1:
            lines.append(f"--- {other_role} R1 KEY OUTPUTS ---")
            lines.append(f"Summary: {other_r1.summary}")
            if other_r1.recommendations:
                lines.append("Recommendations: " + "; ".join(other_r1.recommendations[:3]))
            if other_r1.risks:
                lines.append("Key risks: " + "; ".join(other_r1.risks[:2]))

    if len(lines) <= 2:
        return ""

    return "\n".join(lines)


async def run_board_session(
    data: dict,
    on_status: Optional[Callable] = None,
    on_director_complete: Optional[Callable] = None,
    on_ceo_chunk: Optional[Callable] = None,
    on_director_audio: Optional[Callable] = None,
    on_ceo_audio: Optional[Callable] = None,
    on_round2_complete: Optional[Callable] = None,
    on_round2_audio: Optional[Callable] = None,
    on_round2_status: Optional[Callable] = None,
) -> BoardSession:
    """
    Full board session flow:
    1. Data Gate (validation)
    2. Round 1: Directors analyze in parallel
    3. Round 2: Directors debate sequentially (CSO -> CFO -> CTO)
    4. Conflict Detection (using R1 + R2 data)
    5. CEO Advisory Synthesis (Claude CLI, new session)
    """
    start = time.time()

    # 1. Data Gate
    if on_status:
        await on_status("Validating input data...")
    gate_input = validate_gate_input(data)

    session = BoardSession(input=gate_input, project_id=gate_input.project_id)
    _sessions[session.id] = session

    try:
        # 1b. Memory: load role-filtered extracted facts per director (Mem0 production pattern)
        # Each director gets ONLY their domain's facts (~2K chars), not a 24K raw dump.
        # Source: arxiv.org/abs/2504.19413 — 91% fewer tokens, 26% better accuracy.
        memory_context = None          # CEO synthesis context (full smart context)
        director_contexts: dict[str, str] = {}
        if MEMORY_ENABLED:
            try:
                memory_store = get_memory_store()
                project_id = gate_input.project_id

                # CEO gets full smart context (T1+T2+T4)
                memory_context = await memory_store.get_smart_context(
                    gate_input.company_name, gate_input.problem_statement, project_id
                )

                # Each director gets role-filtered extracted facts only
                for role in ["CSO", "CFO", "CTO"]:
                    director_contexts[role] = await memory_store.get_director_context(
                        role, gate_input.company_name, gate_input.problem_statement, project_id
                    )

                await memory_store.update_company_profile(gate_input.company_name, gate_input)
                if memory_context:
                    session.memory_context_used = memory_context[:300] + "..." if len(memory_context) > 300 else memory_context
                    logger.info(
                        f"Session {session.id}: CEO context={len(memory_context)} chars, "
                        f"director contexts={[f'{r}:{len(director_contexts[r])}' for r in director_contexts]}"
                    )
            except Exception as e:
                logger.warning(f"Memory context retrieval failed: {e}")

        # 2. Round 1: Directors analyze in parallel
        session.status = SessionStatus.DIRECTORS_ROUND1
        if on_status:
            await on_status("Round 1: Directors analyzing independently...")

        directors = get_all_directors()
        tasks = [d.analyze(gate_input, memory_context=director_contexts.get(d.role)) for d in directors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Director task exception: {result}")
                continue
            session.directors.append(result)
            if on_director_complete:
                await on_director_complete(result)

            # TTS for director R1 (non-blocking)
            if result.summary and not result.error:
                try:
                    xtts = get_xtts_manager()
                    audio_url = await asyncio.to_thread(
                        xtts.synthesize, result.summary, result.role.lower()
                    )
                    result.audio_url = audio_url
                    if on_director_audio:
                        await on_director_audio(result.role, audio_url)
                except Exception as e:
                    logger.warning(f"XTTS failed for {result.role}: {e}")

        # 3. Round 2: Sequential debate (CSO -> CFO -> CTO)
        # Google ADK pattern: output_key targeting — each director gets targeted injections
        # based on their declared dependencies, NOT a full context dump.
        # Source: adk.dev/agents/multi-agents/ — "one agent writes to specific state key,
        # subsequent agent reads it" — surgical context, not broadcast.
        session.status = SessionStatus.DIRECTORS_ROUND2
        if on_status:
            await on_status("Round 2: Directors debating...")

        # Build role → R1 response state (ADK's session.state equivalent)
        r1_by_role = {r.role: r for r in session.directors}
        prior_r2: list[Round2Response] = []
        debate_order = ["CSO", "CFO", "CTO"]

        for role in debate_order:
            if role not in r1_by_role:
                continue

            director = next((d for d in directors if d.role == role), None)
            if not director:
                continue

            if on_round2_status:
                await on_round2_status(f"{role} is responding to other directors...")

            # Dependency injection: resolve what this director declared it needs
            # from other R1 analyses (ADK output_key pattern)
            dependency_context = _resolve_dependencies(role, r1_by_role)

            r2_response = await director.debate(
                own_r1=r1_by_role[role],
                all_r1=session.directors,
                prior_r2=prior_r2,
                gate_input=gate_input,
                dependency_context=dependency_context,
            )

            session.round2_responses.append(r2_response)

            if on_round2_complete:
                await on_round2_complete(r2_response)

            # TTS for R2 (non-blocking)
            if r2_response.revised_position and not r2_response.error:
                try:
                    xtts = get_xtts_manager()
                    audio_url = await asyncio.to_thread(
                        xtts.synthesize, r2_response.revised_position, role.lower()
                    )
                    r2_response.audio_url = audio_url
                    if on_round2_audio:
                        await on_round2_audio(role, audio_url)
                except Exception as e:
                    logger.warning(f"XTTS failed for {role} R2: {e}")

            prior_r2.append(r2_response)

        # 4. Conflict Detection (with R2 data for richer analysis)
        session.status = SessionStatus.CONFLICTS_DETECTING
        if on_status:
            await on_status("Detecting conflicts...")

        session.conflicts = await detect_conflicts(
            session.directors,
            round2_responses=session.round2_responses,
        )

        # 5. CEO Advisory Synthesis (Claude CLI — new session)
        session.status = SessionStatus.CEO_THINKING
        if on_status:
            await on_status("CEO (Claude) preparing recommendation...")

        ceo_prompt = build_ceo_prompt(
            gate_input,
            session.directors,
            session.conflicts,
            round2_responses=session.round2_responses,
            memory_context=memory_context,
        )

        # Start new CLI session for board (isolated from Step/Live mode sessions)
        _BOARD_CLIENT_ID = "__board__"
        await cli_new_session(client_id=_BOARD_CLIENT_ID)

        ceo_chunks = []

        async def _on_chunk(line: str):
            ceo_chunks.append(line)
            if on_ceo_chunk:
                await on_ceo_chunk(line)

        ceo_start = time.time()
        ceo_response = await run_claude_streaming(
            ceo_prompt,
            on_chunk=_on_chunk,
            new_session=True,
            client_id=_BOARD_CLIENT_ID,
        )
        ceo_duration = round(time.time() - ceo_start, 2)

        session.ceo_decision = CEORecommendation(
            summary=ceo_response[:500] if ceo_response else "",
            full_response=ceo_response,
            duration_seconds=ceo_duration,
        )

        # TTS for CEO recommendation
        if ceo_response:
            try:
                xtts = get_xtts_manager()
                tts_text = ceo_response[:2000]
                ceo_audio_url = await asyncio.to_thread(
                    xtts.synthesize, tts_text, "ceo"
                )
                session.ceo_decision.audio_url = ceo_audio_url
                if on_ceo_audio:
                    await on_ceo_audio(ceo_audio_url)
            except Exception as e:
                logger.warning(f"XTTS failed for CEO: {e}")

        session.status = SessionStatus.COMPLETED
        session.total_duration_seconds = round(time.time() - start, 2)

        # Save to persistent memory
        if MEMORY_ENABLED:
            try:
                await get_memory_store().save_session(session)
            except Exception as e:
                logger.warning(f"Memory save failed for session {session.id}: {e}")

        if on_status:
            await on_status(f"Board session completed in {session.total_duration_seconds}s")

        logger.info(
            f"Board session {session.id} completed: "
            f"{len(session.directors)} directors R1, "
            f"{len(session.round2_responses)} directors R2, "
            f"{len(session.conflicts)} conflicts, "
            f"total {session.total_duration_seconds}s"
        )

        return session

    except Exception as e:
        session.status = SessionStatus.ERROR
        session.error = str(e)
        session.total_duration_seconds = round(time.time() - start, 2)
        logger.error(f"Board session {session.id} error: {e}", exc_info=True)
        raise


async def run_followup(
    session_id: str,
    question: str,
    on_ceo_chunk: Optional[Callable] = None,
    on_ceo_audio: Optional[Callable] = None,
) -> CEORecommendation:
    """Follow-up question to CEO advisor (uses --continue)."""
    session = await get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    prompt = build_followup_prompt(question)

    ceo_chunks = []

    async def _on_chunk(line: str):
        ceo_chunks.append(line)
        if on_ceo_chunk:
            await on_ceo_chunk(line)

    ceo_start = time.time()
    response = await run_claude_streaming(
        prompt,
        on_chunk=_on_chunk,
        client_id="__board__",
    )
    ceo_duration = round(time.time() - ceo_start, 2)

    decision = CEORecommendation(
        summary=response[:500] if response else "",
        full_response=response,
        duration_seconds=ceo_duration,
    )

    # TTS for follow-up
    if response:
        try:
            xtts = get_xtts_manager()
            tts_text = response[:2000]
            ceo_audio_url = await asyncio.to_thread(
                xtts.synthesize, tts_text, "ceo"
            )
            decision.audio_url = ceo_audio_url
            if on_ceo_audio:
                await on_ceo_audio(ceo_audio_url)
        except Exception as e:
            logger.warning(f"XTTS failed for follow-up CEO: {e}")

    # Append to follow-up history (don't overwrite original ceo_decision)
    session.followup_history.append(decision)
    return decision


async def get_session(session_id: str) -> Optional[BoardSession]:
    session = _sessions.get(session_id)
    if not session and MEMORY_ENABLED:
        try:
            session = await get_memory_store().load_session(session_id)
            if session:
                _sessions[session_id] = session  # cache it
        except Exception as e:
            logger.warning(f"Memory load failed for session {session_id}: {e}")
    return session
