# board/memory_store.py — Persistent memory for Board of Directors sessions
# Layers: SQLite (structured) + Qdrant (semantic, optional) + JSON archives

import json
import logging
import os
from datetime import datetime
from typing import Optional

import aiosqlite

from board.memory_models import SessionSummary
from board.models import BoardSession, DataGateInput
from config import (
    MEMORY_DB_PATH,
    MEMORY_DIR,
    MEMORY_ARCHIVE_DIR,
    MEMORY_QDRANT_DIR,
    MEMORY_MAX_CONTEXT_CHARS,
    MEMORY_TIER_COMPANY,
    MEMORY_TIER_RECENT,
    MEMORY_TIER_DECISIONS,
)

logger = logging.getLogger(__name__)

# Singleton instance
_memory_store: Optional["MemoryStore"] = None


def get_memory_store() -> "MemoryStore":
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


class MemoryStore:
    """Main interface for all board memory operations."""

    def __init__(self):
        self._initialized = False
        self._qdrant_available = False
        self._qdrant_client = None
        self._embedding_model = None

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------

    async def initialize(self):
        """Create SQLite tables, Qdrant collection, archive dirs."""
        if self._initialized:
            return

        # Ensure directories exist
        os.makedirs(MEMORY_DIR, exist_ok=True)
        os.makedirs(MEMORY_ARCHIVE_DIR, exist_ok=True)

        # SQLite
        await self._init_sqlite()

        # Qdrant (optional — graceful fallback)
        await self._init_qdrant()

        self._initialized = True
        logger.info(
            f"MemoryStore initialized. SQLite: {MEMORY_DB_PATH}, "
            f"Qdrant: {'available' if self._qdrant_available else 'fallback to SQLite LIKE'}"
        )

    async def _init_sqlite(self):
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    project_id TEXT,
                    problem_statement TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ceo_summary TEXT DEFAULT '',
                    ceo_full_response TEXT DEFAULT '',
                    total_duration REAL DEFAULT 0,
                    user_choice TEXT,
                    outcome_notes TEXT,
                    memory_context_used TEXT
                );

                CREATE TABLE IF NOT EXISTS director_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    round INTEGER NOT NULL,
                    summary TEXT DEFAULT '',
                    recommendations TEXT DEFAULT '[]',
                    risks TEXT DEFAULT '[]',
                    cross_domain_risks TEXT DEFAULT '[]',
                    key_statements TEXT DEFAULT '[]',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    directors TEXT DEFAULT '[]',
                    description TEXT DEFAULT '',
                    severity TEXT DEFAULT 'medium',
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS business_profiles (
                    company_name TEXT PRIMARY KEY,
                    last_updated TEXT NOT NULL,
                    gate_input_json TEXT DEFAULT '{}',
                    recurring_risks TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS followups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    response TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    company_name TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    phase TEXT DEFAULT '',
                    context_notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_facts (
                    fact_id TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    project_id TEXT,
                    session_id TEXT,
                    fact_text TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    importance TEXT DEFAULT 'medium',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_company ON sessions(company_name);
                CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
                CREATE INDEX IF NOT EXISTS idx_director_results_session ON director_results(session_id);
                CREATE INDEX IF NOT EXISTS idx_conflicts_session ON conflicts(session_id);
                CREATE INDEX IF NOT EXISTS idx_projects_company ON projects(company_name);
                CREATE INDEX IF NOT EXISTS idx_facts_company ON memory_facts(company_name, is_active);
                CREATE INDEX IF NOT EXISTS idx_facts_domain ON memory_facts(company_name, domain, is_active);
            """)
            await db.commit()

    async def _init_qdrant(self):
        """Try to init Qdrant + sentence-transformers. Fall back gracefully."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            os.makedirs(MEMORY_QDRANT_DIR, exist_ok=True)
            self._qdrant_client = QdrantClient(path=MEMORY_QDRANT_DIR)

            # Create collection if not exists
            collections = [c.name for c in self._qdrant_client.get_collections().collections]
            if "board_sessions" not in collections:
                self._qdrant_client.create_collection(
                    collection_name="board_sessions",
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )

            # Load embedding model
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            self._qdrant_available = True
            logger.info("Qdrant + sentence-transformers loaded successfully")
        except ImportError as e:
            logger.warning(f"Qdrant/sentence-transformers not available: {e}. Using SQLite LIKE fallback.")
        except Exception as e:
            logger.warning(f"Qdrant init failed: {e}. Using SQLite LIKE fallback.")

    # ------------------------------------------------------------------
    #  Context retrieval — Smart Tiered Buffer (~30k tokens)
    # ------------------------------------------------------------------

    async def get_smart_context(
        self,
        company_name: str,
        problem: str,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Tiered context loader with relevance scoring.
        Budget: 24000 chars total (~30k tokens):
          T1 (2k)  — company/project profile (always)
          T2 (8k)  — last 3 sessions for this company/project
          T3 (10k) — semantically similar past sessions
          T4 (4k)  — past decisions + outcomes (feedback loop)
        """
        sections = []

        # T1: Company / project profile
        t1 = await self._build_t1_profile(company_name, project_id)
        if t1:
            sections.append(t1)

        # T2: Recent sessions
        t2 = await self._build_t2_recent(company_name, project_id, problem)
        if t2:
            sections.append(t2)

        # T4: Decisions with outcomes
        t4 = await self._build_t4_decisions(company_name, project_id)
        if t4:
            sections.append(t4)

        if not sections:
            return ""

        return "\n\n".join(sections)

    async def _build_t1_profile(self, company_name: str, project_id: Optional[str]) -> str:
        """T1: Company profile + optional project context."""
        lines = ["=== T1: COMPANY PROFILE ==="]

        profile = await self.get_company_profile(company_name)
        if profile:
            gate = profile.get("gate_input", {})
            facts = []
            if gate.get("current_stage"):
                facts.append(f"Stage: {gate['current_stage']}")
            if gate.get("revenue"):
                facts.append(f"Revenue: {gate['revenue']}")
            if gate.get("runway_months"):
                facts.append(f"Runway: {gate['runway_months']} months")
            if gate.get("team_size"):
                facts.append(f"Team: {gate['team_size']} people")
            if gate.get("tech_stack"):
                facts.append(f"Tech: {gate['tech_stack']}")
            if gate.get("competitors"):
                facts.append(f"Competitors: {gate['competitors']}")
            if profile.get("recurring_risks"):
                facts.append(f"Recurring risks: {'; '.join(profile['recurring_risks'][:3])}")
            if facts:
                lines.append(f"Company: {company_name}")
                lines.extend(facts)

        if project_id:
            project = await self.get_project(project_id)
            if project:
                lines.append(f"Active project: {project['name']} ({project.get('phase', '')})")
                if project.get("description"):
                    lines.append(f"Project description: {project['description']}")
                if project.get("context_notes"):
                    lines.append(f"Project notes: {project['context_notes']}")

        result = "\n".join(lines)
        return result[:MEMORY_TIER_COMPANY] if len(result) > MEMORY_TIER_COMPANY else result

    async def _build_t2_recent(
        self, company_name: str, project_id: Optional[str], problem: str
    ) -> str:
        """T2: Last 3 sessions for this company (or project)."""
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if project_id:
                cursor = await db.execute(
                    """SELECT session_id, created_at, problem_statement, ceo_summary, user_choice, outcome_notes
                       FROM sessions WHERE project_id = ?
                       ORDER BY created_at DESC LIMIT 3""",
                    (project_id,),
                )
            else:
                cursor = await db.execute(
                    """SELECT session_id, created_at, problem_statement, ceo_summary, user_choice, outcome_notes
                       FROM sessions WHERE company_name = ?
                       ORDER BY created_at DESC LIMIT 3""",
                    (company_name,),
                )
            rows = await cursor.fetchall()

        if not rows:
            return ""

        lines = ["=== T2: RECENT SESSIONS (last 3) ==="]
        char_count = len(lines[0])

        for row in rows:
            date_str = row["created_at"][:10]
            block = (
                f"\n[{date_str}] Problem: \"{row['problem_statement'][:120]}\"\n"
                f"  CEO summary: {(row['ceo_summary'] or '')[:250]}"
            )
            if row["user_choice"]:
                block += f"\n  Decision: {row['user_choice']}"
            if row["outcome_notes"]:
                block += f"\n  Outcome: {(row['outcome_notes'])[:150]}"

            if char_count + len(block) > MEMORY_TIER_RECENT:
                break
            lines.append(block)
            char_count += len(block)

        return "\n".join(lines)

    async def _build_t3_similar(self, company_name: str, problem: str, budget: int) -> str:
        """T3: Semantically similar past sessions (across all companies)."""
        if self._qdrant_available and self._qdrant_client and self._embedding_model:
            try:
                similar = await self._semantic_search(problem, limit=5)
            except Exception:
                similar = await self._like_search(problem, limit=5)
        else:
            similar = await self._like_search(problem, limit=5)

        # Exclude same company (already in T2)
        similar = [s for s in similar if s.company_name != company_name]
        if not similar:
            return ""

        lines = ["=== T3: SIMILAR PAST SITUATIONS (other companies) ==="]
        char_count = len(lines[0])

        for s in similar:
            date_str = s.created_at[:10]
            block = (
                f"\n[{date_str}] {s.company_name}: \"{s.problem_statement[:120]}\"\n"
                f"  Board conclusion: {s.ceo_summary[:300]}"
            )
            if s.user_choice:
                block += f"\n  They chose: {s.user_choice}"
            if s.outcome_notes:
                block += f"\n  Result: {s.outcome_notes[:150]}"

            if char_count + len(block) > budget:
                break
            lines.append(block)
            char_count += len(block)

        return "\n".join(lines)

    async def _build_t4_decisions(
        self, company_name: str, project_id: Optional[str]
    ) -> str:
        """T4: Past decisions with outcomes — the feedback loop."""
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT created_at, problem_statement, user_choice, outcome_notes
                   FROM sessions
                   WHERE company_name = ? AND user_choice IS NOT NULL AND user_choice != ''
                   ORDER BY created_at DESC LIMIT 10""",
                (company_name,),
            )
            rows = await cursor.fetchall()

        if not rows:
            return ""

        lines = ["=== T4: DECISION LEDGER (choices made + outcomes) ==="]
        char_count = len(lines[0])

        for row in rows:
            date_str = row["created_at"][:10]
            block = (
                f"\n[{date_str}] Problem: \"{row['problem_statement'][:80]}\"\n"
                f"  Chose: {row['user_choice']}"
            )
            if row["outcome_notes"]:
                block += f"\n  Outcome: {row['outcome_notes'][:200]}"

            if char_count + len(block) > MEMORY_TIER_DECISIONS:
                break
            lines.append(block)
            char_count += len(block)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Production context: role-filtered extracted facts (Mem0 pattern)
    # ------------------------------------------------------------------

    async def get_facts_context(
        self,
        company_name: str,
        role: str = "CEO",
        project_id: Optional[str] = None,
    ) -> str:
        """
        Load role-filtered extracted facts for a director's context.
        This is the Mem0 production approach: compressed facts, not raw summaries.
        Budget: ~2K chars (vs 24K raw text).
        CFO → financial+decision facts only
        CTO → technical+team+decision facts only
        CSO → strategic+market+decision facts only
        """
        from board.memory_extractor import format_facts_for_context, DIRECTOR_DOMAINS
        allowed_domains = DIRECTOR_DOMAINS.get(role, set(DIRECTOR_DOMAINS["CEO"]))

        placeholders = ",".join(f"'{d}'" for d in allowed_domains)
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if project_id:
                cursor = await db.execute(
                    f"""SELECT fact_id, fact_text, domain, importance
                        FROM memory_facts
                        WHERE (company_name = ? OR project_id = ?)
                          AND domain IN ({placeholders})
                          AND is_active = 1
                        ORDER BY
                          CASE importance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                          created_at DESC
                        LIMIT 20""",
                    (company_name, project_id),
                )
            else:
                cursor = await db.execute(
                    f"""SELECT fact_id, fact_text, domain, importance
                        FROM memory_facts
                        WHERE company_name = ?
                          AND domain IN ({placeholders})
                          AND is_active = 1
                        ORDER BY
                          CASE importance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                          created_at DESC
                        LIMIT 20""",
                    (company_name,),
                )
            rows = await cursor.fetchall()

        facts = [dict(r) for r in rows]
        return format_facts_for_context(facts, role)

    async def save_facts(
        self,
        company_name: str,
        session_id: str,
        new_facts,  # list[ExtractedFact]
        existing_facts: list[dict],
        operations,  # list[FactOperation]
        project_id: Optional[str] = None,
    ):
        """Apply ADD/UPDATE/DELETE/NOOP operations from conflict resolver."""
        import uuid
        now = datetime.now().isoformat()
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            for op in operations:
                if op.action == "ADD" and op.new_fact_index is not None:
                    fact = new_facts[op.new_fact_index]
                    await db.execute(
                        """INSERT INTO memory_facts
                           (fact_id, company_name, project_id, session_id, fact_text, domain, importance, is_active, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                        (uuid.uuid4().hex[:12], company_name, project_id, session_id,
                         fact.text, fact.domain, fact.importance, now, now),
                    )
                elif op.action == "UPDATE" and op.new_fact_index is not None and op.existing_id:
                    fact = new_facts[op.new_fact_index]
                    await db.execute(
                        """UPDATE memory_facts
                           SET fact_text = ?, importance = ?, updated_at = ?, session_id = ?
                           WHERE fact_id = ?""",
                        (fact.text, fact.importance, now, session_id, op.existing_id),
                    )
                elif op.action == "DELETE" and op.existing_id:
                    # Soft delete — keep for audit trail
                    await db.execute(
                        "UPDATE memory_facts SET is_active = 0, updated_at = ? WHERE fact_id = ?",
                        (now, op.existing_id),
                    )
                # NOOP: do nothing
            await db.commit()
        logger.info(f"Applied {len(operations)} fact operations for {company_name}")

    async def get_existing_facts(self, company_name: str, project_id: Optional[str] = None) -> list[dict]:
        """Get all active facts for conflict resolution."""
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT fact_id, fact_text, domain, importance
                   FROM memory_facts
                   WHERE company_name = ? AND is_active = 1
                   ORDER BY created_at DESC""",
                (company_name,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # Keep backward-compat methods pointing to smart context
    async def get_session_context(self, company_name: str, problem: str, project_id: Optional[str] = None) -> str:
        return await self.get_smart_context(company_name, problem, project_id)

    async def get_director_context(self, role: str, company_name: str, problem: str, project_id: Optional[str] = None) -> str:
        """Director gets role-filtered extracted facts (Mem0 pattern)."""
        facts_context = await self.get_facts_context(company_name, role, project_id)
        # Supplement with T1 profile if facts are empty (first session)
        if not facts_context:
            return await self._build_t1_profile(company_name, project_id)
        return facts_context

    async def _find_relevant_sessions(
        self, company_name: str, problem: str, limit: int = 5
    ) -> list[SessionSummary]:
        """Find relevant past sessions: same company + semantically similar."""
        results: list[SessionSummary] = []

        # 1. Same company sessions (always relevant)
        company_sessions = await self._get_sessions_by_company(company_name, limit=limit)
        results.extend(company_sessions)

        # 2. Semantic search (if Qdrant available)
        if self._qdrant_available and self._qdrant_client and self._embedding_model:
            try:
                semantic = await self._semantic_search(problem, limit=3)
                # Deduplicate
                seen_ids = {s.session_id for s in results}
                for s in semantic:
                    if s.session_id not in seen_ids:
                        results.append(s)
                        seen_ids.add(s.session_id)
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        else:
            # Fallback: SQLite LIKE search
            like_results = await self._like_search(problem, limit=3)
            seen_ids = {s.session_id for s in results}
            for s in like_results:
                if s.session_id not in seen_ids:
                    results.append(s)
                    seen_ids.add(s.session_id)

        return results[:limit]

    async def _get_sessions_by_company(self, company_name: str, limit: int = 5) -> list[SessionSummary]:
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT session_id, created_at, company_name, problem_statement,
                          ceo_summary, user_choice, outcome_notes
                   FROM sessions
                   WHERE company_name = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (company_name, limit),
            )
            rows = await cursor.fetchall()

        summaries = []
        for row in rows:
            conflicts = await self._get_session_conflicts(row["session_id"])
            risks = await self._get_session_top_risks(row["session_id"])
            summaries.append(SessionSummary(
                session_id=row["session_id"],
                created_at=row["created_at"],
                company_name=row["company_name"],
                problem_statement=row["problem_statement"],
                ceo_summary=row["ceo_summary"] or "",
                conflict_types=conflicts,
                top_risks=risks,
                user_choice=row["user_choice"],
                outcome_notes=row["outcome_notes"],
            ))
        return summaries

    async def _like_search(self, problem: str, limit: int = 3) -> list[SessionSummary]:
        """Fallback search using SQLite LIKE on problem_statement + ceo_summary."""
        keywords = [w for w in problem.split() if len(w) > 3][:5]
        if not keywords:
            return []

        conditions = " OR ".join(
            ["problem_statement LIKE ? OR ceo_summary LIKE ?"] * len(keywords)
        )
        params = []
        for kw in keywords:
            params.extend([f"%{kw}%", f"%{kw}%"])

        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"""SELECT session_id, created_at, company_name, problem_statement,
                           ceo_summary, user_choice, outcome_notes
                    FROM sessions
                    WHERE {conditions}
                    ORDER BY created_at DESC
                    LIMIT ?""",
                (*params, limit),
            )
            rows = await cursor.fetchall()

        summaries = []
        for row in rows:
            conflicts = await self._get_session_conflicts(row["session_id"])
            risks = await self._get_session_top_risks(row["session_id"])
            summaries.append(SessionSummary(
                session_id=row["session_id"],
                created_at=row["created_at"],
                company_name=row["company_name"],
                problem_statement=row["problem_statement"],
                ceo_summary=row["ceo_summary"] or "",
                conflict_types=conflicts,
                top_risks=risks,
                user_choice=row["user_choice"],
                outcome_notes=row["outcome_notes"],
            ))
        return summaries

    async def _semantic_search(self, problem: str, limit: int = 3) -> list[SessionSummary]:
        """Qdrant vector search for semantically similar past problems."""
        import asyncio
        embedding = await asyncio.to_thread(
            self._embedding_model.encode, problem
        )

        result = self._qdrant_client.query_points(
            collection_name="board_sessions",
            query=embedding.tolist(),
            limit=limit,
        )

        summaries = []
        for point in result.points:
            payload = point.payload
            summaries.append(SessionSummary(
                session_id=payload.get("session_id", ""),
                created_at=payload.get("created_at", ""),
                company_name=payload.get("company_name", ""),
                problem_statement=payload.get("problem_statement", ""),
                ceo_summary=payload.get("ceo_summary", ""),
                conflict_types=payload.get("conflict_types", []),
                top_risks=payload.get("top_risks", []),
                user_choice=payload.get("user_choice"),
                outcome_notes=payload.get("outcome_notes"),
            ))
        return summaries

    async def _get_session_conflicts(self, session_id: str) -> list[str]:
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            cursor = await db.execute(
                "SELECT type FROM conflicts WHERE session_id = ?", (session_id,)
            )
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def _get_session_top_risks(self, session_id: str, limit: int = 3) -> list[str]:
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            cursor = await db.execute(
                """SELECT risks FROM director_results
                   WHERE session_id = ? AND risks != '[]'
                   LIMIT ?""",
                (session_id, limit),
            )
            rows = await cursor.fetchall()

        all_risks = []
        for row in rows:
            try:
                risks = json.loads(row[0])
                all_risks.extend(risks[:2])
            except (json.JSONDecodeError, TypeError):
                pass
        return all_risks[:limit]

    def _format_session_summary(self, s: SessionSummary) -> str:
        """Format a session summary for CEO context injection."""
        date_str = s.created_at[:10] if len(s.created_at) >= 10 else s.created_at
        lines = [
            f"\n[{date_str}] Problem: \"{s.problem_statement[:100]}\"",
            f"  CEO Recommendation: {s.ceo_summary[:200]}",
        ]
        if s.conflict_types:
            lines.append(f"  Conflicts: {', '.join(s.conflict_types)}")
        if s.top_risks:
            lines.append(f"  Key risks: {'; '.join(s.top_risks[:2])}")
        if s.user_choice:
            lines.append(f"  User decided: {s.user_choice}")
        if s.outcome_notes:
            lines.append(f"  Outcome: {s.outcome_notes}")
        return "\n".join(lines)

    async def _format_director_summary(self, s: SessionSummary, role: str) -> str:
        """Format a session summary filtered for a specific director role."""
        date_str = s.created_at[:10] if len(s.created_at) >= 10 else s.created_at
        lines = [
            f"\n[{date_str}] Problem: \"{s.problem_statement[:100]}\"",
        ]

        # Get this director's past analysis for this session
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT summary, recommendations, risks FROM director_results
                   WHERE session_id = ? AND role = ?
                   ORDER BY round ASC""",
                (s.session_id, role),
            )
            rows = await cursor.fetchall()

        if rows:
            r1 = rows[0]
            lines.append(f"  Your past analysis: {r1['summary'][:150]}")
            try:
                recs = json.loads(r1["recommendations"])
                if recs:
                    lines.append(f"  Your recommendations: {'; '.join(recs[:2])}")
            except (json.JSONDecodeError, TypeError):
                pass

        if s.user_choice:
            lines.append(f"  User decided: {s.user_choice}")
        if s.outcome_notes:
            lines.append(f"  Outcome: {s.outcome_notes}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Save / Load sessions
    # ------------------------------------------------------------------

    async def save_session(self, session: BoardSession):
        """Save completed session to SQLite + JSON archive + Qdrant + extract facts."""
        try:
            await self._save_to_sqlite(session)
            await self._save_to_archive(session)
            await self._save_to_qdrant(session)
            # Async fact extraction (non-blocking — runs after session is saved)
            asyncio.create_task(self._extract_and_save_facts(session))
            logger.info(f"Session {session.id} saved to memory")
        except Exception as e:
            logger.error(f"Failed to save session {session.id} to memory: {e}", exc_info=True)

    async def _extract_and_save_facts(self, session: BoardSession):
        """
        Background task: extract facts from session using Haiku, resolve conflicts, store.
        Mem0 production pattern: store facts, not raw text.
        """
        try:
            from board.memory_extractor import extract_facts_from_session, resolve_fact_conflicts

            company = session.input.company_name
            director_summaries = [d.summary for d in session.directors if d.summary]
            ceo_summary = session.ceo_decision.summary if session.ceo_decision else ""
            user_choice = ""
            outcome_notes = ""

            # Extract new facts
            new_facts = await extract_facts_from_session(
                company_name=company,
                problem_statement=session.input.problem_statement,
                director_summaries=director_summaries,
                ceo_summary=ceo_summary,
                user_choice=user_choice,
                outcome_notes=outcome_notes,
            )

            if not new_facts:
                return

            # Get existing facts for conflict resolution
            existing = await self.get_existing_facts(company, session.project_id)

            # Resolve conflicts: ADD/UPDATE/DELETE/NOOP
            operations = await resolve_fact_conflicts(company, new_facts, existing)

            # Apply operations
            await self.save_facts(
                company_name=company,
                session_id=session.id,
                new_facts=new_facts,
                existing_facts=existing,
                operations=operations,
                project_id=session.project_id,
            )
        except Exception as e:
            logger.warning(f"Background fact extraction failed for session {session.id}: {e}")

    async def _save_to_sqlite(self, session: BoardSession):
        ceo_summary = ""
        ceo_full = ""
        if session.ceo_decision:
            ceo_summary = session.ceo_decision.summary or ""
            ceo_full = session.ceo_decision.full_response or ""

        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            # Main session record
            await db.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, company_name, project_id, problem_statement, created_at,
                    ceo_summary, ceo_full_response, total_duration, memory_context_used)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.id,
                    session.input.company_name,
                    session.project_id,
                    session.input.problem_statement,
                    session.created_at,
                    ceo_summary,
                    ceo_full,
                    session.total_duration_seconds,
                    session.memory_context_used,
                ),
            )

            # Director R1 results
            for d in session.directors:
                statements = [
                    {"tag": s.tag.value, "statement": s.statement, "confidence": s.confidence}
                    for s in d.statements
                ]
                await db.execute(
                    """INSERT INTO director_results
                       (session_id, role, round, summary, recommendations, risks, cross_domain_risks, key_statements)
                       VALUES (?, ?, 1, ?, ?, ?, ?, ?)""",
                    (
                        session.id,
                        d.role,
                        d.summary,
                        json.dumps(d.recommendations),
                        json.dumps(d.risks),
                        json.dumps(d.cross_domain_risks),
                        json.dumps(statements),
                    ),
                )

            # Director R2 results
            for r2 in session.round2_responses:
                recs = r2.revised_recommendations or []
                insights = r2.cross_domain_insights or []
                await db.execute(
                    """INSERT INTO director_results
                       (session_id, role, round, summary, recommendations, risks, key_statements)
                       VALUES (?, ?, 2, ?, ?, ?, ?)""",
                    (
                        session.id,
                        r2.role,
                        r2.revised_position,
                        json.dumps(recs),
                        json.dumps([]),
                        json.dumps(insights),
                    ),
                )

            # Conflicts
            for c in session.conflicts:
                await db.execute(
                    """INSERT INTO conflicts
                       (session_id, type, directors, description, severity)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        session.id,
                        c.type.value,
                        json.dumps(c.directors),
                        c.description,
                        c.severity,
                    ),
                )

            await db.commit()

    async def _save_to_archive(self, session: BoardSession):
        """Save full session dump to JSON archive."""
        now = datetime.now()
        month_dir = os.path.join(MEMORY_ARCHIVE_DIR, now.strftime("%Y%m"))
        os.makedirs(month_dir, exist_ok=True)

        archive_path = os.path.join(month_dir, f"session_{session.id}.json")
        session_data = session.model_dump(mode="json")

        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        # Update index
        index_path = os.path.join(MEMORY_ARCHIVE_DIR, "index.json")
        index = {}
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            except (json.JSONDecodeError, OSError):
                index = {}

        index[session.id] = {
            "company_name": session.input.company_name,
            "problem_statement": session.input.problem_statement[:100],
            "created_at": session.created_at,
            "archive_path": os.path.relpath(archive_path, MEMORY_ARCHIVE_DIR),
        }

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    async def _save_to_qdrant(self, session: BoardSession):
        """Save session vector to Qdrant for semantic search."""
        if not self._qdrant_available or not self._qdrant_client or not self._embedding_model:
            return

        try:
            import asyncio
            from qdrant_client.models import PointStruct

            ceo_summary = session.ceo_decision.summary if session.ceo_decision else ""
            text = f"{session.input.problem_statement} {ceo_summary}"
            embedding = await asyncio.to_thread(self._embedding_model.encode, text)

            conflict_types = [c.type.value for c in session.conflicts]
            top_risks = []
            for d in session.directors:
                top_risks.extend(d.risks[:2])

            point = PointStruct(
                id=hash(session.id) % (2**63),
                vector=embedding.tolist(),
                payload={
                    "session_id": session.id,
                    "company_name": session.input.company_name,
                    "problem_statement": session.input.problem_statement,
                    "ceo_summary": ceo_summary,
                    "created_at": session.created_at,
                    "conflict_types": conflict_types,
                    "top_risks": top_risks[:3],
                },
            )

            self._qdrant_client.upsert(
                collection_name="board_sessions",
                points=[point],
            )
        except Exception as e:
            logger.warning(f"Qdrant save failed: {e}")

    async def load_session(self, session_id: str) -> Optional[BoardSession]:
        """Load a session from JSON archive (full fidelity)."""
        index_path = os.path.join(MEMORY_ARCHIVE_DIR, "index.json")
        if not os.path.exists(index_path):
            return None

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        entry = index.get(session_id)
        if not entry:
            return None

        archive_path = os.path.join(MEMORY_ARCHIVE_DIR, entry["archive_path"])
        if not os.path.exists(archive_path):
            return None

        try:
            with open(archive_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return BoardSession.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from archive: {e}")
            return None

    # ------------------------------------------------------------------
    #  Outcome recording
    # ------------------------------------------------------------------

    async def record_outcome(self, session_id: str, user_choice: str, notes: str = ""):
        """Record what the user decided after the board session."""
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            await db.execute(
                """UPDATE sessions SET user_choice = ?, outcome_notes = ?
                   WHERE session_id = ?""",
                (user_choice, notes, session_id),
            )
            await db.commit()

        # Also update Qdrant payload if available
        if self._qdrant_available and self._qdrant_client:
            try:
                self._qdrant_client.set_payload(
                    collection_name="board_sessions",
                    payload={"user_choice": user_choice, "outcome_notes": notes},
                    points=[hash(session_id) % (2**63)],
                )
            except Exception as e:
                logger.warning(f"Qdrant payload update failed: {e}")

    # ------------------------------------------------------------------
    #  Projects CRUD
    # ------------------------------------------------------------------

    async def create_project(
        self, name: str, company_name: str, description: str = "", phase: str = ""
    ) -> dict:
        import uuid
        project_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            await db.execute(
                """INSERT INTO projects (project_id, name, description, company_name, status, phase, context_notes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'active', ?, '', ?, ?)""",
                (project_id, name, description, company_name, phase, now, now),
            )
            await db.commit()
        return await self.get_project(project_id)

    async def get_project(self, project_id: str) -> Optional[dict]:
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM projects WHERE project_id = ?", (project_id,)
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_projects(self, company_name: Optional[str] = None, status: str = "active") -> list[dict]:
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if company_name:
                cursor = await db.execute(
                    """SELECT * FROM projects WHERE company_name = ? AND status = ?
                       ORDER BY updated_at DESC""",
                    (company_name, status),
                )
            else:
                cursor = await db.execute(
                    "SELECT * FROM projects WHERE status = ? ORDER BY updated_at DESC",
                    (status,),
                )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_project(self, project_id: str, **kwargs) -> Optional[dict]:
        allowed = {"name", "description", "phase", "context_notes", "status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return await self.get_project(project_id)
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            await db.execute(
                f"UPDATE projects SET {set_clause} WHERE project_id = ?",
                (*updates.values(), project_id),
            )
            await db.commit()
        return await self.get_project(project_id)

    # ------------------------------------------------------------------
    #  Company listing (for autocomplete / minimal form)
    # ------------------------------------------------------------------

    async def list_companies(self) -> list[dict]:
        """List all known companies with basic profile for autocomplete."""
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT b.company_name, b.last_updated, b.gate_input_json,
                          COUNT(s.session_id) as session_count,
                          MAX(s.created_at) as last_session
                   FROM business_profiles b
                   LEFT JOIN sessions s ON s.company_name = b.company_name
                   GROUP BY b.company_name
                   ORDER BY last_session DESC"""
            )
            rows = await cursor.fetchall()

        result = []
        for row in rows:
            gate = {}
            try:
                gate = json.loads(row["gate_input_json"])
            except (json.JSONDecodeError, TypeError):
                pass
            result.append({
                "company_name": row["company_name"],
                "last_updated": row["last_updated"],
                "session_count": row["session_count"],
                "last_session": row["last_session"],
                "profile_summary": self._build_profile_summary(gate),
                "gate_input": gate,
            })
        return result

    def _build_profile_summary(self, gate: dict) -> str:
        """One-line summary of what we know about a company."""
        parts = []
        if gate.get("current_stage"):
            parts.append(gate["current_stage"])
        if gate.get("revenue"):
            parts.append(f"Rev: {gate['revenue']}")
        if gate.get("runway_months"):
            parts.append(f"Runway: {gate['runway_months']}m")
        if gate.get("team_size"):
            parts.append(f"Team: {gate['team_size']}")
        return " · ".join(parts) if parts else "Profile loaded"

    # ------------------------------------------------------------------
    #  Company profiles
    # ------------------------------------------------------------------

    async def get_company_profile(self, company_name: str) -> Optional[dict]:
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM business_profiles WHERE company_name = ?",
                (company_name,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "company_name": row["company_name"],
            "last_updated": row["last_updated"],
            "gate_input": json.loads(row["gate_input_json"]),
            "recurring_risks": json.loads(row["recurring_risks"]),
        }

    async def update_company_profile(self, company_name: str, gate_input: DataGateInput):
        """Update or create company profile from gate input data."""
        now = datetime.now().isoformat()
        gate_json = gate_input.model_dump(mode="json")

        # Get existing recurring risks
        existing = await self.get_company_profile(company_name)
        recurring_risks = existing["recurring_risks"] if existing else []

        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            await db.execute(
                """INSERT OR REPLACE INTO business_profiles
                   (company_name, last_updated, gate_input_json, recurring_risks)
                   VALUES (?, ?, ?, ?)""",
                (company_name, now, json.dumps(gate_json, ensure_ascii=False), json.dumps(recurring_risks)),
            )
            await db.commit()

    # ------------------------------------------------------------------
    #  Session listing
    # ------------------------------------------------------------------

    async def list_sessions(self, company_name: Optional[str] = None, limit: int = 10) -> list[dict]:
        """List recent sessions, optionally filtered by company."""
        async with aiosqlite.connect(MEMORY_DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            if company_name:
                cursor = await db.execute(
                    """SELECT session_id, company_name, problem_statement, created_at,
                              ceo_summary, user_choice, total_duration
                       FROM sessions WHERE company_name = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (company_name, limit),
                )
            else:
                cursor = await db.execute(
                    """SELECT session_id, company_name, problem_statement, created_at,
                              ceo_summary, user_choice, total_duration
                       FROM sessions
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                )
            rows = await cursor.fetchall()

        return [dict(row) for row in rows]
