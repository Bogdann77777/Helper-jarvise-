# board/memory_models.py — Pydantic models for memory system

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SessionSummary(BaseModel):
    """Lightweight summary of a past board session (for memory context injection)."""
    session_id: str
    created_at: str
    company_name: str
    problem_statement: str
    ceo_summary: str
    conflict_types: list[str] = []
    top_risks: list[str] = []
    user_choice: Optional[str] = None
    outcome_notes: Optional[str] = None
