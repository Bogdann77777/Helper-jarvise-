# board/models.py — Pydantic модели (контракты данных Board of Directors)

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Data Gate (входные данные) ---

class DataGateInput(BaseModel):
    """Входные данные от пользователя для board-сессии."""
    company_name: str = Field(..., min_length=1, description="Название компании")
    problem_statement: str = Field(..., min_length=10, description="Описание проблемы/вопроса")

    # Project context (optional — if set, memory loads project-scoped context)
    project_id: Optional[str] = None

    # Финансы (хотя бы одно обязательно — если не в памяти)
    revenue: Optional[str] = None
    expenses: Optional[str] = None
    runway_months: Optional[float] = None
    funding: Optional[str] = None

    # Рынок
    market_size: Optional[str] = None
    competitors: Optional[str] = None
    target_audience: Optional[str] = None

    # Продукт/технологии
    tech_stack: Optional[str] = None
    current_stage: Optional[str] = None

    # Команда
    team_size: Optional[int] = None
    team_description: Optional[str] = None

    # Дополнительный контекст
    additional_context: Optional[str] = None

    # Memory-loaded flag (set by server when profile auto-loaded from memory)
    loaded_from_memory: bool = False


# --- Verification Tags ---

class TagType(str, Enum):
    FACT = "FACT"
    ASSUMPTION = "ASSUMPTION"
    ESTIMATE = "ESTIMATE"
    DISPUTED = "DISPUTED"


class TaggedStatement(BaseModel):
    """Утверждение директора с verification tag."""
    tag: TagType
    statement: str
    confidence: float = Field(ge=0.0, le=1.0, description="0.0-1.0")


# --- Systems Thinking Scan (anti-tunnel-vision) ---

class SystemsScan(BaseModel):
    """Pre-analysis: director enumerates ALL facts before focusing on their domain."""
    all_known_facts: list[str] = Field(default_factory=list, description="All facts from the brief, across all domains")
    other_domains: dict[str, str] = Field(default_factory=dict, description="What each other director will analyze")
    my_dependencies: list[str] = Field(default_factory=list, description="My recommendations assume X from CFO/CTO")
    breaks_if: list[str] = Field(default_factory=list, description="My strategy collapses if... (cross-domain risks)")


# --- Director Response ---

class DirectorResponse(BaseModel):
    """Ответ одного директора."""
    role: str = Field(..., description="CSO / CFO / CTO")
    model: str = Field(..., description="Модель, которая сгенерировала ответ")
    summary: str = Field(..., description="Краткое резюме (2-3 предложения)")
    systems_scan: Optional[SystemsScan] = None  # Anti-tunnel-vision pre-analysis
    statements: list[TaggedStatement] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    cross_domain_risks: list[str] = Field(default_factory=list, description="Risks that span multiple domains")
    raw_response: str = Field(default="", description="Сырой ответ модели (для дебага)")
    duration_seconds: float = 0.0
    error: Optional[str] = None
    audio_url: Optional[str] = None


# --- Conflicts ---

class ConflictType(str, Enum):
    BUDGET = "BUDGET"
    TIMELINE = "TIMELINE"
    STRATEGIC = "STRATEGIC"
    PRIORITY = "PRIORITY"


class Conflict(BaseModel):
    """Конфликт между позициями директоров."""
    type: ConflictType
    directors: list[str] = Field(..., description="Какие директора конфликтуют")
    description: str
    severity: str = Field(default="medium", description="low / medium / high")


# --- Round 2: Debate models ---

class ChallengePoint(BaseModel):
    """A challenge from one director to another during Round 2 debate."""
    target_director: str  # CSO/CFO/CTO
    point: str
    reasoning: str
    severity: str = "medium"


class DirectedQuestion(BaseModel):
    """A question directed at a specific director during debate."""
    target_director: str
    question: str


class Round2Response(BaseModel):
    """Response from a director during Round 2 (sequential debate)."""
    role: str
    model: str
    agreements: list[str] = Field(default_factory=list)
    challenges: list[ChallengePoint] = Field(default_factory=list)
    revised_position: str = ""
    questions_for_others: list[DirectedQuestion] = Field(default_factory=list)
    revised_recommendations: list[str] = Field(default_factory=list)
    cross_domain_insights: list[str] = Field(default_factory=list)
    raw_response: str = ""
    duration_seconds: float = 0.0
    error: Optional[str] = None
    audio_url: Optional[str] = None


# --- CEO Recommendation (formerly CEODecision) ---

class CEORecommendation(BaseModel):
    """CEO advisory recommendation (not a final decision — user decides)."""
    summary: str = Field(default="", description="Краткое резюме рекомендации")
    full_response: str = Field(default="", description="Полный ответ CEO")
    duration_seconds: float = 0.0
    audio_url: Optional[str] = None


# Backward compatibility alias
CEODecision = CEORecommendation


# --- Board Session ---

class SessionStatus(str, Enum):
    PENDING = "pending"
    DIRECTORS_ROUND1 = "directors_round1"
    DIRECTORS_ROUND2 = "directors_round2"
    DIRECTORS_RUNNING = "directors_running"  # backward compat
    CONFLICTS_DETECTING = "conflicts_detecting"
    CEO_THINKING = "ceo_thinking"
    COMPLETED = "completed"
    ERROR = "error"


class BoardSession(BaseModel):
    """Полная board-сессия: от входных данных до решения CEO."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: SessionStatus = SessionStatus.PENDING
    input: DataGateInput
    project_id: Optional[str] = None  # Associated project (if any)
    directors: list[DirectorResponse] = Field(default_factory=list)
    round2_responses: list[Round2Response] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    ceo_decision: Optional[CEORecommendation] = None
    followup_history: list[CEORecommendation] = Field(default_factory=list)
    error: Optional[str] = None
    total_duration_seconds: float = 0.0
    memory_context_used: Optional[str] = None  # Audit trail: what memory was injected
