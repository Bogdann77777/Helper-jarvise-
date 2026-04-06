"""
ConversationFSM — Sales Call State Machine.

States define what the agent is trying to accomplish RIGHT NOW.
Each state has its own system prompt section, allowed transitions, and behavior rules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger("sales.fsm")


class CallState(str, Enum):
    """Ordered states of an outbound sales call."""
    CONNECTING   = "connecting"    # Call just initiated, waiting for answer
    INTRO        = "intro"         # Greeting + confirm we have the right person
    HOOK         = "hook"          # 1-2 sentence reason for calling (< 20 sec)
    QUALIFY      = "qualify"       # Understand if there's a fit / pain point
    PITCH        = "pitch"         # Present the value proposition
    OBJECTION    = "objection"     # Handle pushback, then return to pitch or close
    CLOSE        = "close"         # Ask for the next step (meeting/demo/purchase)
    FOLLOWUP     = "followup"      # Confirm details after agreement
    REJECTED     = "rejected"      # Graceful exit when no interest
    VOICEMAIL    = "voicemail"     # Leave a short, compelling voicemail
    ENDED        = "ended"         # Call is over


# Valid transitions: which states can follow which
_VALID_TRANSITIONS: dict[CallState, set[CallState]] = {
    CallState.CONNECTING: {CallState.INTRO, CallState.VOICEMAIL},
    CallState.INTRO:      {CallState.HOOK, CallState.REJECTED, CallState.VOICEMAIL},
    CallState.HOOK:       {CallState.QUALIFY, CallState.OBJECTION, CallState.REJECTED},
    CallState.QUALIFY:    {CallState.PITCH, CallState.OBJECTION, CallState.REJECTED},
    CallState.PITCH:      {CallState.CLOSE, CallState.OBJECTION, CallState.QUALIFY, CallState.REJECTED},
    CallState.OBJECTION:  {CallState.PITCH, CallState.CLOSE, CallState.QUALIFY, CallState.REJECTED},
    CallState.CLOSE:      {CallState.FOLLOWUP, CallState.OBJECTION, CallState.REJECTED},
    CallState.FOLLOWUP:   {CallState.ENDED},
    CallState.REJECTED:   {CallState.ENDED},
    CallState.VOICEMAIL:  {CallState.ENDED},
    CallState.ENDED:      set(),
}

# How many times we allow returning to OBJECTION before forcing a close attempt
MAX_OBJECTION_CYCLES = 3

# Minimum seconds in QUALIFY before moving to PITCH
MIN_QUALIFY_SECONDS = 20.0


@dataclass
class ExtractedFacts:
    """Structured facts extracted from the conversation."""
    prospect_name: Optional[str] = None
    company_name: Optional[str] = None
    role: Optional[str] = None
    pain_point: Optional[str] = None
    interest_level: str = "unknown"   # unknown | low | medium | high
    agreed_next_step: Optional[str] = None
    best_callback_time: Optional[str] = None
    custom: dict = field(default_factory=dict)


@dataclass
class StateSnapshot:
    state: CallState
    entered_at: datetime
    exchange_count: int = 0       # turns in this state
    objection_text: Optional[str] = None  # last objection heard


class ConversationFSM:
    """
    Tracks call state and enforces valid transitions.
    The LLM calls transition_state() via function calling.
    """

    def __init__(self):
        self._state = CallState.CONNECTING
        self._history: list[StateSnapshot] = [
            StateSnapshot(state=CallState.CONNECTING, entered_at=datetime.now())
        ]
        self._objection_count = 0
        self._total_exchanges = 0
        self._facts = ExtractedFacts()
        self._call_start = datetime.now()

    # ── Public API ────────────────────────────────────────────────

    @property
    def state(self) -> CallState:
        return self._state

    @property
    def facts(self) -> ExtractedFacts:
        return self._facts

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self._call_start).total_seconds()

    @property
    def exchanges_in_current_state(self) -> int:
        return self._history[-1].exchange_count

    def transition(self, new_state: CallState, reason: str = "") -> bool:
        """
        Attempt a state transition. Returns True if successful.
        Logs and rejects invalid transitions.
        """
        if new_state == self._state:
            return True  # Already there

        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            logger.warning(
                f"[FSM] INVALID transition {self._state}→{new_state} (reason: {reason}). "
                f"Allowed: {[s.value for s in allowed]}"
            )
            return False

        # Objection cycle guard
        if new_state == CallState.OBJECTION:
            self._objection_count += 1
            if self._objection_count > MAX_OBJECTION_CYCLES:
                logger.info("[FSM] Max objection cycles — forcing CLOSE")
                new_state = CallState.CLOSE
                reason = "max_objections_reached"

        logger.info(f"[FSM] {self._state.value} → {new_state.value} | {reason}")
        self._state = new_state
        self._history.append(
            StateSnapshot(state=new_state, entered_at=datetime.now())
        )
        return True

    def record_exchange(self):
        """Call after each agent turn to track conversation depth."""
        self._total_exchanges += 1
        if self._history:
            self._history[-1].exchange_count += 1

    def update_fact(self, key: str, value: str):
        """Store an extracted fact. key maps to ExtractedFacts fields or custom."""
        if hasattr(self._facts, key):
            setattr(self._facts, key, value)
            logger.debug(f"[FSM] fact: {key}={value!r}")
        else:
            self._facts.custom[key] = value

    def set_objection(self, objection_text: str):
        """Record the current objection for context."""
        if self._history:
            self._history[-1].objection_text = objection_text

    def get_state_context(self) -> dict:
        """
        Returns structured context for the LLM system prompt.
        Injected as a JSON block so the LLM knows what to do.
        """
        return {
            "current_state": self._state.value,
            "elapsed_seconds": round(self.elapsed_seconds),
            "total_exchanges": self._total_exchanges,
            "exchanges_in_state": self.exchanges_in_current_state,
            "objection_count": self._objection_count,
            "facts": {
                "prospect_name": self._facts.prospect_name,
                "company": self._facts.company_name,
                "role": self._facts.role,
                "pain_point": self._facts.pain_point,
                "interest_level": self._facts.interest_level,
            },
            "allowed_transitions": [s.value for s in _VALID_TRANSITIONS.get(self._state, set())],
        }

    def get_history_summary(self) -> str:
        """Compact history for LLM context."""
        parts = []
        for snap in self._history:
            parts.append(f"{snap.state.value}({snap.exchange_count}turns)")
        return " → ".join(parts)

    def is_terminal(self) -> bool:
        return self._state == CallState.ENDED

    def get_call_outcome(self) -> str:
        """Determine call outcome from final state."""
        if not self._history:
            return "unknown"
        last_before_ended = None
        for snap in reversed(self._history):
            if snap.state != CallState.ENDED:
                last_before_ended = snap.state
                break
        outcome_map = {
            CallState.FOLLOWUP:  "converted",
            CallState.REJECTED:  "rejected",
            CallState.VOICEMAIL: "voicemail",
            CallState.CLOSE:     "attempted_close",
        }
        return outcome_map.get(last_before_ended, "incomplete")
