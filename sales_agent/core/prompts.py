"""
Sales Prompt Engine — generates dynamic system prompts based on FSM state.

Architecture:
  - 5-section structure: Identity → Guardrails → Script → Objections → Closing
  - State-specific instructions injected per turn
  - Turn injection: reminder every 5 turns to prevent drift
  - All prompts validated against research: temperature 0.4, max_tokens 80

Research findings applied:
  - Max 2 sentences per response (voice = short)
  - Temperature 0.4 (not 0.7!) for script adherence
  - Inject [INTERRUPTED: text] when barge-in occurs
  - Reminder every 5 turns prevents LLM from going off-script
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .fsm import CallState, ConversationFSM

# ── Load data ─────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"


def _load_objections() -> dict:
    try:
        return json.loads((_DATA_DIR / "objections.json").read_text(encoding="utf-8"))
    except Exception:
        return {"objections": []}


def _load_persona(persona_file: str) -> dict:
    try:
        path = _DATA_DIR / "personas" / persona_file
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── State-specific instructions ───────────────────────────────────────────────

_STATE_INSTRUCTIONS: dict[CallState, str] = {
    CallState.INTRO: """
CURRENT STEP: INTRODUCTION
Your ONLY job right now: confirm you have the right person and get permission to continue.
Script: "Hi, is this [PROSPECT_NAME]? / Great, this is [AGENT_NAME] calling from [COMPANY]. [DISCLOSURE]. I'll only take a minute — is now an okay time?"
- If YES → tell me to transition to HOOK
- If NO / busy → offer specific time options, then transition to CLOSE (for callback)
- If wrong person → ask for the right person or end politely
DO NOT mention the product yet.
""",
    CallState.HOOK: """
CURRENT STEP: HOOK (30 seconds MAX)
Your ONLY job: deliver one compelling reason why they should keep listening.
Use the hook: "[HOOK_LINE]"
Then immediately ask ONE question to gauge interest.
Keep this to 2 sentences total.
- If they engage → transition to QUALIFY
- If immediate rejection → transition to REJECTED (after one gentle push)
""",
    CallState.QUALIFY: """
CURRENT STEP: QUALIFICATION
Your ONLY job: understand if there's a real fit. Ask ONE question at a time.
Required qualifying questions (pick the most relevant based on what you know):
1. "Who currently handles [PAIN_AREA] for you?"
2. "How many [RELEVANT_METRIC] do you manage?"
3. "What's your biggest headache when it comes to [PAIN_AREA]?"
Listen carefully. Acknowledge their answer before asking the next question.
- If qualified (has the pain + authority + interest) → transition to PITCH
- If not qualified → transition to REJECTED politely
- If budget concern → transition to OBJECTION
""",
    CallState.PITCH: """
CURRENT STEP: VALUE PITCH
Your ONLY job: connect their specific pain (from qualification) to your solution.
Use their own words back at them: "You mentioned [PAIN_POINT] — that's exactly why we exist."
Then deliver ONE focused value statement with a specific outcome: "[BENEFIT] — most clients see [METRIC] within [TIMEFRAME]."
Then bridge to close: "Does that sound like the kind of result that would matter to you?"
- If positive response → transition to CLOSE
- If objection → transition to OBJECTION
- If question → answer briefly, stay in PITCH, then bridge to close
""",
    CallState.OBJECTION: """
CURRENT STEP: OBJECTION HANDLING
An objection was raised. Use ONLY the scripted responses from your training data.
DO NOT improvise. DO NOT get defensive.
Pattern: Acknowledge (1 sentence) → Reframe (1 sentence) → Bridge question (1 sentence).
After handling, return to PITCH or CLOSE depending on their response.
If this is the 3rd+ objection → it's time to close anyway or let them go gracefully.
""",
    CallState.CLOSE: """
CURRENT STEP: CLOSING
Your ONLY job: ask for the specific next step. Be direct and offer two options.
Script: "Based on what you've shared — it sounds like [NEXT_STEP] makes sense. Are you available [TIME_OPTION_A] or [TIME_OPTION_B]?"
- If YES → transition to FOLLOWUP
- If soft no / stall → one more push: "Even just 20 minutes — if it doesn't make sense I'll be the first to tell you."
- If hard no → transition to REJECTED gracefully
""",
    CallState.FOLLOWUP: """
CURRENT STEP: CONFIRMING NEXT STEPS
Prospect agreed. Now lock in the details clearly.
Confirm: time, date, email for calendar invite.
Script: "Perfect. I'll send a calendar invite to [EMAIL]. Our team will reach out the day before to confirm. Is there anything specific you'd like us to prepare?"
Then transition to ENDED.
""",
    CallState.REJECTED: """
CURRENT STEP: GRACEFUL EXIT
The prospect said no. End professionally — they may be a future prospect.
Script options:
- "No problem at all. I'll make a note not to bother you again. Have a great day!"
- "Totally understand. If anything changes with [PAIN_AREA], I'm here. Best of luck!"
Do NOT try to pitch again. Transition to ENDED immediately after speaking.
""",
    CallState.VOICEMAIL: """
CURRENT STEP: VOICEMAIL MESSAGE
You reached voicemail. Leave a SHORT, compelling message (max 25 seconds).
Script: "Hi [PROSPECT_NAME], this is [AGENT_NAME] from [COMPANY]. I'm calling about [ONE_LINE_VALUE]. I'll try you again — or you can reach me at [CALLBACK_NUMBER]. Have a great day."
Then transition to ENDED.
""",
}


# ── Main prompt builder ───────────────────────────────────────────────────────

class PromptEngine:
    """Builds the full system prompt for each LLM call."""

    # Function definitions the LLM can call
    FUNCTIONS = [
        {
            "name": "transition_state",
            "description": "Move the conversation to the next logical state",
            "parameters": {
                "type": "object",
                "properties": {
                    "new_state": {
                        "type": "string",
                        "enum": [s.value for s in CallState],
                        "description": "Target state to transition to"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for the transition"
                    }
                },
                "required": ["new_state"]
            }
        },
        {
            "name": "extract_fact",
            "description": "Store a fact about the prospect discovered during conversation",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": ["prospect_name", "company_name", "role", "pain_point",
                                 "interest_level", "agreed_next_step", "best_callback_time"],
                        "description": "Fact category"
                    },
                    "value": {"type": "string", "description": "The extracted value"}
                },
                "required": ["key", "value"]
            }
        },
        {
            "name": "end_call",
            "description": "Signal that the call should be terminated",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": ["converted", "rejected", "voicemail", "callback_scheduled"],
                        "description": "Final call outcome"
                    }
                },
                "required": ["outcome"]
            }
        }
    ]

    def __init__(self, persona_file: str = "template.yaml"):
        self._persona = _load_persona(persona_file)
        self._objections = _load_objections()
        self._build_objection_text()

    def _build_objection_text(self):
        """Pre-build objection response text for injection into prompt."""
        lines = []
        for obj in self._objections.get("objections", []):
            triggers = " / ".join(f'"{t}"' for t in obj.get("triggers", [])[:3])
            response = obj.get("response", "")
            lines.append(f'OBJECTION ({triggers}):\n→ "{response}"')
        self._objection_text = "\n\n".join(lines)

    def _fill_template(self, text: str) -> str:
        """Replace template placeholders with persona values."""
        p = self._persona
        agent = p.get("agent", {})
        product = p.get("product", {})
        target = p.get("target", {})
        call_cfg = p.get("call", {})
        compliance = p.get("compliance", {})

        benefits = product.get("key_benefits", [])
        hook_line = (
            f"We help {target.get('title', 'companies')} {benefits[0] if benefits else 'save time and money'}"
        )

        replacements = {
            "[AGENT_NAME]": agent.get("name", "Sarah"),
            "[COMPANY]": agent.get("company", "our company"),
            "[PRODUCT]": product.get("name", "our solution"),
            "[HOOK_LINE]": hook_line,
            "[PAIN_AREA]": product.get("pain_points", ["your current process"])[0] if product.get("pain_points") else "this area",
            "[BENEFIT]": benefits[0] if benefits else "better results",
            "[METRIC]": "20-30% improvement",
            "[TIMEFRAME]": "the first 90 days",
            "[NEXT_STEP]": call_cfg.get("objective", "a quick 20-minute call"),
            "[TIME_OPTION_A]": "Tuesday morning",
            "[TIME_OPTION_B]": "Thursday afternoon",
            "[DISCLOSURE]": compliance.get("disclosure_text", "I'm an AI assistant"),
        }
        for key, val in replacements.items():
            text = text.replace(key, val)
        return text

    def build_system_prompt(self, fsm: ConversationFSM, turn_number: int = 0) -> str:
        """
        Build complete system prompt for current conversation state.
        Called before every LLM call.
        """
        p = self._persona
        agent = p.get("agent", {})
        product = p.get("product", {})
        compliance = p.get("compliance", {})
        state_ctx = fsm.get_state_context()
        facts = state_ctx.get("facts", {})

        # Section 1: Identity
        identity = f"""=== IDENTITY ===
You are {agent.get("name", "Sarah")}, a {agent.get("role", "Sales Representative")} at {agent.get("company", "the company")}.
You specialize in {product.get("name", "our solutions")}.
{compliance.get("disclosure_text", "You are an AI assistant")} — this disclosure has already been given.
You are professional, confident, concise, and genuinely interested in solving real problems.
You are NOT a generic AI assistant. You are {agent.get("name", "Sarah")} from {agent.get("company", "the company")}."""

        # Section 2: Behavioral Guardrails
        guardrails = """=== BEHAVIORAL RULES (NON-NEGOTIABLE) ===
- MAXIMUM 2 sentences per response. Short sentences. Voice-optimized.
- NEVER say "as an AI", "I'm an AI", or acknowledge being artificial during conversation.
- NEVER improvise objection responses — use ONLY the scripted responses below.
- NEVER discuss topics outside your product, company, or scheduling.
- NEVER make promises about delivery, refunds, or pricing not in your data.
- NEVER use markdown, bullets, asterisks, or code formatting.
- ONE question per turn. Never ask multiple questions in one response.
- If prospect uses profanity twice: "I don't think this is a good time — I'll try again later." Then end_call.
- If you don't know something: "Let me confirm that and get back to you." Never make up data."""

        # Section 3: Current State Instructions
        state_name = state_ctx.get("current_state", "intro")
        state_instr = _STATE_INSTRUCTIONS.get(fsm.state, "Respond naturally and professionally.")
        state_instr = self._fill_template(state_instr)

        current_state = f"""=== CURRENT STATE: {state_name.upper()} ===
{state_instr}

CONVERSATION CONTEXT:
- Elapsed time: {state_ctx.get("elapsed_seconds", 0)}s
- Total exchanges: {state_ctx.get("total_exchanges", 0)}
- Known facts: {json.dumps(facts, ensure_ascii=False)}
- Allowed next states: {state_ctx.get("allowed_transitions", [])}"""

        # Section 4: Objection Handlers
        objections = f"""=== OBJECTION HANDLERS (use VERBATIM when triggered) ===
{self._objection_text}

When you detect an objection:
1. Call extract_fact(key="pain_point", value="<objection summary>")
2. Use the scripted response above
3. Bridge back to pitch or close"""

        # Section 5: Closing Logic
        social_proof = p.get("social_proof", [])
        sp_text = "\n".join(f"- {sp}" for sp in social_proof[:3])

        closing = f"""=== CLOSING & SOCIAL PROOF ===
Social proof you can use:
{sp_text if sp_text else "- We work with leading companies in this space"}

CRITICAL REMINDER: Every response must move the call forward toward the close.
You are here to get a meeting / next step — not to chat indefinitely.
After 15+ exchanges with no progress → escalate to close or exit gracefully."""

        # Turn injection (every 5 turns to prevent LLM drift)
        reminder = ""
        if turn_number > 0 and turn_number % 5 == 0:
            reminder = f"\n\n[SYSTEM REMINDER: You are {agent.get('name', 'Sarah')} from {agent.get('company', 'the company')}. Stay on script. Max 2 sentences.]"

        sections = [identity, guardrails, current_state, objections, closing]
        if reminder:
            sections.append(reminder)

        return "\n\n".join(sections)

    def build_interrupted_message(self, partial_text: str, prospect_response: str) -> str:
        """
        When barge-in occurs, inject context so LLM knows it was interrupted.
        Research finding: must tell LLM what was said before interrupt.
        """
        return (
            f"[INTERRUPTED: You were saying \"{partial_text[:100]}...\" when the prospect spoke. "
            f"They did NOT hear the rest of your message. Respond to: \"{prospect_response}\"]"
        )

    @property
    def agent_name(self) -> str:
        return self._persona.get("agent", {}).get("name", "Sarah")

    @property
    def tts_voice(self) -> str:
        return self._persona.get("tts", {}).get("voice", "anastasia")
