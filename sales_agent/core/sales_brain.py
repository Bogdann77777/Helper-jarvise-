"""
Sales Brain — LLM core for the voice sales agent.

Research-informed settings:
  - Model: Gemini 2.5 Flash (TTFT ~0.56s, best speed/quality balance)
  - Temperature: 0.4 (not 0.7 — script adherence requires low temp)
  - max_tokens: 80 (voice = short. 80 tokens ≈ 60 words ≈ 2 sentences)
  - Streaming: sentence-by-sentence → TTS starts while LLM still generates
  - Fallback: Claude Haiku if Gemini fails (>2.5s or error)
  - Turn injection: system reminder every 5 turns

Architecture:
  Brain receives: prospect utterance + FSM state + barge-in flag
  Brain outputs: sentence stream → TTS queue
  Brain calls: FSM transition functions via function calling
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import AsyncGenerator, Callable

import httpx

from .fsm import CallState, ConversationFSM
from .prompts import PromptEngine
from .text_normalizer import get_normalizer

logger = logging.getLogger("sales.brain")

# ── Model Configuration ───────────────────────────────────────────────────────
_PRIMARY_MODEL  = "google/gemini-2.5-flash"
_FALLBACK_MODEL = "anthropic/claude-haiku-4-5"
_TEMPERATURE    = 0.4   # Low: script adherence. Research: NOT 0.7
_MAX_TOKENS     = 80    # ~60 words = 2 sentences for voice
_LLM_TIMEOUT    = 2.5   # seconds before fallback
_MAX_HISTORY    = 12    # keep last N exchanges (rolling window)

# Pre-scripted filler while LLM thinks (if TTFT > 1.5s)
_FILLERS = [
    "Let me think about that for a moment.",
    "That's a good point.",
    "I appreciate you sharing that.",
]
_filler_idx = 0


class SalesBrain:
    """
    Manages LLM conversation for a single call.
    One instance per CallSession.
    """

    def __init__(
        self,
        openrouter_api_key: str,
        prompt_engine: PromptEngine,
        fsm: ConversationFSM,
        on_state_change: Callable[[CallState, str], None] | None = None,
        on_fact_extracted: Callable[[str, str], None] | None = None,
    ):
        self._api_key = openrouter_api_key
        self._prompt = prompt_engine
        self._fsm = fsm
        self._on_state_change = on_state_change
        self._on_fact_extracted = on_fact_extracted
        self._history: list[dict] = []
        self._turn_number = 0
        self._normalizer = get_normalizer()

    # ── Public API ────────────────────────────────────────────────────────────

    async def respond(
        self,
        prospect_text: str,
        was_interrupted: bool = False,
        interrupted_partial: str = "",
    ) -> AsyncGenerator[str, None]:
        """
        Process prospect utterance, yield normalized sentences for TTS.
        Handles barge-in context, function calls, and state transitions.
        """
        self._turn_number += 1
        self._fsm.record_exchange()

        # Build user message (with barge-in context if needed)
        if was_interrupted and interrupted_partial:
            user_msg = self._prompt.build_interrupted_message(interrupted_partial, prospect_text)
        else:
            user_msg = prospect_text

        self._history.append({"role": "user", "content": user_msg})
        self._trim_history()

        # Build system prompt (state-aware, with periodic reminder)
        system_prompt = self._prompt.build_system_prompt(self._fsm, self._turn_number)

        # Stream LLM response
        t0 = time.time()
        full_response = ""
        filler_sent = False

        try:
            async for sentence in self._stream_llm(system_prompt, self._history):
                elapsed = time.time() - t0

                # If LLM is slow on first response, send filler
                if not filler_sent and elapsed > 1.5 and not full_response:
                    filler = self._get_filler()
                    yield filler
                    filler_sent = True

                normalized = self._normalizer.normalize(sentence)
                if normalized:
                    full_response += sentence + " "
                    yield normalized

        except asyncio.TimeoutError:
            logger.warning(f"[BRAIN] LLM timeout after {time.time()-t0:.1f}s — trying fallback")
            async for sentence in self._stream_llm(system_prompt, self._history, use_fallback=True):
                normalized = self._normalizer.normalize(sentence)
                if normalized:
                    full_response += sentence + " "
                    yield normalized

        except Exception as e:
            logger.error(f"[BRAIN] LLM error: {e}")
            yield "Let me look into that — can you give me just a moment?"
            return

        # Store assistant response in history
        if full_response:
            self._history.append({"role": "assistant", "content": full_response.strip()})

        logger.info(f"[BRAIN] Turn {self._turn_number}: {time.time()-t0:.2f}s | state={self._fsm.state.value}")

    async def get_opening(self) -> AsyncGenerator[str, None]:
        """Generate the opening line for the call."""
        agent_name = self._prompt.agent_name
        # Transition to INTRO
        self._fsm.transition(CallState.INTRO, "call_answered")

        system_prompt = self._prompt.build_system_prompt(self._fsm, 0)
        opening_request = [{"role": "user", "content": "[CALL JUST CONNECTED — the prospect answered. Begin with the intro script.]"}]

        async for sentence in self._stream_llm(system_prompt, opening_request):
            normalized = self._normalizer.normalize(sentence)
            if normalized:
                yield normalized

    # ── Private: LLM Streaming ────────────────────────────────────────────────

    async def _stream_llm(
        self,
        system: str,
        messages: list[dict],
        use_fallback: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Stream from OpenRouter. Yields complete sentences as they arrive.
        Handles function calls (state transitions, fact extraction).
        """
        model = _FALLBACK_MODEL if use_fallback else _PRIMARY_MODEL

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "temperature": _TEMPERATURE,
            "max_tokens": _MAX_TOKENS,
            "stream": True,
            "tools": self._prompt.FUNCTIONS,
            "tool_choice": "auto",
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ceo-voice-agent",
            "X-Title": "CEO Voice Sales Agent",
        }

        buffer = ""
        tool_calls_buffer: list[dict] = []
        in_tool_call = False

        timeout = httpx.Timeout(timeout=_LLM_TIMEOUT + 1.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    # Handle text content
                    content = delta.get("content")
                    if content:
                        buffer += content
                        # Yield complete sentences as they arrive
                        sentences, buffer = self._extract_sentences(buffer)
                        for s in sentences:
                            if s.strip():
                                yield s.strip()

                    # Handle tool calls (function calling)
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        in_tool_call = True
                        for tc in tool_calls:
                            idx = tc.get("index", 0)
                            while len(tool_calls_buffer) <= idx:
                                tool_calls_buffer.append({"id": "", "name": "", "arguments": ""})
                            if tc.get("id"):
                                tool_calls_buffer[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_calls_buffer[idx]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                tool_calls_buffer[idx]["arguments"] += tc["function"]["arguments"]

                # Yield any remaining buffer
                if buffer.strip():
                    yield buffer.strip()

        # Process any collected tool calls
        for tc in tool_calls_buffer:
            if tc["name"]:
                await self._handle_tool_call(tc["name"], tc["arguments"])

    def _extract_sentences(self, text: str) -> tuple[list[str], str]:
        """Extract complete sentences, return (sentences, remaining_buffer)."""
        sentences = []
        # Split on sentence-ending punctuation followed by space or end
        pattern = re.compile(r'(?<=[.!?])\s+')
        parts = pattern.split(text)
        if len(parts) > 1:
            # Last part is incomplete (no terminating punctuation yet)
            sentences = [p for p in parts[:-1] if p.strip()]
            remaining = parts[-1]
        else:
            remaining = text
        return sentences, remaining

    async def _handle_tool_call(self, name: str, arguments_str: str):
        """Process LLM function calls (state transitions, fact extraction)."""
        try:
            args = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError:
            logger.warning(f"[BRAIN] Invalid tool call args: {arguments_str!r}")
            return

        if name == "transition_state":
            new_state_str = args.get("new_state", "")
            reason = args.get("reason", "")
            try:
                new_state = CallState(new_state_str)
                success = self._fsm.transition(new_state, reason)
                if success and self._on_state_change:
                    self._on_state_change(new_state, reason)
            except ValueError:
                logger.warning(f"[BRAIN] Unknown state: {new_state_str}")

        elif name == "extract_fact":
            key = args.get("key", "")
            value = args.get("value", "")
            if key and value:
                self._fsm.update_fact(key, value)
                if self._on_fact_extracted:
                    self._on_fact_extracted(key, value)

        elif name == "end_call":
            outcome = args.get("outcome", "unknown")
            logger.info(f"[BRAIN] end_call called: outcome={outcome}")
            self._fsm.transition(CallState.ENDED, f"agent_ended:{outcome}")

    def _trim_history(self):
        """Keep only last N exchanges to control context size."""
        if len(self._history) > _MAX_HISTORY:
            # Keep system messages and last N exchanges
            self._history = self._history[-_MAX_HISTORY:]

    def _get_filler(self) -> str:
        global _filler_idx
        filler = _FILLERS[_filler_idx % len(_FILLERS)]
        _filler_idx += 1
        return filler

    def get_conversation_transcript(self) -> list[dict]:
        """Return full conversation history for recording."""
        return list(self._history)
