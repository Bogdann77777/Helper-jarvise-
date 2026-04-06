"""
Barge-In Controller — handles prospect interruptions.

Research finding (critical):
  - Must flush TTS in <100ms when prospect speaks
  - Must send [INTERRUPTED: partial_text] to LLM
  - Standard 2026: sub-100ms shutdown of agent audio

Architecture:
  - VAD sends SpeechStarted event
  - BargeInController captures partial TTS text
  - Signals audio_out to stop immediately
  - Passes interrupted context to SalesBrain
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger("sales.barge_in")


@dataclass
class BargeInEvent:
    was_interrupted: bool = False
    partial_agent_text: str = ""
    interrupted_at_ms: float = 0.0


class BargeInController:
    """
    Coordinates between VAD (speech detection) and audio output (TTS playback).

    Usage:
        controller = BargeInController(on_interrupt=my_handler)
        controller.register_tts_tracker(tracker)  # receives current TTS text

        # When VAD detects speech:
        await controller.on_speech_detected()  # triggers interrupt

        # Check before responding:
        event = controller.consume_event()
    """

    def __init__(self, on_interrupt: Callable[[], None] | None = None):
        self._on_interrupt = on_interrupt
        self._current_tts_text = ""
        self._tts_sent_chars = 0
        self._is_agent_speaking = False
        self._interrupt_event: BargeInEvent | None = None
        self._stop_signal: asyncio.Event = asyncio.Event()

    # ── TTS tracking ──────────────────────────────────────────────────────────

    def agent_started_speaking(self, text: str):
        """Called when TTS starts playing a sentence."""
        self._is_agent_speaking = True
        self._current_tts_text = text
        self._stop_signal.clear()
        logger.debug(f"[BARGE_IN] Agent speaking: {text[:50]!r}")

    def agent_finished_speaking(self):
        """Called when TTS finishes entire response."""
        self._is_agent_speaking = False
        self._current_tts_text = ""
        self._tts_sent_chars = 0

    def update_sent_chars(self, chars: int):
        """Track how much of the TTS text has been sent to audio output."""
        self._tts_sent_chars = chars

    # ── Interrupt handling ────────────────────────────────────────────────────

    async def on_speech_detected(self) -> bool:
        """
        Called by VAD when prospect starts speaking.
        Returns True if agent was interrupted (was speaking).
        This is the critical path — must complete in <50ms.
        """
        if not self._is_agent_speaking:
            return False

        # Capture what was being said at interruption point
        partial = self._current_tts_text
        if self._tts_sent_chars > 0 and len(partial) > self._tts_sent_chars:
            partial = partial[:self._tts_sent_chars]

        self._interrupt_event = BargeInEvent(
            was_interrupted=True,
            partial_agent_text=partial,
        )

        # Signal audio output to stop (checked in audio queue)
        self._stop_signal.set()
        self._is_agent_speaking = False

        # Call external handler (e.g., flush WebSocket audio queue)
        if self._on_interrupt:
            try:
                if asyncio.iscoroutinefunction(self._on_interrupt):
                    await self._on_interrupt()
                else:
                    self._on_interrupt()
            except Exception as e:
                logger.warning(f"[BARGE_IN] Interrupt handler error: {e}")

        logger.info(f"[BARGE_IN] Interrupted! Partial: {partial[:60]!r}")
        return True

    def consume_event(self) -> BargeInEvent:
        """
        Consume the interrupt event after STT completes.
        Returns event with was_interrupted=False if no interrupt occurred.
        """
        event = self._interrupt_event or BargeInEvent(was_interrupted=False)
        self._interrupt_event = None
        return event

    @property
    def stop_signal(self) -> asyncio.Event:
        """Audio output checks this to stop playback immediately."""
        return self._stop_signal

    @property
    def is_agent_speaking(self) -> bool:
        return self._is_agent_speaking
