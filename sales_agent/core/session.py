"""
CallSession — ties all Phase 1 components together.

One instance per active call. Manages:
  - ConversationFSM (state machine)
  - SalesBrain (LLM)
  - PromptEngine (system prompts)
  - BargeInController (interruption)
  - CallRecorder (transcript + audio)
  - Audio input chain (VAD + STT)
  - Audio output chain (TTS + queue)

Lifecycle:
  session = CallSession(config)
  await session.start(ws_send)      # sends audio to WebSocket
  await session.feed_audio(pcm)     # from browser/phone
  await session.end()               # finalize recording
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .barge_in import BargeInController
from .call_recorder import CallRecorder
from .fsm import CallState, ConversationFSM
from .prompts import PromptEngine
from .sales_brain import SalesBrain
from .text_normalizer import get_normalizer

logger = logging.getLogger("sales.session")

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_VAD_THRESHOLD  = 0.4
_VAD_MIN_SILENCE = 0.6
_VAD_MIN_SPEECH  = 0.25
_RMS_GATE       = 0.015


class CallSession:
    """Full call session — one per active phone call or browser demo."""

    def __init__(
        self,
        ws_send: Callable,
        openrouter_api_key: str,
        persona_file: str = "template.yaml",
    ):
        self._send = ws_send
        self._api_key = openrouter_api_key

        # Load config
        self._cfg = self._load_config()

        # Core components
        self._fsm = ConversationFSM()
        self._prompt = PromptEngine(persona_file=persona_file)
        self._barge_in = BargeInController(on_interrupt=self._on_interrupt)
        self._recorder = CallRecorder()
        self._brain = SalesBrain(
            openrouter_api_key=openrouter_api_key,
            prompt_engine=self._prompt,
            fsm=self._fsm,
            on_state_change=self._on_state_change,
            on_fact_extracted=self._on_fact_extracted,
        )
        self._normalizer = get_normalizer()

        # Audio components (loaded lazily)
        self._vad = None
        self._stt = None
        self._tts = None
        self._inference_client = None  # PC2 remote STT/TTS

        # Runtime state
        self._audio_out_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._audio_out_task: asyncio.Task | None = None
        self._processing = False
        self._current_tts_text = ""
        self._call_start = time.time()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self):
        """Initialize models and begin the call with opening line."""
        logger.info(f"[SESSION {self._recorder.call_id}] Starting")

        await self._load_models()

        # Start background workers
        self._worker_task = asyncio.create_task(self._pipeline_worker())
        self._audio_out_task = asyncio.create_task(self._audio_out_worker())

        # Send opening greeting
        await self._send({"type": "call_started", "call_id": self._recorder.call_id})
        await self._play_opening()

    async def _load_models(self):
        """Load STT/TTS — local GPU or remote PC2 inference server."""
        loop = asyncio.get_event_loop()

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        # Try PC2 inference server first
        try:
            from inference_server.client import get_inference_client
            client = get_inference_client()
            if await client.check_health():
                self._inference_client = client
                logger.info(f"[SESSION] Using PC2 inference server: {client._url}")
            else:
                logger.info("[SESSION] PC2 server not available — using local models")
        except Exception as e:
            logger.debug(f"[SESSION] Inference client unavailable: {e}")

        # Load VAD (Silero + DeepFilterNet) from translator project
        try:
            translator_path = str(Path(__file__).parent.parent.parent.parent / "translator")
            if translator_path not in sys.path:
                sys.path.insert(0, translator_path)
            from app.pipeline.vad import VAD
            self._vad = VAD(_VAD_THRESHOLD, _VAD_MIN_SILENCE, _VAD_MIN_SPEECH)
            await loop.run_in_executor(None, self._vad.load)
            logger.info("[SESSION] VAD loaded")
        except Exception as e:
            logger.warning(f"[SESSION] VAD load failed: {e} — using RMS gate only")
            self._vad = None

        # Load STT — Kyutai streaming (if no PC2 inference server)
        if not self._inference_client:
            from sales_agent.pipeline.stt_kyutai import get_kyutai_stt
            stt_device = self._cfg.get("stt", {}).get("device", "cuda:0")
            self._stt = get_kyutai_stt(device=stt_device)
            await loop.run_in_executor(None, self._stt.load)
            logger.info(f"[SESSION] Kyutai STT loaded on {stt_device}")

        # Load TTS (XTTS)
        from xtts_manager import get_xtts_manager
        self._tts = get_xtts_manager()
        await loop.run_in_executor(None, self._tts.load)
        logger.info("[SESSION] TTS loaded")

    # ── Audio Input ───────────────────────────────────────────────────────────

    async def feed_audio(self, pcm_bytes: bytes):
        """
        Receive raw PCM audio from browser (float32, 16kHz, mono).
        VAD detects speech, accumulates utterance, puts in queue.
        """
        if self._fsm.is_terminal():
            return

        audio_16k = np.frombuffer(pcm_bytes, dtype=np.float32).copy()

        # Check for barge-in first
        if self._barge_in.is_agent_speaking:
            rms = float(np.sqrt(np.mean(audio_16k ** 2)))
            if rms > _RMS_GATE * 2:
                # Prospect is speaking while agent is playing
                await self._barge_in.on_speech_detected()

        # VAD processing
        if self._vad is not None:
            try:
                speech_active, utterance = self._vad.process_chunk(audio_16k, _RMS_GATE)
                if speech_active:
                    await self._send({"type": "vad", "status": "speaking"})
                if utterance is not None:
                    await self._send({"type": "vad", "status": "silence"})
                    await self._utterance_queue.put(utterance)
            except Exception as e:
                logger.warning(f"[SESSION] VAD error: {e}")
        else:
            # Simple RMS-based detection
            rms = float(np.sqrt(np.mean(audio_16k ** 2)))
            if rms > _RMS_GATE:
                await self._send({"type": "vad", "status": "speaking"})

    # ── Pipeline Worker ───────────────────────────────────────────────────────

    async def _pipeline_worker(self):
        """Process utterances from queue through STT → Brain → TTS."""
        self._utterance_queue: asyncio.Queue = asyncio.Queue()
        while True:
            try:
                audio = await self._utterance_queue.get()
                if audio is None:
                    break
                await self._process_utterance(audio)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SESSION] Worker error: {e}", exc_info=True)

    async def _process_utterance(self, audio: np.ndarray):
        """STT → Brain → TTS pipeline for one prospect utterance."""
        if self._processing:
            logger.debug("[SESSION] Already processing, queuing")
        self._processing = True

        try:
            # ── STT (PC2 or local) ────────────────────────────────────────────
            t0 = time.time()
            if self._inference_client:
                stt_text, confidence = await self._inference_client.transcribe(audio)
                stt_result = {"text": stt_text, "confidence": confidence}
            else:
                stt_result = await asyncio.to_thread(self._stt.transcribe_sync, audio)
                stt_text = stt_result.get("text", "").strip()
            stt_ms = int((time.time() - t0) * 1000)

            if not stt_text or len(stt_text) < 2:
                logger.debug("[SESSION] Empty STT — skipping")
                return

            logger.info(f"[SESSION] STT ({stt_ms}ms): '{stt_text[:80]}'")
            await self._send({"type": "prospect_speech", "text": stt_text})

            # Record prospect turn
            self._recorder.log_turn(
                speaker="prospect",
                text=stt_text,
                state=self._fsm.state.value,
                confidence=stt_result.get("confidence"),
            )

            # ── Barge-in context ──────────────────────────────────────────────
            barge_event = self._barge_in.consume_event()

            # ── LLM + TTS streaming ───────────────────────────────────────────
            await self._clear_audio_queue()  # Clear any queued audio

            agent_text_parts = []
            async for sentence in self._brain.respond(
                prospect_text=stt_text,
                was_interrupted=barge_event.was_interrupted,
                interrupted_partial=barge_event.partial_agent_text,
            ):
                if sentence and sentence.strip():
                    agent_text_parts.append(sentence)
                    await self._send({"type": "agent_speech", "text": sentence})
                    await self._queue_tts(sentence)

            # Record agent turn
            if agent_text_parts:
                full_agent_text = " ".join(agent_text_parts)
                self._recorder.log_turn(
                    speaker="agent",
                    text=full_agent_text,
                    state=self._fsm.state.value,
                )

            # Check if call should end
            if self._fsm.is_terminal():
                await self._end_call()

        finally:
            self._processing = False

    # ── Audio Output ──────────────────────────────────────────────────────────

    async def _play_opening(self):
        """Generate and play the opening line."""
        parts = []
        async for sentence in self._brain.get_opening():
            if sentence.strip():
                parts.append(sentence)
                await self._send({"type": "agent_speech", "text": sentence})
                await self._queue_tts(sentence)

        if parts:
            self._recorder.log_turn(
                speaker="agent",
                text=" ".join(parts),
                state=self._fsm.state.value,
            )

    async def _queue_tts(self, text: str):
        """Add a sentence to the TTS output queue."""
        await self._audio_out_queue.put(("tts", text))

    async def _clear_audio_queue(self):
        """Clear pending audio (called on barge-in)."""
        while not self._audio_out_queue.empty():
            try:
                self._audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._barge_in.stop_signal.set()  # Signal any playing audio to stop

    async def _audio_out_worker(self):
        """Process TTS queue and send audio to WebSocket."""
        while True:
            try:
                item = await self._audio_out_queue.get()
                if item is None:
                    break

                kind, data = item
                if kind == "tts":
                    await self._synthesize_and_send(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SESSION] Audio out error: {e}")

    async def _synthesize_and_send(self, text: str):
        """Synthesize text to speech and send audio chunks via WebSocket."""
        if not text.strip():
            return

        voice = self._prompt.tts_voice
        self._barge_in.agent_started_speaking(text)

        try:
            # Check stop signal before starting
            if self._barge_in.stop_signal.is_set():
                self._barge_in.stop_signal.clear()
                return

            # TTS synthesis (PC2 or local)
            if self._inference_client:
                audio_data, sample_rate = await self._inference_client.synthesize_to_array(text, voice)
            else:
                audio_data, sample_rate = await asyncio.to_thread(
                    self._tts.synthesize_sync, text, voice
                )

            if audio_data is None:
                return

            # Record agent audio
            self._recorder.add_agent_audio(audio_data, sample_rate)

            # Send to WebSocket (in chunks for smooth playback)
            import io
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            # Check stop signal again before sending
            if not self._barge_in.stop_signal.is_set():
                audio_b64 = base64.b64encode(wav_bytes).decode()
                await self._send({
                    "type": "agent_audio",
                    "data": audio_b64,
                    "sample_rate": sample_rate,
                    "text": text,
                })

        except Exception as e:
            logger.error(f"[SESSION] TTS error for '{text[:40]}': {e}")
        finally:
            self._barge_in.agent_finished_speaking()
            self._barge_in.stop_signal.clear()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_state_change(self, new_state: CallState, reason: str):
        """Called by SalesBrain when FSM state changes."""
        asyncio.create_task(self._send({
            "type": "state_change",
            "state": new_state.value,
            "reason": reason,
        }))
        logger.info(f"[SESSION] State → {new_state.value} ({reason})")

    def _on_fact_extracted(self, key: str, value: str):
        """Called when LLM extracts a fact about the prospect."""
        asyncio.create_task(self._send({
            "type": "fact_extracted",
            "key": key,
            "value": value,
        }))

    async def _on_interrupt(self):
        """Called immediately when barge-in detected."""
        await self._clear_audio_queue()
        await self._send({"type": "barge_in", "status": "interrupted"})
        logger.info("[SESSION] Barge-in: audio cleared")

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def end(self):
        """Finalize the call — stop workers, save recordings."""
        logger.info(f"[SESSION {self._recorder.call_id}] Ending")

        # Stop workers
        if self._worker_task:
            self._worker_task.cancel()
        if self._audio_out_task:
            await self._audio_out_queue.put(None)

        await self._end_call()

    async def _end_call(self):
        """Save call data and notify client."""
        outcome = self._fsm.get_call_outcome()
        metadata = self._recorder.finalize(
            outcome=outcome,
            facts=vars(self._fsm.facts),
            fsm_history=self._fsm.get_history_summary(),
        )
        await self._send({
            "type": "call_ended",
            "outcome": outcome,
            "duration_seconds": metadata["duration_seconds"],
            "call_id": self._recorder.call_id,
        })
        logger.info(f"[SESSION] Ended. Outcome: {outcome}")

    @staticmethod
    def _load_config() -> dict:
        try:
            return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
