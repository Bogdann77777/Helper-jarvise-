"""
Call Recorder — stores WAV audio + JSON transcript for every call.

Storage: outputs/calls/{YYYY-MM-DD}/{call_id}/
  - audio.wav         ← combined agent audio (for QA)
  - transcript.json   ← full turn-by-turn log
  - metadata.json     ← outcome, duration, facts, scores
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("sales.recorder")

_OUTPUT_BASE = Path(__file__).parent.parent.parent / "outputs" / "calls"


class CallRecorder:
    """Records transcript and agent audio for one call."""

    def __init__(self):
        self._call_id = str(uuid.uuid4())[:8]
        self._date_str = datetime.now().strftime("%Y-%m-%d")
        self._dir = _OUTPUT_BASE / self._date_str / self._call_id
        self._dir.mkdir(parents=True, exist_ok=True)

        self._transcript: list[dict] = []
        self._agent_audio_chunks: list[np.ndarray] = []
        self._agent_audio_sr: int = 22050
        self._call_start = datetime.now()

        logger.info(f"[RECORDER] Call {self._call_id} → {self._dir}")

    # ── Recording ─────────────────────────────────────────────────────────────

    def log_turn(
        self,
        speaker: str,          # "agent" | "prospect"
        text: str,
        state: str = "",
        confidence: float | None = None,
    ):
        """Log a conversation turn."""
        self._transcript.append({
            "timestamp": (datetime.now() - self._call_start).total_seconds(),
            "speaker": speaker,
            "text": text,
            "state": state,
            "confidence": confidence,
        })

    def add_agent_audio(self, audio: np.ndarray, sample_rate: int):
        """Add an audio chunk from TTS output."""
        if audio is not None and len(audio) > 0:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            self._agent_audio_chunks.append(audio)
            self._agent_audio_sr = sample_rate

    def finalize(
        self,
        outcome: str,
        facts: dict | None = None,
        fsm_history: str = "",
    ) -> dict:
        """
        Save all recordings to disk. Returns metadata dict.
        Call this when the call ends.
        """
        duration = (datetime.now() - self._call_start).total_seconds()

        # Save transcript
        transcript_path = self._dir / "transcript.json"
        transcript_path.write_text(
            json.dumps(self._transcript, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Save agent audio
        if self._agent_audio_chunks:
            try:
                combined = np.concatenate(self._agent_audio_chunks)
                audio_path = self._dir / "agent_audio.wav"
                sf.write(str(audio_path), combined, self._agent_audio_sr)
            except Exception as e:
                logger.warning(f"[RECORDER] Audio save error: {e}")

        # Build metadata
        turns = len(self._transcript)
        agent_turns = sum(1 for t in self._transcript if t["speaker"] == "agent")
        prospect_turns = sum(1 for t in self._transcript if t["speaker"] == "prospect")

        agent_words = sum(len(t["text"].split()) for t in self._transcript if t["speaker"] == "agent")
        prospect_words = sum(len(t["text"].split()) for t in self._transcript if t["speaker"] == "prospect")
        total_words = agent_words + prospect_words
        talk_ratio = round(agent_words / total_words, 2) if total_words > 0 else 0.5

        metadata = {
            "call_id": self._call_id,
            "date": self._date_str,
            "started_at": self._call_start.isoformat(),
            "duration_seconds": round(duration),
            "outcome": outcome,
            "total_turns": turns,
            "agent_turns": agent_turns,
            "prospect_turns": prospect_turns,
            "talk_ratio": talk_ratio,   # target: < 0.4 (agent should talk less)
            "fsm_history": fsm_history,
            "facts": facts or {},
            "transcript_path": str(transcript_path),
        }

        metadata_path = self._dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"[RECORDER] Saved call {self._call_id}: outcome={outcome}, "
            f"duration={duration:.0f}s, turns={turns}, talk_ratio={talk_ratio}"
        )
        return metadata

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def call_id(self) -> str:
        return self._call_id

    @property
    def transcript(self) -> list[dict]:
        return list(self._transcript)
