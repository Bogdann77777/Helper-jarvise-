"""
Kyutai STT Adapter for Sales Agent.

Wraps kyutai/stt-2.6b-en (or multilingual variant) for use in the sales pipeline.
- Input:  float32, 16kHz (from browser/phone)
- Kyutai: expects 24kHz → we resample internally
- Output: {"text", "confidence", "duration"}

Why better than Whisper for phone calls:
  - Streaming capable: processes audio chunks, not full utterance
  - Lower latency: ~200ms vs ~880ms (Whisper + silence wait)
  - Designed for real-time conversation, not transcription

Model: kyutai/stt-2.6b-en       (English, current)
       kyutai/stt-2.6b-multilingual (if exists — check research)
"""
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("sales.stt_kyutai")

# Kyutai expects 24kHz, we receive 16kHz from browser/phone
_INPUT_SR  = 16000
_KYUTAI_SR = 24000

# Default model — English. Override via KYUTAI_STT_MODEL env var.
import os
_DEFAULT_MODEL = os.getenv("KYUTAI_STT_MODEL", "kyutai/stt-2.6b-en")


def _resample_16k_to_24k(audio_16k: np.ndarray) -> np.ndarray:
    """Resample float32 audio from 16kHz to 24kHz (linear interpolation)."""
    target_len = int(len(audio_16k) * _KYUTAI_SR / _INPUT_SR)
    indices = np.linspace(0, len(audio_16k) - 1, target_len)
    return np.interp(indices, np.arange(len(audio_16k)), audio_16k).astype(np.float32)


class KyutaiSTTAdapter:
    """
    Kyutai STT wrapped for the sales agent pipeline.
    Compatible interface with WhisperSTT (same transcribe_sync / transcribe methods).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str = "cuda:0"):
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def load(self):
        """Load model onto GPU. Call once at startup."""
        from transformers import (
            KyutaiSpeechToTextProcessor,
            KyutaiSpeechToTextForConditionalGeneration,
        )
        logger.info(f"[KYUTAI STT] Loading {self._model_name} on {self._device}...")
        t0 = time.time()
        self._processor = KyutaiSpeechToTextProcessor.from_pretrained(self._model_name)
        self._model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            self._model_name,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        logger.info(f"[KYUTAI STT] Loaded in {time.time()-t0:.1f}s")

    def transcribe_sync(self, audio_16k: np.ndarray) -> dict:
        """
        Synchronous transcription. Called from asyncio.to_thread().
        audio_16k: float32, 16kHz, mono numpy array
        """
        if self._model is None:
            raise RuntimeError("Kyutai STT not loaded — call load() first")

        duration = len(audio_16k) / _INPUT_SR

        # Resample 16kHz → 24kHz
        audio_24k = _resample_16k_to_24k(audio_16k)

        import torch.nn.functional as F

        inputs = self._processor(
            audio_24k,
            sampling_rate=_KYUTAI_SR,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}

        t0 = time.time()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=True,
            )
        inference_ms = int((time.time() - t0) * 1000)

        # Confidence from token probabilities
        confidence = 0.0
        if out.scores:
            probs = [F.softmax(s.float(), dim=-1).max().item() for s in out.scores]
            confidence = sum(probs) / len(probs)

        text = self._processor.batch_decode(
            out.sequences, skip_special_tokens=True
        )[0].strip()

        logger.info(f"[KYUTAI STT] {inference_ms}ms | dur={duration:.1f}s | conf={confidence:.2f} | '{text[:80]}'")

        return {
            "text":       text,
            "confidence": confidence,
            "duration":   duration,
            "latency_ms": inference_ms,
        }

    async def transcribe(self, audio_16k: np.ndarray) -> dict:
        """Async wrapper for use in async pipeline."""
        return await asyncio.to_thread(self.transcribe_sync, audio_16k)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# ── Singleton ─────────────────────────────────────────────────────────────────
_instance: KyutaiSTTAdapter | None = None


def get_kyutai_stt(device: str = "cuda:0") -> KyutaiSTTAdapter:
    global _instance
    if _instance is None:
        _instance = KyutaiSTTAdapter(device=device)
    return _instance
