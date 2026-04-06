"""
Inference Client — used by PC1 apps to call PC2's STT/TTS API.

Usage in any app:
    from inference_server.client import get_inference_client
    client = get_inference_client()

    # STT
    text, confidence = await client.transcribe(audio_float32)

    # TTS
    wav_bytes, sample_rate = await client.synthesize("Hello there", voice="anastasia")

Set in .env on PC1:
    INFERENCE_SERVER=http://192.168.1.XXX:8010

Falls back to local models if server unavailable.
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from typing import Optional

import httpx
import numpy as np
import soundfile as sf

logger = logging.getLogger("inference_client")

_DEFAULT_URL = "http://localhost:8010"
_TIMEOUT = 10.0   # seconds


class InferenceClient:
    """HTTP client for the remote Inference Server on PC2."""

    def __init__(self, server_url: str = ""):
        self._url = (server_url or os.getenv("INFERENCE_SERVER", _DEFAULT_URL)).rstrip("/")
        self._available: Optional[bool] = None
        logger.info(f"[INFERENCE] Server: {self._url}")

    async def check_health(self) -> bool:
        """Check if server is up and models are ready."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._url}/health")
                data = resp.json()
                self._available = data.get("status") == "ready"
                return self._available
        except Exception:
            self._available = False
            return False

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "en",
    ) -> tuple[str, Optional[float]]:
        """
        Send audio to PC2 for transcription.
        Returns (text, confidence).
        """
        # Encode audio as base64 WAV
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._url}/stt",
                    json={
                        "audio_b64": audio_b64,
                        "sample_rate": sample_rate,
                        "language": language,
                    }
                )
                resp.raise_for_status()
                data = resp.json()
                logger.debug(f"[INFERENCE] STT: '{data['text'][:60]}' ({data['latency_ms']}ms)")
                return data["text"], data.get("confidence")
        except Exception as e:
            logger.error(f"[INFERENCE] STT failed: {e}")
            return "", None

    async def synthesize(
        self,
        text: str,
        voice: str = "anastasia",
        language: str = "en",
    ) -> tuple[Optional[bytes], int]:
        """
        Send text to PC2 for synthesis.
        Returns (wav_bytes, sample_rate).
        """
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._url}/tts",
                    json={"text": text, "voice": voice, "language": language}
                )
                resp.raise_for_status()
                data = resp.json()
                wav_bytes = base64.b64decode(data["audio_b64"])
                logger.debug(f"[INFERENCE] TTS: {data['latency_ms']}ms, {data['duration_seconds']:.1f}s audio")
                return wav_bytes, data["sample_rate"]
        except Exception as e:
            logger.error(f"[INFERENCE] TTS failed: {e}")
            return None, 22050

    async def synthesize_to_array(
        self,
        text: str,
        voice: str = "anastasia",
    ) -> tuple[Optional[np.ndarray], int]:
        """Synthesize and return as numpy array (for pipeline compatibility)."""
        wav_bytes, sr = await self.synthesize(text, voice)
        if wav_bytes is None:
            return None, sr
        buf = io.BytesIO(wav_bytes)
        audio, sample_rate = sf.read(buf, dtype="float32")
        return audio, sample_rate

    async def list_speakers(self) -> list[dict]:
        """Get available voice clones from PC2."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._url}/speakers")
                return resp.json().get("speakers", [])
        except Exception:
            return []

    @property
    def is_remote(self) -> bool:
        """True if using remote PC2, False if localhost."""
        return "localhost" not in self._url and "127.0.0.1" not in self._url


# ── Singleton ─────────────────────────────────────────────────────────────────
_client: InferenceClient | None = None


def get_inference_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient()
    return _client
