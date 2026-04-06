"""
Audio Codec — converts phone audio formats to/from our pipeline format.

Phone audio realities:
  - Telnyx streams OPUS 16kHz (recommended) or G.711 μ-law 8kHz
  - Our STT (Whisper) expects: PCM float32, 16kHz, mono
  - Our TTS (XTTS) outputs: PCM float32, 22050Hz, mono
  - Telnyx expects back: linear PCM 16-bit, 8kHz or 16kHz

This file handles all conversions so the rest of the codebase
works with clean numpy float32 arrays regardless of phone format.
"""
import audioop
import struct

import numpy as np


# ── μ-law (G.711) Conversion ─────────────────────────────────────────────────

def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Convert G.711 μ-law encoded bytes → linear PCM 16-bit."""
    return audioop.ulaw2lin(mulaw_bytes, 2)  # 2 = 16-bit output


def pcm16_to_mulaw(pcm16_bytes: bytes) -> bytes:
    """Convert linear PCM 16-bit → G.711 μ-law."""
    return audioop.lin2ulaw(pcm16_bytes, 2)


def mulaw_bytes_to_float32(mulaw_bytes: bytes) -> np.ndarray:
    """
    μ-law 8kHz bytes → float32 numpy array (still 8kHz after this).
    Call resample_8k_to_16k() after.
    """
    pcm16 = mulaw_to_pcm16(mulaw_bytes)
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    return samples / 32768.0  # normalize to [-1, 1]


# ── Resampling ────────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple resampling via linear interpolation. Good for voice."""
    if from_sr == to_sr:
        return audio
    target_len = int(len(audio) * to_sr / from_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def resample_8k_to_16k(audio_8k: np.ndarray) -> np.ndarray:
    return resample(audio_8k, 8000, 16000)


def resample_16k_to_8k(audio_16k: np.ndarray) -> np.ndarray:
    return resample(audio_16k, 16000, 8000)


def resample_22k_to_16k(audio_22k: np.ndarray) -> np.ndarray:
    return resample(audio_22k, 22050, 16000)


def resample_22k_to_8k(audio_22k: np.ndarray) -> np.ndarray:
    return resample(audio_22k, 22050, 8000)


# ── PCM float32 ↔ bytes ──────────────────────────────────────────────────────

def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """float32 [-1,1] → signed int16 bytes (for Telnyx output)."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Signed int16 bytes → float32 [-1,1]."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


# ── OPUS via WebSocket (Telnyx streams raw PCM in media events) ──────────────

def decode_telnyx_media(payload: bytes, codec: str = "PCMU") -> np.ndarray:
    """
    Decode audio from a Telnyx media WebSocket event.

    Telnyx sends: base64-encoded audio in `media` events
    Codec options:
      - "PCMU" → G.711 μ-law, 8kHz (default Telnyx)
      - "PCMA" → G.711 A-law, 8kHz
      - "opus" → OPUS, 8kHz or 16kHz (requires opuslib)

    Returns: float32 numpy array at 16kHz for our STT.
    """
    if codec == "PCMU":
        audio_8k = mulaw_bytes_to_float32(payload)
        return resample_8k_to_16k(audio_8k)

    elif codec == "PCMA":
        pcm16 = audioop.alaw2lin(payload, 2)
        audio_8k = pcm16_bytes_to_float32(pcm16)
        return resample_8k_to_16k(audio_8k)

    elif codec == "linear16":
        # Already PCM16 — just normalize
        return pcm16_bytes_to_float32(payload)

    else:
        raise ValueError(f"Unsupported codec: {codec}")


def encode_for_telnyx(audio_float32: np.ndarray, input_sr: int, codec: str = "PCMU") -> bytes:
    """
    Encode our TTS output for sending back to Telnyx.
    TTS outputs 22050Hz → resample → encode.
    """
    # Resample to 8kHz for μ-law
    audio_8k = resample(audio_float32, input_sr, 8000)

    if codec == "PCMU":
        pcm16 = float32_to_pcm16_bytes(audio_8k)
        return pcm16_to_mulaw(pcm16)

    elif codec == "linear16":
        audio_16k = resample(audio_float32, input_sr, 16000)
        return float32_to_pcm16_bytes(audio_16k)

    else:
        raise ValueError(f"Unsupported output codec: {codec}")
