"""
remote_ai_client.py — HTTP client for remote STT/TTS server at 192.168.1.243.

STT: POST http://192.168.1.243:8001/transcribe  (multipart audio)
TTS: POST http://192.168.1.243:8002/synthesize  (JSON → WAV bytes)

Drop-in replacement for local STTManager / Qwen3TTSManager.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
STT_URL = os.environ.get("STT_REMOTE_URL", "http://192.168.1.243:8001")
TTS_URL = os.environ.get("TTS_REMOTE_URL", "http://192.168.1.243:8002")
TIMEOUT_STT = 120   # 2 min — large audio file transcription
TIMEOUT_TTS = 60    # 1 min — TTS generation


# ── STT ───────────────────────────────────────────────────────────────────────

def remote_transcribe(audio_path: str | Path) -> dict:
    """
    Transcribe audio file via remote STT server.
    Returns: {"text": str, "language": str, "duration": float, "elapsed": float}
    """
    url = f"{STT_URL}/transcribe"
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"[remote_stt] Transcribing {audio_path.name} via {STT_URL}")
    with open(audio_path, "rb") as f:
        resp = requests.post(
            url,
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=TIMEOUT_STT,
        )
    resp.raise_for_status()
    result = resp.json()
    logger.info(f"[remote_stt] Done: {result.get('text','')[:80]}")
    return result


def remote_transcribe_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """Transcribe audio bytes via remote STT server."""
    url = f"{STT_URL}/transcribe"
    logger.info(f"[remote_stt] Transcribing {len(audio_bytes)//1024}KB via {STT_URL}")
    resp = requests.post(
        url,
        files={"file": (filename, audio_bytes, "audio/wav")},
        timeout=TIMEOUT_STT,
    )
    resp.raise_for_status()
    result = resp.json()
    logger.info(f"[remote_stt] Done: {result.get('text','')[:80]}")
    return result


# ── TTS ───────────────────────────────────────────────────────────────────────

def remote_synthesize(
    text: str,
    speaker: str = "Ryan",
    language: str = "Russian",
    instruct: str = "Speak in a warm, professional voice.",
    mode: str = "custom",
    save_to: Optional[str | Path] = None,
) -> dict:
    """
    Synthesize speech via remote TTS server.
    Returns: {"url": str, "file": str, "gen_sec": float, "audio_dur": float}
    """
    url = f"{TTS_URL}/synthesize"
    logger.info(f"[remote_tts] Synthesizing: speaker={speaker} lang={language} text={text[:60]}...")

    resp = requests.post(
        url,
        json={
            "text": text,
            "speaker": speaker,
            "language": language,
            "instruct": instruct,
            "mode": mode,
        },
        timeout=TIMEOUT_TTS,
    )
    resp.raise_for_status()

    wav_bytes = resp.content
    elapsed = float(resp.headers.get("X-Elapsed", "0"))

    # Save WAV file
    if save_to:
        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
        file_str = str(out_path)
        url_str = f"/static/audio/{out_path.name}"
    else:
        audio_dir = Path(__file__).parent / "static" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        fname = f"rtts_{uuid.uuid4().hex[:8]}.wav"
        out_path = audio_dir / fname
        out_path.write_bytes(wav_bytes)
        file_str = str(out_path)
        url_str = f"/static/audio/{fname}"

    # Estimate audio duration from file size (PCM 24kHz mono 16-bit ≈ 48000 bytes/sec)
    audio_dur = len(wav_bytes) / 48000

    logger.info(f"[remote_tts] Done: {len(wav_bytes)//1024}KB, ~{audio_dur:.1f}s audio, elapsed={elapsed}s")
    return {
        "url": url_str,
        "file": file_str,
        "gen_sec": elapsed,
        "audio_dur": round(audio_dur, 2),
        "rtf": round(elapsed / audio_dur, 3) if audio_dur > 0 else 0,
    }


def remote_synthesize_clone(
    text: str,
    ref_audio: str | Path,
    ref_text: str,
    language: str = "Russian",
    save_to: Optional[str | Path] = None,
) -> dict:
    """Zero-shot voice clone via remote TTS server."""
    import base64
    ref_audio = Path(ref_audio)
    if not ref_audio.exists():
        raise FileNotFoundError(f"ref_audio not found: {ref_audio}")

    ref_b64 = base64.b64encode(ref_audio.read_bytes()).decode()
    url = f"{TTS_URL}/synthesize"

    resp = requests.post(
        url,
        json={
            "text": text,
            "mode": "clone",
            "language": language,
            "ref_audio_b64": ref_b64,
            "ref_text": ref_text,
        },
        timeout=TIMEOUT_TTS,
    )
    resp.raise_for_status()

    wav_bytes = resp.content
    elapsed = float(resp.headers.get("X-Elapsed", "0"))

    if save_to:
        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
        file_str = str(out_path)
        url_str = f"/static/audio/{out_path.name}"
    else:
        audio_dir = Path(__file__).parent / "static" / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        fname = f"rclone_{uuid.uuid4().hex[:8]}.wav"
        out_path = audio_dir / fname
        out_path.write_bytes(wav_bytes)
        file_str = str(out_path)
        url_str = f"/static/audio/{fname}"

    audio_dur = len(wav_bytes) / 48000
    return {
        "url": url_str,
        "file": file_str,
        "gen_sec": elapsed,
        "audio_dur": round(audio_dur, 2),
        "rtf": round(elapsed / audio_dur, 3) if audio_dur > 0 else 0,
    }


# ── Health checks ─────────────────────────────────────────────────────────────

def check_remote_health() -> dict:
    """Check both remote servers. Returns {stt_ok, tts_ok, details}."""
    results = {}
    for name, base in [("stt", STT_URL), ("tts", TTS_URL)]:
        try:
            r = requests.get(f"{base}/health", timeout=5)
            r.raise_for_status()
            results[f"{name}_ok"] = True
            results[f"{name}_info"] = r.json()
        except Exception as e:
            results[f"{name}_ok"] = False
            results[f"{name}_error"] = str(e)
    return results
