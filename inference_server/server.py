"""
Inference Server — runs on PC2 (RTX 3070 Ti 8GB).
Exposes STT + TTS via HTTP API for all apps on PC1 (and others).

Models loaded:
  STT: Whisper large-v3 (~3GB VRAM)
  TTS: Qwen3TTS 1.7B (~3GB VRAM)
  Total: ~6GB → fits in 8GB

Endpoints:
  POST /stt          ← audio bytes → text
  POST /tts          ← text → WAV bytes
  GET  /health       ← model status
  GET  /speakers     ← list available voice clones

Start on PC2:
  python inference_server/server.py
  → http://0.0.0.0:8010

PC1 apps use:
  INFERENCE_SERVER=http://192.168.x.x:8010  (set in .env)
"""
import asyncio
import base64
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("inference_server")

# ── Add parent to path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Models (loaded at startup) ────────────────────────────────────────────────
_stt = None
_tts = None
_models_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _stt, _tts, _models_ready
    logger.info("Loading models on GPU...")

    loop = asyncio.get_event_loop()

    # STT — Kyutai streaming (faster than Whisper for real-time calls)
    try:
        from sales_agent.pipeline.stt_kyutai import get_kyutai_stt
        stt_model = os.getenv("KYUTAI_STT_MODEL", "kyutai/stt-2.6b-en")
        _stt = get_kyutai_stt(device="cuda:0")
        _stt._model_name = stt_model
        await loop.run_in_executor(None, _stt.load)
        logger.info(f"✅ STT (Kyutai {stt_model}) loaded on GPU")
    except Exception as e:
        logger.error(f"❌ STT load failed: {e}")

    # TTS — Qwen3TTS
    try:
        from qwen3tts_manager import get_qwen3tts_manager
        _tts = get_qwen3tts_manager()
        # Force GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        await loop.run_in_executor(None, _tts.load)
        logger.info("✅ TTS (Qwen3TTS) loaded on GPU")
    except Exception as e:
        logger.warning(f"⚠️  Qwen3TTS failed, trying XTTS: {e}")
        try:
            from xtts_manager import get_xtts_manager
            _tts = get_xtts_manager()
            await loop.run_in_executor(None, _tts.load)
            logger.info("✅ TTS (XTTS v2) loaded on GPU")
        except Exception as e2:
            logger.error(f"❌ TTS load failed: {e2}")

    _models_ready = bool(_stt and _tts)
    logger.info(f"Inference Server ready. STT={'✅' if _stt else '❌'} TTS={'✅' if _tts else '❌'}")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Inference Server", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Request/Response models ───────────────────────────────────────────────────

class STTRequest(BaseModel):
    audio_b64: str       # base64-encoded float32 PCM or WAV bytes
    sample_rate: int = 16000
    language: str = "en"

class STTResponse(BaseModel):
    text: str
    confidence: float | None = None
    duration_seconds: float | None = None
    latency_ms: int = 0

class TTSRequest(BaseModel):
    text: str
    voice: str = "anastasia"   # voice clone name from voices/
    language: str = "en"
    speed: float = 1.0

class TTSResponse(BaseModel):
    audio_b64: str      # base64-encoded WAV bytes
    sample_rate: int
    duration_seconds: float
    latency_ms: int

class HealthResponse(BaseModel):
    status: str
    stt_ready: bool
    tts_ready: bool
    stt_model: str
    tts_model: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ready" if _models_ready else "loading",
        stt_ready=_stt is not None,
        tts_ready=_tts is not None,
        stt_model=getattr(_stt, "model_name", "kyutai/stt-2.6b-en") if _stt else "none",
        tts_model="qwen3tts-1.7b" if "qwen" in type(_tts).__name__.lower() else "xtts-v2",
    )


@app.get("/speakers")
async def list_speakers():
    """List available voice clones from voices/ folder."""
    voices_dir = Path(__file__).parent.parent / "voices"
    speakers = []
    for f in voices_dir.glob("*.wav"):
        speakers.append({"name": f.stem, "file": f.name})
    return {"speakers": speakers}


@app.post("/stt", response_model=STTResponse)
async def transcribe(req: STTRequest):
    """Transcribe audio. Send float32 PCM as base64."""
    if not _stt:
        raise HTTPException(503, "STT model not ready")

    t0 = time.time()
    try:
        audio_bytes = base64.b64decode(req.audio_b64)
        # Try WAV first, fallback to raw float32
        try:
            buf = io.BytesIO(audio_bytes)
            audio_data, sr = sf.read(buf, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
        except Exception:
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _stt.transcribe_sync, audio_data)
        latency_ms = int((time.time() - t0) * 1000)

        return STTResponse(
            text=result.get("text", ""),
            confidence=result.get("confidence"),
            duration_seconds=result.get("duration"),
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"STT error: {e}")
        raise HTTPException(500, str(e))


@app.post("/tts", response_model=TTSResponse)
async def synthesize(req: TTSRequest):
    """Synthesize speech. Returns WAV as base64."""
    if not _tts:
        raise HTTPException(503, "TTS model not ready")

    t0 = time.time()
    try:
        loop = asyncio.get_event_loop()
        audio_data, sample_rate = await loop.run_in_executor(
            None, _tts.synthesize_sync, req.text, req.voice
        )

        buf = io.BytesIO()
        sf.write(buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()

        latency_ms = int((time.time() - t0) * 1000)
        duration = len(audio_data) / sample_rate

        return TTSResponse(
            audio_b64=base64.b64encode(wav_bytes).decode(),
            sample_rate=sample_rate,
            duration_seconds=round(duration, 2),
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(500, str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("INFERENCE_PORT", "8010"))
    host = os.getenv("INFERENCE_HOST", "0.0.0.0")
    logger.info(f"Starting Inference Server on {host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=False)
