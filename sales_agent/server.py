"""
Sales Agent Server — FastAPI + WebSocket for Phase 1 browser demo.

Windows fix: WindowsSelectorEventLoopPolicy (research finding)
Port: 8001 (separate from CLI helper on 8000)

WebSocket protocol:
  Client → Server:
    {type: "audio", data: base64_pcm_float32_16kHz}
    {type: "start_call"}
    {type: "end_call"}

  Server → Client:
    {type: "call_started", call_id: str}
    {type: "vad", status: "speaking"|"silence"}
    {type: "prospect_speech", text: str}
    {type: "agent_speech", text: str}
    {type: "agent_audio", data: base64_wav, sample_rate: int, text: str}
    {type: "state_change", state: str, reason: str}
    {type: "fact_extracted", key: str, value: str}
    {type: "barge_in", status: str}
    {type: "call_ended", outcome: str, duration_seconds: int, call_id: str}
    {type: "error", message: str}
"""
import asyncio
import base64
import json
import logging
import os
import sys
from pathlib import Path

# Windows asyncio fix (research finding: IocpProactor issues)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

logger = logging.getLogger("sales.server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Config ────────────────────────────────────────────────────────────────────

_CFG_PATH = Path(__file__).parent / "config.yaml"
_cfg = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}
_PORT = _cfg.get("server", {}).get("port", 8001)
_HOST = _cfg.get("server", {}).get("host", "0.0.0.0")
_PERSONA_FILE = _cfg.get("agent", {}).get("persona_file", "template.yaml")

# API key
_OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not _OPENROUTER_KEY:
    # Try loading from parent .env
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        _OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
    except ImportError:
        pass

# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Sales Agent starting on port {_PORT}")
    if not _OPENROUTER_KEY:
        logger.error("OPENROUTER_API_KEY not set!")
    yield
    logger.info("Sales Agent shutting down")


app = FastAPI(title="AI Voice Sales Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (demo UI)
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Active sessions
_sessions: dict[str, object] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the demo UI."""
    html_path = _static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Sales Agent Demo</h1><p>index.html not found</p>")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_calls": len(_sessions),
        "persona": _PERSONA_FILE,
        "openrouter_configured": bool(_OPENROUTER_KEY),
    }


@app.websocket("/ws/call")
async def call_websocket(ws: WebSocket):
    """Main WebSocket endpoint for a sales call."""
    await ws.accept()
    session_id = id(ws)
    session = None

    async def ws_send(data: dict):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass

    logger.info(f"[WS {session_id}] Connected")

    try:
        async for raw_msg in ws.iter_text():
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "start_call":
                # Initialize new call session
                if session is not None:
                    await session.end()

                from core.session import CallSession
                session = CallSession(
                    ws_send=ws_send,
                    openrouter_api_key=_OPENROUTER_KEY,
                    persona_file=_PERSONA_FILE,
                )
                _sessions[str(session_id)] = session
                await session.start()

            elif msg_type == "audio":
                # Receive PCM audio from browser
                if session is None:
                    await ws_send({"type": "error", "message": "Call not started"})
                    continue
                try:
                    pcm_bytes = base64.b64decode(msg.get("data", ""))
                    await session.feed_audio(pcm_bytes)
                except Exception as e:
                    logger.warning(f"[WS {session_id}] Audio error: {e}")

            elif msg_type == "end_call":
                if session:
                    await session.end()
                    session = None
                    _sessions.pop(str(session_id), None)

            elif msg_type == "ping":
                await ws_send({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"[WS {session_id}] Disconnected")
    except Exception as e:
        logger.error(f"[WS {session_id}] Error: {e}", exc_info=True)
    finally:
        if session:
            await session.end()
        _sessions.pop(str(session_id), None)
        logger.info(f"[WS {session_id}] Session cleaned up")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=_HOST,
        port=_PORT,
        reload=False,
        log_level="info",
    )
