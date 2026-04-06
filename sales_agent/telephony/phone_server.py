"""
Phone Server — FastAPI routes for Telnyx telephony integration.

Adds to the main sales_agent/server.py:
  POST /telnyx/events  — Telnyx webhook (call.answered, call.hangup, AMD, etc.)
  WS   /telnyx/stream  — Telnyx media streaming WebSocket

When you get Telnyx:
  1. Set in .env: TELNYX_API_KEY, TELNYX_FROM_NUMBER, TELNYX_CONNECTION_ID
  2. In Telnyx dashboard: set webhook URL = https://your-server.com/telnyx/events
  3. Set stream URL = wss://your-server.com/telnyx/stream
  4. Start campaign manager → it dials automatically

For testing with ngrok:
  1. Run: ngrok http 8001
  2. Copy https://xxxx.ngrok.io URL
  3. Set in Telnyx dashboard as webhook URL
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from .call_manager import get_call_manager
from .telnyx_adapter import TelnyxStreamAdapter

logger = logging.getLogger("sales.phone_server")

router = APIRouter(prefix="/telnyx", tags=["telephony"])

# Active phone call sessions
_phone_sessions: dict[str, object] = {}  # call_control_id → CallSession


# ── Webhook Endpoint ──────────────────────────────────────────────────────────

@router.post("/events")
async def telnyx_webhook(request: Request):
    """
    Receives all Telnyx call events.
    Telnyx sends here: call.initiated, call.answered, call.hangup, AMD, recording, etc.
    """
    try:
        body = await request.json()
    except Exception:
        return {"status": "error"}

    event_type = body.get("data", {}).get("event_type", "")
    payload = body.get("data", {}).get("payload", {})
    call_id = payload.get("call_control_id", "")

    logger.info(f"[TELNYX EVENT] {event_type} call_id={call_id}")

    manager = get_call_manager()

    if event_type == "call.initiated":
        # Outbound call started (phone is ringing)
        logger.info(f"[TELNYX] Call initiated to {payload.get('to')}")

    elif event_type == "call.answered":
        # Human (or machine) picked up
        manager.on_call_answered(call_id)
        # AMD will fire shortly after — wait for it before starting pitch

    elif event_type == "call.machine.premium.detection.ended":
        # AMD result — decide whether to pitch or leave voicemail
        result = payload.get("result", "not_sure")
        is_human = manager.on_amd_result(call_id, result)

        session = _phone_sessions.get(call_id)
        if session:
            if is_human:
                # Human answered — start sales script
                await session.on_human_confirmed()
            else:
                # Voicemail — switch to voicemail script
                await session.on_voicemail_detected(result)

    elif event_type == "call.hangup":
        # Call ended
        record = manager.on_call_ended(call_id)
        session = _phone_sessions.pop(call_id, None)
        if session:
            await session.end()
        logger.info(f"[TELNYX] Hangup: {call_id} duration={record.duration_seconds if record else 0}s")

    elif event_type == "call.recording.saved":
        # Telnyx finished saving the call recording
        recording_url = payload.get("recording_urls", {}).get("mp3", "")
        logger.info(f"[TELNYX] Recording saved: {recording_url}")
        # TODO: download and store locally for analytics

    return {"status": "ok"}


# ── Media Streaming WebSocket ─────────────────────────────────────────────────

@router.websocket("/stream")
async def telnyx_stream(ws: WebSocket):
    """
    Telnyx opens this WebSocket when a call connects.
    Bidirectional audio streaming: phone audio IN, agent audio OUT.
    """
    await ws.accept()
    call_control_id = None

    try:
        # First message from Telnyx contains call info
        first_msg = json.loads(await ws.receive_text())
        if first_msg.get("event") == "connected":
            # Telnyx protocol: next message will be "start" with call info
            start_msg = json.loads(await ws.receive_text())
            call_control_id = (
                start_msg.get("start", {}).get("callSid")
                or start_msg.get("streamSid", "unknown")
            )
            logger.info(f"[TELNYX STREAM] Call {call_control_id} streaming started")

        # Get or create CallSession for this call
        session = _phone_sessions.get(call_control_id)
        if session is None:
            # Inbound call or session not yet created
            session = await _create_phone_session(call_control_id, ws)
            if call_control_id:
                _phone_sessions[call_control_id] = session

        # Create adapter and handle the stream
        adapter = TelnyxStreamAdapter(
            call_sid=call_control_id or "unknown",
            on_audio=session.feed_audio,
            on_hangup=session.end,
            on_amd=lambda r: session.on_voicemail_detected(r) if r not in ("human_residence", "human_business", "not_sure") else session.on_human_confirmed(),
        )

        # Inject send_audio into session so agent audio goes to phone
        session._telnyx_adapter = adapter

        await adapter.handle_websocket(ws)

    except WebSocketDisconnect:
        logger.info(f"[TELNYX STREAM] Disconnected: {call_control_id}")
    except Exception as e:
        logger.error(f"[TELNYX STREAM] Error: {e}", exc_info=True)
    finally:
        if call_control_id:
            _phone_sessions.pop(call_control_id, None)


async def _create_phone_session(call_control_id: str, ws: WebSocket) -> object:
    """Create a CallSession for a phone call."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    persona = os.getenv("SALES_PERSONA", "template.yaml")

    # Wrap WebSocket send for our session
    async def ws_send(data: dict):
        pass  # Phone calls don't need WebSocket UI events (no browser)

    from sales_agent.core.session import CallSession
    session = CallSession(
        ws_send=ws_send,
        openrouter_api_key=api_key,
        persona_file=persona,
    )
    return session


def get_phone_sessions() -> dict:
    return _phone_sessions
