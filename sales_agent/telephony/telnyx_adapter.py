"""
Telnyx WebSocket Adapter — handles Telnyx media streaming protocol.

How Telnyx works:
  1. You buy a number on Telnyx
  2. You create a TeXML app, set WebSocket URL = your server
  3. When someone calls (or you dial out) → Telnyx opens WebSocket to your server
  4. Telnyx sends JSON events: call.answered, media, call.hangup, etc.
  5. You respond with media events to send audio back

WebSocket message format (Telnyx bidirectional streaming):
  Incoming: {"event":"media","media":{"payload":"base64...","chunk":"1","timestamp":"..."}}
  Outgoing: {"event":"media","media":{"payload":"base64..."}}

Setup (do once in Telnyx dashboard):
  1. Connections → Create TeXML Application
  2. Enable "Streaming" → WebSocket URL: wss://your-server.com/telnyx/stream
  3. Codec: PCMU (G.711 μ-law 8kHz) — universal, no extra setup
  4. Enable DTMF detection (for IVR navigation)
  5. Enable "Record calls" for QA

AMD (Answering Machine Detection):
  Telnyx Premium AMD: free, auto-enabled on premium numbers
  Event: call.machine.premium.detection.ended
  Results: human_residence | human_business | machine | silence | fax_detected | not_sure
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Callable

from .codec import decode_telnyx_media, encode_for_telnyx

logger = logging.getLogger("sales.telnyx")

# Audio chunk size Telnyx sends (20ms at 8kHz = 160 samples)
CHUNK_MS = 20
TELNYX_SR = 8000


class TelnyxStreamAdapter:
    """
    Handles one Telnyx WebSocket streaming connection.
    Translates Telnyx protocol → our CallSession audio interface.

    Usage:
        adapter = TelnyxStreamAdapter(
            call_sid="...",
            on_audio=session.feed_audio,
            on_hangup=handle_hangup,
        )
        await adapter.handle_websocket(ws)
    """

    def __init__(
        self,
        call_sid: str,
        on_audio: Callable,          # async fn(pcm_bytes: bytes) called for each chunk
        on_hangup: Callable,         # async fn() called when call ends
        on_amd: Callable | None = None,  # async fn(result: str) called with AMD result
        codec: str = "PCMU",
    ):
        self.call_sid = call_sid
        self._on_audio = on_audio
        self._on_hangup = on_hangup
        self._on_amd = on_amd
        self._codec = codec
        self._ws = None
        self._stream_sid: str | None = None
        self._call_active = False
        self._sequence = 0

    async def handle_websocket(self, ws) -> None:
        """
        Main WebSocket handler loop for one Telnyx call.
        Call this from your FastAPI WebSocket endpoint.
        """
        self._ws = ws
        self._call_active = True
        logger.info(f"[TELNYX {self.call_sid}] WebSocket connected")

        try:
            async for raw in ws.iter_text():
                await self._handle_event(raw)
        except Exception as e:
            logger.error(f"[TELNYX {self.call_sid}] WS error: {e}")
        finally:
            self._call_active = False
            await self._on_hangup()
            logger.info(f"[TELNYX {self.call_sid}] WebSocket closed")

    async def _handle_event(self, raw: str) -> None:
        """Parse and dispatch Telnyx event."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        event = msg.get("event", "")

        if event == "connected":
            # Telnyx handshake
            logger.debug(f"[TELNYX] Connected: {msg}")

        elif event == "start":
            # Stream started — save stream SID for sending audio back
            self._stream_sid = msg.get("start", {}).get("streamSid") or msg.get("streamSid")
            self._codec = msg.get("start", {}).get("mediaFormat", {}).get("encoding", "PCMU")
            logger.info(f"[TELNYX {self.call_sid}] Stream started: {self._stream_sid} codec={self._codec}")

        elif event == "media":
            # Incoming audio chunk
            payload_b64 = msg.get("media", {}).get("payload", "")
            if payload_b64:
                raw_bytes = base64.b64decode(payload_b64)
                # Decode to float32 16kHz numpy array
                audio_16k = decode_telnyx_media(raw_bytes, codec=self._codec)
                # Pass to our pipeline
                await self._on_audio(audio_16k.tobytes())

        elif event == "stop":
            logger.info(f"[TELNYX {self.call_sid}] Stream stopped")
            self._call_active = False

        elif event == "dtmf":
            digit = msg.get("dtmf", {}).get("digit", "")
            logger.info(f"[TELNYX {self.call_sid}] DTMF: {digit}")
            # Can be used for IVR navigation (pressing 1, 2, etc.)

        elif event == "call.machine.premium.detection.ended":
            # AMD result
            result = msg.get("payload", {}).get("result", "not_sure")
            logger.info(f"[TELNYX {self.call_sid}] AMD: {result}")
            if self._on_amd:
                await self._on_amd(result)

    async def send_audio(self, audio_float32, input_sr: int = 22050) -> None:
        """
        Send TTS audio back to Telnyx → prospect hears agent voice.
        Encodes our TTS output to phone format and sends via WebSocket.
        """
        if not self._ws or not self._stream_sid or not self._call_active:
            return

        encoded = encode_for_telnyx(audio_float32, input_sr, codec=self._codec)

        # Send in 20ms chunks (Telnyx expects chunked audio)
        chunk_size = (TELNYX_SR // 50)  # 20ms at 8kHz = 160 bytes (PCMU)
        for i in range(0, len(encoded), chunk_size):
            chunk = encoded[i:i + chunk_size]
            payload = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {
                    "payload": base64.b64encode(chunk).decode()
                }
            }
            try:
                await self._ws.send_text(json.dumps(payload))
                self._sequence += 1
            except Exception as e:
                logger.warning(f"[TELNYX] Send error: {e}")
                break

    async def send_clear(self) -> None:
        """
        Clear Telnyx audio buffer — used on barge-in.
        Stops any audio currently playing on the phone.
        """
        if not self._ws or not self._stream_sid:
            return
        try:
            await self._ws.send_text(json.dumps({
                "event": "clear",
                "streamSid": self._stream_sid,
            }))
        except Exception:
            pass

    async def hangup(self) -> None:
        """Send hangup signal — end the call from our side."""
        if not self._ws:
            return
        try:
            await self._ws.send_text(json.dumps({"event": "stop", "streamSid": self._stream_sid}))
        except Exception:
            pass
