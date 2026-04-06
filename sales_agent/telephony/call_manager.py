"""
Call Manager — Telnyx REST API for outbound calling.

What this does:
  - Dials outbound calls via Telnyx API
  - Tracks call state (ringing / answered / voicemail / ended)
  - Supports AMD result handling
  - Manages concurrent call limits

Setup (when you have Telnyx API key):
  1. Get API key from telnyx.com → API Keys
  2. Buy a phone number (local US number for better pickup rate)
  3. Create TeXML Application → set WebSocket URL
  4. Set TELNYX_API_KEY and TELNYX_FROM_NUMBER in .env

The actual call flow:
  1. POST /calls → Telnyx dials the number
  2. Telnyx connects to your WebSocket (TelnyxStreamAdapter)
  3. AMD fires → if human: start sales script, if machine: leave voicemail
  4. Call ends → webhook to /telnyx/events
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx

logger = logging.getLogger("sales.call_manager")

# ── Config ────────────────────────────────────────────────────────────────────
TELNYX_API_BASE = "https://api.telnyx.com/v2"
MAX_CONCURRENT_CALLS = 5   # start conservative, increase after testing


@dataclass
class CallRecord:
    """Tracks state of one outbound call."""
    call_id: str
    contact_id: int
    phone_number: str
    campaign_id: int
    status: str = "dialing"         # dialing | ringing | answered | voicemail | completed | failed | no_answer
    amd_result: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    answered_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: int = 0
    telnyx_call_control_id: Optional[str] = None


class TelnyxCallManager:
    """
    Manages outbound calls via Telnyx REST API.
    Works without API key — all methods gracefully no-op when key missing.
    """

    def __init__(self):
        self._api_key = os.getenv("TELNYX_API_KEY", "")
        self._from_number = os.getenv("TELNYX_FROM_NUMBER", "")
        self._connection_id = os.getenv("TELNYX_CONNECTION_ID", "")  # TeXML App ID
        self._active_calls: dict[str, CallRecord] = {}

        if not self._api_key:
            logger.warning("[TELNYX] No API key — telephony disabled. Set TELNYX_API_KEY in .env")
        else:
            logger.info(f"[TELNYX] Ready. From: {self._from_number}")

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._from_number)

    @property
    def active_call_count(self) -> int:
        return len(self._active_calls)

    # ── Outbound Dialing ──────────────────────────────────────────────────────

    async def dial(
        self,
        to_number: str,
        contact_id: int,
        campaign_id: int,
        webhook_url: str,
    ) -> Optional[str]:
        """
        Initiate an outbound call.
        Returns call_control_id (Telnyx call ID) or None on failure.

        When ready to deploy: just set TELNYX_API_KEY + TELNYX_FROM_NUMBER in .env
        """
        if not self.is_configured:
            logger.error("[TELNYX] Cannot dial — not configured")
            return None

        if self.active_call_count >= MAX_CONCURRENT_CALLS:
            logger.warning(f"[TELNYX] Max concurrent calls ({MAX_CONCURRENT_CALLS}) reached")
            return None

        payload = {
            "connection_id": self._connection_id,
            "to": to_number,
            "from": self._from_number,
            "from_display_name": os.getenv("TELNYX_CALLER_NAME", ""),
            "stream_url": webhook_url + "/telnyx/stream",
            "stream_bidirectional_mode": "rtp",
            "answering_machine_detection": "premium",  # Free AMD!
            "answering_machine_detection_config": {
                "after_seconds": 4,
                "silence_threshold": 2000,
            },
            "record": "record-from-answer",  # Record for QA
            "record_format": "mp3",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{TELNYX_API_BASE}/calls",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    }
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})
                call_control_id = data.get("call_control_id")

                if call_control_id:
                    record = CallRecord(
                        call_id=call_control_id,
                        contact_id=contact_id,
                        phone_number=to_number,
                        campaign_id=campaign_id,
                        telnyx_call_control_id=call_control_id,
                    )
                    self._active_calls[call_control_id] = record
                    logger.info(f"[TELNYX] Dialing {to_number} → call_id={call_control_id}")
                    return call_control_id

        except httpx.HTTPStatusError as e:
            logger.error(f"[TELNYX] Dial failed: {e.response.status_code} {e.response.text}")
        except Exception as e:
            logger.error(f"[TELNYX] Dial error: {e}")

        return None

    async def hangup(self, call_control_id: str) -> bool:
        """Hang up an active call."""
        if not self.is_configured:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{TELNYX_API_BASE}/calls/{call_control_id}/actions/hangup",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                resp.raise_for_status()
                return True
        except Exception as e:
            logger.warning(f"[TELNYX] Hangup error: {e}")
            return False

    # ── Event Handling ────────────────────────────────────────────────────────

    def on_call_answered(self, call_control_id: str):
        if call_control_id in self._active_calls:
            record = self._active_calls[call_control_id]
            record.status = "answered"
            record.answered_at = datetime.now()
            logger.info(f"[TELNYX] Answered: {call_control_id}")

    def on_amd_result(self, call_control_id: str, result: str):
        """
        AMD result: human_residence | human_business | machine | silence | fax_detected | not_sure
        """
        if call_control_id in self._active_calls:
            self._active_calls[call_control_id].amd_result = result
        logger.info(f"[TELNYX] AMD {call_control_id}: {result}")
        return result in ("human_residence", "human_business", "not_sure")  # True = proceed with pitch

    def on_call_ended(self, call_control_id: str) -> Optional[CallRecord]:
        record = self._active_calls.pop(call_control_id, None)
        if record:
            record.status = "completed"
            record.ended_at = datetime.now()
            if record.answered_at:
                record.duration_seconds = int(
                    (record.ended_at - record.answered_at).total_seconds()
                )
        return record

    # ── Number Formatting ─────────────────────────────────────────────────────

    @staticmethod
    def format_e164(phone: str) -> str:
        """Convert various phone formats to E.164 (+12125551234)."""
        digits = ''.join(c for c in phone if c.isdigit())
        if len(digits) == 10:
            return f"+1{digits}"  # US number
        if len(digits) == 11 and digits[0] == '1':
            return f"+{digits}"
        return f"+{digits}"  # Hope for the best


# Singleton
_manager: TelnyxCallManager | None = None


def get_call_manager() -> TelnyxCallManager:
    global _manager
    if _manager is None:
        _manager = TelnyxCallManager()
    return _manager
