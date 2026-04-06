"""
Call Scheduler — TCPA-compliant outbound call timing.

TCPA Rules (enforced here):
  - Calling hours: 8am–9pm LOCAL time of recipient
  - No weekends (optional, configurable)
  - Retry delays: no-answer → 4hr, voicemail → 24hr, max 3 attempts
  - DNC check before every call (in db.py)

This scheduler:
  1. Gets contacts ready to call from DB
  2. Checks local time compliance for each contact's timezone
  3. Returns a batch for the campaign manager to dial
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .db import get_contacts_to_call, is_dnc

logger = logging.getLogger("sales.scheduler")

# TCPA calling window (caller's local time)
CALL_WINDOW_START = time(8, 0)   # 8:00 AM
CALL_WINDOW_END   = time(21, 0)  # 9:00 PM

# Retry delays
RETRY_NO_ANSWER_MINUTES  = 4 * 60    # 4 hours
RETRY_VOICEMAIL_MINUTES  = 24 * 60   # 24 hours
RETRY_BUSY_MINUTES       = 60        # 1 hour
MAX_ATTEMPTS             = 3


def is_calling_allowed(timezone_str: str) -> bool:
    """
    Check if it's legal/appropriate to call this timezone right now.
    Returns True if within calling window.
    """
    try:
        tz = ZoneInfo(timezone_str)
        local_time = datetime.now(tz).time()
        return CALL_WINDOW_START <= local_time <= CALL_WINDOW_END
    except Exception as e:
        logger.warning(f"[SCHEDULER] Invalid timezone {timezone_str}: {e}")
        # Default conservative: don't call if timezone unknown
        return False


def get_contacts_for_campaign(campaign_id: int, batch_size: int = 5) -> list[dict]:
    """
    Get contacts ready to call right now (timezone-filtered).
    Returns up to batch_size contacts.
    """
    # Get candidates from DB (already filters by status, retry timing, attempts)
    candidates = get_contacts_to_call(campaign_id, limit=batch_size * 3)

    ready = []
    for contact in candidates:
        phone = contact.get("phone", "")

        # Double-check DNC (DB already filters, this is belt+suspenders)
        if is_dnc(phone):
            logger.info(f"[SCHEDULER] Skip DNC: {phone}")
            continue

        # Timezone compliance
        tz = contact.get("timezone", "America/New_York")
        if not is_calling_allowed(tz):
            logger.debug(f"[SCHEDULER] Skip (outside hours) {phone} tz={tz}")
            continue

        ready.append(contact)
        if len(ready) >= batch_size:
            break

    logger.info(f"[SCHEDULER] Campaign {campaign_id}: {len(ready)}/{len(candidates)} contacts ready")
    return ready


def get_retry_delay(outcome: str) -> int | None:
    """
    Returns retry delay in minutes for a given call outcome.
    Returns None if no retry (converted, rejected, max_retries).
    """
    delays = {
        "no_answer": RETRY_NO_ANSWER_MINUTES,
        "voicemail": RETRY_VOICEMAIL_MINUTES,
        "busy":      RETRY_BUSY_MINUTES,
        "failed":    RETRY_BUSY_MINUTES,
    }
    return delays.get(outcome)
