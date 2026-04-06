"""
Campaign Manager — orchestrates outbound calling campaigns.

Responsibilities:
  - Poll scheduler for contacts to call
  - Dial via Telnyx (when configured) or simulate for testing
  - Handle concurrency limits
  - Update DB after each call
  - Send Telegram progress reports

CLI usage (for testing):
  python -m campaign.manager --campaign 1 --dry-run   ← show who would be called
  python -m campaign.manager --campaign 1              ← start calling
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("sales.campaign_manager")


class CampaignManager:
    """Manages an outbound calling campaign."""

    def __init__(
        self,
        campaign_id: int,
        server_url: str,          # Your public URL (ngrok or VPS)
        poll_interval: int = 30,  # seconds between batches
        max_concurrent: int = 3,  # concurrent calls
        dry_run: bool = False,    # don't actually dial
    ):
        self.campaign_id = campaign_id
        self.server_url = server_url.rstrip("/")
        self.poll_interval = poll_interval
        self.max_concurrent = max_concurrent
        self.dry_run = dry_run
        self._running = False
        self._active: set[asyncio.Task] = set()

        # Import here to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from campaign.db import init_db
        init_db()

    async def start(self):
        """Start the campaign loop. Runs until stopped."""
        self._running = True
        logger.info(f"[CAMPAIGN {self.campaign_id}] Started. dry_run={self.dry_run}")

        await self._send_tg(f"📞 Campaign {self.campaign_id} started")

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error(f"[CAMPAIGN] Tick error: {e}", exc_info=True)
            await asyncio.sleep(self.poll_interval)

    def stop(self):
        self._running = False
        logger.info(f"[CAMPAIGN {self.campaign_id}] Stopping")

    async def _tick(self):
        """One poll cycle: get contacts, dial them."""
        from campaign.scheduler import get_contacts_for_campaign, get_retry_delay
        from campaign.db import update_contact_status, log_call

        # How many slots available?
        slots = self.max_concurrent - len(self._active)
        if slots <= 0:
            return

        contacts = get_contacts_for_campaign(self.campaign_id, batch_size=slots)
        if not contacts:
            logger.debug(f"[CAMPAIGN {self.campaign_id}] No contacts ready")
            return

        for contact in contacts:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would call: {contact['first_name']} {contact['phone']}")
                continue

            task = asyncio.create_task(self._dial_contact(contact))
            self._active.add(task)
            task.add_done_callback(self._active.discard)

    async def _dial_contact(self, contact: dict):
        """Attempt to call one contact."""
        from campaign.db import update_contact_status, log_call
        from telephony.call_manager import get_call_manager

        phone = contact["phone"]
        contact_id = contact["id"]
        persona_file = contact.get("persona_file", "template.yaml")

        logger.info(f"[CAMPAIGN] Dialing {contact.get('first_name', '')} {phone}")
        update_contact_status(contact_id, "dialing")

        manager = get_call_manager()

        if not manager.is_configured:
            # Simulation mode — log as if called
            logger.info(f"[CAMPAIGN] SIMULATION (no Telnyx key): {phone}")
            await asyncio.sleep(2)
            update_contact_status(contact_id, "pending", next_retry_minutes=None)
            return

        call_id = await manager.dial(
            to_number=phone,
            contact_id=contact_id,
            campaign_id=self.campaign_id,
            webhook_url=self.server_url,
        )

        if not call_id:
            update_contact_status(contact_id, "failed", next_retry_minutes=60)
            log_call(contact_id, self.campaign_id, "failed")
            return

        # The rest is handled by webhook events (phone_server.py)
        logger.info(f"[CAMPAIGN] Dialing in progress: {call_id}")

    async def _send_tg(self, message: str):
        """Send Telegram notification."""
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from tg_send import tg_msg
            tg_msg(message)
        except Exception:
            pass

    async def send_daily_report(self):
        """Send campaign stats to Telegram."""
        from campaign.db import get_campaign_stats
        stats = get_campaign_stats(self.campaign_id)
        msg = (
            f"📊 Campaign {self.campaign_id} Daily Report\n"
            f"Total calls: {stats['total_calls']}\n"
            f"Converted: {stats['converted']} ({stats['conversion_rate']}%)\n"
            f"Answer rate: {stats['answer_rate']}%\n"
            f"Voicemail: {stats['voicemail']}\n"
            f"Avg duration: {stats['avg_duration']:.0f}s\n"
            f"Avg score: {stats['avg_score']:.1f}/10" if stats.get('avg_score') else ""
        )
        await self._send_tg(msg)


def __main__():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=int, required=True)
    parser.add_argument("--server-url", default=os.getenv("SERVER_URL", "http://localhost:8001"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    manager = CampaignManager(
        campaign_id=args.campaign,
        server_url=args.server_url,
        max_concurrent=args.concurrency,
        dry_run=args.dry_run,
    )
    asyncio.run(manager.start())


if __name__ == "__main__":
    __main__()
