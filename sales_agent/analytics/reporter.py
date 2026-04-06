"""
Analytics Reporter — daily/weekly Telegram reports + hypothesis testing.

Reports:
  - Daily: conversion rate, avg score, top objections, best calls
  - Weekly: A/B test results, trend analysis, script recommendations
  - On converted: immediate Telegram celebrate
  - On score < 4: Telegram alert (agent performing badly)

Hypothesis testing:
  Record what we think will work → compare to reality → iterate
  Example: "Variant B (new opening) will convert better than A"
  → after 20 calls: B converts 12% vs A 8% → confirm hypothesis
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("sales.reporter")


def _tg(msg: str):
    """Send Telegram message."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from tg_send import tg_msg
        tg_msg(msg)
    except Exception as e:
        logger.warning(f"[REPORTER] Telegram error: {e}")


def send_call_result(
    outcome: str,
    score: dict | None,
    contact_name: str = "",
    duration_seconds: int = 0,
    campaign_name: str = "",
):
    """Send immediate notification after each call."""
    outcome_emoji = {
        "converted":  "🎉",
        "rejected":   "❌",
        "voicemail":  "📬",
        "no_answer":  "📵",
        "failed":     "⚠️",
    }.get(outcome, "📞")

    msg_parts = [f"{outcome_emoji} Call completed: {outcome.upper()}"]
    if contact_name:
        msg_parts.append(f"👤 {contact_name}")
    if campaign_name:
        msg_parts.append(f"📋 {campaign_name}")
    if duration_seconds:
        msg_parts.append(f"⏱ {duration_seconds}s")

    if score:
        overall = score.get("overall_score", 0)
        msg_parts.append(f"⭐ Score: {overall:.1f}/10")
        if score.get("improvements"):
            msg_parts.append(f"💡 {score['improvements'][0]}")

    _tg("\n".join(msg_parts))


def send_daily_report(campaign_id: int | None = None):
    """Send daily summary report."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from campaign.db import get_campaign_stats, get_campaigns

        today = datetime.now().strftime("%A, %B %d, %Y")
        report = [f"📊 Sales Agent Daily Report\n{today}"]

        campaigns = get_campaigns() if campaign_id is None else []
        if campaign_id:
            campaigns = [{"id": campaign_id, "name": f"Campaign {campaign_id}"}]

        for camp in campaigns:
            stats = get_campaign_stats(camp["id"])
            if stats["total_calls"] == 0:
                continue

            report.append(
                f"\n📋 {camp['name']}\n"
                f"  Calls: {stats['total_calls']}\n"
                f"  Converted: {stats['converted']} ({stats['conversion_rate']}%)\n"
                f"  Answer rate: {stats['answer_rate']}%\n"
                f"  Voicemail: {stats['voicemail']}\n"
                f"  Avg duration: {stats.get('avg_duration', 0):.0f}s\n"
                f"  Avg score: {stats.get('avg_score', 0) or 0:.1f}/10"
            )

        _tg("\n".join(report))

    except Exception as e:
        logger.error(f"[REPORTER] Daily report error: {e}")


def send_ab_report(campaign_id: int):
    """
    Compare A/B variant performance.
    Shows which script is winning and by how much.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from campaign.db import get_conn

        with get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    ab_variant,
                    COUNT(*) as calls,
                    SUM(CASE WHEN outcome='converted' THEN 1 ELSE 0 END) as converted,
                    AVG(score) as avg_score,
                    AVG(duration_seconds) as avg_duration
                FROM call_log
                WHERE campaign_id = ? AND ab_variant IS NOT NULL AND ab_variant != ''
                GROUP BY ab_variant
                ORDER BY calls DESC
            """, (campaign_id,)).fetchall()

        if not rows:
            _tg(f"📊 A/B Report Campaign {campaign_id}: No data yet")
            return

        report = [f"🧪 A/B Test Report — Campaign {campaign_id}"]
        best_conversion = 0
        best_variant = ""

        for row in rows:
            calls = row["calls"]
            converted = row["converted"] or 0
            rate = converted / calls * 100 if calls > 0 else 0
            avg_score = row["avg_score"] or 0
            report.append(
                f"\n  Variant: {row['ab_variant']}\n"
                f"  Calls: {calls} | Converted: {converted} ({rate:.1f}%)\n"
                f"  Avg score: {avg_score:.1f}/10 | Avg duration: {row['avg_duration']:.0f}s"
            )
            if rate > best_conversion:
                best_conversion = rate
                best_variant = row["ab_variant"]

        if best_variant:
            report.append(f"\n🏆 WINNER so far: {best_variant} ({best_conversion:.1f}% conversion)")

        _tg("\n".join(report))

    except Exception as e:
        logger.error(f"[REPORTER] A/B report error: {e}")
