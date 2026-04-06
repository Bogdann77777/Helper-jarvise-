"""
marketing_factory/pipeline.py — Orchestrator: FactoryInput → 7 PDFs → Telegram.

Stage flow:
  Stage 1: Market Research  (Perplexity, sync, ~20s)
  Stage 2: VOC              (Perplexity, parallel with Stage 3, ~25s)
  Stage 3: Competitor CI    (Perplexity, parallel with Stage 2)
  Stage 4: Customer Journey Map  (Claude, sync, uses 1+2+3, ~20s)
  Stage 5: Copywriting AGENT    (Claude + Perplexity tools, agentic loop, ~60s)
  Stage 6: Creative Brief AGENT (Claude + Perplexity tools, agentic loop, ~60s)
  Stage 7: Measurement Framework (Claude, sync, uses 1+4+5, ~20s)
  Render:  7 PDFs via fpdf2
  Deliver: Telegram (optional)
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketing_factory.prompts import (
    run_market_research,
    run_voc_research,
    run_competitor_analysis,
    run_customer_journey,
    run_measurement_framework,
    MarketResearchOutput,
    VOCOutput,
    CompetitorAnalysisOutput,
    CustomerJourneyOutput,
    CopywritingOutput,
    CopyVariant,
)
from marketing_factory.copywriting_agent import run_copywriting_agent
from marketing_factory.creative_agent import run_creative_agent
from marketing_factory.pdf_renderer import (
    render_market_research_pdf,
    render_voc_pdf,
    render_competitor_pdf,
    render_journey_pdf,
    render_copywriting_pdf,
    render_creative_pdf,
    render_measurement_pdf,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Input dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FactoryInput:
    product_category: str      # "luxury hair salon services"
    product_description: str   # "Premium keratin treatments, $200-400, Asheville NC"
    market: str                # "Asheville, North Carolina"
    platform: str              # "TikTok"
    duration_sec: int = 30     # video duration for creative brief


# ─────────────────────────────────────────────────────────────────────────────
#  Context builders — compact summaries for Claude stages
# ─────────────────────────────────────────────────────────────────────────────

def voc_to_brief(voc: VOCOutput) -> str:
    """Converts VOCOutput to compact text brief for copy/creative prompts."""
    d = voc.model_dump()
    lines: list[str] = []

    lines.append(f"TARGET: {d.get('market', '')}")

    lines.append("\nTOP PAIN POINTS:")
    for theme in d.get("dominant_pain_points", [])[:5]:
        intensity = theme.get("pain_intensity", 0)
        trend = theme.get("temporal_trend", "")
        lines.append(f"  - {theme.get('theme','')} (intensity {intensity}/10{', trend: '+trend if trend else ''})")
        quotes = theme.get("verbatim_quotes", [])
        if quotes:
            lines.append(f'    Quote: "{quotes[0]}"')

    lines.append("\nBUYING LANGUAGE (use these exact phrases):")
    phrases = d.get("buying_language", [])[:10]
    lines.append("  " + " | ".join(f'"{p}"' for p in phrases))

    forces = d.get("jtbd_forces", {})
    push = forces.get("push_from_current", [])[:3]
    pull = forces.get("pull_to_new", [])[:3]
    if push:
        lines.append("\nJTBD PUSH (away from current):")
        for item in push:
            lines.append(f"  - {item}")
    if pull:
        lines.append("\nJTBD PULL (toward new):")
        for item in pull:
            lines.append(f"  - {item}")

    sentiment = d.get("sentiment_distribution", {})
    if sentiment:
        lines.append(f"\nSENTIMENT: {sentiment.get('five_star_pct',0):.0f}% 5-star praise: '{sentiment.get('most_praised','')}'")
        lines.append(f"  {sentiment.get('one_two_star_pct',0):.0f}% 1-2 star criticism: '{sentiment.get('most_criticized','')}'")

    return "\n".join(lines)


def ci_to_context(ci: CompetitorAnalysisOutput) -> tuple[str, str]:
    """Returns (dominant_narrative, white_space) strings from CI output."""
    d = ci.model_dump()
    dominant = d.get("dominant_ad_narrative", "")
    white_space_list = d.get("white_space_positioning", [])
    white_space = "; ".join(white_space_list[:3]) if white_space_list else ""
    return dominant, white_space


def build_research_context(
    market: MarketResearchOutput,
    voc: VOCOutput,
    ci: CompetitorAnalysisOutput,
) -> str:
    """Builds compact research context string for Claude stages (CJM, Copy, Creative, Measurement)."""
    lines: list[str] = []

    # Market data
    md = market.model_dump()
    lines.append(f"MARKET: {md.get('market_name','')} | TAM=${md.get('tam_usd_billions',0):.1f}B | "
                 f"Stage: {md.get('market_stage','')} | CAGR {md.get('cagr_projected_pct',0):.1f}%")

    segs = md.get("customer_segments", [])[:3]
    if segs:
        lines.append("TOP SEGMENTS:")
        for s in segs:
            psych = s.get("psychographic_profile", "")
            media = ", ".join(s.get("media_consumption", [])[:2]) if s.get("media_consumption") else ""
            lines.append(f"  - {s.get('name','')} ({s.get('size_pct',0):.0f}%) | WTP: {s.get('willingness_to_pay','')} | "
                        f"Psycho: {psych[:80] if psych else 'N/A'} | Media: {media}")

    pestel = md.get("pestel", {})
    if pestel:
        top_pestel = [(k, v) for k, v in pestel.items() if isinstance(v, dict) and v.get("impact_score", 0) >= 7]
        if top_pestel:
            lines.append("HIGH-IMPACT PESTEL FACTORS:")
            for k, v in top_pestel[:3]:
                lines.append(f"  - {k.upper()} ({v['impact_score']}/10): {v.get('implication','')[:80]}")

    # VOC summary
    vd = voc.model_dump()
    pains = vd.get("dominant_pain_points", [])[:3]
    if pains:
        lines.append("TOP CUSTOMER PAINS:")
        for p in pains:
            lines.append(f"  - {p.get('theme','')} | trigger: {p.get('trigger','')[:60]}")

    buying_phrases = vd.get("buying_language", [])[:6]
    if buying_phrases:
        lines.append(f"BUYING LANGUAGE: {' | '.join(buying_phrases)}")

    # CI summary
    cd = ci.model_dump()
    lines.append(f"DOMINANT NARRATIVE: {cd.get('dominant_ad_narrative','')[:120]}")
    white_space = cd.get("white_space_positioning", [])[:3]
    if white_space:
        lines.append(f"WHITE SPACE: {' | '.join(white_space)}")

    return "\n".join(lines)


def build_journey_context(journey: CustomerJourneyOutput) -> str:
    """Compact journey summary for copy/creative/measurement prompts."""
    d = journey.model_dump()
    lines: list[str] = []

    lines.append(f"DOMINANT PERSONA: {d.get('dominant_persona','')}")
    lines.append(f"BIGGEST DROP-OFF: {d.get('biggest_drop_off_stage','')} stage")
    lines.append(f"CRITICAL PATH: {d.get('critical_path','')}")

    lines.append("WINNING SEQUENCE:")
    for i, step in enumerate(d.get("winning_touchpoint_sequence", [])[:5], 1):
        lines.append(f"  {i}. {step}")

    gaps = d.get("competitor_journey_gaps", [])
    if gaps:
        lines.append(f"COMPETITOR GAPS TO EXPLOIT: {' | '.join(gaps[:3])}")

    return "\n".join(lines)


def build_copy_context(copy: CopywritingOutput) -> str:
    """Compact copy summary for measurement prompt (A/B test hypotheses)."""
    d = copy.model_dump()
    lines: list[str] = []

    top = d.get("top_recommendation", {})
    if top:
        lines.append(f"TOP RECOMMENDATION:")
        lines.append(f"  Hook: {top.get('hook','')[:100]}")
        lines.append(f"  Hook type: {top.get('hook_type','')}")
        lines.append(f"  Proof element: {top.get('proof_element','')}")
        lines.append(f"  Score: {top.get('score',0):.1f}/10")

    by_temp = d.get("by_temperature", {})
    for temp in ["cold", "warm", "retarget"]:
        block = by_temp.get(temp, {})
        variants = block.get("variants", [])
        if variants:
            best = max(variants, key=lambda x: x.get("score", 0))
            lines.append(f"{temp.upper()} best: [{best.get('hook_type','')}] {best.get('hook','')[:80]}")

    by_formula = d.get("by_formula", {})
    if by_formula:
        lines.append("FORMULAS TESTED: " + " | ".join(by_formula.keys()))

    return "\n".join(lines)


def build_market_context_for_measurement(market: MarketResearchOutput) -> str:
    """Market data summary for measurement framework prompt."""
    d = market.model_dump()
    lines: list[str] = []

    segs = d.get("customer_segments", [])
    wtp_values = [s.get("willingness_to_pay", "") for s in segs[:3]]
    lines.append(f"MARKET: {d.get('market_name','')} | TAM=${d.get('tam_usd_billions',0):.1f}B | "
                 f"CAGR {d.get('cagr_projected_pct',0):.1f}%")
    lines.append(f"WTP RANGE: {' / '.join(wtp_values)}")
    lines.append(f"MARKET STAGE: {d.get('market_stage','')}")

    # Channel preferences from segments
    channels: list[str] = []
    for s in segs:
        channels.extend(s.get("channel_preference", []))
    unique_channels = list(dict.fromkeys(channels))[:6]
    if unique_channels:
        lines.append(f"KEY CHANNELS FROM SEGMENTS: {', '.join(unique_channels)}")

    seasonality = d.get("seasonality_index", {})
    if seasonality:
        peak = max(seasonality.items(), key=lambda x: x[1]) if seasonality else None
        if peak:
            lines.append(f"PEAK MONTH: {peak[0]} (index {peak[1]})")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def run_factory(
    inp: FactoryInput,
    output_dir: Optional[Path] = None,
    send_telegram: bool = True,
) -> list[Path]:
    """
    Full pipeline: FactoryInput → 7 PDFs → optional Telegram delivery.
    Returns list of 7 PDF paths.
    """
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "factory" / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        print(f"[factory] {msg}", flush=True)

    # ── Stage 1: Market Research ───────────────────────────────────────────
    _log("Stage 1/7 — Market Research (Perplexity)...")
    market = await asyncio.to_thread(
        run_market_research,
        inp.product_category,
        inp.market,
        inp.platform,
    )
    _log(f"  Market: {market.market_name}  TAM=${market.tam_usd_billions:.1f}B  "
         f"CAGR {market.cagr_projected_pct:.1f}%"
         f"{'  PESTEL: ok' if market.pestel else ''}"
         f"{'  Seasonality: ok' if market.seasonality_index else ''}")

    # ── Stages 2+3: VOC + Competitor CI (parallel) ────────────────────────
    _log("Stage 2+3/7 — VOC + Competitor CI (parallel, Perplexity)...")
    voc, ci = await asyncio.gather(
        asyncio.to_thread(
            run_voc_research,
            inp.product_description,
            inp.market,
            inp.product_category,
        ),
        asyncio.to_thread(
            run_competitor_analysis,
            inp.product_category,
            inp.market,
            inp.platform,
        ),
    )
    _log(f"  VOC: {len(voc.dominant_pain_points)} pain themes, "
         f"{len(voc.buying_language)} buying phrases"
         f"{'  Sentiment: ok' if voc.sentiment_distribution else ''}")
    _log(f"  CI: {len(ci.competitors)} competitors"
         f"{'  SOV: ok' if ci.share_of_voice_estimate else ''}"
         f"{'  SWOT: ok' if ci.swot_summary else ''}")

    # ── Stage 4: Customer Journey Map (Claude) ────────────────────────────
    _log("Stage 4/7 — Customer Journey Map (Claude)...")
    research_context = build_research_context(market, voc, ci)
    journey = await asyncio.to_thread(
        run_customer_journey,
        inp.product_description,
        inp.market,
        inp.product_category,
        research_context,
    )
    _log(f"  Journey: {len(journey.journey_stages)} stages  "
         f"Drop-off: {journey.biggest_drop_off_stage}  "
         f"Gaps: {len(journey.competitor_journey_gaps)}")

    # ── Stage 5: Copywriting AGENT (Claude + Perplexity tools) ───────────
    _log("Stage 5/7 — Copywriting AGENT (Claude + live research tools)...")
    voc_brief          = voc_to_brief(voc)
    dominant_narrative, white_space = ci_to_context(ci)
    journey_ctx        = build_journey_context(journey)
    copy = await asyncio.to_thread(
        run_copywriting_agent,
        inp.product_description,
        inp.platform,
        voc_brief,
        dominant_narrative,
        white_space,
        journey_ctx,
        inp.product_category,
        inp.market,
    )
    top = copy.top_recommendation
    _log(f"  Top hook [{top.hook_type}] (score {top.score:.1f}): {top.hook[:60]}...")
    _log(f"  Formulas: {list(copy.by_formula.keys()) if copy.by_formula else 'N/A'}")

    # ── Stage 6: Creative Brief AGENT (Claude + Perplexity tools) ────────
    _log("Stage 6/7 — Creative Brief AGENT (Claude + live research tools)...")
    creative = await asyncio.to_thread(
        run_creative_agent,
        inp.product_description,
        inp.platform,
        voc_brief,
        top,
        inp.duration_sec,
        journey_ctx,
        inp.product_category,
    )
    _log(f"  Concepts: {len(creative.concepts)}  "
         f"Top concept: {creative.top_concept_id}  "
         f"Aspect ratios: {[s.ratio for s in creative.aspect_ratio_specs]}")

    # ── Stage 7: Measurement Framework (Claude) ───────────────────────────
    _log("Stage 7/7 — Measurement Framework (Claude)...")
    market_ctx = build_market_context_for_measurement(market)
    copy_ctx   = build_copy_context(copy)
    measurement = await asyncio.to_thread(
        run_measurement_framework,
        inp.product_description,
        inp.market,
        inp.platform,
        market_ctx,
        copy_ctx,
        journey_ctx,
    )
    _log(f"  CAC target: ${measurement.cac_target_usd:.0f}  "
         f"LTV:CAC {measurement.ltv_cac_ratio_target:.1f}x  "
         f"ROMI {measurement.romi_target_pct:.0f}%  "
         f"A/B tests: {len(measurement.ab_test_roadmap)}")

    # ── Render 7 PDFs ─────────────────────────────────────────────────────
    _log("Rendering 7 PDFs...")
    slug = inp.product_category.replace(" ", "_")[:30]

    pdf_paths = [
        render_market_research_pdf(market, output_dir / f"{slug}_1_market.pdf"),
        render_voc_pdf(voc, output_dir / f"{slug}_2_voc.pdf"),
        render_competitor_pdf(ci, output_dir / f"{slug}_3_ci.pdf"),
        render_journey_pdf(journey, output_dir / f"{slug}_4_journey.pdf"),
        render_copywriting_pdf(copy.model_dump(), output_dir / f"{slug}_5_copy.pdf"),
        render_creative_pdf(creative.model_dump(), output_dir / f"{slug}_6_creative.pdf"),
        render_measurement_pdf(measurement, output_dir / f"{slug}_7_measurement.pdf"),
    ]
    _log(f"  7 PDFs → {output_dir}")

    # ── Telegram delivery ─────────────────────────────────────────────────
    if send_telegram:
        try:
            from tg_send import tg_msg, tg_file
            summary = (
                f"Marketing Factory v2 DONE\n"
                f"Product: {inp.product_description}\n"
                f"Market: {inp.market} / {inp.platform}\n"
                f"TAM: ${market.tam_usd_billions:.1f}B  CAGR: {market.cagr_projected_pct:.1f}%\n"
                f"VOC: {len(voc.dominant_pain_points)} pain points\n"
                f"Journey: {journey.biggest_drop_off_stage} = biggest drop-off\n"
                f"Top hook [{top.hook_type}] score {top.score:.1f}/10: {top.hook[:80]}\n"
                f"CAC target: ${measurement.cac_target_usd:.0f}  LTV:CAC {measurement.ltv_cac_ratio_target:.1f}x\n"
                f"PDFs: {output_dir}"
            )
            tg_msg(summary)
            for p in pdf_paths:
                tg_file(str(p))
            _log("  Telegram: summary + 7 PDFs sent")
        except Exception as e:
            _log(f"  Telegram send failed (non-fatal): {e}")

    return pdf_paths
