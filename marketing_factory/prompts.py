"""
marketing_factory/prompts.py — Production Prompt Map + Pydantic Schemas v2

Stage 1 — Market Research        (Perplexity sonar-pro)
  Framework: McKinsey Pyramid + TAM/SAM/SOM + PESTEL + Seasonality
  Output: market sizing, CAGR, 3-5 segments with psychographics, PESTEL, seasonality index

Stage 2 — Voice of Customer      (Perplexity sonar-pro)
  Framework: Gartner VOC + JTBD 4-Forces + Sentiment Distribution
  Sources: Reddit, Yelp, Google Reviews, App Store, booking platforms
  Min quality: 300+ public data points synthesized, 10+ themes, verbatim quotes

Stage 3 — Competitor Intelligence (Perplexity sonar-pro)
  Framework: CI Intelligence Cycle + Porter's Five Forces + SWOT
  Sources: Meta Ad Library, TikTok Creative Center, Google Ads, brand sites

Stage 4 — Customer Journey Map   (Claude via OpenRouter) NEW
  Framework: McKinsey CJM + RACE + Moments of Truth
  5 stages × 3+ touchpoints each, competitor journey gaps, winning sequence

Stage 5 — Copywriting            (Claude via OpenRouter, was Perplexity)
  Framework: AIDA + PAS + BAB + ELM, 9-hook taxonomy, proof elements, urgency

Stage 6 — Creative Brief         (Claude via OpenRouter, was Perplexity)
  Framework: Performance Creative Director, 2-3 concepts, aspect ratios, benchmarks

Stage 7 — Measurement Framework  (Claude via OpenRouter) NEW
  Framework: Blended MMM + MTA, CAC/LTV/ROMI, A/B test roadmap, 30/60/90 milestones
"""

from __future__ import annotations

import json
import time
import requests
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, model_validator

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PERPLEXITY_API_KEY, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# ─── API constants ────────────────────────────────────────────────────────────
SONAR_PRO          = "sonar-pro"
CLAUDE_MODEL       = "anthropic/claude-sonnet-4-6"   # via OpenRouter (confirmed working)
MAX_RETRIES        = 3
SCHEMA_TIMEOUT     = 90
FACTORY_MAX_TOKENS = 4000   # Perplexity stages 1-3
CLAUDE_MAX_TOKENS  = 8000   # Claude stages 4-7 (more room for quality)


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 1 — MARKET RESEARCH (enhanced)
#  McKinsey Pyramid + TAM/SAM/SOM + PESTEL + Seasonality + Psychographics
# ─────────────────────────────────────────────────────────────────────────────

class CustomerSegment(BaseModel):
    name: str
    size_pct: float                    = Field(..., ge=0, le=100)
    purchase_drivers: list[str]        = Field(..., min_length=3)
    willingness_to_pay: str
    channel_preference: list[str]      = Field(..., min_length=1)
    psychographic_profile: Optional[str] = None   # NEW: values, identity, lifestyle
    media_consumption: Optional[list[str]] = None  # NEW: ["TikTok 2h/day", "Instagram 1h/day"]


class MarketResearchOutput(BaseModel):
    market_name: str
    tam_usd_billions: float  = Field(..., description="Total Addressable Market, USD billions")
    sam_usd_billions: float  = Field(..., description="Serviceable Addressable Market, USD billions")
    som_usd_millions: float  = Field(..., description="Serviceable Obtainable Market, USD millions")
    cagr_historical_pct: float
    cagr_projected_pct: float
    market_stage: str        = Field(..., pattern="^(emerging|growth|mature|declining)$")
    top_purchase_drivers: list[str]   = Field(..., min_length=5)
    customer_segments: list[CustomerSegment] = Field(..., min_length=3)
    key_market_trends: list[str]      = Field(..., min_length=3)
    strategic_implications: list[str] = Field(..., min_length=3, max_length=3)
    data_confidence: float            = Field(..., ge=0.0, le=1.0)
    pestel: Optional[dict]            = None   # NEW: {political: {impact_score, implication}, ...}
    seasonality_index: Optional[dict] = None   # NEW: {Jan: 80, Feb: 90, ..., Dec: 120}

    @model_validator(mode="after")
    def tam_gte_sam(self) -> "MarketResearchOutput":
        if self.tam_usd_billions < self.sam_usd_billions:
            raise ValueError("TAM must be >= SAM")
        return self


MARKET_RESEARCH_SYSTEM = """\
You are a McKinsey Senior Partner specializing in market intelligence and go-to-market strategy.
Your methodology: Pyramid Principle (conclusion first, then data), TAM/SAM/SOM via both
top-down (industry reports) AND bottom-up (unit economics) approaches, evidence-based.
Quality bar: Fortune 500 board presentation. Every number needs a source basis.

CRITICAL OUTPUT RULE: Return ONLY a JSON object — no prose, no markdown, no explanation.
The JSON must match this exact schema:

{
  "market_name": "string",
  "tam_usd_billions": <number>,
  "sam_usd_billions": <number>,
  "som_usd_millions": <number>,
  "cagr_historical_pct": <number>,
  "cagr_projected_pct": <number>,
  "market_stage": "emerging|growth|mature|declining",
  "top_purchase_drivers": ["string", ...],
  "customer_segments": [
    {
      "name": "string",
      "size_pct": <number 0-100>,
      "purchase_drivers": ["string", ...],
      "willingness_to_pay": "string",
      "channel_preference": ["string", ...],
      "psychographic_profile": "string — values, identity, lifestyle summary",
      "media_consumption": ["TikTok 2h/day", "Instagram 1h/day", ...]
    }
  ],
  "key_market_trends": ["string", ...],
  "strategic_implications": ["string", "string", "string"],
  "data_confidence": <number 0.0-1.0>,
  "pestel": {
    "political": {"impact_score": <1-10>, "implication": "string"},
    "economic": {"impact_score": <1-10>, "implication": "string"},
    "social": {"impact_score": <1-10>, "implication": "string"},
    "technological": {"impact_score": <1-10>, "implication": "string"},
    "environmental": {"impact_score": <1-10>, "implication": "string"},
    "legal": {"impact_score": <1-10>, "implication": "string"}
  },
  "seasonality_index": {"Jan": <50-200>, "Feb": <...>, ..., "Dec": <...>}
}

JSON only. No text before or after the JSON object.\
"""

MARKET_RESEARCH_USER = """\
Research the {product_category} market in {market} for {platform} advertising.

Required data points:

1. MARKET SIZING (both methodologies):
   - Top-down: industry reports → narrow to geography/demo
   - Bottom-up: (# potential customers) x (avg annual spend)
   - TAM = global, SAM = {market} geography + target demo, SOM = capturable in 12 months

2. GROWTH RATE:
   - Historical CAGR last 3 years, Projected CAGR next 5 years, Market stage

3. CUSTOMER SEGMENTS (minimum 3):
   - Name + size % of market
   - 3-6 purchase drivers per segment (functional + emotional + social)
   - Willingness-to-pay range
   - Preferred marketing channels
   - Psychographic profile: core values, identity aspirations, lifestyle markers
   - Media consumption: specific platforms + estimated daily hours

4. PURCHASE DRIVERS (minimum 5, ranked by impact):
   Include rational (quality, convenience, price) AND emotional (status, identity, fear)

5. MARKET TRENDS (minimum 3):
   Macro forces shaping buyer behavior over next 2-3 years

6. STRATEGIC IMPLICATIONS (exactly 3, Pyramid Principle format):
   Start with SO WHAT conclusion, then supporting evidence

7. PESTEL SCAN (all 6 forces):
   - Political: regulations, licensing, government programs
   - Economic: disposable income trends, price sensitivity, recession sensitivity
   - Social: demographic shifts, cultural drivers, lifestyle changes
   - Technological: new channels, AI disruption, platform changes
   - Environmental: sustainability pressure, eco-credentials
   - Legal: compliance requirements, data privacy, advertising regulations
   Rate each 1-10 for impact and state the strategic implication.

8. SEASONALITY INDEX (monthly demand, 100 = annual average):
   Jan through Dec — what months are peak vs trough?

Return strict JSON only.\
"""

MARKET_RESEARCH_SCHEMA = MarketResearchOutput.model_json_schema()


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 2 — VOICE OF CUSTOMER (enhanced)
#  Gartner VOC + JTBD 4-Forces + Sentiment Distribution + Temporal Trends
# ─────────────────────────────────────────────────────────────────────────────

class VOCTheme(BaseModel):
    theme: str
    category: str        = Field(..., pattern="^(pain_point|desire|fear|job_to_be_done)$")
    pain_intensity: int  = Field(..., ge=1, le=10)
    frequency_score: int = Field(..., ge=1, le=10)
    verbatim_quotes: list[str]  = Field(..., min_length=2)
    exact_phrases: list[str]    = Field(..., min_length=2)
    jtbd_job: Optional[str]     = None
    trigger: Optional[str]      = None
    segment_correlation: Optional[list[str]] = None   # NEW: which segments feel this most
    temporal_trend: Optional[str] = None              # NEW: "growing"|"stable"|"declining"


class JTBDForces(BaseModel):
    push_from_current: list[str] = Field(..., min_length=3)
    pull_to_new: list[str]       = Field(..., min_length=3)
    anxiety_about_new: list[str] = Field(..., min_length=2)
    habit_inertia: list[str]     = Field(..., min_length=2)


class VOCOutput(BaseModel):
    product_description: str
    market: str
    total_themes_analyzed: int  = Field(..., ge=10)
    dominant_pain_points: list[VOCTheme] = Field(..., min_length=5)
    top_desires: list[VOCTheme]          = Field(..., min_length=3)
    key_fears: list[VOCTheme]            = Field(..., min_length=2)
    jtbd_jobs: list[VOCTheme]            = Field(..., min_length=3)
    jtbd_forces: JTBDForces
    buying_language: list[str]           = Field(..., min_length=10)
    language_to_avoid: list[str]         = Field(..., min_length=5)
    sources_synthesized: list[str]       = Field(..., min_length=3)
    sentiment_distribution: Optional[dict] = None  # NEW: {five_star_pct, one_two_star_pct, most_praised, most_criticized}


VOC_SYSTEM = """\
You are a Gartner-certified VOC Research Director with 15 years of consumer insights experience.
Your methodology: multi-channel synthesis (Direct + Indirect + Inferred signals), JTBD 4-Forces
(Push/Pull/Anxiety/Habit from Switch Interview framework by Bob Moesta).
Quality bar: 300+ public data points synthesized, 10+ distinct themes, all quotes verbatim.

SOURCE HIERARCHY (search all channels):
  Direct: App Store reviews, Google Reviews, Yelp, Trustpilot, booking platform reviews
  Indirect: Reddit discussions, Facebook Groups, YouTube comments, Twitter/X threads
  Inferred: behavioral triggers, switching patterns, search intent signals

CRITICAL OUTPUT RULE: Return ONLY a JSON object — no prose, no markdown, no explanation.
The JSON must match this exact schema:

{
  "product_description": "string",
  "market": "string",
  "total_themes_analyzed": <integer, min 10>,
  "dominant_pain_points": [
    {
      "theme": "string",
      "category": "pain_point|desire|fear|job_to_be_done",
      "pain_intensity": <1-10>,
      "frequency_score": <1-10>,
      "verbatim_quotes": ["exact customer words", ...],
      "exact_phrases": ["specific vocabulary", ...],
      "jtbd_job": "When X, I want to Y, so I can Z",
      "trigger": "event or situation that prompted search",
      "segment_correlation": ["segment name 1", "segment name 2"],
      "temporal_trend": "growing|stable|declining"
    }
  ],
  "top_desires": [...],
  "key_fears": [...],
  "jtbd_jobs": [...],
  "jtbd_forces": {
    "push_from_current": ["string", ...],
    "pull_to_new": ["string", ...],
    "anxiety_about_new": ["string", ...],
    "habit_inertia": ["string", ...]
  },
  "buying_language": ["exact phrase", ...],
  "language_to_avoid": ["clinical word", ...],
  "sources_synthesized": ["Reddit/r/hair", "Yelp reviews Asheville NC", ...],
  "sentiment_distribution": {
    "five_star_pct": <number>,
    "one_two_star_pct": <number>,
    "most_praised": "string",
    "most_criticized": "string"
  }
}

JSON only. All quotes must be real customer language — never paraphrase or invent.\
"""

VOC_USER = """\
Conduct Voice of Customer research for {product_description} in {market}.

Search across ALL channels (mandatory):

DIRECT SOURCES:
- Yelp: search "{product_description} {market}" — read reviews, filter 1-3 stars AND 4-5 stars
- Google Reviews: top 5 competitors in {market}
- Booking platforms: Vagaro, StyleSeat, Fresha — reviews for category in {market}
- App Store / Google Play: service apps in this category

INDIRECT SOURCES:
- Reddit: r/[relevant subreddit] + r/AskReddit for "{product_category} {market}"
- Facebook Groups: local community groups in {market}
- YouTube comments: tutorial/review videos in this category

SYNTHESIS REQUIREMENTS:

1. PAIN POINTS (minimum 5, score each 1-10 for intensity AND frequency):
   - What frustrates customers about current {product_category} options?
   - What triggers switching from one provider to another?
   Include 2+ verbatim quotes and 2+ exact phrases per theme.
   Add: which customer segments feel this most (segment_correlation)
   Add: temporal_trend — is this pain growing, stable, or declining over 12 months?

2. DESIRES (minimum 3): What makes a 5-star review?

3. FEARS (minimum 2): What stops them from trying new providers?

4. JOBS TO BE DONE (minimum 3, JTBD format):
   "When [situation], I want to [motivation], so I can [outcome]"
   Cover: functional + emotional + social job

5. JTBD 4-FORCES:
   Push (3+) / Pull (3+) / Anxiety (2+) / Habit (2+)

6. CUSTOMER LANGUAGE:
   10+ exact phrases they use | 5+ clinical phrases they find inauthentic

7. SENTIMENT DISTRIBUTION:
   - What % of reviews are 5-star? What are they celebrating?
   - What % are 1-2 star? What is the single biggest complaint?

Report total themes analyzed (must be 10+). Return strict JSON only.\
"""

VOC_SCHEMA = VOCOutput.model_json_schema()


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 3 — COMPETITOR INTELLIGENCE (enhanced)
#  CI Intelligence Cycle + Porter's Five Forces + SWOT + SOV + Creative Fatigue
# ─────────────────────────────────────────────────────────────────────────────

class CompetitorProfile(BaseModel):
    name: str
    market_position: str          = Field(..., pattern="^(leader|challenger|niche|new_entrant)$")
    primary_ad_angle: str
    messaging_pillars: list[str]  = Field(..., min_length=3)
    channel_mix: list[str]        = Field(..., min_length=2)
    price_positioning: str        = Field(..., pattern="^(budget|mid-market|premium|luxury)$")
    target_audience: str
    key_strengths: list[str]      = Field(..., min_length=2)
    critical_weaknesses: list[str] = Field(..., min_length=2)
    opportunity_gap: str
    sample_ad_hooks: list[str]    = Field(..., min_length=2)
    creative_fatigue_months: Optional[int] = None    # NEW: est. months running same creative
    customer_complaints: Optional[list[str]] = None  # NEW: what their customers complain about


class CompetitorAnalysisOutput(BaseModel):
    product_category: str
    market: str
    analysis_date: str
    competitive_intensity: str    = Field(..., pattern="^(low|medium|high|very_high)$")
    market_leader: str
    competitors: list[CompetitorProfile] = Field(..., min_length=5)
    dominant_ad_narrative: str
    white_space_positioning: list[str] = Field(..., min_length=3)
    recommended_differentiation: str
    porter_five_forces_summary: dict
    share_of_voice_estimate: Optional[dict] = None   # NEW: {brand: pct_estimate}
    swot_summary: Optional[dict] = None              # NEW: {strengths, weaknesses, opportunities, threats}


CI_SYSTEM = """\
You are a Competitive Intelligence Director applying the CI Intelligence Cycle methodology.
Framework: Plan → Collect → Analyze → Disseminate with Porter's Five Forces overlay.
Quality bar: actionable strategic intelligence. Most valuable output = what competitors are NOT doing.

SOURCE PRIORITY:
  1. Meta Ad Library — active ads right now
  2. TikTok Creative Center — top performing ads in category
  3. Google search ads — competitor headlines
  4. Competitor websites — core value propositions
  5. Industry news — recent campaigns, pricing changes

CRITICAL OUTPUT RULE: Return ONLY a JSON object — no prose, no markdown, no explanation.
The JSON must match this exact schema:

{
  "product_category": "string",
  "market": "string",
  "analysis_date": "YYYY-MM-DD",
  "competitive_intensity": "low|medium|high|very_high",
  "market_leader": "string",
  "competitors": [
    {
      "name": "string",
      "market_position": "leader|challenger|niche|new_entrant",
      "primary_ad_angle": "string",
      "messaging_pillars": ["string", ...],
      "channel_mix": ["TikTok (40%)", "Instagram (35%)", ...],
      "price_positioning": "budget|mid-market|premium|luxury",
      "target_audience": "string",
      "key_strengths": ["string", ...],
      "critical_weaknesses": ["string", ...],
      "opportunity_gap": "string",
      "sample_ad_hooks": ["hook1", "hook2"],
      "creative_fatigue_months": <integer, estimate>,
      "customer_complaints": ["what their customers say negatively", ...]
    }
  ],
  "dominant_ad_narrative": "string",
  "white_space_positioning": ["angle1", "angle2", "angle3"],
  "recommended_differentiation": "string",
  "porter_five_forces_summary": {
    "supplier_power": "low|medium|high",
    "buyer_power": "low|medium|high",
    "new_entrants": "low|medium|high",
    "substitutes": "low|medium|high",
    "rivalry": "low|medium|high"
  },
  "share_of_voice_estimate": {"Brand A": 35, "Brand B": 25, "Brand C": 20, "Others": 20},
  "swot_summary": {
    "strengths": ["market opportunity strength 1", ...],
    "weaknesses": ["entering challenger weakness 1", ...],
    "opportunities": ["market gap 1", ...],
    "threats": ["competitive threat 1", ...]
  }
}

JSON only.\
"""

CI_USER = """\
Conduct competitive intelligence analysis for {product_category} in {market}, \
focusing on {platform} advertising.

COLLECTION (all sources):
1. Meta Ad Library: active ads for "{product_category}" in "{market}"
2. TikTok Creative Center: category trends, top ads, hooks being tested
3. Google: "{product_category} {market}" — read ads and landing page headlines
4. Top 5 competitor websites: hero message, unique claim, price point
5. Recent news: funding rounds, new entrants, pricing changes

ANALYSIS REQUIREMENTS (minimum 5 competitors):

For EACH competitor:
  - Primary ad angle: most repeated positioning claim
  - Messaging pillars: 3-5 themes consistently hit
  - Channel mix: where they advertise + estimated % allocation
  - Strengths: what they do well
  - Critical weaknesses: what they IGNORE or do POORLY (this is our attack vector)
  - Opportunity gap: customer need they leave unaddressed
  - Sample ad hooks: 2 actual headlines from their ads
  - Creative fatigue: estimate months running same creative style
  - Customer complaints: what do THEIR customers complain about in reviews?

SYNTHESIS:

1. Dominant narrative: what story does the ENTIRE market tell? (sea of sameness)

2. White space: minimum 3 positioning angles NO competitor currently owns
   Based on: gaps in messaging + unmet needs from VOC

3. Recommended differentiation: single most defensible angle
   Must be: unclaimed + customer-demanded + ownable long-term

4. Porter's Five Forces: calibration of competitive dynamics

5. Share of Voice estimate: rough % estimate for top competitors

6. SWOT summary: for a new entrant to this market

Return strict JSON only.\
"""

CI_SCHEMA = CompetitorAnalysisOutput.model_json_schema()


# ─────────────────────────────────────────────────────────────────────────────
#  PERPLEXITY STRUCTURED CALL — with JSON Schema + Pydantic validation
# ─────────────────────────────────────────────────────────────────────────────

def _call_perplexity_structured(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    schema_name: str,
    timeout: int = SCHEMA_TIMEOUT,
) -> dict:
    payload = {
        "model": SONAR_PRO,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": FACTORY_MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
            },
        },
    }
    resp = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    if "<think>" in content:
        content = content.split("</think>")[-1].strip()
    return json.loads(content)


def _research_with_validation(
    system_prompt: str,
    user_prompt: str,
    json_schema: dict,
    schema_name: str,
    pydantic_model: type[BaseModel],
) -> tuple[BaseModel, list[str]]:
    from pydantic import ValidationError

    last_error: str = ""
    citations: list[str] = []

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = user_prompt
        if last_error:
            prompt = (
                f"{user_prompt}\n\n"
                f"PREVIOUS ATTEMPT FAILED VALIDATION. Fix these errors:\n{last_error}\n"
                f"Return corrected JSON only."
            )

        print(f"[factory/prompts] {schema_name} attempt {attempt}/{MAX_RETRIES}...")
        try:
            raw = _call_perplexity_structured(
                system_prompt, prompt, json_schema, schema_name
            )
            result = pydantic_model.model_validate(raw)
            print(f"[factory/prompts] {schema_name} ✓ validated")
            return result, citations
        except requests.exceptions.Timeout:
            last_error = "Request timed out. Return shorter but complete JSON."
            print(f"[factory/prompts] {schema_name} timeout (attempt {attempt})")
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}. Return valid JSON only."
            print(f"[factory/prompts] {schema_name} JSON error: {e}")
        except ValidationError as e:
            last_error = str(e)
            print(f"[factory/prompts] {schema_name} validation error: {e}")
        except requests.HTTPError as e:
            raise RuntimeError(f"Perplexity API error: {e}") from e

        if attempt < MAX_RETRIES:
            time.sleep(2)

    raise RuntimeError(
        f"{schema_name}: failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLAUDE / OPENROUTER CALL — for stages 4-7
#  Uses response_format: json_object + Pydantic validation
# ─────────────────────────────────────────────────────────────────────────────

def _call_openrouter_claude(
    system_prompt: str,
    user_prompt: str,
    timeout: int = SCHEMA_TIMEOUT,
) -> dict:
    """Call Claude via OpenRouter with JSON output enforcement."""
    payload = {
        "model": CLAUDE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": CLAUDE_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        OPENROUTER_BASE_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # Strip markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return json.loads(content)


def _claude_with_validation(
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    pydantic_model: type[BaseModel],
) -> BaseModel:
    """Claude API call via OpenRouter with Pydantic validation + retry (max 3)."""
    from pydantic import ValidationError

    last_error: str = ""

    for attempt in range(1, MAX_RETRIES + 1):
        prompt = user_prompt
        if last_error:
            prompt = (
                f"{user_prompt}\n\n"
                f"PREVIOUS ATTEMPT FAILED VALIDATION. Fix these errors:\n{last_error}\n"
                f"Return corrected JSON only."
            )

        print(f"[factory/claude] {schema_name} attempt {attempt}/{MAX_RETRIES}...")
        try:
            raw = _call_openrouter_claude(system_prompt, prompt)
            result = pydantic_model.model_validate(raw)
            print(f"[factory/claude] {schema_name} ✓ validated")
            return result
        except requests.exceptions.Timeout:
            last_error = "Request timed out. Return shorter but complete JSON."
            print(f"[factory/claude] {schema_name} timeout (attempt {attempt})")
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}. Return valid JSON only."
            print(f"[factory/claude] {schema_name} JSON error: {e}")
        except ValidationError as e:
            last_error = str(e)
            print(f"[factory/claude] {schema_name} validation error: {e}")
        except requests.HTTPError as e:
            raise RuntimeError(f"OpenRouter API error: {e}") from e

        if attempt < MAX_RETRIES:
            time.sleep(2)

    raise RuntimeError(
        f"{schema_name}: failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — Perplexity stages (1-3)
# ─────────────────────────────────────────────────────────────────────────────

def run_market_research(
    product_category: str,
    market: str,
    platform: str,
) -> MarketResearchOutput:
    """Stage 1: Market sizing + segments + PESTEL + seasonality."""
    return _research_with_validation(
        system_prompt=MARKET_RESEARCH_SYSTEM,
        user_prompt=MARKET_RESEARCH_USER.format(
            product_category=product_category,
            market=market,
            platform=platform,
        ),
        json_schema=MARKET_RESEARCH_SCHEMA,
        schema_name="marketresearch",
        pydantic_model=MarketResearchOutput,
    )[0]


def run_voc_research(
    product_description: str,
    market: str,
    product_category: str,
) -> VOCOutput:
    """Stage 2: Voice of Customer — JTBD, pain points, verbatim language."""
    return _research_with_validation(
        system_prompt=VOC_SYSTEM,
        user_prompt=VOC_USER.format(
            product_description=product_description,
            market=market,
            product_category=product_category,
        ),
        json_schema=VOC_SCHEMA,
        schema_name="vocresearch",
        pydantic_model=VOCOutput,
    )[0]


def run_competitor_analysis(
    product_category: str,
    market: str,
    platform: str,
) -> CompetitorAnalysisOutput:
    """Stage 3: Competitor intel — ad angles, white space, SOV, SWOT."""
    return _research_with_validation(
        system_prompt=CI_SYSTEM,
        user_prompt=CI_USER.format(
            product_category=product_category,
            market=market,
            platform=platform,
        ),
        json_schema=CI_SCHEMA,
        schema_name="competitoranalysis",
        pydantic_model=CompetitorAnalysisOutput,
    )[0]


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 4 — CUSTOMER JOURNEY MAP (NEW — Claude)
#  McKinsey CJM + RACE Framework + Moments of Truth
# ─────────────────────────────────────────────────────────────────────────────

class TouchpointDetail(BaseModel):
    channel: str              # "Instagram Reels", "Google Search"
    message: str              # brand message at this touchpoint
    customer_emotion: str     # "curious", "skeptical", "excited"
    friction_point: str       # main barrier at this touchpoint
    kpi: str                  # "CTR", "Time on page", "Conversion rate"
    optimization_action: str  # what specifically improves this touchpoint


class JourneyStage(BaseModel):
    stage_name: str                             # Awareness|Consideration|Decision|Retention|Advocacy
    customer_mindset: str                       # internal monologue — exact words
    time_to_next_stage: str                     # "2-7 days", "same session"
    touchpoints: list[TouchpointDetail]         = Field(..., min_length=3)
    moment_of_truth: str                        # the critical decision point in this stage
    drop_off_risk: str                          # what kills conversion at this stage
    winning_content_type: str                   # "UGC testimonial video", "social proof screenshot"


class CustomerJourneyOutput(BaseModel):
    product_category: str
    market: str
    dominant_persona: str                       # primary customer archetype
    journey_stages: list[JourneyStage]          = Field(..., min_length=5)
    critical_path: str                          # shortest path from awareness to purchase
    biggest_drop_off_stage: str                 # where most customers abandon
    winning_touchpoint_sequence: list[str]      = Field(..., min_length=5)  # optimal sequence
    competitor_journey_gaps: list[str]          = Field(..., min_length=3)  # what competitors miss


CJM_SYSTEM = """\
You are a McKinsey Customer Experience Practice Lead with 15 years of B2C journey design.
Your framework: RACE (Reach-Act-Convert-Engage) + Moments of Truth (Zero/First/Second/Ultimate).
You map EVERY touchpoint where a customer interacts with a brand, and identify WHERE they leave.

The most valuable output: WHERE customers abandon and WHAT competitors are failing to do at each stage.

CRITICAL OUTPUT RULE: Return ONLY a JSON object. No prose, no markdown.

JSON schema:
{
  "product_category": "string",
  "market": "string",
  "dominant_persona": "string (primary customer archetype from the segments)",
  "journey_stages": [
    {
      "stage_name": "Awareness|Consideration|Decision|Retention|Advocacy",
      "customer_mindset": "exact internal monologue — 'I wonder if...' or 'I need to find...'",
      "time_to_next_stage": "string (e.g. '2-7 days' or 'same session')",
      "touchpoints": [
        {
          "channel": "specific platform or format",
          "message": "what the brand communicates here",
          "customer_emotion": "one word emotion",
          "friction_point": "what stops or delays them",
          "kpi": "metric to track success",
          "optimization_action": "single most impactful improvement"
        }
      ],
      "moment_of_truth": "the critical make-or-break decision point",
      "drop_off_risk": "specific reason most customers leave at this stage",
      "winning_content_type": "most effective content format for this stage"
    }
  ],
  "critical_path": "shortest sequence from first touch to purchase",
  "biggest_drop_off_stage": "stage name where most customers abandon",
  "winning_touchpoint_sequence": ["step 1", "step 2", "step 3", "step 4", "step 5"],
  "competitor_journey_gaps": ["gap 1 (touchpoint competitors miss)", "gap 2", "gap 3"]
}

JSON only. Be specific — no generic marketing platitudes.\
"""

def _build_cjm_user_prompt(
    product_description: str,
    market: str,
    product_category: str,
    research_context: str,
) -> str:
    return f"""\
Map the complete customer journey for: {product_description}
Market: {market} | Category: {product_category}

RESEARCH CONTEXT (use this data to ground your journey map):
{research_context}

TASK: Build a precise, data-driven customer journey map.

For each of the 5 stages (Awareness → Consideration → Decision → Retention → Advocacy):

AWARENESS stage:
- How does this customer first encounter this category? (trigger events)
- Which platforms? What search queries? What triggers the need?
- Zero Moment of Truth: what online research do they do BEFORE engaging?
- Drop-off risk: why do many never move past awareness?

CONSIDERATION stage:
- What criteria do they compare? (make it specific to this product/market)
- Which review sites? Which friends? Which content?
- First Moment of Truth: seeing the product for real — what makes them consider?
- Drop-off risk: what makes them choose a competitor?

DECISION stage:
- What is the final trigger? (urgency, social proof, price, guarantee?)
- Second Moment of Truth: the purchase experience
- Drop-off risk: what cart abandonment or last-minute doubt occurs?

RETENTION stage:
- Post-purchase: what keeps them coming back?
- What defines a 5-star experience vs. a 1-star complaint?
- Drop-off risk: what causes churn?

ADVOCACY stage:
- What prompts them to refer? Review? Share?
- Ultimate Moment of Truth: what makes them a brand evangelist?
- Drop-off risk: why do satisfied customers NOT refer?

COMPETITOR GAPS:
- Which 3+ touchpoints are competitors systematically missing?
- Where is there white space in the journey for differentiation?

WINNING SEQUENCE:
- What is the optimal 5-step touchpoint sequence for fastest path to purchase?
- Ground this in the VOC data (actual customer language and triggers)

Return strict JSON only.\
"""


def run_customer_journey(
    product_description: str,
    market: str,
    product_category: str,
    research_context: str,
) -> CustomerJourneyOutput:
    """Stage 4: Customer Journey Map — 5 stages × 3+ touchpoints, competitor gaps."""
    return _claude_with_validation(
        system_prompt=CJM_SYSTEM,
        user_prompt=_build_cjm_user_prompt(
            product_description, market, product_category, research_context
        ),
        schema_name="customerjourneymap",
        pydantic_model=CustomerJourneyOutput,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 5 — AD COPYWRITING (enhanced — now Claude via OpenRouter)
#  AIDA + PAS + BAB + ELM + 9-hook taxonomy + proof elements + urgency
# ─────────────────────────────────────────────────────────────────────────────

class CopyVariant(BaseModel):
    hook: str
    body: str
    cta: str
    emotion: str
    score: float       = Field(..., ge=0.0, le=10.0)
    hook_type: str     # one of 9 types (see system prompt)
    proof_element: str  # "87% see results in 2 weeks" or "Before: X | After: Y"
    urgency_mechanic: str  # "Only 3 spots left" or "Limited time offer" or "none"
    ad_angle: str      # pain | transformation | authority | social_proof


class TemperatureBlock(BaseModel):
    elm_route: str
    variants: list[CopyVariant] = Field(..., min_length=2, max_length=4)


class CopyMetadata(BaseModel):
    platform: str
    framework_used: str
    total_variants_generated: int


class ByTemperature(BaseModel):
    cold: TemperatureBlock
    warm: TemperatureBlock
    retarget: TemperatureBlock


class FormulaBlock(BaseModel):
    formula: str              # "AIDA" | "PAS" | "BAB"
    audience_awareness: str   # "unaware" | "problem_aware" | "solution_aware"
    variants: list[CopyVariant] = Field(..., min_length=2)


class CopywritingOutput(BaseModel):
    metadata: CopyMetadata
    top_recommendation: CopyVariant
    by_temperature: ByTemperature
    by_formula: dict          # {"AIDA": FormulaBlock, "PAS": FormulaBlock, "BAB": FormulaBlock}


COPY_SYSTEM = """\
You are a direct-response copywriter at the level of Gary Halbert and Eugene Schwartz.
Your copy converts because it speaks the customer's exact language, names their exact pain,
and makes them feel understood before asking for anything.

FRAMEWORK: AIDA + PAS + BAB + ELM (Elaboration Likelihood Model)
- Cold audience (unaware): AIDA + ELM peripheral route — pattern-interrupt, emotion, identity
- Warm audience (problem-aware): PAS — problem agitate solution, ELM central route
- Retarget (solution-aware): BAB — before/after/bridge + urgency + loss aversion
- Never use one formula for all audiences. Diagnose awareness first.

9 HOOK TYPES (always label each variant):
  1. pain_agitation — "You've tried everything and nothing works..."
  2. transformation_reveal — "From [bad state] to [dream state] in [timeframe]"
  3. social_proof — "10,000 women in [city] use this to..."
  4. curiosity_gap — "The one thing [category] experts don't tell you"
  5. pattern_interrupt — starts with unexpected or provocative statement
  6. authority — "After 15 years and 1000 clients..."
  7. loss_aversion — "Every day you wait, [negative consequence]"
  8. number_statistic — "87% of [target] report [outcome] within [timeframe]"
  9. question — direct question that forces mental engagement

PROOF ELEMENTS (required on every variant):
  - Statistics: "87% see results in 2 weeks" (use real numbers if VOC provides them)
  - Before/after: "Before: [pain] | After: [outcome]"
  - Social validation: "Join 2,000+ [customers] in [market]"

URGENCY MECHANICS (required on retarget variants):
  - Scarcity: "Only 3 spots left this month"
  - Deadline: "Offer expires [timeframe]"
  - Loss aversion: "Every [day/week] without this costs you [consequence]"
  - FOMO: "While spots are still available"

CRITICAL OUTPUT RULE: Return ONLY a JSON object. No prose, no markdown.
"""

def _build_copy_user_prompt(
    product_description: str,
    platform: str,
    voc_brief: str,
    dominant_narrative: str,
    white_space: str,
    journey_context: str,
) -> str:
    return f"""\
Create {platform} ad copy for: {product_description}

VOC INTELLIGENCE (use these exact phrases and address these exact pains):
{voc_brief}

COMPETITOR CONTEXT:
Dominant narrative (break away from this): {dominant_narrative}
White space to exploit: {white_space}

CUSTOMER JOURNEY CONTEXT:
{journey_context}

Platform: {platform}

DELIVERABLES:

1. BY TEMPERATURE (cold/warm/retarget):
   - Cold: 2-4 variants using AIDA, hook types: pain_agitation + curiosity_gap minimum
   - Warm: 2-4 variants using PAS, hook types: transformation_reveal + number_statistic minimum
   - Retarget: 2-4 variants using BAB, hook types: loss_aversion + social_proof minimum
   Every variant MUST have: hook_type, proof_element, urgency_mechanic, ad_angle

2. BY FORMULA:
   - AIDA block (2+ variants) — for unaware cold audience
   - PAS block (2+ variants) — for problem-aware audience
   - BAB block (2+ variants) — for solution-aware audience
   Each block: formula name + audience_awareness + variants

3. TOP RECOMMENDATION: best single variant across all (highest conversion potential)

JSON schema to follow exactly:
{{
  "metadata": {{
    "platform": "{platform}",
    "framework_used": "AIDA+PAS+BAB+ELM",
    "total_variants_generated": <integer>
  }},
  "top_recommendation": {{
    "hook": "<first 3s / headline>",
    "body": "<main copy 2-3 sentences>",
    "cta": "<call to action>",
    "emotion": "<primary emotion>",
    "score": <0.0-10.0>,
    "hook_type": "<one of 9 types>",
    "proof_element": "<specific proof>",
    "urgency_mechanic": "<urgency or 'none'>",
    "ad_angle": "pain|transformation|authority|social_proof"
  }},
  "by_temperature": {{
    "cold": {{
      "elm_route": "peripheral",
      "variants": [{{...}}, ...]
    }},
    "warm": {{
      "elm_route": "central",
      "variants": [{{...}}, ...]
    }},
    "retarget": {{
      "elm_route": "peripheral",
      "variants": [{{...}}, ...]
    }}
  }},
  "by_formula": {{
    "AIDA": {{
      "formula": "AIDA",
      "audience_awareness": "unaware",
      "variants": [{{...}}, ...]
    }},
    "PAS": {{
      "formula": "PAS",
      "audience_awareness": "problem_aware",
      "variants": [{{...}}, ...]
    }},
    "BAB": {{
      "formula": "BAB",
      "audience_awareness": "solution_aware",
      "variants": [{{...}}, ...]
    }}
  }}
}}

Return strict JSON only.\
"""


def run_copywriting(
    product_description: str,
    platform: str,
    voc_brief: str,
    dominant_narrative: str,
    white_space: str,
    journey_context: str = "",
) -> CopywritingOutput:
    """Stage 5: Generate ad copy variants (AIDA+PAS+BAB+ELM) across 3 audience temperatures."""
    return _claude_with_validation(
        system_prompt=COPY_SYSTEM,
        user_prompt=_build_copy_user_prompt(
            product_description, platform, voc_brief,
            dominant_narrative, white_space, journey_context,
        ),
        schema_name="copywriting",
        pydantic_model=CopywritingOutput,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 6 — CREATIVE BRIEF (enhanced — Claude via OpenRouter)
#  Multiple concepts + aspect ratios + performance benchmarks + Kling
# ─────────────────────────────────────────────────────────────────────────────

class EmotionalArc(BaseModel):
    start: str
    middle: str
    end: str


class StoryboardScene(BaseModel):
    scene_number: int
    visual_description: str
    voiceover: str
    text_overlay: str
    sfx: str


class DesignSpec(BaseModel):
    color_palette: list[str] = Field(..., min_length=3)
    typography: dict
    visual_style_guide: str


class GenerationPrompts(BaseModel):
    midjourney: str
    stable_diffusion: str
    runway: str
    kling: str            # NEW: Kling 3.0 prompt


class AspectRatioSpec(BaseModel):
    ratio: str            # "9:16" | "1:1" | "16:9"
    platform_placement: str  # "TikTok feed", "Instagram Story"
    crop_notes: str       # what to reframe or adjust


class PerformanceBenchmarks(BaseModel):
    ctr_target_pct: float           # e.g. 2.5
    hook_retention_3s_pct: float    # % watching past 3 seconds
    completion_rate_pct: float      # % watching full video


class ConceptVariant(BaseModel):
    concept_id: str          # "A", "B", "C"
    hook_type: str
    video_style: str
    emotional_arc: EmotionalArc
    creative_concept: str
    best_for_audience: str   # "cold traffic" | "retarget" | "warm"
    storyboard: list[StoryboardScene] = Field(..., min_length=4, max_length=8)


class CreativeMeta(BaseModel):
    platform: str
    duration_sec: int


class CreativeBriefOutput(BaseModel):
    metadata: CreativeMeta
    concepts: list[ConceptVariant]  = Field(..., min_length=2, max_length=3)  # 2-3 concepts
    top_concept_id: str              # "A" | "B" | "C"
    design_spec: DesignSpec
    generation_prompts: GenerationPrompts
    aspect_ratio_specs: list[AspectRatioSpec] = Field(..., min_length=2)
    performance_benchmarks: PerformanceBenchmarks


CREATIVE_SYSTEM = """\
You are a Creative Director at a top-tier performance marketing agency.
You write video ad briefs that production teams execute in 1 day. Every creative decision
is data-driven: customer language, proven hooks, competitor white space.

FORMAT: Native-first (no polished corporate look), authentic, platform-native.
RULE 1: Hook must create pattern interrupt in first 2 seconds.
RULE 2: Generate 2-3 DISTINCT concept variants for A/B testing (different hook types).
RULE 3: All benchmarks must be realistic for the specific platform.
RULE 4: Include aspect ratio specs for multi-placement deployment.

CRITICAL OUTPUT RULE: Return ONLY a JSON object. No prose, no markdown.
"""

def _build_creative_user_prompt(
    product_description: str,
    platform: str,
    voc_brief: str,
    top_copy,
    duration_sec: int,
    journey_context: str,
) -> str:
    return f"""\
Write a {platform} video creative brief for: {product_description}

VOC INTELLIGENCE (speak this language, address these pains):
{voc_brief}

TOP PERFORMING COPY (build the primary video concept around this):
Hook: {top_copy.hook}
Body: {top_copy.body}
CTA:  {top_copy.cta}
Hook type: {top_copy.hook_type}

CUSTOMER JOURNEY CONTEXT (which stage each concept targets):
{journey_context}

Platform: {platform} | Duration: {duration_sec}s

DELIVERABLES:

1. CONCEPTS (generate 2-3 distinct concepts):
   - Concept A: primary (based on top performing copy above)
   - Concept B: alternative hook type for A/B test
   - Concept C (optional): third angle for testing
   Each concept: different hook_type, video_style, emotional_arc
   Label best_for_audience: cold traffic | warm | retarget

2. STORYBOARD (4-8 scenes per concept):
   Scene 1 (0-2s): HOOK — pattern interrupt, before any branding
   Scene 2 (2-5s): PROBLEM — name the pain in customer's exact words
   Scene 3-N (5-25s): SOLUTION + PROOF
   Final scene: CTA

3. DESIGN SPEC:
   Color palette: 3-4 hex codes (native, not corporate)
   Typography: heading/body/accent fonts (system fonts preferred for native feel)
   Visual style: specific aesthetic — "shot on iPhone, bathroom lighting, authentic"

4. AI GENERATION PROMPTS:
   - Midjourney: key visual for thumbnail
   - Stable Diffusion: product/lifestyle thumbnail prompt
   - Runway Gen-4: text-to-video prompt for Scene 1 (hook)
   - Kling 3.0: video generation prompt optimized for Kling's strengths

5. ASPECT RATIO SPECS:
   - 9:16 (TikTok/Reels/Stories): primary — crop/safe zones notes
   - 1:1 (Instagram feed/Facebook): what to adjust
   - 16:9 (YouTube pre-roll) if applicable

6. PERFORMANCE BENCHMARKS (realistic for {platform}):
   - CTR target % (industry average for this category/platform)
   - Hook retention: % watching past 3 seconds (good = 30%+)
   - Completion rate: % watching full video (good = 15%+ for 30s)

JSON schema to follow exactly:
{{
  "metadata": {{"platform": "{platform}", "duration_sec": {duration_sec}}},
  "concepts": [
    {{
      "concept_id": "A",
      "hook_type": "string",
      "video_style": "string",
      "emotional_arc": {{"start": "string", "middle": "string", "end": "string"}},
      "creative_concept": "2-3 sentence description",
      "best_for_audience": "cold traffic|warm|retarget",
      "storyboard": [
        {{
          "scene_number": 1,
          "visual_description": "string",
          "voiceover": "string",
          "text_overlay": "string",
          "sfx": "string"
        }}
      ]
    }}
  ],
  "top_concept_id": "A",
  "design_spec": {{
    "color_palette": ["#hex1", "#hex2", "#hex3"],
    "typography": {{"heading": "string", "body": "string", "accent": "string"}},
    "visual_style_guide": "string"
  }},
  "generation_prompts": {{
    "midjourney": "string",
    "stable_diffusion": "string",
    "runway": "string",
    "kling": "string"
  }},
  "aspect_ratio_specs": [
    {{"ratio": "9:16", "platform_placement": "string", "crop_notes": "string"}},
    {{"ratio": "1:1", "platform_placement": "string", "crop_notes": "string"}}
  ],
  "performance_benchmarks": {{
    "ctr_target_pct": <float>,
    "hook_retention_3s_pct": <float>,
    "completion_rate_pct": <float>
  }}
}}

Return strict JSON only.\
"""


def run_creative_brief(
    product_description: str,
    platform: str,
    voc_brief: str,
    top_copy: CopyVariant,
    duration_sec: int = 30,
    journey_context: str = "",
) -> CreativeBriefOutput:
    """Stage 6: Generate video creative brief — 2-3 concepts, storyboards, AI prompts."""
    return _claude_with_validation(
        system_prompt=CREATIVE_SYSTEM,
        user_prompt=_build_creative_user_prompt(
            product_description, platform, voc_brief, top_copy, duration_sec, journey_context,
        ),
        schema_name="creativebrief",
        pydantic_model=CreativeBriefOutput,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  STAGE 7 — MEASUREMENT FRAMEWORK (NEW — Claude)
#  Blended MMM + MTA, CAC/LTV/ROMI, A/B test roadmap, 30/60/90 milestones
# ─────────────────────────────────────────────────────────────────────────────

class PlatformKPIs(BaseModel):
    platform: str
    primary_kpi: str
    primary_target: str        # "CTR > 2.5%"
    secondary_kpis: list[str]  = Field(..., min_length=3)
    industry_benchmark: str


class FunnelMetric(BaseModel):
    funnel_stage: str          # Awareness | Consideration | Decision | Retention | Advocacy
    kpi: str
    target: str
    measurement_method: str
    tool: str                  # "Meta Ads Manager", "Google Analytics 4"


class ABTestPlan(BaseModel):
    test_name: str
    hypothesis: str
    variable_tested: str
    control: str               # baseline
    variant: str               # what we're testing
    success_metric: str
    min_sample_size: int
    expected_lift_pct: float
    priority: int              # 1 = highest priority


class MeasurementFrameworkOutput(BaseModel):
    product_category: str
    platform: str
    attribution_model: str         # e.g. "Last-click for direct, MMM for brand"
    cac_target_usd: float
    ltv_cac_ratio_target: float    # 3.0 = LTV is 3x CAC
    romi_target_pct: float         # 300 = 3x ROMI
    platform_kpis: list[PlatformKPIs]  = Field(..., min_length=1)
    funnel_metrics: list[FunnelMetric] = Field(..., min_length=5)
    ab_test_roadmap: list[ABTestPlan]  = Field(..., min_length=5)
    day_30_milestones: list[str]   = Field(..., min_length=3)
    day_60_milestones: list[str]   = Field(..., min_length=3)
    day_90_milestones: list[str]   = Field(..., min_length=3)
    weekly_review_checklist: list[str] = Field(..., min_length=7)
    budget_allocation_pct: dict    # {channel: pct} — should sum to 100


MEASUREMENT_SYSTEM = """\
You are a McKinsey/Gartner-level Marketing Analytics Director specializing in performance marketing measurement.
Your framework: Blended MMM (Marketing Mix Modeling) + MTA (Multi-Touch Attribution).
You build measurement systems that connect marketing activity directly to revenue.

CORE PRINCIPLES:
1. If a metric doesn't connect to revenue, it doesn't belong in the framework
2. Blend MMM (strategic planning) + MTA (tactical optimization) — never rely on one alone
3. A/B tests must have specific hypotheses derived from the actual copy/creative variants generated
4. Milestones must be specific, measurable, and sequential

CRITICAL OUTPUT RULE: Return ONLY a JSON object. No prose, no markdown.

JSON schema:
{
  "product_category": "string",
  "platform": "string",
  "attribution_model": "string (recommendation and rationale in one sentence)",
  "cac_target_usd": <float>,
  "ltv_cac_ratio_target": <float>,
  "romi_target_pct": <float>,
  "platform_kpis": [
    {
      "platform": "string",
      "primary_kpi": "string",
      "primary_target": "string (e.g. CTR > 2.5%)",
      "secondary_kpis": ["string", ...],
      "industry_benchmark": "string"
    }
  ],
  "funnel_metrics": [
    {
      "funnel_stage": "Awareness|Consideration|Decision|Retention|Advocacy",
      "kpi": "string",
      "target": "string",
      "measurement_method": "string",
      "tool": "string"
    }
  ],
  "ab_test_roadmap": [
    {
      "test_name": "string",
      "hypothesis": "string (if we change X, then Y will improve by Z%)",
      "variable_tested": "string",
      "control": "string",
      "variant": "string",
      "success_metric": "string",
      "min_sample_size": <integer>,
      "expected_lift_pct": <float>,
      "priority": <1-5>
    }
  ],
  "day_30_milestones": ["specific measurable milestone", ...],
  "day_60_milestones": ["specific measurable milestone", ...],
  "day_90_milestones": ["specific measurable milestone", ...],
  "weekly_review_checklist": ["specific action item", ...],
  "budget_allocation_pct": {"platform1": <pct>, "platform2": <pct>}
}

JSON only.\
"""

def _build_measurement_user_prompt(
    product_description: str,
    market: str,
    platform: str,
    market_data_context: str,
    copy_context: str,
    journey_context: str,
) -> str:
    return f"""\
Build a complete marketing measurement framework for:
Product: {product_description}
Market: {market}
Primary Platform: {platform}

MARKET DATA CONTEXT:
{market_data_context}

AD COPY VARIANTS GENERATED (use these for A/B test hypotheses):
{copy_context}

CUSTOMER JOURNEY CONTEXT (use for funnel metric alignment):
{journey_context}

DELIVERABLES:

1. ATTRIBUTION MODEL RECOMMENDATION:
   - Recommend: Last-click OR Data-driven OR Blended MMM+MTA
   - Be specific about WHY for this product/market (B2C local service = different than e-commerce)
   - Tool recommendation (GA4, Meta Pixel, Northbeam, Triple Whale, etc.)

2. CAC TARGET:
   - Derive from market WTP data — what's an acceptable acquisition cost?
   - LTV:CAC ratio target (standard: 3:1 minimum, 5:1 = healthy)
   - ROMI target % (300% = 3x return on ad spend)

3. PLATFORM KPIs (for {platform} + 1-2 secondary platforms):
   - Primary KPI with specific numerical target
   - 3+ secondary KPIs
   - Industry benchmark comparison

4. FUNNEL METRICS (all 5 stages):
   For each: KPI + numerical target + how to measure + which tool
   Connect to the customer journey touchpoints identified earlier

5. A/B TEST ROADMAP (5 tests, priority ranked 1-5):
   Derive hypotheses directly from the copy variants generated:
   - Test 1: hook type comparison (use the actual hooks generated)
   - Test 2: formula comparison (AIDA vs PAS vs BAB)
   - Test 3: proof element type
   - Test 4: urgency mechanic
   - Test 5: audience temperature targeting
   For each test: specific hypothesis, control, variant, success metric, sample size

6. 30/60/90 DAY MILESTONES:
   Day 30: data collection + baseline establishment goals
   Day 60: optimization + iteration goals
   Day 90: scale + profitability goals
   Make each milestone SPECIFIC with numbers

7. WEEKLY REVIEW CHECKLIST (7+ items):
   Specific data points to check each week
   Include: ad performance, funnel metrics, creative fatigue signals

8. BUDGET ALLOCATION:
   % split across channels (should sum to 100)
   Base on competitive channel mix + journey touchpoints

Return strict JSON only.\
"""


def run_measurement_framework(
    product_description: str,
    market: str,
    platform: str,
    market_data_context: str,
    copy_context: str,
    journey_context: str,
) -> MeasurementFrameworkOutput:
    """Stage 7: Build complete marketing measurement framework — KPIs, A/B tests, milestones."""
    return _claude_with_validation(
        system_prompt=MEASUREMENT_SYSTEM,
        user_prompt=_build_measurement_user_prompt(
            product_description, market, platform,
            market_data_context, copy_context, journey_context,
        ),
        schema_name="measurementframework",
        pydantic_model=MeasurementFrameworkOutput,
    )
