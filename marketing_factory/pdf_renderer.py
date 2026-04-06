"""
marketing_factory/pdf_renderer.py — 5 PDF render functions (fpdf2).

Design: A4 portrait, Helvetica only (built-in, zero font deps), max 2 pages.
Color palette mirrors tech-passport design:
  teal    #01696F  →  (1, 105, 111)
  dark    #1E1C16  →  (30, 28, 22)
  muted   #6E6D68  →  (110, 109, 104)
  bg      #F5F4F0  →  (245, 244, 240)
  white              (255, 255, 255)
  green   #3D7020  →  (61, 112, 32)
  orange  #C96200  →  (201, 98, 0)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from fpdf import FPDF, XPos, YPos

# ─── Palette ──────────────────────────────────────────────────────────────────
C_TEAL   = (1,   105, 111)
C_DARK   = (30,  28,  22)
C_MUTED  = (110, 109, 104)
C_BG     = (245, 244, 240)
C_WHITE  = (255, 255, 255)
C_GREEN  = (61,  112, 32)
C_ORANGE = (201, 98,  0)
C_LIGHT  = (230, 239, 238)   # very light teal for alternating rows

PAGE_W   = 210   # A4 mm
MARGIN   = 15
INNER_W  = PAGE_W - 2 * MARGIN   # 180mm


def _safe(text: str) -> str:
    """Replace non-latin-1 chars with ASCII equivalents for Helvetica compatibility."""
    return (str(text)
        .replace("\u2014", "-").replace("\u2013", "-")    # em/en dash
        .replace("\u2018", "'").replace("\u2019", "'")    # smart single quotes
        .replace("\u201c", '"').replace("\u201d", '"')    # smart double quotes
        .replace("\u2022", "-").replace("\u25aa", "-")    # bullets
        .replace("\u2026", "...")                          # ellipsis
        .replace("\u2192", "->").replace("\u2190", "<-")  # arrows
        .encode("latin-1", errors="replace").decode("latin-1")  # catch-all
    )


# ─── Base PDF class ───────────────────────────────────────────────────────────
class _BasePDF(FPDF):
    def __init__(self, title: str, subtitle: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._title    = _safe(title)
        self._subtitle = _safe(subtitle)
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        # Teal header bar
        self.set_fill_color(*C_TEAL)
        self.rect(0, 0, PAGE_W, 18, style="F")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*C_WHITE)
        self.set_y(5)
        self.cell(INNER_W + 2 * MARGIN, 8,
                  f"{self._title.upper()}  ·  {self._subtitle}  ·  {date.today():%B %d, %Y}",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_text_color(*C_DARK)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_MUTED)
        self.cell(0, 6, f"Page {self.page_no()} - Marketing Factory - AI-generated report",
                  align="C")

    # ── Helpers ────────────────────────────────────────────────────────────────
    def section_header(self, text: str, color: tuple = C_TEAL):
        self.set_fill_color(*color)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 9)
        self.cell(INNER_W, 7, f"  {_safe(text).upper()}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*C_DARK)
        self.ln(2)

    def kv_row(self, label: str, value: str, fill: bool = False):
        if fill:
            self.set_fill_color(*C_BG)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C_MUTED)
        self.cell(45, 6, _safe(label), fill=fill,
                  new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*C_DARK)
        self.multi_cell(INNER_W - 45, 6, _safe(value), fill=fill)
        self.ln(1)

    def bullet(self, text: str, color: tuple = C_TEAL, indent: int = 4):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*color)
        self.cell(indent, 5, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_text_color(*C_DARK)
        self.multi_cell(INNER_W - indent, 5, _safe(text))

    def quote_box(self, text: str):
        """Bordered quote box for verbatim VOC quotes."""
        self.set_fill_color(*C_LIGHT)
        self.set_draw_color(*C_TEAL)
        self.set_line_width(0.4)
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*C_MUTED)
        self.multi_cell(INNER_W, 5, f'  "{_safe(text)}"', border=1, fill=True)
        self.set_line_width(0.2)
        self.set_draw_color(0, 0, 0)
        self.ln(1)

    def callout(self, text: str, color: tuple = C_TEAL):
        self.set_fill_color(*color)
        self.rect(MARGIN, self.get_y(), 2, 14, style="F")
        self.set_x(MARGIN + 4)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_DARK)
        self.multi_cell(INNER_W - 4, 6, _safe(text), fill=False)
        self.ln(2)

    def table_header(self, cols: list[tuple[str, float]]):
        """cols = [(label, width_mm), ...]"""
        self.set_fill_color(*C_TEAL)
        self.set_text_color(*C_WHITE)
        self.set_font("Helvetica", "B", 7.5)
        for label, w in cols:
            self.cell(w, 6, f" {_safe(label)}", border=1, fill=True,
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln()
        self.set_text_color(*C_DARK)

    def table_row(self, values: list[str], cols: list[tuple[str, float]], fill: bool = False):
        if fill:
            self.set_fill_color(*C_BG)
        self.set_font("Helvetica", "", 7.5)
        for val, (_, w) in zip(values, cols):
            self.cell(w, 6, f" {_safe(str(val))[:40]}", border=1, fill=fill,
                      new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln()


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 1 — MARKET RESEARCH
# ─────────────────────────────────────────────────────────────────────────────

def render_market_research_pdf(data: Any, output_path: Path) -> Path:
    """Renders MarketResearchOutput → PDF. data can be Pydantic model or dict."""
    d = data if isinstance(data, dict) else data.model_dump()

    pdf = _BasePDF(title=d.get("market_name", "Market"), subtitle="Market Research")
    pdf.add_page()

    # ── KPI boxes ─────────────────────────────────────────────────────────────
    pdf.section_header("Market Sizing")
    kpis = [
        ("TAM",  f"${d.get('tam_usd_billions', 0):.1f}B",  C_TEAL),
        ("SAM",  f"${d.get('sam_usd_billions', 0):.1f}B",  C_GREEN),
        ("SOM",  f"${d.get('som_usd_millions', 0):.0f}M",  C_ORANGE),
        ("CAGR", f"{d.get('cagr_projected_pct', 0):.1f}%", C_TEAL),
    ]
    box_w = INNER_W / 4
    for label, value, color in kpis:
        pdf.set_fill_color(*color)
        pdf.set_text_color(*C_WHITE)
        pdf.set_font("Helvetica", "B", 14)
        x0 = pdf.get_x()
        y0 = pdf.get_y()
        pdf.cell(box_w - 1, 14, value, fill=True, align="C",
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_x(x0)
        pdf.set_y(y0 + 14)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(box_w - 1, 5, label, align="C",
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_y(y0)
        pdf.set_x(x0 + box_w)
    pdf.set_y(pdf.get_y() + 20)
    pdf.ln(2)

    # ── Stage ─────────────────────────────────────────────────────────────────
    stage = d.get("market_stage", "").upper()
    hist  = d.get("cagr_historical_pct", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 5, _safe(f"Market stage: {stage}   |   Historical CAGR: {hist:.1f}%   |   "
                         f"Data confidence: {d.get('data_confidence', 0)*100:.0f}%"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Customer segments ─────────────────────────────────────────────────────
    pdf.section_header("Customer Segments")
    cols = [("Segment", 55), ("Size %", 20), ("Top Driver", 65), ("WTP", 40)]
    pdf.table_header(cols)
    for i, seg in enumerate(d.get("customer_segments", [])[:5]):
        driver = seg.get("purchase_drivers", ["-"])[0]
        pdf.table_row(
            [seg.get("name",""), f"{seg.get('size_pct',0):.0f}%",
             driver, seg.get("willingness_to_pay","")],
            cols, fill=(i % 2 == 0)
        )
    pdf.ln(3)

    # ── Purchase drivers ──────────────────────────────────────────────────────
    pdf.section_header("Top Purchase Drivers")
    for drv in d.get("top_purchase_drivers", [])[:7]:
        pdf.bullet(drv)
    pdf.ln(3)

    # ── Strategic implications ────────────────────────────────────────────────
    pdf.section_header("Strategic Implications", color=C_DARK)
    for i, imp in enumerate(d.get("strategic_implications", [])[:3], 1):
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*C_TEAL)
        pdf.cell(8, 5, f"{i}.", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*C_DARK)
        pdf.multi_cell(INNER_W - 8, 5, _safe(imp))
        pdf.ln(2)

    # ── Trends ────────────────────────────────────────────────────────────────
    if d.get("key_market_trends"):
        pdf.section_header("Key Trends")
        for trend in d.get("key_market_trends", [])[:5]:
            pdf.bullet(trend, color=C_MUTED)

    pdf.output(str(output_path))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 2 — VOICE OF CUSTOMER
# ─────────────────────────────────────────────────────────────────────────────

def render_voc_pdf(data: Any, output_path: Path) -> Path:
    d = data if isinstance(data, dict) else data.model_dump()

    pdf = _BasePDF(title=d.get("market", "VOC"), subtitle="Voice of Customer")
    pdf.add_page()

    # ── Meta ──────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*C_MUTED)
    n_themes = d.get("total_themes_analyzed", 0)
    sources  = ", ".join(d.get("sources_synthesized", [])[:4])
    pdf.cell(0, 5, _safe(f"Themes analyzed: {n_themes}   |   Sources: {sources}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Top pain points ───────────────────────────────────────────────────────
    pdf.section_header("Top Pain Points  (intensity × frequency)")
    for theme in d.get("dominant_pain_points", [])[:5]:
        intensity = theme.get("pain_intensity", 0)
        frequency = theme.get("frequency_score", 0)
        score     = intensity * frequency

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*C_DARK)
        pdf.cell(INNER_W - 30, 5, _safe(theme.get("theme", "")), new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_ORANGE)
        pdf.cell(30, 5, f"Score: {score}  ({intensity}×{frequency})",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")

        for quote in theme.get("verbatim_quotes", [])[:2]:
            pdf.quote_box(quote)

        if theme.get("trigger"):
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*C_MUTED)
            pdf.cell(0, 4, _safe(f"  Trigger: {theme['trigger']}"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # ── JTBD Forces ───────────────────────────────────────────────────────────
    pdf.section_header("JTBD 4-Forces  (Switch Interview)")
    forces = d.get("jtbd_forces", {})
    half_w = INNER_W / 2 - 2
    y_start = pdf.get_y()

    # Left column: Push + Pull
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*C_ORANGE)
    pdf.cell(half_w, 5, "PUSH (away from current)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for item in forces.get("push_from_current", [])[:3]:
        pdf.bullet(item, color=C_ORANGE)
    pdf.ln(2)
    pdf.set_text_color(*C_GREEN)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.cell(half_w, 5, "PULL (toward new solution)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for item in forces.get("pull_to_new", [])[:3]:
        pdf.bullet(item, color=C_GREEN)

    # Right column: Anxiety + Habit
    y_right = y_start
    pdf.set_xy(MARGIN + half_w + 4, y_right)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(half_w, 5, "ANXIETY (fear of switching)",
             new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_xy(MARGIN + half_w + 4, y_right + 5)
    for item in forces.get("anxiety_about_new", [])[:2]:
        pdf.set_x(MARGIN + half_w + 4)
        pdf.bullet(item, color=C_MUTED)
    pdf.set_x(MARGIN + half_w + 4)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(half_w, 5, "HABIT INERTIA",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for item in forces.get("habit_inertia", [])[:2]:
        pdf.bullet(item, color=C_MUTED)
    pdf.ln(4)

    # ── Buying language ───────────────────────────────────────────────────────
    pdf.section_header("Buying Language  (exact phrases for ad copy)")
    phrases = d.get("buying_language", [])[:12]
    # chips layout
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(*C_DARK)
    x_cur = MARGIN
    for phrase in phrases:
        chip_w = min(pdf.get_string_width(phrase) + 6, INNER_W)
        if x_cur + chip_w > MARGIN + INNER_W:
            x_cur = MARGIN
            pdf.ln(7)
        pdf.set_xy(x_cur, pdf.get_y())
        pdf.set_fill_color(*C_LIGHT)
        pdf.cell(chip_w, 6, phrase, border=1, fill=True)
        x_cur += chip_w + 2
    pdf.ln(8)

    # ── Avoid ─────────────────────────────────────────────────────────────────
    if d.get("language_to_avoid"):
        pdf.section_header("Language to Avoid", color=C_MUTED)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*C_MUTED)
        avoid = "  //  ".join(d.get("language_to_avoid", [])[:6])
        pdf.cell(0, 5, avoid, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.output(str(output_path))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 3 — COMPETITOR INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

def render_competitor_pdf(data: Any, output_path: Path) -> Path:
    d = data if isinstance(data, dict) else data.model_dump()

    pdf = _BasePDF(title=d.get("product_category", "CI"), subtitle="Competitor Intelligence")
    pdf.add_page()

    # ── Meta ──────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*C_MUTED)
    intensity = d.get("competitive_intensity", "").upper()
    leader    = d.get("market_leader", "")
    pdf.cell(0, 5, _safe(f"Market: {d.get('market','')}   |   Intensity: {intensity}   |   Leader: {leader}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Competitors table ─────────────────────────────────────────────────────
    pdf.section_header("Competitor Landscape")
    cols = [("Brand", 32), ("Position", 22), ("Ad Angle", 54), ("Price", 22), ("Gap", 50)]
    pdf.table_header(cols)
    for i, comp in enumerate(d.get("competitors", [])[:7]):
        pdf.table_row(
            [comp.get("name",""),
             comp.get("market_position",""),
             comp.get("primary_ad_angle",""),
             comp.get("price_positioning",""),
             comp.get("opportunity_gap","")],
            cols, fill=(i % 2 == 0)
        )
    pdf.ln(4)

    # ── Dominant narrative ────────────────────────────────────────────────────
    pdf.section_header("Dominant Ad Narrative  (sea of sameness)")
    pdf.callout(d.get("dominant_ad_narrative", ""), color=C_MUTED)

    # ── White space ───────────────────────────────────────────────────────────
    pdf.section_header("White Space Positioning  (unclaimed angles)")
    for i, angle in enumerate(d.get("white_space_positioning", [])[:4], 1):
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*C_GREEN)
        pdf.cell(6, 5, f"{i}.", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*C_DARK)
        pdf.multi_cell(INNER_W - 6, 5, _safe(angle))
        pdf.ln(1)

    # ── Recommended differentiation ───────────────────────────────────────────
    pdf.section_header("Recommended Differentiation", color=C_GREEN)
    pdf.set_fill_color(*C_LIGHT)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*C_DARK)
    pdf.multi_cell(INNER_W, 7, _safe(f"  {d.get('recommended_differentiation','')}"), fill=True)
    pdf.ln(3)

    # ── Porter's Five Forces ─────────────────────────────────────────────────
    forces = d.get("porter_five_forces_summary", {})
    if forces:
        pdf.section_header("Porter's Five Forces")
        labels = [("Supplier Power", "supplier_power"),
                  ("Buyer Power",    "buyer_power"),
                  ("New Entrants",   "new_entrants"),
                  ("Substitutes",    "substitutes"),
                  ("Rivalry",        "rivalry")]
        box_w = INNER_W / 5
        for label, key in labels:
            level = forces.get(key, "?").upper()
            color = C_GREEN if level == "LOW" else (C_ORANGE if level == "MEDIUM" else (201,50,50))
            pdf.set_fill_color(*color)
            pdf.set_text_color(*C_WHITE)
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(box_w - 1, 8, level, fill=True, align="C",
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_MUTED)
        for label, _ in labels:
            pdf.cell(box_w - 1, 4, label, align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln()

    pdf.output(str(output_path))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 4 — CUSTOMER JOURNEY MAP (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def render_journey_pdf(data: Any, output_path: Path) -> Path:
    """Renders CustomerJourneyOutput → PDF."""
    d = data if isinstance(data, dict) else data.model_dump()

    pdf = _BasePDF(title=d.get("product_category", "Journey"), subtitle="Customer Journey Map")
    pdf.add_page()

    # ── Persona + meta ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 5,
             _safe(f"Dominant persona: {d.get('dominant_persona','')}   |   "
                   f"Market: {d.get('market','')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # ── Critical path callout ─────────────────────────────────────────────
    pdf.section_header("Critical Path to Purchase")
    pdf.callout(d.get("critical_path", ""), color=C_GREEN)

    # ── Biggest drop-off ──────────────────────────────────────────────────
    drop_off = d.get("biggest_drop_off_stage", "")
    if drop_off:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*C_ORANGE)
        pdf.cell(0, 5, _safe(f"  Biggest drop-off: {drop_off} stage"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # ── Journey stages ────────────────────────────────────────────────────
    pdf.section_header("5-Stage Journey Map")

    STAGE_COLORS = {
        "awareness": C_TEAL,
        "consideration": C_GREEN,
        "decision": C_ORANGE,
        "retention": (80, 80, 160),
        "advocacy": (160, 80, 160),
    }

    for stage in d.get("journey_stages", []):
        stage_name = stage.get("stage_name", "")
        color = STAGE_COLORS.get(stage_name.lower(), C_TEAL)

        # Stage header
        pdf.set_fill_color(*color)
        pdf.set_text_color(*C_WHITE)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(INNER_W, 6, f"  {_safe(stage_name.upper())}  "
                 f"| time to next: {_safe(stage.get('time_to_next_stage',''))}",
                 fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*C_DARK)

        # Mindset quote
        mindset = stage.get("customer_mindset", "")
        if mindset:
            pdf.quote_box(mindset)

        # Touchpoints table
        touchpoints = stage.get("touchpoints", [])
        if touchpoints:
            cols = [("Channel", 35), ("Message", 55), ("Friction", 45), ("KPI", 45)]
            pdf.table_header(cols)
            for i, tp in enumerate(touchpoints[:4]):
                pdf.table_row(
                    [tp.get("channel", ""),
                     tp.get("message", ""),
                     tp.get("friction_point", ""),
                     tp.get("kpi", "")],
                    cols, fill=(i % 2 == 0)
                )
            pdf.ln(1)

        # Moment of truth + drop-off
        mot = stage.get("moment_of_truth", "")
        if mot:
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*C_TEAL)
            pdf.cell(25, 4, "  MOT:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*C_DARK)
            pdf.multi_cell(INNER_W - 25, 4, _safe(mot))

        drop = stage.get("drop_off_risk", "")
        if drop:
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*C_ORANGE)
            pdf.cell(25, 4, "  DROP-OFF:", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(*C_DARK)
            pdf.multi_cell(INNER_W - 25, 4, _safe(drop))

        pdf.ln(3)

    # ── Winning sequence ──────────────────────────────────────────────────
    sequence = d.get("winning_touchpoint_sequence", [])
    if sequence:
        pdf.section_header("Winning Touchpoint Sequence", color=C_GREEN)
        for i, step in enumerate(sequence[:7], 1):
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*C_GREEN)
            pdf.cell(8, 5, f"{i}.", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*C_DARK)
            pdf.multi_cell(INNER_W - 8, 5, _safe(step))
        pdf.ln(2)

    # ── Competitor journey gaps ───────────────────────────────────────────
    gaps = d.get("competitor_journey_gaps", [])
    if gaps:
        pdf.section_header("Competitor Journey Gaps  (our opportunity)", color=C_ORANGE)
        for gap in gaps[:5]:
            pdf.callout(gap, color=C_ORANGE)

    pdf.output(str(output_path))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 4 — COPYWRITING
# ─────────────────────────────────────────────────────────────────────────────

def render_copywriting_pdf(data: dict, output_path: Path) -> Path:
    pdf = _BasePDF(title=data.get("metadata", {}).get("platform","Copy"),
                   subtitle="Ad Copywriting")
    pdf.add_page()

    meta = data.get("metadata", {})
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*C_MUTED)
    pdf.cell(0, 5, _safe(
             f"Platform: {meta.get('platform','')}   |   "
             f"Framework: {meta.get('framework_used','')}   |   "
             f"Variants generated: {meta.get('total_variants_generated','')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Top recommendation ────────────────────────────────────────────────────
    top = data.get("top_recommendation", {})
    if top:
        pdf.section_header("Top Recommendation  (overall winner)")
        pdf.set_fill_color(*C_LIGHT)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*C_TEAL)
        pdf.multi_cell(INNER_W, 6, _safe(top.get("hook", top.get("headline", ""))), fill=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*C_DARK)
        pdf.multi_cell(INNER_W, 5, _safe(top.get("body", "")))
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*C_ORANGE)
        pdf.cell(0, 5, _safe(f"CTA: {top.get('cta','')}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        score = top.get("score", top.get("total_score", 0))
        if score:
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*C_MUTED)
            pdf.cell(0, 4, f"Score: {score:.1f}/10",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # ── Variants by temperature ───────────────────────────────────────────────
    by_temp = data.get("by_temperature", {})
    TEMP_COLORS = {"cold": C_TEAL, "warm": C_ORANGE, "retarget": (130, 60, 170)}

    for temp, color in TEMP_COLORS.items():
        block = by_temp.get(temp, {})
        if not block:
            continue
        pdf.section_header(f"{temp.upper()} audience", color=color)

        elm = block.get("elm_route", "")
        if elm:
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*C_MUTED)
            pdf.cell(0, 4, _safe(f"ELM route: {elm}"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        variants = block.get("variants", [])
        # show top 2 per temperature
        for v in sorted(variants,
                        key=lambda x: x.get("score", x.get("total_score", 0)),
                        reverse=True)[:2]:
            v_score = v.get("score", v.get("total_score", 0))
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*C_DARK)
            hook = v.get("hook", v.get("headline", ""))
            pdf.multi_cell(INNER_W - 25, 5, _safe(hook))
            # score pill
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*color)
            pdf.cell(0, 4, _safe(f"  {v_score:.1f}/10  *  {v.get('emotion','')}"),
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

    pdf.output(str(output_path))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 6 — CREATIVE BRIEF
# ─────────────────────────────────────────────────────────────────────────────

def render_creative_pdf(data: dict, output_path: Path) -> Path:  # noqa: C901
    pdf = _BasePDF(title=data.get("metadata", {}).get("platform","Creative"),
                   subtitle="Creative Brief")
    pdf.add_page()

    meta = data.get("metadata", {})

    # Support both new (concepts list) and old (concept dict) format
    concepts = data.get("concepts", [])
    if not concepts:
        old_concept = data.get("concept", {})
        if old_concept:
            concepts = [{
                "concept_id": "A",
                "hook_type": old_concept.get("hook_type", ""),
                "video_style": old_concept.get("video_style", ""),
                "emotional_arc": old_concept.get("emotional_arc", {}),
                "creative_concept": old_concept.get("creative_concept", ""),
                "best_for_audience": "cold traffic",
                "storyboard": data.get("storyboard", []),
            }]

    top_concept_id = data.get("top_concept_id", "A")

    # ── Meta row ──────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*C_MUTED)
    benchmarks = data.get("performance_benchmarks", {})
    bench_str = ""
    if benchmarks:
        bench_str = (f"  |  CTR: {benchmarks.get('ctr_target_pct',0):.1f}%"
                     f"  Hook ret: {benchmarks.get('hook_retention_3s_pct',0):.0f}%"
                     f"  Completion: {benchmarks.get('completion_rate_pct',0):.0f}%")
    pdf.cell(0, 5, _safe(f"Platform: {meta.get('platform','')}  |  {meta.get('duration_sec',30)}s"
                         f"  |  Concepts: {len(concepts)}  |  Top: {top_concept_id}{bench_str}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    # ── Each concept ──────────────────────────────────────────────────────────
    CONCEPT_COLORS = {"A": C_TEAL, "B": C_GREEN, "C": C_ORANGE}
    for concept in concepts[:3]:
        cid = concept.get("concept_id", "A")
        color = CONCEPT_COLORS.get(cid, C_TEAL)
        is_top = (cid == top_concept_id)
        header_label = f"CONCEPT {cid}{'  * TOP PICK' if is_top else ''}"
        pdf.section_header(
            f"{header_label}  |  {concept.get('hook_type','')}  |  "
            f"{concept.get('video_style','')}  |  {concept.get('best_for_audience','')}",
            color=color
        )
        arc = concept.get("emotional_arc", {})
        if arc:
            arc_str = f"Arc: {arc.get('start','')} -> {arc.get('middle','')} -> {arc.get('end','')}"
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*C_MUTED)
            pdf.cell(0, 4, _safe(arc_str), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if concept.get("creative_concept"):
            pdf.callout(concept["creative_concept"], color=color)
        storyboard = concept.get("storyboard", [])
        if storyboard:
            cols = [("Sc.", 10), ("Visual", 58), ("Voiceover/Text", 62), ("SFX", 50)]
            pdf.table_header(cols)
            for i, scene in enumerate(storyboard[:8]):
                pdf.table_row(
                    [str(scene.get("scene_number", i+1)),
                     scene.get("visual_description", ""),
                     scene.get("voiceover", "") or scene.get("text_overlay", ""),
                     scene.get("sfx", "")],
                    cols, fill=(i % 2 == 0)
                )
            pdf.ln(2)

    # ── Design spec ───────────────────────────────────────────────────────────
    design = data.get("design_spec", {})
    if design:
        pdf.section_header("Design Spec", color=C_DARK)
        for label, value in [
            ("Color palette",  ", ".join(design.get("color_palette", [])[:4])),
            ("Typography",     str(design.get("typography", {}))),
            ("Visual style",   design.get("visual_style_guide", design.get("visual_style",""))),
        ]:
            pdf.kv_row(label, str(value)[:120], fill=True)
        pdf.ln(2)

    # ── Aspect ratio specs ────────────────────────────────────────────────────
    aspect_specs = data.get("aspect_ratio_specs", [])
    if aspect_specs:
        pdf.section_header("Aspect Ratio Specs", color=C_DARK)
        cols = [("Ratio", 20), ("Placement", 60), ("Crop Notes", 100)]
        pdf.table_header(cols)
        for i, spec in enumerate(aspect_specs[:4]):
            pdf.table_row(
                [spec.get("ratio",""), spec.get("platform_placement",""), spec.get("crop_notes","")],
                cols, fill=(i % 2 == 0)
            )
        pdf.ln(2)

    # ── AI generation prompts ─────────────────────────────────────────────────
    gen_prompts = data.get("generation_prompts", {})
    if gen_prompts:
        pdf.section_header("AI Generation Prompts", color=C_DARK)
        for tool, prompt_text in gen_prompts.items():
            if not prompt_text:
                continue
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(*C_TEAL)
            pdf.cell(0, 5, f"  {tool.upper()}:",
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*C_MUTED)
            pdf.multi_cell(INNER_W, 4.5, _safe(str(prompt_text)[:300]))
            pdf.ln(1)

    pdf.output(str(output_path))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
#  PDF 7 — MEASUREMENT FRAMEWORK (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def render_measurement_pdf(data: Any, output_path: Path) -> Path:
    """Renders MeasurementFrameworkOutput → PDF."""
    d = data if isinstance(data, dict) else data.model_dump()

    pdf = _BasePDF(title=d.get("product_category", "Measurement"),
                   subtitle="Measurement Framework")
    pdf.add_page()

    # ── KPI cards: CAC / LTV:CAC / ROMI / Attribution ─────────────────────────
    pdf.section_header("Revenue Targets")
    kpis = [
        ("CAC Target",  f"${d.get('cac_target_usd', 0):.0f}",      C_TEAL),
        ("LTV:CAC",     f"{d.get('ltv_cac_ratio_target', 0):.1f}x", C_GREEN),
        ("ROMI Target", f"{d.get('romi_target_pct', 0):.0f}%",      C_ORANGE),
    ]
    box_w = INNER_W / 3
    for label, value, color in kpis:
        pdf.set_fill_color(*color)
        pdf.set_text_color(*C_WHITE)
        pdf.set_font("Helvetica", "B", 13)
        x0 = pdf.get_x()
        y0 = pdf.get_y()
        pdf.cell(box_w - 1, 12, _safe(value), fill=True, align="C",
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_x(x0)
        pdf.set_y(y0 + 12)
        pdf.set_font("Helvetica", "", 7)
        pdf.cell(box_w - 1, 4, label, align="C",
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_y(y0)
        pdf.set_x(x0 + box_w)
    pdf.set_y(pdf.get_y() + 18)
    attr = d.get("attribution_model", "")
    if attr:
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*C_MUTED)
        pdf.multi_cell(INNER_W, 4.5, _safe(f"Attribution model: {attr}"))
    pdf.ln(3)

    # ── Platform KPIs table ───────────────────────────────────────────────────
    platform_kpis = d.get("platform_kpis", [])
    if platform_kpis:
        pdf.section_header("Platform KPIs")
        cols = [("Platform", 35), ("Primary KPI", 45), ("Target", 35), ("Benchmark", 65)]
        pdf.table_header(cols)
        for i, pk in enumerate(platform_kpis[:4]):
            pdf.table_row(
                [pk.get("platform",""), pk.get("primary_kpi",""),
                 pk.get("primary_target",""), pk.get("industry_benchmark","")],
                cols, fill=(i % 2 == 0)
            )
        pdf.ln(3)

    # ── Funnel metrics table ───────────────────────────────────────────────────
    funnel_metrics = d.get("funnel_metrics", [])
    if funnel_metrics:
        pdf.section_header("Funnel Metrics")
        cols = [("Stage", 35), ("KPI", 45), ("Target", 35), ("Tool", 65)]
        pdf.table_header(cols)
        for i, fm in enumerate(funnel_metrics[:5]):
            pdf.table_row(
                [fm.get("funnel_stage",""), fm.get("kpi",""),
                 fm.get("target",""), fm.get("tool","")],
                cols, fill=(i % 2 == 0)
            )
        pdf.ln(3)

    # ── A/B test roadmap ──────────────────────────────────────────────────────
    ab_tests = d.get("ab_test_roadmap", [])
    if ab_tests:
        pdf.section_header("A/B Test Roadmap  (priority ranked)")
        cols = [("P", 8), ("Test", 38), ("Hypothesis", 80), ("Success Metric", 37), ("Lift%", 17)]
        pdf.table_header(cols)
        for i, test in enumerate(sorted(ab_tests, key=lambda x: x.get("priority", 99))[:5]):
            pdf.table_row(
                [str(test.get("priority","")),
                 test.get("test_name",""),
                 test.get("hypothesis",""),
                 test.get("success_metric",""),
                 f"{test.get('expected_lift_pct',0):.0f}%"],
                cols, fill=(i % 2 == 0)
            )
        pdf.ln(3)

    # ── 30/60/90 Day Milestones ───────────────────────────────────────────────
    pdf.section_header("30 / 60 / 90 Day Milestones", color=C_DARK)
    milestone_sets = [
        ("30 DAYS", d.get("day_30_milestones", []), C_TEAL),
        ("60 DAYS", d.get("day_60_milestones", []), C_GREEN),
        ("90 DAYS", d.get("day_90_milestones", []), C_ORANGE),
    ]
    col_w = INNER_W / 3 - 1
    y_start = pdf.get_y()
    x_positions = [MARGIN, MARGIN + col_w + 1, MARGIN + 2 * (col_w + 1)]

    for col_idx, (label, items, color) in enumerate(milestone_sets):
        x = x_positions[col_idx]
        pdf.set_xy(x, y_start)
        pdf.set_fill_color(*color)
        pdf.set_text_color(*C_WHITE)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_w, 6, f"  {label}", fill=True,
                 new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.set_xy(x, y_start + 6)
        pdf.set_text_color(*C_DARK)
        pdf.set_font("Helvetica", "", 7)
        for item in items[:4]:
            pdf.set_x(x)
            pdf.multi_cell(col_w, 4.5, _safe(f"- {item}"))

    max_items = max(len(d.get("day_30_milestones", [])),
                    len(d.get("day_60_milestones", [])),
                    len(d.get("day_90_milestones", [])), 1)
    pdf.set_y(y_start + 6 + min(max_items, 4) * 5 + 5)
    pdf.ln(3)

    # ── Budget allocation ─────────────────────────────────────────────────────
    budget = d.get("budget_allocation_pct", {})
    if budget:
        pdf.section_header("Budget Allocation")
        ch_list = list(budget.items())[:6]
        bw = INNER_W / max(len(ch_list), 1)
        for ch, pct in ch_list:
            pdf.set_fill_color(*C_TEAL)
            pdf.set_text_color(*C_WHITE)
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.cell(bw - 1, 8, _safe(f"{ch[:10]}  {pct:.0f}%"), fill=True, align="C",
                     new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(9)

    # ── Weekly review checklist ───────────────────────────────────────────────
    checklist = d.get("weekly_review_checklist", [])
    if checklist:
        pdf.section_header("Weekly Review Checklist", color=C_MUTED)
        half = (len(checklist) + 1) // 2
        left_items  = checklist[:half]
        right_items = checklist[half:]
        y_cl = pdf.get_y()
        for item in left_items:
            pdf.bullet(item, color=C_TEAL, indent=3)
        right_y = y_cl
        for item in right_items:
            pdf.set_xy(MARGIN + INNER_W // 2, right_y)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*C_TEAL)
            pdf.cell(3, 5, "-", new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.set_text_color(*C_DARK)
            pdf.multi_cell(INNER_W // 2 - 3, 5, _safe(item))
            right_y = pdf.get_y()

    pdf.output(str(output_path))
    return output_path
