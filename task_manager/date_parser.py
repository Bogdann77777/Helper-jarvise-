"""
date_parser.py — Russian/Ukrainian natural language date parsing.
"""
import re
from datetime import date, datetime
from typing import Optional

import dateparser

_LANGUAGES = ["ru", "uk"]
_SETTINGS = {
    "PREFER_DATES_FROM": "future",
    "RETURN_AS_TIMEZONE_AWARE": False,
    "DATE_ORDER": "DMY",
}

# Prepositions to strip before parsing
_PREP_RE = re.compile(
    r"^(до|к|на|в|во|с|по|через|не позднее|не пізніше)\s+",
    re.IGNORECASE | re.UNICODE,
)

URGENCY_WORDS = {
    "срочно", "срочное", "urgent", "asap", "немедленно",
    "сегодня же", "прямо сейчас", "терміново",
}


def parse_date(text: str) -> Optional[date]:
    """
    Parse a Russian/Ukrainian date expression.
    Returns date object or None.

    Examples:
      "до 15 апреля"       → date(2026, 4, 15)
      "через 3 дня"        → date.today() + 3
      "следующей неделе"   → next Monday
      "завтра"             → tomorrow
      "20.04"              → date(2026, 4, 20)
      "срочно"             → None  (urgency, not a date)
    """
    if not text:
        return None

    clean = _PREP_RE.sub("", text.strip())

    # Urgency words are not dates
    if clean.lower() in URGENCY_WORDS:
        return None

    settings = dict(_SETTINGS)
    settings["RELATIVE_BASE"] = datetime.now()

    # DD.MM or DD/MM → explicit date (dateparser misreads as time HH:MM)
    _ddmm = re.match(r"^(\d{1,2})[./](\d{1,2})$", clean)
    if _ddmm:
        day, month = int(_ddmm.group(1)), int(_ddmm.group(2))
        try:
            today = date.today()
            candidate = date(today.year, month, day)
            if candidate < today:
                candidate = date(today.year + 1, month, day)
            return candidate
        except ValueError:
            pass

    try:
        result = dateparser.parse(clean, languages=_LANGUAGES, settings=settings)
        if result:
            return result.date()
        # Retry without stripping preposition (handles "следующий вторник" etc.)
        if clean != text.strip():
            result2 = dateparser.parse(text.strip(), languages=_LANGUAGES, settings=settings)
            if result2:
                return result2.date()
    except Exception:
        pass
    return None


def is_urgent(text: str) -> bool:
    """Returns True if text contains urgency markers."""
    low = text.lower()
    return any(w in low for w in URGENCY_WORDS)


def date_to_iso(d: date) -> str:
    return d.isoformat()


def format_date_ru(iso: str) -> str:
    """Format ISO date to Russian short form: '15 апр', '20 мая'"""
    MONTHS = [
        "", "янв", "фев", "мар", "апр", "май", "июн",
        "июл", "авг", "сен", "окт", "ноя", "дек",
    ]
    try:
        d = date.fromisoformat(iso)
        today = date.today()
        if d == today:
            return "сегодня"
        if (d - today).days == 1:
            return "завтра"
        if (d - today).days == -1:
            return "вчера"
        label = f"{d.day} {MONTHS[d.month]}"
        if d.year != today.year:
            label += f" {d.year}"
        if d < today:
            label += " ⚠️"  # overdue
        return label
    except Exception:
        return iso
