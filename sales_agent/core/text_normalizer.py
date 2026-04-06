"""
Text Normalizer — prepares LLM output for TTS.

Problems solved:
  - XTTS/Qwen3 TTS choke on "$500", "ROI", "Q3 2026", etc.
  - Numbers become "five hundred" (XTTS reads "500" as "five zero zero")
  - Abbreviations become letter-by-letter for XTTS
  - Adds natural pauses via punctuation (SSML not supported in XTTS)
  - Strips markdown, code blocks, bullets (LLM sometimes adds them)
"""
import re


# ── Abbreviation pronunciation map ───────────────────────────────────────────
_ABBREVS = {
    "ROI": "R-O-I",
    "KPI": "K-P-I",
    "CEO": "C-E-O",
    "CFO": "C-F-O",
    "CTO": "C-T-O",
    "HR": "H-R",
    "IT": "I-T",
    "AI": "A-I",
    "ML": "M-L",
    "SaaS": "sass",
    "B2B": "B-to-B",
    "B2C": "B-to-C",
    "CRM": "C-R-M",
    "ERP": "E-R-P",
    "API": "A-P-I",
    "SDK": "S-D-K",
    "MVP": "M-V-P",
    "Q1": "Q-one",
    "Q2": "Q-two",
    "Q3": "Q-three",
    "Q4": "Q-four",
    "USA": "U-S-A",
    "UK": "U-K",
    "EU": "E-U",
    "LLC": "L-L-C",
    "Inc": "Inc.",
    "Ltd": "Limited",
    "HVAC": "H-VAC",
    "A/B": "A-B",
    "vs": "versus",
    "etc": "et cetera",
    "e.g": "for example",
    "i.e": "that is",
}

# Compiled regex for abbreviations (whole word only, case-sensitive)
_ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(_ABBREVS, key=len, reverse=True)) + r')\b'
)

# ── Number words ──────────────────────────────────────────────────────────────
_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _num_to_words(n: int) -> str:
    """Convert integer to English words (up to 999,999,999)."""
    if n < 0:
        return "negative " + _num_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        rest = _num_to_words(n % 100)
        return _ONES[n // 100] + " hundred" + (" " + rest if rest else "")
    if n < 1_000_000:
        rest = _num_to_words(n % 1000)
        return _num_to_words(n // 1000) + " thousand" + (" " + rest if rest else "")
    if n < 1_000_000_000:
        rest = _num_to_words(n % 1_000_000)
        return _num_to_words(n // 1_000_000) + " million" + (" " + rest if rest else "")
    rest = _num_to_words(n % 1_000_000_000)
    return _num_to_words(n // 1_000_000_000) + " billion" + (" " + rest if rest else "")


def _convert_number(match: re.Match) -> str:
    """Regex replacement function for numbers."""
    text = match.group(0)
    # Strip formatting characters
    clean = re.sub(r'[,\s]', '', text)
    try:
        n = int(clean)
        return _num_to_words(n)
    except ValueError:
        return text


def _convert_currency(match: re.Match) -> str:
    """$500 → 'five hundred dollars' | $1.5M → 'one point five million dollars'"""
    symbol = match.group(1)  # $, €, £
    amount_str = match.group(2).replace(',', '')
    suffix_map = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    suffix = match.group(3).upper() if match.group(3) else ''

    currency_word = {'$': 'dollars', '€': 'euros', '£': 'pounds'}.get(symbol, 'dollars')

    try:
        if '.' in amount_str:
            # Handle decimals like $1.5M
            major, minor = amount_str.split('.', 1)
            major_n = int(major)
            minor_n = int(minor)
            if suffix:
                result = _num_to_words(major_n) + f" point {_num_to_words(minor_n)}"
                multiplier = suffix_map.get(suffix, 1)
                suffix_word = {1_000: 'thousand', 1_000_000: 'million', 1_000_000_000: 'billion'}.get(multiplier, '')
                return result + (" " + suffix_word if suffix_word else "") + " " + currency_word
            return _num_to_words(major_n) + " " + currency_word
        else:
            n = int(amount_str)
            if suffix:
                multiplier = suffix_map.get(suffix, 1)
                suffix_word = {1_000: 'thousand', 1_000_000: 'million', 1_000_000_000: 'billion'}.get(multiplier, '')
                return _num_to_words(n) + " " + suffix_word + " " + currency_word
            return _num_to_words(n) + " " + currency_word
    except (ValueError, KeyError):
        return match.group(0)


def _convert_percent(match: re.Match) -> str:
    """20% → 'twenty percent'"""
    num = match.group(1).replace(',', '')
    try:
        if '.' in num:
            parts = num.split('.')
            return _num_to_words(int(parts[0])) + " point " + _num_to_words(int(parts[1])) + " percent"
        return _num_to_words(int(num)) + " percent"
    except ValueError:
        return match.group(0)


# ── Main normalizer ───────────────────────────────────────────────────────────

class TextNormalizer:
    """
    Cleans LLM output for TTS synthesis.
    Call normalize(text) before passing to XTTS/Qwen3.
    """

    # Patterns applied in order
    _CURRENCY_RE   = re.compile(r'([$€£])(\d[\d,]*(?:\.\d+)?)([KMBkmb]?)\b')
    _PERCENT_RE    = re.compile(r'(\d[\d,]*(?:\.\d+)?)%')
    _NUMBER_RE     = re.compile(r'\b\d[\d,]{2,}\b')   # 3+ digit numbers (1,000 etc.)
    _SIMPLE_NUM_RE = re.compile(r'\b(\d{2,})\b')       # 2+ digit standalone numbers
    _MARKDOWN_RE   = re.compile(r'[*_`#~]|^\s*[-•]\s', re.MULTILINE)
    _URL_RE        = re.compile(r'https?://\S+')
    _ELLIPSIS_RE   = re.compile(r'\.{3,}')
    _MULTI_SPACE   = re.compile(r'  +')

    def normalize(self, text: str) -> str:
        if not text or not text.strip():
            return text

        # 1. Strip markdown artifacts
        text = self._MARKDOWN_RE.sub('', text)
        text = self._URL_RE.sub('', text)

        # 2. Currency (before general numbers)
        text = self._CURRENCY_RE.sub(_convert_currency, text)

        # 3. Percentages
        text = self._PERCENT_RE.sub(_convert_percent, text)

        # 4. Large numbers with commas (1,000 → one thousand)
        text = self._NUMBER_RE.sub(_convert_number, text)

        # 5. Remaining 2+ digit numbers
        text = self._SIMPLE_NUM_RE.sub(lambda m: _num_to_words(int(m.group(1))), text)

        # 6. Abbreviations
        text = _ABBREV_PATTERN.sub(lambda m: _ABBREVS.get(m.group(1), m.group(1)), text)

        # 7. Ellipsis → natural pause (comma works better in XTTS)
        text = self._ELLIPSIS_RE.sub(',', text)

        # 8. Add pause after closing questions (before ?)
        # "Would you be available Tuesday?" → "Would you be available Tuesday —?"
        # Actually, just ensure question marks have space
        text = re.sub(r'\?(?!\s)', '? ', text)

        # 9. Clean up whitespace
        text = self._MULTI_SPACE.sub(' ', text).strip()

        return text

    def split_sentences(self, text: str) -> list[str]:
        """
        Split normalized text into sentences for streaming TTS.
        Each sentence is sent to TTS as soon as LLM generates it.
        """
        # Split on sentence boundaries, keep delimiters
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]


# Singleton
_normalizer: TextNormalizer | None = None


def get_normalizer() -> TextNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = TextNormalizer()
    return _normalizer
