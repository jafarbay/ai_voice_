"""Detect Russian phone numbers spoken as words.

When a phone number is dictated over the phone, Whisper transcribes it as
a long run of Russian number words ("плюс семь девятьсот двадцать семь
четыреста пять девяносто один двадцать два"). The regex detector finds
nothing because there are no digits in the transcript, and the LLM
detector often normalizes the value to "+79274059122" which is no longer
a literal substring of the text, so its span is dropped.

This detector handles exactly that case: it scans the text for maximal
runs of consecutive "phone-like" tokens (number words + an optional
leading "плюс"), estimates how many digits such a run represents, and if
it crosses the phone-length threshold (>= ``MIN_PHONE_DIGITS``) emits a
PIISpan covering the full char range of that run.

The goal is NOT to parse the number precisely — only to identify the
contiguous span in the transcript that needs to be redacted. The
resulting char span can then be mapped to audio time by
``span_to_time.spans_to_time_pairs`` just like any other PII span.
"""

from __future__ import annotations

import re

from app.pipeline.pii.types import PIISpan, PIIType

SOURCE = "word_phone"
CONFIDENCE = 0.85

# Minimum number of spoken digits to qualify as a phone number. Russian
# mobile numbers have 11 digits including the country code; we accept 10+
# to be robust to transcription that drops "восемь"/"плюс семь".
MIN_PHONE_DIGITS = 10

# Digit words -> their numeric value. We accept common gender variants
# (один/одна, два/две) since spoken numbers mix them freely.
_UNITS: dict[str, int] = {
    "ноль": 0,
    "один": 1, "одна": 1, "одну": 1,
    "два": 2, "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
}

_TEENS: dict[str, int] = {
    "десять": 10,
    "одиннадцать": 11,
    "двенадцать": 12,
    "тринадцать": 13,
    "четырнадцать": 14,
    "пятнадцать": 15,
    "шестнадцать": 16,
    "семнадцать": 17,
    "восемнадцать": 18,
    "девятнадцать": 19,
}

_TENS: dict[str, int] = {
    "двадцать": 20,
    "тридцать": 30,
    "сорок": 40,
    "пятьдесят": 50,
    "шестьдесят": 60,
    "семьдесят": 70,
    "восемьдесят": 80,
    "девяносто": 90,
}

_HUNDREDS: dict[str, int] = {
    "сто": 100,
    "двести": 200,
    "триста": 300,
    "четыреста": 400,
    "пятьсот": 500,
    "шестьсот": 600,
    "семьсот": 700,
    "восемьсот": 800,
    "девятьсот": 900,
}

_THOUSANDS: dict[str, int] = {
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
}

# Words that are considered part of a spoken phone sequence.
_PHONE_PREFIX: set[str] = {"плюс"}

# All number-like words we treat as part of a run.
NUMBER_WORDS: dict[str, int] = {
    **_UNITS, **_TEENS, **_TENS, **_HUNDREDS, **_THOUSANDS,
}

# Tokens we allow *inside* a phone run without breaking it. These are
# sometimes emitted by Whisper between digit groups. Kept conservative
# — if we get false positives we can trim this further.
_FILLER_WORDS: set[str] = set()

_WORD_RE = re.compile(r"[а-яёА-ЯЁ]+")


def _token_digits(word: str) -> str:
    """Return the digit string this single word contributes to a phone number.

    Examples:
        "семь"       -> "7"
        "девятьсот"  -> "900"
        "двадцать"   -> "20"
        "плюс"       -> ""   (prefix, no digits but valid in run)
        "лет"        -> ""   (non-number word)
    """
    if word in _UNITS:
        return str(_UNITS[word])
    if word in _TEENS:
        return str(_TEENS[word])
    if word in _TENS:
        return str(_TENS[word])
    if word in _HUNDREDS:
        return str(_HUNDREDS[word])
    if word in _THOUSANDS:
        return str(_THOUSANDS[word])
    return ""


def _is_phone_token(word: str) -> bool:
    """True if the token may participate in a spoken-phone sequence."""
    return (
        word in NUMBER_WORDS
        or word in _PHONE_PREFIX
        or word in _FILLER_WORDS
    )


def detect_word_phones(text: str) -> list[PIISpan]:
    """Find Russian phone numbers written as word sequences.

    Returns a list of PIISpan with ``type=PHONE`` and ``source="word_phone"``
    whose char range covers the full spoken run, including any leading
    "плюс".
    """
    if not text:
        return []

    lowered = text.lower()
    tokens = [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(lowered)]
    if not tokens:
        return []

    spans: list[PIISpan] = []

    i = 0
    n = len(tokens)
    while i < n:
        word, _, _ = tokens[i]
        if not _is_phone_token(word):
            i += 1
            continue

        # Start a run at i, extend while consecutive tokens are phone-like.
        j = i
        digits = ""
        while j < n and _is_phone_token(tokens[j][0]):
            digits += _token_digits(tokens[j][0])
            j += 1

        if len(digits) >= MIN_PHONE_DIGITS:
            start_char = tokens[i][1]
            end_char = tokens[j - 1][2]
            # Use original-case slice from the input text.
            matched = text[start_char:end_char]
            spans.append(
                PIISpan(
                    type=PIIType.PHONE,
                    text=matched,
                    start_char=start_char,
                    end_char=end_char,
                    source=SOURCE,
                    confidence=CONFIDENCE,
                )
            )

        # Advance past this run (or past the single non-qualifying token).
        i = j if j > i else i + 1

    return spans


__all__ = ["detect_word_phones", "NUMBER_WORDS", "MIN_PHONE_DIGITS", "SOURCE"]
