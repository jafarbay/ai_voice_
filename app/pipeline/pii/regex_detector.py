"""Regex-based detector for structured Russian PII.

Handles: EMAIL, PHONE (RU), INN (10/12 digits), SNILS (11 digits, XXX-XXX-XXX XX),
PASSPORT (4 digits + 6 digits).

All patterns return *all* matches; overlap resolution is the merger's job.
"""

from __future__ import annotations

import re

from app.pipeline.pii.types import PIISpan, PIIType

SOURCE = "regex"
CONFIDENCE = 0.99

# --- Patterns ---------------------------------------------------------------

# Email: RFC-light. Restrict to ASCII to avoid matching Russian letters.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Russian phone. Accepts +7 / 8 prefix and any mix of spaces / dashes / parens.
# Example: +7 (900) 123-45-67, 8 900 123 45 67, 89001234567.
_PHONE_RE = re.compile(
    r"(?:\+7|8)[\s\-\(]*\d{3}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}"
)

# Passport series+number: 4 digits + optional space + 6 digits.
# Must be bounded by non-digit chars to avoid biting into longer number strings.
_PASSPORT_RE = re.compile(r"(?<!\d)\d{4}\s?\d{6}(?!\d)")

# SNILS: 11 digits in XXX-XXX-XXX XX form (or with any whitespace/dash between).
# Require a specific pattern (not just 11 digits) so we don't mislabel plain numbers.
_SNILS_RE = re.compile(
    r"(?<!\d)\d{3}[\s\-]\d{3}[\s\-]\d{3}[\s\-]\d{2}(?!\d)"
)

# INN: exactly 10 or 12 digits. We use lookarounds to anchor at non-digit
# boundaries; merger handles overlaps with passport (10 digits is unambiguous
# only if not part of a 10+/12+ run).
_INN_RE = re.compile(r"(?<!\d)(?:\d{12}|\d{10})(?!\d)")


# --- Public API -------------------------------------------------------------


def detect_regex(text: str) -> list[PIISpan]:
    """Run all regex patterns and return collected PIISpans.

    Spans may overlap; ``merger.merge_spans`` resolves conflicts.
    """
    spans: list[PIISpan] = []

    for m in _EMAIL_RE.finditer(text):
        spans.append(_make_span(PIIType.EMAIL, m.group(), m.start(), m.end()))

    for m in _PHONE_RE.finditer(text):
        match_text = m.group()
        # Normalize and verify we actually have 11 digits starting with 7/8.
        digits = re.sub(r"\D", "", match_text)
        if len(digits) == 11 and digits[0] in ("7", "8"):
            spans.append(_make_span(PIIType.PHONE, match_text, m.start(), m.end()))

    for m in _PASSPORT_RE.finditer(text):
        spans.append(_make_span(PIIType.PASSPORT, m.group(), m.start(), m.end()))

    for m in _SNILS_RE.finditer(text):
        spans.append(_make_span(PIIType.SNILS, m.group(), m.start(), m.end()))

    for m in _INN_RE.finditer(text):
        spans.append(_make_span(PIIType.INN, m.group(), m.start(), m.end()))

    return spans


def _make_span(pii_type: PIIType, matched: str, start: int, end: int) -> PIISpan:
    return PIISpan(
        type=pii_type,
        text=matched,
        start_char=start,
        end_char=end,
        source=SOURCE,
        confidence=CONFIDENCE,
    )
