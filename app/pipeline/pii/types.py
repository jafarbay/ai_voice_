"""Re-exports of PII types / constants used by detectors and merger.

This module is deliberately tiny: detectors import PIIType / PIISpan from
`app.schemas`, but re-export them here so internal pipeline code can stay
decoupled from the outward-facing schema module if it grows in the future.
"""

from __future__ import annotations

from app.schemas import PIISource, PIISpan, PIIType

__all__ = ["PIIType", "PIISpan", "PIISource"]

# Priority for conflict resolution in merger: higher wins.
# ``word_phone`` sits between regex and llm: it's nearly as precise as
# regex on its narrow target (long runs of Russian number words) and
# much more reliable than the LLM, which tends to normalize phone
# numbers into digits that are no longer literal substrings of the text.
SOURCE_PRIORITY: dict[str, int] = {
    "regex": 4,
    "word_phone": 3,
    "llm": 2,
    "natasha": 1,
}
