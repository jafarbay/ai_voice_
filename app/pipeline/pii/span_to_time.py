"""Map PII character spans onto audio time intervals.

The transcript text we feed to PII detectors is a simple space-joined
concatenation of Whisper word tokens (``' '.join(w.text for w in words)``).
We deliberately do NOT use ``TranscriptResult.text`` because Whisper may add
punctuation/spacing that desyncs character offsets from word boundaries.

For every ``PIISpan`` we look at which words its ``[start_char, end_char)``
interval overlaps (inclusive of touching boundaries only on the left side,
half-open range on the right), and take ``min(word.start)`` / ``max(word.end)``
as the audio time window.
"""

from __future__ import annotations

import logging

from app.pipeline.pii.types import PIISpan
from app.schemas import TimeInterval, Word

log = logging.getLogger(__name__)


def build_full_text(words: list[Word]) -> str:
    """Reconstruct the transcript text exactly as fed to the PII detectors.

    Uses a single space as a separator — this must match what the orchestrator
    feeds to regex / Natasha / LLM detectors so character offsets line up with
    the word char-index produced by :func:`build_word_char_index`.
    """
    return " ".join(_word_surface(w) for w in words)


def _word_surface(w: Word) -> str:
    """Return the plain surface form of a word with no leading/trailing spaces.

    Whisper sometimes emits tokens with a leading space (e.g. ``" Иван"``); we
    strip it so the single-space join is uniform.
    """
    return w.word.strip()


def build_word_char_index(
    words: list[Word],
) -> tuple[str, list[tuple[int, int, int]]]:
    """Return ``(full_text, [(word_idx, char_start, char_end), ...])``.

    ``char_start`` / ``char_end`` are half-open offsets into ``full_text``
    referring to that word's surface substring (no surrounding spaces).
    """
    parts: list[str] = []
    index: list[tuple[int, int, int]] = []
    cursor = 0
    for i, w in enumerate(words):
        surface = _word_surface(w)
        if i > 0:
            # Single-space separator between words. Account for it in the cursor.
            parts.append(" ")
            cursor += 1
        parts.append(surface)
        start = cursor
        end = cursor + len(surface)
        index.append((i, start, end))
        cursor = end
    return "".join(parts), index


def spans_to_time_pairs(
    spans: list[PIISpan],
    words: list[Word],
    full_text: str | None = None,
) -> list[tuple[PIISpan, TimeInterval]]:
    """Like :func:`spans_to_time`, but return ``(span, interval)`` pairs.

    Skipped spans (no word overlap) are omitted from the result, so the
    caller can zip the surviving spans with their audio intervals.
    """
    if not spans:
        return []
    if not words:
        log.warning(
            "spans_to_time_pairs called with empty word list; returning no intervals"
        )
        return []

    reconstructed, index = build_word_char_index(words)
    if full_text is not None and full_text != reconstructed:
        # Not fatal — detectors may have run on a text that was constructed
        # the same way; we still warn because char offsets may drift.
        log.warning(
            "full_text passed to spans_to_time does not match build_full_text(words); "
            "char offsets may be misaligned",
        )

    pairs: list[tuple[PIISpan, TimeInterval]] = []
    for span in spans:
        matched: list[int] = []
        for word_idx, c_start, c_end in index:
            # Half-open interval overlap on [span.start_char, span.end_char) vs
            # [c_start, c_end).
            if span.start_char < c_end and c_start < span.end_char:
                matched.append(word_idx)

        if not matched:
            log.warning(
                "span %s [%d:%d] '%s' did not overlap any word; skipping",
                span.type, span.start_char, span.end_char, span.text,
            )
            continue

        t_start = min(words[i].start for i in matched)
        t_end = max(words[i].end for i in matched)
        interval = TimeInterval(
            start=float(t_start),
            end=float(t_end),
            pii_type=span.type,
            text=span.text,
            source=span.source,
        )
        pairs.append((span, interval))

    return pairs


def spans_to_time(
    spans: list[PIISpan],
    words: list[Word],
    full_text: str | None = None,
) -> list[TimeInterval]:
    """Convert character-indexed PII spans into audio time intervals.

    ``full_text`` is accepted for API compatibility but is not strictly needed
    — we rebuild the word char-index from ``words`` so that char offsets are
    guaranteed consistent with the text used upstream (provided the caller
    also used :func:`build_full_text`).
    """
    return [interval for _span, interval in spans_to_time_pairs(spans, words, full_text)]


__all__ = [
    "build_full_text",
    "build_word_char_index",
    "spans_to_time",
    "spans_to_time_pairs",
]
