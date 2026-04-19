"""Unit tests for app.pipeline.pii.span_to_time."""

from __future__ import annotations

from app.pipeline.pii.span_to_time import (
    build_full_text,
    build_word_char_index,
    spans_to_time,
)
from app.schemas import PIISpan, PIIType, Word


def _words() -> list[Word]:
    return [
        Word(word="Меня", start=0.0, end=0.4, probability=1.0),
        Word(word="зовут", start=0.4, end=0.8, probability=1.0),
        Word(word="Иван", start=0.8, end=1.1, probability=1.0),
        Word(word="Петров", start=1.1, end=1.5, probability=1.0),
    ]


def test_build_full_text():
    assert build_full_text(_words()) == "Меня зовут Иван Петров"


def test_build_word_char_index_offsets_match_surface():
    words = _words()
    full_text, index = build_word_char_index(words)
    assert full_text == "Меня зовут Иван Петров"
    # Each recorded offset must point back at the word's surface form.
    for i, ((word_idx, c_start, c_end), w) in enumerate(zip(index, words, strict=True)):
        assert word_idx == i
        assert full_text[c_start:c_end] == w.word


def test_single_word_span():
    words = _words()
    # Span covers only "Иван"
    span = PIISpan(
        type=PIIType.PERSON,
        text="Иван",
        start_char=11,
        end_char=15,
        source="natasha",
        confidence=0.9,
    )
    intervals = spans_to_time([span], words, build_full_text(words))
    assert len(intervals) == 1
    iv = intervals[0]
    assert iv.start == 0.8
    assert iv.end == 1.1
    assert iv.pii_type == PIIType.PERSON
    assert iv.text == "Иван"
    assert iv.source == "natasha"


def test_multi_word_span():
    words = _words()
    full_text = build_full_text(words)
    # "Иван Петров" -> 11..22
    assert full_text[11:22] == "Иван Петров"
    span = PIISpan(
        type=PIIType.PERSON,
        text="Иван Петров",
        start_char=11,
        end_char=22,
        source="natasha",
    )
    intervals = spans_to_time([span], words, full_text)
    assert len(intervals) == 1
    iv = intervals[0]
    assert iv.start == 0.8
    assert iv.end == 1.5


def test_span_over_word_boundary():
    """Span whose char range crosses the separating space between two words
    should capture both adjacent words."""
    words = _words()
    full_text = build_full_text(words)
    # Char indices for "Меня зовут Иван Петров":
    #   "Меня" 0..4, " " 4, "зовут" 5..10, " " 10, "Иван" 11..15, " " 15, "Петров" 16..22
    # Range 9..12 covers last char of "зовут" + space + first char of "Иван".
    assert full_text[9:12] == "т И"
    span = PIISpan(
        type=PIIType.PERSON,
        text=full_text[9:12],
        start_char=9,
        end_char=12,
        source="llm",
    )
    intervals = spans_to_time([span], words, full_text)
    assert len(intervals) == 1
    iv = intervals[0]
    # Should span from start of "зовут" (0.4) to end of "Иван" (1.1).
    assert iv.start == 0.4
    assert iv.end == 1.1


def test_no_match_skipped():
    """Span whose char positions lie outside the full_text must be skipped,
    not crash."""
    words = _words()
    full_text = build_full_text(words)
    bogus = PIISpan(
        type=PIIType.PERSON,
        text="ghost",
        start_char=1000,
        end_char=1005,
        source="llm",
    )
    intervals = spans_to_time([bogus], words, full_text)
    assert intervals == []


def test_empty_spans_returns_empty():
    assert spans_to_time([], _words(), build_full_text(_words())) == []


def test_empty_words_returns_empty():
    span = PIISpan(
        type=PIIType.PERSON, text="x", start_char=0, end_char=1, source="regex"
    )
    assert spans_to_time([span], [], "x") == []
