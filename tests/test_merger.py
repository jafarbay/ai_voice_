"""Unit tests for merger.merge_spans."""

from __future__ import annotations

from app.pipeline.pii.merger import merge_spans
from app.schemas import PIISpan, PIIType


def _mk(type_, start, end, text=None, source="regex", conf=0.9):
    if text is None:
        text = "x" * (end - start)
    return PIISpan(
        type=type_,
        text=text,
        start_char=start,
        end_char=end,
        source=source,
        confidence=conf,
    )


def test_empty_input():
    assert merge_spans([]) == []


def test_non_overlapping_preserved_and_sorted():
    a = _mk(PIIType.PHONE, 0, 5)
    b = _mk(PIIType.EMAIL, 20, 30)
    c = _mk(PIIType.INN, 40, 50)
    out = merge_spans([b, a, c])
    assert [s.start_char for s in out] == [0, 20, 40]
    assert len(out) == 3


def test_exact_duplicates_collapsed():
    a = _mk(PIIType.PHONE, 0, 10, text="+79001234567")
    b = _mk(PIIType.PHONE, 0, 10, text="+79001234567")
    out = merge_spans([a, b])
    assert len(out) == 1


def test_regex_beats_natasha_on_overlap():
    regex_span = _mk(PIIType.PHONE, 5, 17, source="regex", conf=0.99)
    natasha_span = _mk(PIIType.PERSON, 10, 20, source="natasha", conf=0.85)
    out = merge_spans([natasha_span, regex_span])
    assert len(out) == 1
    assert out[0].source == "regex"
    assert out[0].type == PIIType.PHONE


def test_llm_beats_natasha_on_overlap():
    llm_span = _mk(PIIType.PHONE, 5, 17, source="llm", conf=0.8)
    natasha_span = _mk(PIIType.PERSON, 10, 20, source="natasha", conf=0.85)
    out = merge_spans([natasha_span, llm_span])
    assert len(out) == 1
    assert out[0].source == "llm"


def test_regex_beats_llm_on_overlap():
    regex_span = _mk(PIIType.PHONE, 5, 17, source="regex")
    llm_span = _mk(PIIType.PHONE, 10, 20, source="llm")
    out = merge_spans([llm_span, regex_span])
    assert len(out) == 1
    assert out[0].source == "regex"


def test_same_source_longer_wins():
    short = _mk(PIIType.PERSON, 5, 10, source="natasha")
    longer = _mk(PIIType.PERSON, 5, 20, source="natasha")
    out = merge_spans([short, longer])
    assert len(out) == 1
    assert out[0].end_char == 20


def test_output_sorted_by_start_char():
    spans = [
        _mk(PIIType.EMAIL, 50, 60, source="regex"),
        _mk(PIIType.PHONE, 0, 10, source="regex"),
        _mk(PIIType.PERSON, 20, 30, source="natasha"),
    ]
    out = merge_spans(spans)
    assert [s.start_char for s in out] == sorted(s.start_char for s in out)


def test_adjacent_but_not_overlapping_both_kept():
    a = _mk(PIIType.PHONE, 0, 10, source="regex")
    b = _mk(PIIType.EMAIL, 10, 20, source="regex")  # touches but does not overlap
    out = merge_spans([a, b])
    assert len(out) == 2
