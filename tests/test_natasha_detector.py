"""Integration tests for the Natasha NER detector.

Natasha downloads ~70 MB of embedding/tagger weights on first use; after
caching subsequent runs take <1 s.
"""

from __future__ import annotations

import pytest

from app.pipeline.pii.natasha_detector import detect_natasha
from app.schemas import PIIType


@pytest.mark.integration
def test_person_and_location():
    text = "Меня зовут Иван Петров, я живу в Москве."
    spans = detect_natasha(text)
    types = {s.type for s in spans}
    assert PIIType.PERSON in types, f"Expected PERSON, got {[s.type for s in spans]}"
    # LOC may be mapped to ADDRESS.
    assert PIIType.ADDRESS in types, f"Expected ADDRESS (from LOC), got {[s.type for s in spans]}"


@pytest.mark.integration
def test_source_and_spans_point_to_text():
    text = "Иван Петров живёт в Казани."
    spans = detect_natasha(text)
    assert spans, "Expected at least one span"
    for span in spans:
        assert span.source == "natasha"
        # Char offsets must point at the matched text.
        assert text[span.start_char : span.end_char] == span.text


def test_empty_text_returns_empty():
    assert detect_natasha("") == []
    assert detect_natasha("   ") == []
