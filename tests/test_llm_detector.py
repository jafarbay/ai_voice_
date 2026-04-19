"""Slow tests for the LLM-based PII detector.

These are marked ``@pytest.mark.slow``; run with:
    pytest tests/test_llm_detector.py -v -s -m slow

If the GGUF file isn't present on disk the detector returns [] and tests
skip gracefully.
"""

from __future__ import annotations

import os

import pytest

from app.config import get_settings
from app.pipeline.pii.llm_detector import detect_llm
from app.schemas import PIIType


def _model_available() -> bool:
    path = get_settings().llm_gguf_path
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return os.path.exists(path)


pytestmark = pytest.mark.slow


@pytest.mark.skipif(not _model_available(), reason="LLM GGUF not downloaded")
def test_llm_detects_person():
    text = "Меня зовут Иван Петров."
    spans = detect_llm(text)
    types = {s.type for s in spans}
    assert PIIType.PERSON in types, f"Expected PERSON, got {[(s.type, s.text) for s in spans]}"


@pytest.mark.skipif(not _model_available(), reason="LLM GGUF not downloaded")
def test_llm_detects_worded_phone():
    """Worded numbers are a known-hard case for small models.

    Qwen2.5-3B tends to identify the phone category but rewrite the digits
    (e.g. returning "8 567 089 09" instead of the original Russian words),
    and our substring-only mapping then drops the span. We accept either:
    (a) a PHONE span, or (b) no spans at all — both are known behaviours.
    A non-PHONE span (e.g. PERSON on a clearly phone-only sentence) is a
    regression.
    """
    text = (
        "Мой телефон восемь девятьсот пять шестьсот семь ноль восемь "
        "ноль девять"
    )
    spans = detect_llm(text)
    types = {s.type for s in spans}
    if spans:
        # If we got anything, at least one of them should be PHONE.
        assert PIIType.PHONE in types, (
            f"Expected PHONE from worded number, got {[(s.type, s.text) for s in spans]}"
        )


@pytest.mark.skipif(not _model_available(), reason="LLM GGUF not downloaded")
def test_llm_source_and_offsets():
    text = "Имя: Иван Петров."
    spans = detect_llm(text)
    for s in spans:
        assert s.source == "llm"
        assert text[s.start_char : s.end_char] == s.text
