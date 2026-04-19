"""Unit tests for the regex-based PII detector."""

from __future__ import annotations

from app.pipeline.pii.regex_detector import detect_regex
from app.schemas import PIIType


def _types(spans) -> list[PIIType]:
    return [s.type for s in spans]


def test_phone_plain():
    text = "Позвоните +7 900 123 45 67"
    spans = detect_regex(text)
    phones = [s for s in spans if s.type == PIIType.PHONE]
    assert len(phones) == 1
    # Span must point at the phone substring.
    assert "900" in phones[0].text
    assert text[phones[0].start_char : phones[0].end_char] == phones[0].text


def test_email():
    text = "Моя почта ivan@example.com"
    spans = detect_regex(text)
    emails = [s for s in spans if s.type == PIIType.EMAIL]
    assert len(emails) == 1
    assert emails[0].text == "ivan@example.com"


def test_inn_10_digits():
    text = "ИНН 1234567890"
    spans = detect_regex(text)
    inns = [s for s in spans if s.type == PIIType.INN]
    assert len(inns) == 1
    assert inns[0].text == "1234567890"


def test_snils():
    text = "СНИЛС 123-456-789 12"
    spans = detect_regex(text)
    snils = [s for s in spans if s.type == PIIType.SNILS]
    assert len(snils) == 1
    assert "123" in snils[0].text


def test_passport():
    text = "Паспорт 1234 567890"
    spans = detect_regex(text)
    passports = [s for s in spans if s.type == PIIType.PASSPORT]
    assert len(passports) == 1
    assert passports[0].text == "1234 567890"


def test_mixed_three_types():
    text = "Контакт: +7 999 111 22 33, ivan@mail.ru, ИНН 9876543210"
    spans = detect_regex(text)
    types = set(_types(spans))
    # At minimum all three distinct types must show up.
    assert PIIType.PHONE in types
    assert PIIType.EMAIL in types
    assert PIIType.INN in types
    # Check we've got at least three spans (could be more if overlaps).
    assert len(spans) >= 3


def test_confidence_and_source():
    spans = detect_regex("email: a@b.co")
    assert all(s.source == "regex" for s in spans)
    assert all(s.confidence > 0.9 for s in spans)


def test_no_false_positive_empty():
    assert detect_regex("") == []
    assert detect_regex("просто текст без PII") == []
