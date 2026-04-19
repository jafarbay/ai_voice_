"""Tests for the spoken-Russian phone number detector."""
from __future__ import annotations

from app.pipeline.pii.word_phone_detector import detect_word_phones
from app.pipeline.pii.types import PIIType

USER_REAL_PHRASE = (
    "Мой номер телефона. Это плюс семь девятьсот двадцать семь "
    "четыреста пять девяносто один двадцать два."
)


def test_user_real_phone_detected():
    spans = detect_word_phones(USER_REAL_PHRASE)
    assert len(spans) == 1
    s = spans[0]
    assert s.type == PIIType.PHONE
    assert s.source == "word_phone"
    matched = USER_REAL_PHRASE[s.start_char:s.end_char]
    assert matched.lower().startswith("плюс семь")
    assert matched.lower().endswith("двадцать два")


def test_no_false_positive_on_short_age():
    assert detect_word_phones("Мне семь лет.") == []
    assert detect_word_phones("Мне двадцать пять лет.") == []


def test_no_false_positive_on_empty_text():
    assert detect_word_phones("") == []
    assert detect_word_phones("Просто слова без чисел.") == []


def test_two_phones_in_text():
    text = (
        "Первый номер плюс семь девятьсот двадцать семь четыреста пять "
        "девяносто один двадцать два, а второй восемь девятьсот пятнадцать "
        "ноль ноль ноль ноль один ноль один."
    )
    spans = detect_word_phones(text)
    assert len(spans) >= 1
    matched_first = text[spans[0].start_char:spans[0].end_char].lower()
    assert "плюс семь" in matched_first


def test_phone_without_plus_prefix():
    text = "Звоните восемь девятьсот пятнадцать ноль ноль ноль ноль один ноль один."
    spans = detect_word_phones(text)
    assert len(spans) == 1
    assert spans[0].type == PIIType.PHONE


def test_confidence_and_source():
    spans = detect_word_phones(USER_REAL_PHRASE)
    assert spans[0].confidence == 0.85
    assert spans[0].source == "word_phone"
