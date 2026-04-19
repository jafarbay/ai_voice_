"""Unit tests for ``app.pipeline.align.assign_speakers``.

These are pure-Python tests; no audio/models are loaded.
"""

from __future__ import annotations

import pytest

from app.pipeline.align import assign_speakers
from app.schemas import SpeakerSegment, Word


def _w(text: str, start: float, end: float) -> Word:
    return Word(word=text, start=start, end=end, probability=0.9)


def test_single_speaker_stub() -> None:
    """Stub case: one segment covering the whole audio → every word gets it."""
    words = [_w(f"w{i}", float(i), float(i) + 1.0) for i in range(5)]
    segments = [SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")]

    result = assign_speakers(words, segments)

    assert len(result) == 5
    assert all(w.speaker == "SPEAKER_00" for w in result)
    # Underlying word fields must be preserved.
    assert [w.word for w in result] == [f"w{i}" for i in range(5)]
    assert [w.start for w in result] == [float(i) for i in range(5)]


def test_multiple_segments() -> None:
    """Words split across two segments by midpoint."""
    words_a = [_w("a0", 0.0, 1.0), _w("a1", 1.0, 2.0)]  # midpoints 0.5, 1.5 → A
    words_b = [_w("b0", 3.0, 4.0), _w("b1", 4.0, 5.0)]  # midpoints 3.5, 4.5 → B
    words = words_a + words_b

    segments = [
        SpeakerSegment(start=0.0, end=2.5, speaker="A"),
        SpeakerSegment(start=2.5, end=6.0, speaker="B"),
    ]

    result = assign_speakers(words, segments)
    speakers = [w.speaker for w in result]

    assert speakers == ["A", "A", "B", "B"]


def test_word_outside_segments() -> None:
    """Word past the end of all segments falls back to the nearest one."""
    words = [_w("late", 10.0, 11.0)]  # midpoint 10.5, beyond segment end 5
    segments = [SpeakerSegment(start=0.0, end=5.0, speaker="X")]

    result = assign_speakers(words, segments)

    assert len(result) == 1
    assert result[0].speaker == "X"


def test_empty_segments_fallback() -> None:
    """No segments at all → everyone defaults to SPEAKER_00 (defensive)."""
    words = [_w("only", 0.0, 1.0)]
    result = assign_speakers(words, segments=[])
    assert result[0].speaker == "SPEAKER_00"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
