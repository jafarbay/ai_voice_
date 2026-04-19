"""Attach speaker labels to STT words.

Given the list of ``Word`` objects from faster-whisper and the list of
``SpeakerSegment`` objects from diarization (possibly a stub), assign a
speaker label to each word. We use the **midpoint** of the word's time
interval as the probe — this avoids edge-flicker when a word straddles a
segment boundary.

Rules:

* If the midpoint falls inside a segment ``[start, end)``, pick that
  segment's speaker.
* Otherwise (word extends past the last segment, or tiny gap between
  segments), fall back to the **nearest** segment by distance from the
  midpoint.
* If there are no segments at all, every word gets ``SPEAKER_00``.
"""

from __future__ import annotations

from app.schemas import SpeakerSegment, Word, WordWithSpeaker

_FALLBACK_SPEAKER = "SPEAKER_00"


def _segment_distance(midpoint: float, segment: SpeakerSegment) -> float:
    """Return 0 if midpoint is inside the segment, else the gap to it."""
    if segment.start <= midpoint < segment.end:
        return 0.0
    if midpoint < segment.start:
        return segment.start - midpoint
    return midpoint - segment.end


def _pick_speaker(midpoint: float, segments: list[SpeakerSegment]) -> str:
    """Return the speaker label for the segment closest to ``midpoint``."""
    # Prefer a strict containment match first (handles the common path
    # without the cost of a full scan-with-min on large inputs).
    for seg in segments:
        if seg.start <= midpoint < seg.end:
            return seg.speaker

    # No segment contains the midpoint → pick the nearest by edge distance.
    nearest = min(segments, key=lambda s: _segment_distance(midpoint, s))
    return nearest.speaker


def assign_speakers(
    words: list[Word],
    segments: list[SpeakerSegment],
) -> list[WordWithSpeaker]:
    """Return a new list of ``WordWithSpeaker`` with a speaker per word.

    The input ``words`` list is not mutated; we emit fresh pydantic
    instances so the output type is precise (``speaker: str`` is required
    on ``WordWithSpeaker`` whereas it is optional on ``Word``).
    """
    if not segments:
        return [
            WordWithSpeaker(
                word=w.word,
                start=w.start,
                end=w.end,
                probability=w.probability,
                speaker=_FALLBACK_SPEAKER,
            )
            for w in words
        ]

    result: list[WordWithSpeaker] = []
    for w in words:
        midpoint = (w.start + w.end) / 2.0
        speaker = _pick_speaker(midpoint, segments)
        result.append(
            WordWithSpeaker(
                word=w.word,
                start=w.start,
                end=w.end,
                probability=w.probability,
                speaker=speaker,
            )
        )
    return result
