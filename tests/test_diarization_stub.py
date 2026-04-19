"""Integration test for the diarization fallback.

When ``ENABLE_DIARIZATION=False`` the ``diarize`` function returns exactly
one ``SPEAKER_00`` segment covering the whole recording. No audio is
actually analysed in that code path, so this runs in milliseconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import get_settings
from app.pipeline.diarization import diarize

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.wav"
SAMPLE_DURATION_SEC = 17.81


def test_stub_returns_single_speaker_segment() -> None:
    settings = get_settings()
    # This test exercises the fallback path; skip if pyannote is enabled.
    if settings.enable_diarization:
        pytest.skip("ENABLE_DIARIZATION=true — fallback test not applicable")

    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}. Run `python scripts/gen_sample.py` first.")

    segments = diarize(FIXTURE, duration_sec=SAMPLE_DURATION_SEC)

    assert len(segments) == 1, f"stub must return exactly one segment, got {len(segments)}"
    seg = segments[0]
    assert seg.speaker == "SPEAKER_00"
    assert seg.start == 0.0
    assert seg.end == pytest.approx(SAMPLE_DURATION_SEC)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
