"""Real pyannote diarization integration. CUDA-only; skipped otherwise."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

cuda = pytest.importorskip("torch").cuda
if not cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from app.config import get_settings
from app.pipeline import diarization


def test_get_diarizer_returns_pipeline_or_none():
    """Should return a pyannote Pipeline instance or None (HF gate / no token)."""
    diar = diarization.get_diarizer()
    assert diar is None or hasattr(diar, "__call__")


@pytest.mark.skipif(
    not Path("tests/fixtures/two_speakers.wav").exists(),
    reason="two-speaker fixture not present",
)
def test_diarize_two_speakers_returns_distinct_labels(tmp_path):
    """A 2-speaker fixture should yield at least 2 distinct speaker IDs."""
    src = Path("tests/fixtures/two_speakers.wav")
    dst = tmp_path / "two.wav"
    shutil.copy(src, dst)
    segments = diarization.diarize(dst, duration_sec=10.0)
    speakers = {s.speaker for s in segments}
    assert len(speakers) >= 2, f"expected ≥2 speakers, got {speakers}"
