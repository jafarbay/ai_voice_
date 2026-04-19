"""Integration test: audio_prep + faster-whisper STT.

Runs on `tests/fixtures/sample.wav`, checks that we get a plausible word
list with monotonically increasing timestamps and that the transcript
mentions at least one of the expected name tokens (case-insensitive).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.audio_prep import normalize_audio
from app.pipeline.stt import get_transcriber

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.wav"


@pytest.fixture(scope="module")
def normalized_wav(tmp_path_factory) -> Path:
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}. Run `python scripts/gen_sample.py` first.")
    out_dir = tmp_path_factory.mktemp("stt")
    out_path = out_dir / "input.wav"
    duration = normalize_audio(FIXTURE, out_path)
    assert duration > 0.5, f"normalized audio too short: {duration}s"
    return out_path


def test_stt_on_sample(normalized_wav: Path) -> None:
    transcriber = get_transcriber()
    result = transcriber.transcribe(normalized_wav)

    # Report what we got so pytest -s shows it for manual inspection.
    print(f"\n[stt] device={transcriber.device} compute_type={transcriber.compute_type}")
    print(f"[stt] language={result.language} duration={result.duration:.2f}s words={len(result.words)}")
    print(f"[stt] text: {result.text!r}")
    print("[stt] first 10 words:")
    for w in result.words[:10]:
        print(f"  {w.start:6.2f}-{w.end:6.2f} p={w.probability:.2f} {w.word!r}")

    assert result.text, "transcript text is empty"
    assert len(result.words) > 5, f"expected > 5 words, got {len(result.words)}"

    for i, w in enumerate(result.words):
        assert w.start < w.end, f"word {i} has non-increasing times: {w}"
        assert 0.0 <= w.probability <= 1.0, f"word {i} probability out of range: {w}"

    lower = result.text.lower()
    assert ("иван" in lower) or ("петров" in lower), (
        f"transcript missing expected name tokens, got: {result.text!r}"
    )
