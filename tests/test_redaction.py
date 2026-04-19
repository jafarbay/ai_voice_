"""Unit tests for app.pipeline.redaction.redact_audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline.redaction import redact_audio
from app.schemas import PIIType, TimeInterval


SR = 16000
DURATION = 2.0
TONE_FREQ = 440.0


def _make_tone_wav(tmp_path: Path) -> Path:
    n = int(SR * DURATION)
    t = np.arange(n, dtype=np.float32) / SR
    audio = (0.5 * np.sin(2.0 * np.pi * TONE_FREQ * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(str(path), audio, SR, subtype="PCM_16")
    return path


def _interval(start: float, end: float) -> TimeInterval:
    return TimeInterval(
        start=start,
        end=end,
        pii_type=PIIType.PERSON,
        text="X",
        source="regex",
    )


def test_beep_replaces_interval(tmp_path):
    src = _make_tone_wav(tmp_path)
    dst = tmp_path / "out.wav"
    intervals = [_interval(0.5, 1.0)]
    redact_audio(src, dst, intervals, beep_freq_hz=1000.0, beep_amplitude=0.2, fade_ms=0.0)

    audio, sr = sf.read(str(dst))
    assert sr == SR
    start_sample = int(0.5 * sr)
    end_sample = int(1.0 * sr)
    redacted = audio[start_sample:end_sample]

    # FFT: the dominant frequency inside the redacted window must be ~1 kHz.
    spectrum = np.abs(np.fft.rfft(redacted))
    freqs = np.fft.rfftfreq(redacted.shape[0], d=1.0 / sr)
    peak_freq = freqs[int(np.argmax(spectrum))]
    assert abs(peak_freq - 1000.0) < 5.0, f"expected ~1000 Hz, got {peak_freq:.2f} Hz"

    # And outside the interval the original 440 Hz should still dominate.
    before = audio[: start_sample]
    spec_before = np.abs(np.fft.rfft(before))
    freqs_before = np.fft.rfftfreq(before.shape[0], d=1.0 / sr)
    peak_before = freqs_before[int(np.argmax(spec_before))]
    assert abs(peak_before - TONE_FREQ) < 5.0


def test_fade_no_clicks(tmp_path):
    src = _make_tone_wav(tmp_path)
    dst = tmp_path / "out.wav"
    intervals = [_interval(0.5, 1.0)]
    # Explicitly enable 10 ms fade.
    redact_audio(src, dst, intervals, fade_ms=10.0, beep_amplitude=0.2)

    audio, sr = sf.read(str(dst))
    start_sample = int(0.5 * sr)
    end_sample = int(1.0 * sr)

    # With fade-in, the first beep sample must be ~0 (not full amplitude),
    # so the discontinuity at the boundary is small.
    last_orig = audio[start_sample - 1]
    first_beep = audio[start_sample]
    assert abs(first_beep) < 0.05, (
        f"first beep sample should be near zero after fade-in, got {first_beep:.4f}"
    )
    # And the jump at the boundary should be well under the tone's peak.
    assert abs(first_beep - last_orig) < 0.6

    # Symmetric check on the trailing boundary.
    last_beep = audio[end_sample - 1]
    first_after = audio[end_sample]
    assert abs(last_beep) < 0.05, (
        f"last beep sample should be near zero after fade-out, got {last_beep:.4f}"
    )
    assert abs(first_after - last_beep) < 0.6


def test_empty_intervals_noop(tmp_path):
    src = _make_tone_wav(tmp_path)
    dst = tmp_path / "out.wav"
    redact_audio(src, dst, [])

    a_in, sr_in = sf.read(str(src))
    a_out, sr_out = sf.read(str(dst))
    assert sr_in == sr_out
    assert a_in.shape == a_out.shape
    # 16-bit PCM round-trip introduces tiny quantization noise (< 1/32768 ~ 3e-5).
    assert np.max(np.abs(a_in - a_out)) < 1e-3


def test_preserves_length(tmp_path):
    src = _make_tone_wav(tmp_path)
    dst = tmp_path / "out.wav"
    intervals = [_interval(0.2, 0.4), _interval(1.1, 1.3)]
    redact_audio(src, dst, intervals)

    a_in, sr_in = sf.read(str(src))
    a_out, sr_out = sf.read(str(dst))
    assert len(a_in) == len(a_out)
    assert sr_in == sr_out


def test_out_of_bounds_clipped(tmp_path):
    src = _make_tone_wav(tmp_path)
    dst = tmp_path / "out.wav"
    # DURATION = 2.0 s; request an interval that extends past the end.
    intervals = [_interval(1.8, 5.0)]
    redact_audio(src, dst, intervals, fade_ms=0.0, beep_amplitude=0.2)

    audio, sr = sf.read(str(dst))
    assert len(audio) == int(SR * DURATION)
    # The last portion (from 1.8 s to end) should be a 1 kHz beep now.
    tail = audio[int(1.8 * sr):]
    spectrum = np.abs(np.fft.rfft(tail))
    freqs = np.fft.rfftfreq(tail.shape[0], d=1.0 / sr)
    peak_freq = freqs[int(np.argmax(spectrum))]
    assert abs(peak_freq - 1000.0) < 10.0


def test_missing_input_raises(tmp_path):
    dst = tmp_path / "out.wav"
    with pytest.raises(FileNotFoundError):
        redact_audio(tmp_path / "nope.wav", dst, [])
