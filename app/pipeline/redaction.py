"""Audio redaction: replace PII time intervals with a 1 kHz beep.

Uses ``soundfile`` + ``numpy``. Input is expected to be mono float audio (after
:mod:`app.pipeline.audio_prep`), but stereo input is tolerated by mixing to
mono. A short (10 ms by default) linear fade in/out is applied at each beep
boundary to suppress audible clicks at sample discontinuities.

Output is written as 16-bit PCM WAV at the same sample rate as the input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from app.schemas import TimeInterval

log = logging.getLogger(__name__)

_DEFAULT_FADE_MS = 10.0


def _apply_fade(beep: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply a linear fade in/out to the beep buffer in place."""
    length = beep.shape[0]
    if length == 0 or fade_samples <= 0:
        return beep

    # If beep is shorter than 2 * fade, scale the fade down to half the length
    # so in and out don't overlap destructively.
    fs = min(fade_samples, length // 2)
    if fs <= 0:
        return beep

    fade_in = np.linspace(0.0, 1.0, fs, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fs, dtype=np.float32)
    beep[:fs] *= fade_in
    beep[length - fs:] *= fade_out
    return beep


def _make_beep(length: int, sr: int, freq_hz: float, amplitude: float) -> np.ndarray:
    t = np.arange(length, dtype=np.float32) / float(sr)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def redact_audio(
    input_wav: Path,
    output_wav: Path,
    intervals: Iterable[TimeInterval],
    beep_freq_hz: float = 1000.0,
    beep_amplitude: float = 0.2,
    fade_ms: float = _DEFAULT_FADE_MS,
) -> None:
    """Replace each interval with a faded 1 kHz sine beep and write the result.

    Parameters
    ----------
    input_wav:
        Path to the source WAV file (expected mono 16 kHz float32 after
        ``audio_prep``; stereo input is mixed to mono).
    output_wav:
        Destination path. Written as 16-bit PCM WAV.
    intervals:
        PII time intervals to beep out.
    beep_freq_hz:
        Beep sine frequency in Hz. Default 1000 Hz.
    beep_amplitude:
        Peak amplitude in the [-1, 1] float domain. Default 0.2.
    fade_ms:
        Linear fade in/out duration in milliseconds applied at each beep
        boundary to prevent audible clicks. Default 10 ms. Pass 0 to disable.
    """
    input_wav = Path(input_wav)
    output_wav = Path(output_wav)
    if not input_wav.exists():
        raise FileNotFoundError(f"audio not found: {input_wav}")

    audio, sr = sf.read(str(input_wav), always_2d=False)

    # Ensure float32 mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=True)

    total_samples = audio.shape[0]
    fade_samples = int(round(fade_ms * 1e-3 * sr)) if fade_ms > 0 else 0

    for interval in intervals:
        start_sample = int(interval.start * sr)
        end_sample = int(interval.end * sr)
        # Clip to audio bounds.
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(0, min(end_sample, total_samples))
        length = end_sample - start_sample
        if length <= 0:
            log.warning(
                "skipping redaction interval [%.3f, %.3f] — empty after clipping",
                interval.start, interval.end,
            )
            continue

        beep = _make_beep(length, sr, beep_freq_hz, beep_amplitude)
        beep = _apply_fade(beep, fade_samples)
        audio[start_sample:end_sample] = beep

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio, sr, subtype="PCM_16")


__all__ = ["redact_audio"]
