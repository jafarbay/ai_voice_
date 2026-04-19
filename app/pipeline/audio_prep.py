"""Audio normalization step.

Takes any audio file supported by ffmpeg and converts it into the canonical
format the rest of the pipeline expects: 16 kHz mono PCM16 WAV.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


def _ffmpeg_binary() -> str:
    """Return the path to the ffmpeg executable, raising a clear error if missing."""
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install it (https://ffmpeg.org) and make sure "
            "the `ffmpeg` executable is reachable."
        )
    return binary


def normalize_audio(input_path: Path, output_path: Path) -> float:
    """Convert `input_path` into a 16 kHz mono PCM16 WAV at `output_path`.

    Returns
    -------
    float
        Duration of the normalized audio in seconds (derived from the produced
        WAV via `soundfile`, so it reflects the file we will actually feed to
        downstream steps).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"input audio not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", str(TARGET_CHANNELS),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )

    # Derive duration from the produced WAV to avoid an extra ffprobe dependency.
    import soundfile as sf

    info = sf.info(str(output_path))
    return float(info.frames) / float(info.samplerate)
