"""Speaker diarization via pyannote.audio 3.1.

Loaded lazily. If the model load fails (no token, terms not accepted, no
CUDA, network down) we log a warning and fall back to the single-speaker
stub so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import get_settings
from app.schemas import SpeakerSegment

log = logging.getLogger(__name__)

_pipeline: Any | None = None
_load_failed = False
_lock = Lock()


def get_diarizer() -> Any | None:
    """Lazy singleton. Returns a pyannote Pipeline or None on failure."""
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline

    with _lock:
        if _pipeline is not None or _load_failed:
            return _pipeline

        settings = get_settings()
        token = settings.huggingface_token
        if not token:
            log.warning("HUGGINGFACE_TOKEN missing; diarization disabled")
            _load_failed = True
            return None

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            log.warning("pyannote.audio not installed: %s", exc)
            _load_failed = True
            return None

        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
        except Exception as exc:
            log.warning("pyannote pipeline load failed: %s", exc)
            _load_failed = True
            return None

        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            log.info("pyannote diarization on CUDA")
        else:
            log.info("pyannote diarization on CPU (no CUDA)")

        _pipeline = pipeline
        return _pipeline


def diarize(wav_path: Path, duration_sec: float) -> list[SpeakerSegment]:
    """Return speaker segments for ``wav_path``.

    Falls back to a single SPEAKER_00 segment if diarization is disabled
    or the pipeline fails to load.
    """
    settings = get_settings()
    if not settings.enable_diarization:
        return [SpeakerSegment(start=0.0, end=float(duration_sec), speaker="SPEAKER_00")]

    pipeline = get_diarizer()
    if pipeline is None:
        return [SpeakerSegment(start=0.0, end=float(duration_sec), speaker="SPEAKER_00")]

    try:
        annotation = pipeline(str(wav_path))
    except Exception as exc:
        log.warning("pyannote inference failed: %s", exc)
        return [SpeakerSegment(start=0.0, end=float(duration_sec), speaker="SPEAKER_00")]

    segments = [
        SpeakerSegment(start=float(turn.start), end=float(turn.end), speaker=str(label))
        for turn, _, label in annotation.itertracks(yield_label=True)
    ]
    if not segments:
        return [SpeakerSegment(start=0.0, end=float(duration_sec), speaker="SPEAKER_00")]
    return segments
