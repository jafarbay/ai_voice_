"""Filesystem layout helpers for a single job.

Every job gets its own directory under ``settings.data_dir``:

    data/<job_id>/
        input<ext>               # original upload (extension varies)
        input.wav                # normalized 16 kHz mono PCM
        redacted.wav             # audio with PII intervals beeped out
        transcript_full.json     # full transcript with word timings
        transcript_redacted.json # transcript with PII words replaced
        events.jsonl             # append-only pipeline / PII event log

The ``JobPaths`` object centralizes these paths so the orchestrator, API
routes and tests share a single source of truth.
"""

from __future__ import annotations

from pathlib import Path


class JobPaths:
    """Concrete filesystem locations for a single job id.

    ``data_dir`` is the project-wide data root (``settings.data_dir``); the
    job's own files live under ``<data_dir>/<job_id>/``.
    """

    def __init__(self, job_id: str, data_dir: Path) -> None:
        self.job_id = job_id
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / job_id

        # File slots — ``input_raw`` has no extension; use ``input_with_ext``
        # when you need the real upload path.
        self.input_raw = self.root / "input"
        self.input_wav = self.root / "input.wav"
        self.redacted_wav = self.root / "redacted.wav"
        self.transcript_full = self.root / "transcript_full.json"
        self.transcript_redacted = self.root / "transcript_redacted.json"
        self.events = self.root / "events.jsonl"

    def ensure(self) -> None:
        """Create the job directory (and parents) if it doesn't exist yet."""
        self.root.mkdir(parents=True, exist_ok=True)

    def input_with_ext(self, ext: str) -> Path:
        """Return the ``input<ext>`` path, tolerating ``ext`` with/without dot."""
        if ext and not ext.startswith("."):
            ext = "." + ext
        return self.root / f"input{ext}"


__all__ = ["JobPaths"]
