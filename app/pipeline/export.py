"""Pack a job's artifacts into a single in-memory ZIP."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from app.config import get_settings
from app.storage.files import JobPaths


def build_zip(job_id: str) -> bytes:
    """Return a ZIP byte blob with the anonymized artifacts for ``job_id``.

    The archive deliberately excludes the original input audio and the
    full (non-redacted) transcript — those contain raw PII and must not
    leave the server. Only the redacted audio, the redacted transcript
    and the PII event log are shipped.
    """
    paths = JobPaths(job_id, Path(get_settings().data_dir))
    candidates = [
        paths.redacted_wav,
        paths.transcript_redacted,
        paths.events,
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in candidates:
            if src.exists():
                zf.write(src, arcname=f"{job_id}/{src.name}")

    return buf.getvalue()
