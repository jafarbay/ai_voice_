"""Pack a job's artifacts into a single in-memory ZIP."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from app.config import get_settings
from app.storage.files import JobPaths


def build_zip(job_id: str) -> bytes:
    """Return a ZIP byte blob with all available artifacts for ``job_id``."""
    paths = JobPaths(job_id, Path(get_settings().data_dir))
    candidates = [
        paths.input_wav,
        paths.redacted_wav,
        paths.transcript_full,
        paths.transcript_redacted,
        paths.events,
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for src in candidates:
            if src.exists():
                zf.write(src, arcname=f"{job_id}/{src.name}")

    return buf.getvalue()
