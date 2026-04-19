"""REST API for job upload, status polling and artifact retrieval.

``POST /jobs`` saves the upload to ``data/<job_id>/input.<ext>``, inserts
a ``queued`` row in SQLite, and schedules :func:`run_pipeline` via
:class:`fastapi.BackgroundTasks`. Because ``run_pipeline`` is sync,
FastAPI runs it in the default threadpool. All IO paths come from
:class:`app.storage.files.JobPaths`; all DB calls go through
:mod:`app.db`.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response

from app import db
from app.config import get_settings
from app.pipeline.export import build_zip
from app.pipeline.orchestrator import run_pipeline
from app.schemas import (
    EventItem,
    EventsResponse,
    JobCreatedResponse,
    JobInfoResponse,
)
from app.storage.files import JobPaths

router = APIRouter(prefix="/jobs", tags=["jobs"])


ALLOWED_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".opus", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB

TranscriptVersion = Literal["full", "redacted"]
AudioVersion = Literal["original", "redacted"]


def _job_row_to_info(
    row: dict, *, pii_count: int | None = None
) -> JobInfoResponse:
    """Adapt a raw ``jobs`` row dict to the public response schema."""
    return JobInfoResponse(
        id=row["id"],
        created_at=row["created_at"],
        status=row["status"],
        input_filename=row.get("input_filename"),
        duration_sec=row.get("duration_sec"),
        error=row.get("error"),
        pii_count=pii_count,
    )


def _event_row_to_item(row: dict) -> EventItem:
    """Adapt a raw ``events`` row dict to the public item schema."""
    return EventItem(
        id=row["id"],
        timestamp=row["timestamp"],
        pii_type=row["pii_type"],
        text=row["text"],
        start_sec=row.get("start_sec"),
        end_sec=row.get("end_sec"),
        source=row.get("source"),
        confidence=row.get("confidence"),
    )


@router.post(
    "",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload an audio file and kick off the anonymization pipeline.",
)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to anonymize."),
) -> JobCreatedResponse:
    """Accept an upload, persist it, and schedule the pipeline in background."""
    filename = file.filename or "input"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed: {sorted(ALLOWED_EXTS)}"
            ),
        )

    # Prepare filesystem layout for the new job.
    job_id = uuid.uuid4().hex[:12]
    settings = get_settings()
    paths = JobPaths(job_id, Path(settings.data_dir))
    paths.ensure()

    # Stream the upload to disk in chunks so we can enforce a hard size limit
    # without reading the whole file into memory first. We deliberately
    # save under ``upload<ext>`` (not ``input<ext>``) so a ``.wav`` upload
    # does not collide with the normalized ``input.wav`` ffmpeg target
    # produced by the pipeline (ffmpeg refuses to overwrite its input).
    target = paths.root / f"upload{ext}"
    total = 0
    chunk_size = 1024 * 1024  # 1 MiB
    try:
        with open(target, "wb") as fh:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    fh.close()
                    try:
                        target.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} "
                            f"MB limit"
                        ),
                    )
                fh.write(chunk)
    finally:
        await file.close()

    if total == 0:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    # Create the DB row before scheduling the background task so the caller
    # can always poll /jobs/{id} right after the response.
    db.create_job(job_id, input_filename=filename)

    background_tasks.add_task(run_pipeline, job_id, target)

    return JobCreatedResponse(job_id=job_id, status="queued", filename=filename)


@router.get(
    "",
    response_model=list[JobInfoResponse],
    summary="List the 50 most recent jobs.",
)
async def list_jobs() -> list[JobInfoResponse]:
    """Return metadata for the latest jobs (newest first)."""
    rows = db.list_jobs(50)
    return [_job_row_to_info(r) for r in rows]


@router.get(
    "/{job_id}",
    response_model=JobInfoResponse,
    summary="Get metadata and current status for one job.",
)
async def get_job(job_id: str) -> JobInfoResponse:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )
    # Count events for this job (cheap thanks to idx_events_job).
    conn = db.get_connection()
    try:
        cur = conn.execute(
            "SELECT COUNT(*) AS c FROM events WHERE job_id = ?", (job_id,)
        )
        pii_count = int(cur.fetchone()["c"])
    finally:
        conn.close()
    return _job_row_to_info(row, pii_count=pii_count)


@router.get(
    "/{job_id}/transcript",
    summary="Download the transcript (full or redacted) as JSON.",
)
async def get_transcript(
    job_id: str,
    version: TranscriptVersion = Query(
        default="full",
        description="Which transcript variant to return.",
    ),
) -> Response:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )
    if row["status"] != "done":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Transcript not available yet: job status is "
                f"'{row['status']}'"
            ),
        )

    settings = get_settings()
    paths = JobPaths(job_id, Path(settings.data_dir))
    target = (
        paths.transcript_full
        if version == "full"
        else paths.transcript_redacted
    )
    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcript file for version='{version}' not found on disk",
        )
    return Response(
        content=target.read_bytes(),
        media_type="application/json",
    )


@router.get(
    "/{job_id}/audio",
    summary="Download job audio (original normalized or redacted) as WAV.",
)
async def get_audio(
    job_id: str,
    version: AudioVersion = Query(
        default="redacted",
        description="Which audio artifact to return.",
    ),
) -> FileResponse:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )

    settings = get_settings()
    paths = JobPaths(job_id, Path(settings.data_dir))

    if version == "original":
        # Prefer the normalized 16 kHz mono wav produced by audio_prep
        # (consistent format for any uploader). Fall back to the raw
        # upload (upload.<ext>) if normalization has not happened yet.
        target: Path | None = None
        if paths.input_wav.exists():
            target = paths.input_wav
        elif paths.root.exists():
            for candidate in sorted(paths.root.glob("upload.*")):
                if candidate.suffix.lower() in ALLOWED_EXTS:
                    target = candidate
                    break
        if target is None or not target.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Original audio not found",
            )
        media_type = (
            "audio/wav"
            if target.suffix.lower() == ".wav"
            else "application/octet-stream"
        )
        return FileResponse(
            str(target),
            media_type=media_type,
            filename=f"{job_id}_original{target.suffix.lower()}",
        )

    # version == "redacted"
    target = paths.redacted_wav
    if not target.exists():
        # Distinguish "not ready yet" from "gone".
        if row["status"] != "done":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Redacted audio not available yet: job status is "
                    f"'{row['status']}'"
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Redacted audio not found",
        )
    return FileResponse(
        str(target),
        media_type="audio/wav",
        filename=f"{job_id}_redacted.wav",
    )


@router.get(
    "/{job_id}/events",
    response_model=EventsResponse,
    summary="List all PII events detected for a job.",
)
async def get_events(job_id: str) -> EventsResponse:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )
    rows = db.list_events(job_id)
    events = [_event_row_to_item(r) for r in rows]
    return EventsResponse(job_id=job_id, count=len(events), events=events)


@router.get(
    "/{job_id}/export",
    summary="Download all job artifacts as a ZIP.",
)
async def export_job(job_id: str) -> Response:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found",
        )
    blob = build_zip(job_id)
    headers = {"Content-Disposition": f'attachment; filename="{job_id}.zip"'}
    return Response(content=blob, media_type="application/zip", headers=headers)


__all__ = ["router"]
