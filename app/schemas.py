"""Pydantic schemas shared by the pipeline and the API.

Keep this module free of heavy imports so FastAPI startup stays fast.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "failed"]


class PIIType(str, Enum):
    """All PII categories the service can detect and redact."""

    PASSPORT = "PASSPORT"
    INN = "INN"
    SNILS = "SNILS"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    PERSON = "PERSON"


PIISource = Literal["regex", "natasha", "llm", "word_phone"]


class PIISpan(BaseModel):
    """A detected PII span (character offsets in the transcript text)."""

    type: PIIType
    text: str
    start_char: int
    end_char: int
    source: PIISource = "regex"
    confidence: float = 1.0


class Word(BaseModel):
    """Single word from the STT step with its timing and confidence."""

    word: str = Field(..., description="Surface form as produced by Whisper (may include leading space).")
    start: float = Field(..., description="Start time in seconds relative to the audio file.")
    end: float = Field(..., description="End time in seconds relative to the audio file.")
    probability: float = Field(
        default=0.0,
        description="Whisper word-level probability in [0, 1]; 0.0 if not provided by the model.",
    )
    speaker: str | None = Field(
        default=None,
        description="Speaker label attached by diarization/alignment.",
    )


class TranscriptResult(BaseModel):
    """Whisper output for a single audio file."""

    language: str = Field(..., description="ISO language code detected / forced for the audio.")
    duration: float = Field(..., description="Audio duration in seconds.")
    text: str = Field(..., description="Concatenated transcript text.")
    words: list[Word] = Field(default_factory=list, description="Word-level timestamps.")


class SpeakerSegment(BaseModel):
    """A contiguous audio interval attributed to a single speaker."""

    start: float = Field(..., description="Segment start time in seconds.")
    end: float = Field(..., description="Segment end time in seconds.")
    speaker: str = Field(..., description="Speaker label, e.g. SPEAKER_00.")


class WordWithSpeaker(Word):
    """A Word with a required speaker label assigned by the align step."""

    speaker: str = Field(..., description="Speaker label assigned during alignment.")


class PIIEntity(BaseModel):
    """A detected PII span (character offsets in the transcript text)."""

    type: str
    text: str
    start_char: int
    end_char: int
    start_sec: float | None = None
    end_sec: float | None = None
    source: Literal["regex", "natasha", "llm", "word_phone"] | str = "regex"
    confidence: float = 1.0


class TimeInterval(BaseModel):
    """A PII interval expressed in audio time (seconds), ready for redaction."""

    start: float = Field(..., description="Interval start time in seconds.")
    end: float = Field(..., description="Interval end time in seconds.")
    pii_type: PIIType = Field(..., description="PII category of the underlying span.")
    text: str = Field(..., description="Original surface text of the span.")
    source: PIISource = Field(default="regex", description="Detector source that produced the span.")


# --- API response schemas ------------------------------------------------


class JobCreatedResponse(BaseModel):
    """Payload returned by ``POST /jobs`` right after upload."""

    job_id: str = Field(..., description="Short hex identifier for the created job.")
    status: JobStatus = Field(
        default="queued",
        description="Initial job status; the pipeline runs in the background.",
    )
    filename: str | None = Field(
        default=None,
        description="Original upload filename echoed back to the client.",
    )


class JobInfoResponse(BaseModel):
    """Metadata for one job (returned by ``GET /jobs/{id}`` and list)."""

    id: str = Field(..., description="Job id.")
    created_at: str = Field(..., description="Creation timestamp (ISO-8601 UTC).")
    status: JobStatus = Field(..., description="Current job status.")
    input_filename: str | None = Field(
        default=None,
        description="Original filename provided by the client (for display only).",
    )
    duration_sec: float | None = Field(
        default=None,
        description="Duration of the normalized input audio in seconds.",
    )
    error: str | None = Field(
        default=None,
        description="Error message if ``status == 'failed'``.",
    )
    pii_count: int | None = Field(
        default=None,
        description="Number of PII events recorded for the job (None if not yet computed).",
    )


class EventItem(BaseModel):
    """One row from the ``events`` table, adapted for API responses."""

    id: int = Field(..., description="Autoincrement event id.")
    timestamp: str = Field(..., description="Event creation timestamp (ISO-8601 UTC).")
    pii_type: str = Field(..., description="PII category.")
    text: str = Field(..., description="Raw matched text.")
    start_sec: float | None = Field(
        default=None, description="Start time of the PII interval in seconds."
    )
    end_sec: float | None = Field(
        default=None, description="End time of the PII interval in seconds."
    )
    source: str | None = Field(
        default=None, description="Which detector produced the span."
    )
    confidence: float | None = Field(
        default=None, description="Detector confidence in [0, 1]."
    )


class EventsResponse(BaseModel):
    """Envelope returned by ``GET /jobs/{id}/events``."""

    job_id: str
    count: int
    events: list[EventItem] = Field(default_factory=list)


class EventOut(BaseModel):
    """Flat row from the ``events`` table (one PII detection)."""

    id: int = Field(..., description="Autoincrement event id.")
    job_id: str = Field(..., description="Owning job id.")
    timestamp: str = Field(..., description="Event creation timestamp (ISO-8601 UTC).")
    pii_type: str = Field(..., description="PII category.")
    text: str = Field(..., description="Raw matched text.")
    start_sec: float | None = Field(
        default=None, description="Start time of the PII interval in seconds."
    )
    end_sec: float | None = Field(
        default=None, description="End time of the PII interval in seconds."
    )
    source: str = Field(..., description="Which detector produced the span.")
    confidence: float | None = Field(
        default=None, description="Detector confidence in [0, 1]."
    )
