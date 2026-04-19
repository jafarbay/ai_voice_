"""End-to-end anonymization pipeline for a single job.

Flow:

    audio_prep  ->  stt  ->  diarization  ->  align
                                |
                                v
                           full_text
                                |
            regex + natasha + llm  ==>  merger  ==>  merged spans
                                                        |
                                                        v
                                              spans_to_time_pairs
                                                        |
                     +----------------------------------+
                     |                                  |
                     v                                  v
                redaction.wav                 transcript_redacted.json
                     |
                     v
                events.jsonl + SQLite

All I/O goes through :class:`app.storage.files.JobPaths`; all persistence
goes through :mod:`app.db`; all events go through
:class:`app.pipeline.events.EventLogger`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app import db
from app.config import get_settings
from app.pipeline import audio_prep, diarization, redaction, stt
from app.pipeline.align import assign_speakers
from app.pipeline.events import EventLogger
from app.pipeline.pii import merger, natasha_detector, regex_detector, span_to_time
from app.pipeline.pii import llm_detector, word_phone_detector
from app.pipeline.pii.types import PIISpan
from app.schemas import TranscriptResult, WordWithSpeaker
from app.storage.files import JobPaths

log = logging.getLogger(__name__)


def _build_pii_char_map(
    full_text: str, merged_spans: list[PIISpan]
) -> list[str | None]:
    """Return a per-character list assigning a PII type (or ``None``) to each char."""
    pii_map: list[str | None] = [None] * len(full_text)
    for span in merged_spans:
        start = max(0, span.start_char)
        end = min(len(full_text), span.end_char)
        tag = span.type.value if hasattr(span.type, "value") else str(span.type)
        for i in range(start, end):
            pii_map[i] = tag
    return pii_map


def _word_pii_tag(
    word_start: int, word_end: int, pii_map: list[str | None]
) -> str | None:
    """Pick a PII tag for a word: the first non-None char tag inside its range."""
    for i in range(word_start, min(word_end, len(pii_map))):
        if pii_map[i] is not None:
            return pii_map[i]
    return None


def write_transcript_full(
    path: Path,
    words_with_speakers: list[WordWithSpeaker],
    transcript: TranscriptResult,
    merged_spans: list[PIISpan],
) -> None:
    """Write ``transcript_full.json`` with word timings and PII markers."""
    full_text, index = span_to_time.build_word_char_index(words_with_speakers)
    pii_map = _build_pii_char_map(full_text, merged_spans)

    speakers: list[str] = []
    for w in words_with_speakers:
        if w.speaker and w.speaker not in speakers:
            speakers.append(w.speaker)

    words_json: list[dict[str, Any]] = []
    for (w_idx, c_start, c_end), w in zip(index, words_with_speakers, strict=True):
        tag = _word_pii_tag(c_start, c_end, pii_map)
        words_json.append(
            {
                "word": w.word,
                "start": w.start,
                "end": w.end,
                "speaker": w.speaker,
                "probability": w.probability,
                "pii_type": tag,
            }
        )

    payload = {
        "language": transcript.language,
        "duration": transcript.duration,
        "speakers": speakers,
        "words": words_json,
        "text": full_text,
        "pii_count": len(merged_spans),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def write_transcript_redacted(
    path: Path,
    words_with_speakers: list[WordWithSpeaker],
    transcript: TranscriptResult,
    merged_spans: list[PIISpan],
) -> None:
    """Write ``transcript_redacted.json``: PII words replaced with ``[PII:TYPE]``."""
    full_text, index = span_to_time.build_word_char_index(words_with_speakers)
    pii_map = _build_pii_char_map(full_text, merged_spans)

    speakers: list[str] = []
    redacted_words: list[dict[str, Any]] = []
    redacted_tokens: list[str] = []
    # Track the last PII type emitted so adjacent PII words collapse into
    # a single [PII:TYPE] in the human-readable text (but keep one row per
    # word in the JSON array so the UI can still highlight per-word).
    last_text_tag: str | None = None

    for (w_idx, c_start, c_end), w in zip(index, words_with_speakers, strict=True):
        tag = _word_pii_tag(c_start, c_end, pii_map)
        if w.speaker and w.speaker not in speakers:
            speakers.append(w.speaker)

        surface = w.word if tag is None else f"[PII:{tag}]"
        redacted_words.append(
            {
                "word": surface,
                "original_word": w.word,
                "start": w.start,
                "end": w.end,
                "speaker": w.speaker,
                "probability": w.probability,
                "pii_type": tag,
            }
        )

        if tag is None:
            redacted_tokens.append(w.word)
            last_text_tag = None
        else:
            # Collapse a run of PII words of the same type into a single token.
            if tag != last_text_tag:
                redacted_tokens.append(f"[PII:{tag}]")
            last_text_tag = tag

    payload = {
        "language": transcript.language,
        "duration": transcript.duration,
        "speakers": speakers,
        "words": redacted_words,
        "text": " ".join(redacted_tokens),
        "pii_count": len(merged_spans),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def run_pipeline(job_id: str, input_file: Path) -> None:
    """Execute the full anonymization pipeline for ``job_id``.

    Expects ``db.create_job(job_id, ...)`` to have been called beforehand.
    Mutations to the ``jobs`` row happen here (status transitions) and
    events are appended both to ``events.jsonl`` and the ``events`` table.
    """
    settings = get_settings()
    paths = JobPaths(job_id, Path(settings.data_dir))
    paths.ensure()

    conn = db.get_connection()
    logger = EventLogger(job_id, paths.events, conn)

    try:
        # Ensure row exists even if caller forgot (orchestrator may be
        # invoked directly for smoke tests).
        if db.get_job(job_id, conn=conn) is None:
            db.create_job(job_id, input_filename=Path(input_file).name, conn=conn)

        db.update_job_status(job_id, "running", conn=conn)
        logger.pipeline_event("STARTED", input=str(input_file))

        # 1. Audio prep -------------------------------------------------
        duration = audio_prep.normalize_audio(Path(input_file), paths.input_wav)
        logger.pipeline_event("AUDIO_PREP_DONE", duration_sec=duration)

        # 2. STT --------------------------------------------------------
        transcriber = stt.get_transcriber()
        transcript: TranscriptResult = transcriber.transcribe(paths.input_wav)
        logger.pipeline_event(
            "STT_DONE",
            words=len(transcript.words),
            language=transcript.language,
            transcript_duration=transcript.duration,
        )

        # 3. Diarization + alignment -----------------------------------
        segments = diarization.diarize(paths.input_wav, transcript.duration)
        words_with_speakers = assign_speakers(transcript.words, segments)
        logger.pipeline_event("DIARIZATION_DONE", segments=len(segments))

        # 4. Build full text for PII detection -------------------------
        full_text = span_to_time.build_full_text(transcript.words)

        # 5. PII detection (regex + natasha + optional LLM) ------------
        spans: list[PIISpan] = []
        try:
            spans.extend(regex_detector.detect_regex(full_text))
        except Exception as exc:  # noqa: BLE001
            logger.pipeline_event("REGEX_ERROR", error=str(exc))
        try:
            spans.extend(word_phone_detector.detect_word_phones(full_text))
        except Exception as exc:  # noqa: BLE001
            logger.pipeline_event("WORD_PHONE_ERROR", error=str(exc))
        try:
            spans.extend(natasha_detector.detect_natasha(full_text))
        except Exception as exc:  # noqa: BLE001
            logger.pipeline_event("NATASHA_ERROR", error=str(exc))
        try:
            spans.extend(llm_detector.get_llm_detector().detect(full_text))
        except Exception as exc:  # noqa: BLE001 - LLM is optional
            logger.pipeline_event("LLM_ERROR", error=str(exc))

        merged = merger.merge_spans(spans)
        logger.pipeline_event(
            "PII_DETECTED", count=len(merged), raw=len(spans)
        )

        # 6. Spans -> audio time intervals -----------------------------
        pairs = span_to_time.spans_to_time_pairs(
            merged, transcript.words, full_text
        )
        intervals = [iv for _s, iv in pairs]

        # 7. Redact audio ----------------------------------------------
        redaction.redact_audio(paths.input_wav, paths.redacted_wav, intervals)
        logger.pipeline_event("REDACTION_DONE", intervals=len(intervals))

        # 8. Write transcripts -----------------------------------------
        write_transcript_full(
            paths.transcript_full, words_with_speakers, transcript, merged
        )
        write_transcript_redacted(
            paths.transcript_redacted, words_with_speakers, transcript, merged
        )

        # 9. One event per PII span with its audio interval ------------
        for span, interval in pairs:
            pii_type = (
                span.type.value if hasattr(span.type, "value") else str(span.type)
            )
            logger.emit(
                pii_type=pii_type,
                text=span.text,
                start_sec=interval.start,
                end_sec=interval.end,
                source=str(span.source),
                confidence=float(span.confidence),
            )

        db.update_job_status(
            job_id, "done", duration_sec=duration, conn=conn
        )
        logger.pipeline_event("COMPLETED", pii_count=len(merged))

    except Exception as exc:
        log.exception("pipeline failed for job %s", job_id)
        try:
            db.update_job_status(
                job_id, "failed", error=str(exc), conn=conn
            )
            logger.pipeline_event("FAILED", error=str(exc))
        except Exception:  # noqa: BLE001
            log.exception("failed to record failure state for job %s", job_id)
        raise
    finally:
        conn.close()


__all__ = [
    "run_pipeline",
    "write_transcript_full",
    "write_transcript_redacted",
]
