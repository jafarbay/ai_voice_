"""End-to-end smoke test of the full anonymization pipeline.

Marked ``integration`` because it loads Whisper (and, if available,
llama-cpp GGUF) and therefore takes tens of seconds.

Run with:

    python -m pytest tests/test_pipeline_smoke.py -v -s -m integration
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import soundfile as sf

from app import db
from app.config import get_settings
from app.pipeline import orchestrator
from app.storage.files import JobPaths

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.wav"
JOB_ID = "test-job-smoke-001"


@pytest.mark.integration
def test_full_pipeline_on_sample():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")

    settings = get_settings()
    db.init_schema()
    paths = JobPaths(JOB_ID, Path(settings.data_dir))

    # Clean leftover from previous runs so assertions are deterministic.
    if paths.root.exists():
        shutil.rmtree(paths.root)

    # Remove any previous DB rows for this job so assertions are clean.
    conn = db.get_connection()
    try:
        conn.execute("DELETE FROM events WHERE job_id = ?", (JOB_ID,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (JOB_ID,))
        conn.commit()
    finally:
        conn.close()

    db.create_job(JOB_ID, input_filename=FIXTURE.name)

    try:
        orchestrator.run_pipeline(JOB_ID, FIXTURE)

        # --- File artifacts ------------------------------------------------
        assert paths.transcript_full.exists(), "transcript_full.json missing"
        assert paths.transcript_redacted.exists(), "transcript_redacted.json missing"
        assert paths.redacted_wav.exists(), "redacted.wav missing"
        assert paths.events.exists(), "events.jsonl missing"
        assert paths.input_wav.exists(), "input.wav missing"

        with open(paths.transcript_full, encoding="utf-8") as fh:
            full = json.load(fh)
        assert "words" in full and isinstance(full["words"], list)
        assert full["words"], "transcript_full has zero words"

        with open(paths.transcript_redacted, encoding="utf-8") as fh:
            redacted = json.load(fh)
        assert "words" in redacted

        # Redacted WAV duration should match the sample (~17.81 s for the
        # canonical fixture); tolerate ±1 s for non-canonical fixtures.
        info = sf.info(str(paths.redacted_wav))
        red_duration = float(info.frames) / float(info.samplerate)
        assert red_duration > 0.5, f"redacted.wav too short: {red_duration}"

        # events.jsonl has at least STARTED + COMPLETED + something in between.
        with open(paths.events, encoding="utf-8") as fh:
            event_lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        assert len(event_lines) > 0, "events.jsonl is empty"
        parsed = [json.loads(ln) for ln in event_lines]
        kinds = [e.get("kind") for e in parsed]
        assert "STARTED" in kinds
        assert "COMPLETED" in kinds

        # --- DB state ------------------------------------------------------
        job_row = db.get_job(JOB_ID)
        assert job_row is not None, "job row missing"
        assert job_row["status"] == "done", f"status={job_row['status']}, err={job_row.get('error')}"

        events = db.list_events(JOB_ID)
        assert len(events) >= 1, "SQLite events table has no rows for the job"

        # sample.wav contains "Иван Петров" so PERSON must be detected by
        # either natasha or LLM.
        pii_types = {e["pii_type"] for e in events}
        assert "PERSON" in pii_types, f"expected PERSON in {pii_types}"

        # Debug dump for `-s` inspection.
        print(f"\n[smoke] pii events: {len(events)}")
        for e in events:
            print(
                f"  {e['pii_type']:<8} {e['start_sec']:.2f}-{e['end_sec']:.2f} "
                f"src={e['source']} text={e['text']!r}"
            )
    finally:
        # Cleanup: remove artifacts + DB rows so re-runs are clean.
        if paths.root.exists():
            shutil.rmtree(paths.root, ignore_errors=True)
        conn = db.get_connection()
        try:
            conn.execute("DELETE FROM events WHERE job_id = ?", (JOB_ID,))
            conn.execute("DELETE FROM jobs WHERE id = ?", (JOB_ID,))
            conn.commit()
        finally:
            conn.close()
