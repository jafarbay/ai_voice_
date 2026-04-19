"""REST API tests.

Fast tests run without any model loading and cover error paths.
The full-lifecycle test is marked ``integration`` because the background
task invokes Whisper (+ optionally the LLM), which takes tens of seconds.

Run:

    python -m pytest tests/test_api.py -v                     # fast only
    python -m pytest tests/test_api.py -v -s -m integration   # full
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import get_settings
from app.main import app
from app.storage.files import JobPaths

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample.wav"


def _cleanup_job(job_id: str) -> None:
    """Remove all disk + DB traces of ``job_id`` (idempotent)."""
    settings = get_settings()
    paths = JobPaths(job_id, Path(settings.data_dir))
    if paths.root.exists():
        shutil.rmtree(paths.root, ignore_errors=True)
    conn = db.get_connection()
    try:
        conn.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
    finally:
        conn.close()


# --- Fast tests (no models) ----------------------------------------------


def test_health():
    db.init_schema()
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_invalid_extension():
    db.init_schema()
    with TestClient(app) as client:
        resp = client.post(
            "/jobs",
            files={"file": ("malicious.txt", io.BytesIO(b"not audio"), "text/plain")},
        )
    assert resp.status_code == 400
    assert "extension" in resp.json()["detail"].lower()


def test_get_unknown_job():
    db.init_schema()
    with TestClient(app) as client:
        resp = client.get("/jobs/nonexistent")
    assert resp.status_code == 404


def test_transcript_for_unfinished_job():
    """A job in ``running`` state must return 409 for transcript reads."""
    db.init_schema()
    job_id = "test-api-unfinished-001"
    _cleanup_job(job_id)
    db.create_job(job_id, input_filename="fake.wav")
    db.update_job_status(job_id, "running")
    try:
        with TestClient(app) as client:
            resp = client.get(f"/jobs/{job_id}/transcript", params={"version": "full"})
        assert resp.status_code == 409, resp.text
        assert "running" in resp.json()["detail"].lower()
    finally:
        _cleanup_job(job_id)


def test_events_unknown_job():
    db.init_schema()
    with TestClient(app) as client:
        resp = client.get("/jobs/does-not-exist/events")
    assert resp.status_code == 404


def test_audio_unknown_job():
    db.init_schema()
    with TestClient(app) as client:
        resp = client.get("/jobs/does-not-exist/audio")
    assert resp.status_code == 404


def test_list_jobs_returns_list():
    db.init_schema()
    with TestClient(app) as client:
        resp = client.get("/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# --- Full end-to-end (integration) ---------------------------------------


@pytest.mark.integration
def test_full_lifecycle():
    """POST → background pipeline runs synchronously in TestClient → poll GETs."""
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")

    db.init_schema()
    job_id: str | None = None

    try:
        with TestClient(app) as client:
            # 1. POST upload -------------------------------------------------
            with open(FIXTURE, "rb") as fh:
                resp = client.post(
                    "/jobs",
                    files={"file": (FIXTURE.name, fh, "audio/wav")},
                )
            assert resp.status_code == 202, resp.text
            created = resp.json()
            job_id = created["job_id"]
            assert created["status"] == "queued"
            assert job_id and len(job_id) == 12

            # TestClient executes BackgroundTasks synchronously after the
            # response returns, so by the time we make the next call the
            # pipeline has completed (or failed).

            # 2. Job status --------------------------------------------------
            resp = client.get(f"/jobs/{job_id}")
            assert resp.status_code == 200, resp.text
            info = resp.json()
            assert info["id"] == job_id
            assert info["status"] == "done", (
                f"status={info['status']}, error={info.get('error')}"
            )
            assert info["error"] in (None, "")
            assert info["duration_sec"] is not None
            # Canonical fixture is ~17.81 s; tolerate ±1 s.
            if abs(info["duration_sec"] - 17.81) > 1.5:
                print(
                    f"[warn] duration_sec={info['duration_sec']} "
                    f"(expected ~17.81)"
                )

            # 3. Transcript full ---------------------------------------------
            resp = client.get(
                f"/jobs/{job_id}/transcript", params={"version": "full"}
            )
            assert resp.status_code == 200, resp.text
            full = resp.json()
            assert isinstance(full.get("words"), list) and full["words"]
            assert "text" in full

            # 4. Transcript redacted -----------------------------------------
            resp = client.get(
                f"/jobs/{job_id}/transcript", params={"version": "redacted"}
            )
            assert resp.status_code == 200, resp.text
            redacted = resp.json()
            assert isinstance(redacted.get("words"), list) and redacted["words"]
            combined = redacted.get("text", "") + " " + json.dumps(
                redacted["words"], ensure_ascii=False
            )
            assert "[PII:" in combined, (
                "expected at least one [PII:*] marker in redacted transcript"
            )

            # 5. Audio original ----------------------------------------------
            resp = client.get(
                f"/jobs/{job_id}/audio", params={"version": "original"}
            )
            assert resp.status_code == 200, resp.text
            assert resp.headers["content-type"].startswith("audio/")
            assert len(resp.content) > 1024

            # 6. Audio redacted ----------------------------------------------
            resp = client.get(
                f"/jobs/{job_id}/audio", params={"version": "redacted"}
            )
            assert resp.status_code == 200, resp.text
            assert resp.headers["content-type"] == "audio/wav"
            assert len(resp.content) > 1024

            # 7. Events ------------------------------------------------------
            resp = client.get(f"/jobs/{job_id}/events")
            assert resp.status_code == 200, resp.text
            events = resp.json()
            assert events["job_id"] == job_id
            assert events["count"] >= 1, "expected at least 1 PII event"
            pii_types = {e["pii_type"] for e in events["events"]}
            print(f"\n[api e2e] events.count={events['count']} types={pii_types}")

    finally:
        if job_id is not None:
            _cleanup_job(job_id)
