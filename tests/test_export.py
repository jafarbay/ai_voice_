"""Tests for the per-job ZIP export builder."""
from __future__ import annotations

import io
import zipfile

import pytest

from app import db
from app.pipeline.export import build_zip


@pytest.fixture()
def stub_job(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "app.db"))
    import app.config as cfg
    cfg._settings = None
    db.init_schema()
    job_id = "exp_test"
    db.create_job(job_id, input_filename="x.wav")
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "input.wav").write_bytes(b"RIFF....input")
    (job_dir / "redacted.wav").write_bytes(b"RIFF....redacted")
    (job_dir / "transcript_full.json").write_text('{"text": "full"}', encoding="utf-8")
    (job_dir / "transcript_redacted.json").write_text('{"text": "red"}', encoding="utf-8")
    (job_dir / "events.jsonl").write_text('{"event": 1}\n', encoding="utf-8")
    return job_id


def test_zip_contains_all_artifacts(stub_job):
    blob = build_zip(stub_job)
    with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
        names = set(zf.namelist())
    required_suffixes = [
        "/input.wav",
        "/redacted.wav",
        "/transcript_full.json",
        "/transcript_redacted.json",
        "/events.jsonl",
    ]
    for suffix in required_suffixes:
        assert any(n.endswith(suffix) for n in names), f"missing {suffix} in {names}"


def test_zip_skips_missing_files(stub_job, tmp_path):
    (tmp_path / stub_job / "redacted.wav").unlink()
    blob = build_zip(stub_job)
    with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
        names = set(zf.namelist())
    assert not any(n.endswith("/redacted.wav") for n in names)
    assert any(n.endswith("/input.wav") for n in names)
