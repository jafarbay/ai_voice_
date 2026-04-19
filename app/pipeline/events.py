"""Structured event logging for a single pipeline run.

Two event streams:

- **PII events** (:meth:`EventLogger.emit`) are written to both
  ``events.jsonl`` (append, UTF-8, ``ensure_ascii=False``) *and* the
  SQLite ``events`` table. One row per detected PII entity.

- **Pipeline events** (:meth:`EventLogger.pipeline_event`) mark phase
  transitions (STARTED, STT_DONE, ...). They go only to ``events.jsonl``
  so a human can scan a chronological trace without polluting the PII
  table. The SQLite counterpart may be added later without API change.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from app import db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventLogger:
    """Write PII and pipeline events for a single job."""

    def __init__(
        self,
        job_id: str,
        jsonl_path: Path,
        db_conn: sqlite3.Connection,
    ) -> None:
        self.job_id = job_id
        self.jsonl_path = Path(jsonl_path)
        self.db_conn = db_conn
        self._lock = Lock()
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # -- internal --------------------------------------------------------

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    # -- public ----------------------------------------------------------

    def emit(
        self,
        pii_type: str,
        text: str,
        start_sec: float | None,
        end_sec: float | None,
        source: str | None,
        confidence: float | None,
    ) -> None:
        """Record a PII detection in both JSONL and SQLite."""
        ts = _now_iso()
        record = {
            "timestamp": ts,
            "kind": "PII",
            "job_id": self.job_id,
            "pii_type": pii_type,
            "text": text,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "source": source,
            "confidence": confidence,
        }
        self._write_jsonl(record)
        db.insert_event(
            job_id=self.job_id,
            pii_type=pii_type,
            text=text,
            start_sec=start_sec,
            end_sec=end_sec,
            source=source,
            confidence=confidence,
            timestamp=ts,
            conn=self.db_conn,
        )

    def pipeline_event(self, kind: str, **extra: Any) -> None:
        """Record a pipeline lifecycle event (jsonl-only)."""
        record = {
            "timestamp": _now_iso(),
            "kind": kind,
            "job_id": self.job_id,
        }
        if extra:
            record.update(extra)
        self._write_jsonl(record)


__all__ = ["EventLogger"]
