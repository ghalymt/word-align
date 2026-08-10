"""SQLite project store — persist jobs, stages, QA issues, settings.

Enables crash recovery, resumable jobs, and project history.

Schema:
    projects   - one row per audio file processed
    jobs       - one row per pipeline run (re-runs create new jobs)
    stages     - one row per stage within a job (vosk, whisperx, mfa, etc.)
    qa_issues  - QA issues linked to a job
    settings   - key-value store for global settings
    glossary   - project-level and global glossary entries
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    audio_path  TEXT NOT NULL,
    transcript_path TEXT,
    language    TEXT,
    mode        TEXT DEFAULT 'reference',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(id),
    status      TEXT DEFAULT 'pending',   -- pending|running|completed|failed|cancelled
    profile     TEXT DEFAULT '{}',
    started_at  REAL,
    completed_at REAL,
    error       TEXT,
    output_files TEXT DEFAULT '[]',       -- JSON array of paths
    word_count  INTEGER DEFAULT 0,
    segment_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs(id),
    stage_name  TEXT NOT NULL,             -- vosk, whisperx, mfa, segmentation, etc.
    status      TEXT DEFAULT 'pending',    -- pending|running|completed|failed|skipped
    started_at  REAL,
    completed_at REAL,
    duration_seconds REAL,
    cache_key   TEXT,
    output_path TEXT,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS qa_issues (
    id          TEXT PRIMARY KEY,
    job_id      TEXT NOT NULL REFERENCES jobs(id),
    category    TEXT NOT NULL,             -- agreement|repetition|timing|semantic
    severity    TEXT DEFAULT 'low',        -- low|medium|high
    confidence  REAL DEFAULT 0.0,
    start_time  REAL,
    end_time    REAL,
    word_ids    TEXT DEFAULT '[]',         -- JSON array
    original_text TEXT,
    suggested_text TEXT,
    explanation TEXT,
    sources     TEXT DEFAULT '[]',         -- JSON array
    status      TEXT DEFAULT 'open',       -- open|accepted|edited|dismissed
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS glossary (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT REFERENCES projects(id),  -- NULL = global
    term        TEXT NOT NULL,
    correct_form TEXT,
    context     TEXT,
    created_at  REAL NOT NULL,
    UNIQUE(project_id, term)
);

CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_stages_job ON stages(job_id);
CREATE INDEX IF NOT EXISTS idx_qa_job ON qa_issues(job_id);
"""


@dataclass
class ProjectRecord:
    id: str
    name: str
    audio_path: str
    transcript_path: Optional[str] = None
    language: Optional[str] = None
    mode: str = "reference"
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class JobRecord:
    id: str
    project_id: str
    status: str = "pending"
    profile: dict = field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    output_files: list = field(default_factory=list)
    word_count: int = 0
    segment_count: int = 0


class ProjectStore:
    """SQLite-backed project persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or self._default_db_path())
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    @staticmethod
    def _default_db_path() -> Path:
        home = Path.home()
        d = home / ".wordalign"
        d.mkdir(parents=True, exist_ok=True)
        return d / "projects.db"

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        # Record schema version
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            ("schema_version", str(SCHEMA_VERSION), time.time())
        )
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ---- Projects ----

    def create_project(self, project: ProjectRecord) -> None:
        conn = self._get_conn()
        now = time.time()
        project.created_at = project.created_at or now
        project.updated_at = now
        conn.execute(
            "INSERT INTO projects (id, name, audio_path, transcript_path, language, mode, created_at, updated_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (project.id, project.name, project.audio_path, project.transcript_path,
             project.language, project.mode, project.created_at, project.updated_at,
             json.dumps(project.metadata))
        )
        conn.commit()

    def get_project(self, project_id: str) -> Optional[ProjectRecord]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            return None
        return ProjectRecord(
            id=row["id"], name=row["name"], audio_path=row["audio_path"],
            transcript_path=row["transcript_path"], language=row["language"],
            mode=row["mode"], created_at=row["created_at"], updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def list_projects(self, limit: int = 50) -> list[ProjectRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            ProjectRecord(
                id=r["id"], name=r["name"], audio_path=r["audio_path"],
                transcript_path=r["transcript_path"], language=r["language"],
                mode=r["mode"], created_at=r["created_at"], updated_at=r["updated_at"],
                metadata=json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    def update_project(self, project_id: str, **kwargs) -> None:
        conn = self._get_conn()
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k == "metadata":
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        sets.append("updated_at = ?")
        vals.append(time.time())
        vals.append(project_id)
        conn.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()

    def delete_project(self, project_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM qa_issues WHERE job_id IN (SELECT id FROM jobs WHERE project_id = ?)", (project_id,))
        conn.execute("DELETE FROM stages WHERE job_id IN (SELECT id FROM jobs WHERE project_id = ?)", (project_id,))
        conn.execute("DELETE FROM jobs WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM glossary WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()

    # ---- Jobs ----

    def create_job(self, job: JobRecord) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO jobs (id, project_id, status, profile, started_at, completed_at, error, output_files, word_count, segment_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job.id, job.project_id, job.status, json.dumps(job.profile),
             job.started_at, job.completed_at, job.error,
             json.dumps(job.output_files), job.word_count, job.segment_count)
        )
        conn.commit()

    def update_job(self, job_id: str, **kwargs) -> None:
        conn = self._get_conn()
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k in ("profile", "output_files"):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(job_id)
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            return None
        return JobRecord(
            id=row["id"], project_id=row["project_id"], status=row["status"],
            profile=json.loads(row["profile"] or "{}"),
            started_at=row["started_at"], completed_at=row["completed_at"],
            error=row["error"],
            output_files=json.loads(row["output_files"] or "[]"),
            word_count=row["word_count"], segment_count=row["segment_count"],
        )

    def list_jobs(self, project_id: str) -> list[JobRecord]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM jobs WHERE project_id = ? ORDER BY started_at DESC", (project_id,)
        ).fetchall()
        return [
            JobRecord(
                id=r["id"], project_id=r["project_id"], status=r["status"],
                profile=json.loads(r["profile"] or "{}"),
                started_at=r["started_at"], completed_at=r["completed_at"],
                error=r["error"],
                output_files=json.loads(r["output_files"] or "[]"),
                word_count=r["word_count"], segment_count=r["segment_count"],
            )
            for r in rows
        ]

    # ---- Stages ----

    def record_stage(self, job_id: str, stage_name: str, status: str,
                     started_at: Optional[float] = None,
                     completed_at: Optional[float] = None,
                     duration_seconds: Optional[float] = None,
                     cache_key: Optional[str] = None,
                     output_path: Optional[str] = None,
                     metadata: Optional[dict] = None) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "INSERT INTO stages (job_id, stage_name, status, started_at, completed_at, duration_seconds, cache_key, output_path, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, stage_name, status, started_at, completed_at,
             duration_seconds, cache_key, output_path,
             json.dumps(metadata or {}))
        )
        conn.commit()
        return cursor.lastrowid

    def update_stage(self, stage_id: int, **kwargs) -> None:
        conn = self._get_conn()
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k == "metadata":
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(stage_id)
        conn.execute(f"UPDATE stages SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()

    def list_stages(self, job_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM stages WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- QA Issues ----

    def save_qa_issue(self, job_id: str, issue: dict) -> None:
        conn = self._get_conn()
        issue_id = issue.get("id", f"issue_{int(time.time()*1000)}")
        conn.execute(
            "INSERT OR REPLACE INTO qa_issues (id, job_id, category, severity, confidence, start_time, end_time, word_ids, original_text, suggested_text, explanation, sources, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (issue_id, job_id, issue.get("category", "unknown"),
             issue.get("severity", "low"), issue.get("confidence", 0.0),
             issue.get("start", 0.0), issue.get("end", 0.0),
             json.dumps(issue.get("word_ids", [])),
             issue.get("original_text", ""),
             issue.get("suggested_text"),
             issue.get("explanation", ""),
             json.dumps(issue.get("sources", [])),
             issue.get("status", "open"), time.time())
        )
        conn.commit()

    def list_qa_issues(self, job_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM qa_issues WHERE job_id = ? ORDER BY start_time", (job_id,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["word_ids"] = json.loads(d.get("word_ids") or "[]")
            d["sources"] = json.loads(d.get("sources") or "[]")
            result.append(d)
        return result

    def update_qa_issue(self, issue_id: str, **kwargs) -> None:
        conn = self._get_conn()
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k in ("word_ids", "sources"):
                v = json.dumps(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        vals.append(issue_id)
        conn.execute(f"UPDATE qa_issues SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()

    # ---- Settings ----

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time())
        )
        conn.commit()

    # ---- Glossary ----

    def add_glossary_entry(self, term: str, correct_form: Optional[str] = None,
                           context: Optional[str] = None,
                           project_id: Optional[str] = None) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO glossary (project_id, term, correct_form, context, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (project_id, term.lower(), correct_form, context, time.time())
        )
        conn.commit()

    def list_glossary(self, project_id: Optional[str] = None) -> list[dict]:
        conn = self._get_conn()
        if project_id:
            rows = conn.execute(
                "SELECT * FROM glossary WHERE project_id IS NULL OR project_id = ? ORDER BY term",
                (project_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM glossary WHERE project_id IS NULL ORDER BY term"
            ).fetchall()
        return [dict(r) for r in rows]

    def remove_glossary_entry(self, entry_id: int) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM glossary WHERE id = ?", (entry_id,))
        conn.commit()
