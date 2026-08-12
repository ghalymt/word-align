"""Crash recovery — resume incomplete jobs.

Depends on the SQLite project store. On restart, detects jobs stuck in
``pending``/``running`` state and offers to continue from the last completed
stage.

Recovery flow:
1. List incomplete jobs from ProjectStore
2. For each, inspect stage records to find the last completed stage
3. Compute the resume point (stage index) + cache key
4. The caller can re-run the pipeline from that stage
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from .database import JobRecord, ProjectRecord, ProjectStore


@dataclass
class IncompleteJob:
    """A job that did not finish cleanly."""
    job: JobRecord
    project: Optional[ProjectRecord]
    last_completed_stage: Optional[str]
    last_completed_stage_id: Optional[int]
    stages_completed: int
    stages_total: int
    interrupted_at: float
    age_seconds: float


@dataclass
class RecoveryReport:
    """Summary of all recoverable jobs."""
    incomplete: List[IncompleteJob] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.incomplete)

    def resume_point(self, job_id: str) -> Optional[str]:
        """Return the stage name to resume from, or None if not resumable."""
        for inc in self.incomplete:
            if inc.job.id == job_id:
                return inc.last_completed_stage
        return None


class CrashRecovery:
    """Detect and manage interrupted pipeline jobs."""

    STALE_AFTER_SECONDS = 60  # jobs older than this are considered stale

    def __init__(self, store: Optional[ProjectStore] = None):
        self.store = store or ProjectStore()

    def find_incomplete_jobs(self, stale_after_seconds: Optional[float] = None) -> RecoveryReport:
        """Scan for jobs stuck in pending/running state."""
        stale_after = stale_after_seconds or self.STALE_AFTER_SECONDS
        report = RecoveryReport()
        now = time.time()

        for project in self.store.list_projects(limit=200):
            for job in self.store.list_jobs(project.id):
                if job.status not in ("pending", "running"):
                    continue
                started = job.started_at or now
                age = now - started
                if age < stale_after:
                    # Too fresh to be a crash — might still be running
                    continue

                stages = self.store.list_stages(job.id)
                completed = [s for s in stages if s["status"] == "completed"]
                last_completed = completed[-1] if completed else None

                report.incomplete.append(IncompleteJob(
                    job=job,
                    project=project,
                    last_completed_stage=last_completed["stage_name"] if last_completed else None,
                    last_completed_stage_id=last_completed["id"] if last_completed else None,
                    stages_completed=len(completed),
                    stages_total=len(stages),
                    interrupted_at=started + age,
                    age_seconds=age,
                ))
        return report

    def mark_job_resumed(self, job_id: str) -> None:
        """Mark a job as running again after recovery."""
        self.store.update_job(job_id, status="running", started_at=time.time())

    def mark_job_cancelled(self, job_id: str, reason: str = "recovery skipped") -> None:
        """Mark a job as cancelled so it stops appearing in recovery scans."""
        self.store.update_job(job_id, status="cancelled", error=reason,
                              completed_at=time.time())

    def resume_plan(self, job_id: str) -> dict:
        """Build a resume plan for a job: which stage to restart from.

        Returns
        -------
        dict with:
        - resume_stage: stage name to re-run (or None if fresh start)
        - cache_key: optional cache key of the last completed stage output
        - message: human-readable guidance
        """
        job = self.store.get_job(job_id)
        if not job:
            return {"resume_stage": None, "cache_key": None,
                    "message": "Job not found."}
        if job.status not in ("pending", "running"):
            return {"resume_stage": None, "cache_key": None,
                    "message": f"Job is {job.status}; nothing to recover."}

        stages = self.store.list_stages(job_id)
        completed = [s for s in stages if s["status"] == "completed"]
        if not completed:
            return {"resume_stage": None, "cache_key": None,
                    "message": "No completed stages; restart from the beginning."}

        last = completed[-1]
        return {
            "resume_stage": last["stage_name"],
            "cache_key": last.get("cache_key"),
            "message": f"Last completed stage: {last['stage_name']} "
                       f"({len(completed)}/{len(stages)} stages). "
                       f"Resume from the next stage.",
        }
