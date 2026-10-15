"""Run an out-of-process backend worker and collect its JSON payload.

Every worker prints one sentinel-tagged JSON line to stdout and logs to
stderr. This helper launches it with UTF-8 I/O (Windows otherwise writes
cp1252 and dies on the first non-Latin path), polls for cancellation, and
turns failures into :class:`WorkerError` with the tail of stderr.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


class WorkerError(RuntimeError):
    """The worker could not start, failed, or printed no payload."""


class WorkerCancelled(WorkerError):
    """The run was cancelled while the worker was busy."""


def parse_payload(stdout: str, sentinel: str) -> Dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(sentinel):
            return json.loads(line[len(sentinel):])
    raise WorkerError("worker produced no JSON payload")


def run_worker(python: str, worker: Path, args: Sequence[str], sentinel: str,
               cancel_event=None, env: Optional[Dict[str, str]] = None,
               timeout: Optional[float] = None, name: str = "worker",
               on_stderr_line=None) -> Dict:
    """Run *worker* under *python* with *args*; return its JSON payload."""
    cmd: List[str] = [python, str(worker), *[str(a) for a in args]]
    full_env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    if env:
        full_env.update({k: str(v) for k, v in env.items() if v is not None})
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding="utf-8",
                                errors="replace", env=full_env)
    except OSError as exc:
        raise WorkerError(f"{name}: could not launch {python}: {exc}") from exc
    started = time.time()
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            cancelled = cancel_event is not None and cancel_event.is_set()
            expired = timeout is not None and time.time() - started > timeout
            if cancelled or expired:
                proc.terminate()
                try:
                    proc.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                if cancelled:
                    raise WorkerCancelled(f"{name}: cancelled")
                raise WorkerError(f"{name}: timed out after {timeout:.0f} s")
    if on_stderr_line is not None:
        for line in stderr.splitlines():
            on_stderr_line(line)
    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-8:])
        raise WorkerError(f"{name} failed (exit {proc.returncode}):\n{tail}")
    try:
        return parse_payload(stdout, sentinel)
    except (WorkerError, json.JSONDecodeError) as exc:
        tail = "\n".join(stderr.strip().splitlines()[-5:])
        raise WorkerError(f"{name}: {exc}\n{tail}") from exc
