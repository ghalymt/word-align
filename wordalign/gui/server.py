"""Local HTTP API server for the WordAlign GUI.

Runs a Flask/FastAPI server that the web GUI can talk to.
Provides endpoints for: file selection, pipeline execution,
progress events, QA issues, model management, and hardware info.

Usage:
    python -m wordalign.gui.server                  # default port 5575
    python -m wordalign.gui.server --port 8080      # custom port

Then open http://127.0.0.1:5575/ in a browser (the server serves the GUI
itself; opening app.html from disk will not connect, because the API sends
no CORS headers).
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..models.paths import (
    _default_llm_model_path,
    _default_llm_mtp_model_path,
    _llama_cpp_dir,
)


_ACTIVE_RUNS: dict[str, dict[str, Any]] = {}
_UPLOAD_DIRS: dict[str, str] = {}
_RUNS_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Job queue. Every run used to get its own thread, so two GUI jobs loaded
# their GPU models at the same time and one (or both) ran out of memory.
# Jobs now wait in a FIFO and WORDALIGN_MAX_CONCURRENT_JOBS workers (default
# 1) take them in order. Inside a run, GPU engines additionally hold the
# process-wide GPU slot (core.gpu_scheduler), so raising the job limit lets
# CPU stages overlap without putting two models on the GPU.
# ---------------------------------------------------------------------------
_QUEUE: "collections.deque[tuple[str, dict]]" = collections.deque()
_QUEUE_COND = threading.Condition(_RUNS_LOCK)
_WORKERS: list = []


def _max_concurrent_jobs() -> int:
    try:
        return max(1, int(os.environ.get("WORDALIGN_MAX_CONCURRENT_JOBS", "1")))
    except ValueError:
        return 1


def _queue_position(job_id: str) -> Optional[int]:
    """1-based place in the queue, or None if the job is not waiting."""
    with _RUNS_LOCK:
        for n, (queued_id, _) in enumerate(_QUEUE, 1):
            if queued_id == job_id:
                return n
    return None


def _submit_job(job_id: str, params: dict) -> int:
    """Queue a job; start the workers on first use. Returns its position."""
    with _QUEUE_COND:
        _ACTIVE_RUNS[job_id] = {"runner": None, "cancel_requested": False,
                                "status": "queued"}
        _QUEUE.append((job_id, params))
        position = len(_QUEUE)
        while len(_WORKERS) < _max_concurrent_jobs():
            worker = threading.Thread(target=_queue_worker, daemon=True,
                                      name=f"wordalign-job-{len(_WORKERS) + 1}")
            _WORKERS.append(worker)
            worker.start()
        _QUEUE_COND.notify()
    return position


def _cleanup_job(job_id: str, params: dict) -> None:
    owned_upload_dir = params.get("_owned_upload_dir")
    if owned_upload_dir:
        shutil.rmtree(owned_upload_dir, ignore_errors=True)
    with _RUNS_LOCK:
        upload_id = params.get("upload_id")
        if upload_id:
            _UPLOAD_DIRS.pop(str(upload_id), None)
        _ACTIVE_RUNS.pop(job_id, None)


def _cancel_queued_job(job_id: str, params: dict) -> None:
    """Finish a job that was cancelled before it started."""
    try:
        from ..core.database import ProjectStore
        store = ProjectStore()
        store.update_job(job_id, status="cancelled", completed_at=time.time(),
                         error="Cancelled before it started")
        store.close()
    except Exception:
        pass
    _cleanup_job(job_id, params)


def _queue_worker() -> None:
    while True:
        with _QUEUE_COND:
            while not _QUEUE:
                _QUEUE_COND.wait()
            job_id, params = _QUEUE.popleft()
            state = _ACTIVE_RUNS.get(job_id)
            cancelled = state is None or state.get("cancel_requested")
            if not cancelled:
                state["status"] = "running"
        if cancelled:
            _cancel_queued_job(job_id, params)
            continue
        try:
            from ..core.database import ProjectStore
            store = ProjectStore()
            store.update_job(job_id, status="running", started_at=time.time())
            store.close()
        except Exception:
            pass
        try:
            PipelineAPIHandler._run_pipeline_thread(None, job_id, params)
        except Exception:
            pass   # _run_pipeline_thread records its own failures
        finally:
            _cleanup_job(job_id, params)


class _JobSink:
    def __init__(self, job_id: str, store):
        from ..core.events import CollectSink
        self.job_id = job_id
        self.store = store
        self.collector = CollectSink()
        self.stage_ids: dict[str, int] = {}
        self.active_stage: str | None = None
        self.active_stage_id: int | None = None

    def emit(self, event) -> None:
        from ..core.events import (CancelledEvent, ErrorEvent, StageCompleted,
                                    StageStarted)
        self.collector.emit(event)
        now = time.time()
        if isinstance(event, StageStarted):
            self.active_stage = event.stage
            stage_id = self.store.record_stage(
                self.job_id, event.stage, "running", started_at=now)
            self.active_stage_id = int(stage_id or 0)
            self.stage_ids[event.stage] = self.active_stage_id
        elif isinstance(event, StageCompleted):
            stage_id = self.stage_ids.get(event.stage)
            if stage_id is None:
                self.stage_ids[event.stage] = self.store.record_stage(
                    self.job_id, event.stage, "completed",
                    started_at=now, completed_at=now,
                    duration_seconds=event.duration_seconds)
            else:
                self.store.update_stage(
                    stage_id, status="completed", completed_at=now,
                    duration_seconds=event.duration_seconds)
            if self.active_stage == event.stage:
                self.active_stage = None
                self.active_stage_id = None
        elif isinstance(event, ErrorEvent):
            stage_id = self.stage_ids.get(event.stage)
            if stage_id is None:
                self.stage_ids[event.stage] = self.store.record_stage(
                    self.job_id, event.stage, "failed", started_at=now,
                    completed_at=now)
            else:
                self.store.update_stage(
                    stage_id, status="failed", completed_at=now)
            if self.active_stage == event.stage:
                self.active_stage = None
                self.active_stage_id = None
        elif isinstance(event, CancelledEvent):
            if self.active_stage_id is not None:
                self.store.update_stage(
                    self.active_stage_id, status="cancelled", completed_at=now)
            self.store.update_job(self.job_id, status="cancelled",
                                  completed_at=now)


_LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}
# Largest request body a refused request still has read off the socket.
_REFUSE_DRAIN_LIMIT = 64 * 1024


def _split_host(value: str):
    """'127.0.0.1:5575' / '[::1]:5575' / 'localhost' -> (host, port|None)."""
    value = (value or "").strip().lower()
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port = value.partition(":")
    return host, (int(port) if port.isdigit() else None)


def _allowed_hosts(bound_host: str) -> set:
    """Host names this server answers to.

    Loopback names always; the bound address when it is a specific one; and
    WORDALIGN_ALLOWED_HOSTS (comma-separated) for deliberate LAN setups.
    """
    hosts = set(_LOOPBACK_NAMES)
    if bound_host and bound_host not in ("0.0.0.0", "::", ""):
        hosts.add(bound_host.lower())
    extra = os.environ.get("WORDALIGN_ALLOWED_HOSTS", "")
    hosts.update(h.strip().lower() for h in extra.split(",") if h.strip())
    return hosts


class PipelineAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the GUI API — no external dependencies."""

    def _foreign_request(self, state_changing: bool) -> Optional[str]:
        """Why this request must be refused, or None if it may proceed.

        The API listens on localhost, but any website open in the browser
        can still send requests to it. Two checks keep those out:

        * Host must name this server (blocks DNS rebinding, where an
          attacker's domain is re-pointed at 127.0.0.1).
        * A state-changing request that comes from a browser (it carries
          Origin or Sec-Fetch-Site) must come from this server's own
          origin -- the GUI page. Requests without either header come from
          scripts or curl, which a website cannot make the browser send.
        """
        bound_host, port = self.server.server_address[:2]
        allowed = _allowed_hosts(str(bound_host))
        host, host_port = _split_host(self.headers.get("Host", ""))
        if host not in allowed or host_port not in (None, port):
            return "unexpected Host header"
        if not state_changing:
            return None
        if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
            return "cross-site request"
        origin = self.headers.get("Origin")
        if origin is None:
            return None
        parsed = urlparse(origin)
        if (parsed.scheme != "http" or parsed.hostname not in allowed
                or (parsed.port or 80) != port):
            return "cross-origin request"
        return None

    def _refuse(self, reason: str) -> None:
        # A small body is drained so the connection stays usable; a large
        # one (say, a refused multi-GB upload) is not read at all -- the
        # reply closes the connection instead.
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = -1
        drain = 0 <= length <= _REFUSE_DRAIN_LIMIT
        if drain and length:
            self.rfile.read(length)
        body = json.dumps({"error": f"Forbidden: {reason}"}).encode("utf-8")
        self.send_response(403)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if not drain:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Deliberately no Access-Control-* headers: the GUI is served
        # same-origin from "/", so no cross-origin access is needed, and a
        # wildcard would let any website open in the browser call this
        # localhost API.
        self.end_headers()
        self.wfile.write(body)

    # The API's JSON bodies are a few KB; uploads use their own streaming path.
    _MAX_JSON_BODY = 1 << 20

    def _read_body(self) -> dict | None:
        """The JSON object sent ({} if missing or not an object), or None
        if the body is too large to read."""
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        if length > self._MAX_JSON_BODY:
            self.close_connection = True      # leave it unread
            return None
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _handle_upload(self):
        """Receive a multipart upload, streaming each file to a temp dir."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json({"error": "Expected multipart upload"}, 400)
            return
        length = int(self.headers.get("Content-Length", 0) or 0)
        boundary = content_type.split("boundary=")[1].strip().strip('"') \
            if "boundary=" in content_type else ""
        if not boundary:
            self._send_json({"error": "No boundary"}, 400)
            return
        import tempfile
        upload_dir = tempfile.mkdtemp(prefix="wordalign_upload_")
        try:
            saved_files = _stream_multipart(self.rfile, length,
                                            boundary.encode(), upload_dir)
        except ValueError as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            self._send_json({"error": f"Malformed upload: {exc}"}, 400)
            return
        if not saved_files:
            shutil.rmtree(upload_dir, ignore_errors=True)
        upload_id = uuid.uuid4().hex if saved_files else None
        if upload_id:
            with _RUNS_LOCK:
                _UPLOAD_DIRS[upload_id] = upload_dir
        self._send_json({
            "files": saved_files,
            "upload_id": upload_id,
            "ok": len(saved_files) > 0,
        })

    def do_GET(self):
        reason = self._foreign_request(state_changing=False)
        if reason:
            self._refuse(reason)
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        if path == "/" or path == "":
            # Serve the main app HTML
            gui_path = Path(__file__).parent / "app.html"
            # PyInstaller frozen: html may be in _MEIPASS
            if not gui_path.exists():
                import sys as _sys
                base = getattr(_sys, '_MEIPASS', None)
                if base:
                    gui_path = Path(base) / "wordalign" / "gui" / "app.html"
            if gui_path.exists():
                html = gui_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            else:
                self._send_json({"error": "app.html not found"}, 404)
                return

        if path == "/api/health":
            self._send_json({"status": "ok", "version": __version__})

        elif path == "/api/model-paths":
            from ..models.paths import ModelPaths
            mp = ModelPaths.auto_detect()
            self._send_json(mp.to_dict())

        elif path == "/api/hardware":
            from ..runtimes.detector import HardwareDetector
            detector = HardwareDetector()
            info = detector.detect()
            self._send_json({
                "os": info.os,
                "cpu_count": info.cpu_count,
                "ram_gb": round(info.ram_gb, 1),
                "cpu_brand": info.cpu_brand,
                "gpu": info.gpu_name,
                "gpu_vram_gb": info.gpu_vram_gb,
                "cuda_available": info.cuda_available,
                "ffmpeg_available": info.ffmpeg_available,
                "disk_free_gb": round(info.disk_free_gb, 1),
            })

        elif path == "/api/engines":
            from ..engines.adapters.whisperx_adapter import WhisperXAdapter
            from ..engines.adapters.vosk_adapter import VoskAdapter
            from ..engines.adapters.qwen_adapter import QwenAdapter
            from ..engines.adapters.mfa_adapter import MFAAdapter
            from ..engines.adapters.yamnet_adapter import YAMNetAdapter
            from ..plugins.registry import PluginRegistry
            reg = PluginRegistry()
            reg.register(WhisperXAdapter())
            reg.register(VoskAdapter())
            reg.register(QwenAdapter())
            reg.register(MFAAdapter())
            reg.register(YAMNetAdapter())
            engines = []
            for plugin_id in reg.list_plugins():
                plugin = reg.get(plugin_id)
                desc = plugin.descriptor()
                try:
                    health = plugin.health_check()
                except Exception as exc:
                    from ..plugins.base import HealthStatus
                    health = HealthStatus(
                        ready=False, runtime_status="broken",
                        missing_components=["health_check"],
                        message=str(exc))
                engines.append({
                    "id": desc.engine_id,
                    "name": desc.display_name,
                    "version": desc.version,
                    "capabilities": desc.capabilities.labels(),
                    "runtime_type": desc.runtime_type,
                    "languages": desc.supported_languages,
                    "models": desc.models,
                    "hardware": desc.hardware,
                    "ready": health.ready,
                    "status": health.runtime_status,
                })
            self._send_json({"engines": engines})

        elif path == "/api/models":
            from ..models.catalog import ModelCatalog
            catalog_path = str(Path(__file__).parent.parent / "data" / "model_catalog.json")
            catalog = ModelCatalog(catalog_path)
            models = []
            for entry in catalog.list_all():
                models.append({
                    "id": entry.id,
                    "name": entry.display_name,
                    "display_name": entry.display_name,
                    "plugin_id": entry.plugin_id,
                    "size_bytes": entry.size_bytes,
                    "size_gb": round(entry.size_bytes / 1e9, 2),
                    "languages": entry.languages,
                    "recommended": entry.recommended,
                    "url": entry.source_repo,
                })
            self._send_json({"models": models})

        elif path == "/api/projects":
            from ..core.database import ProjectStore
            store = ProjectStore()
            projects = store.list_projects()
            store.close()
            self._send_json({"projects": [vars(p) for p in projects]})

        elif path == "/api/profiles":
            profiles_dir = Path(__file__).parent.parent / "profiles"
            profiles = []
            for f in profiles_dir.glob("*.json"):
                try:
                    with open(f, encoding="utf-8") as fh:
                        data = json.load(fh)
                    profiles.append({"name": data.get("name", f.stem), "file": f.name, "data": data})
                except Exception:
                    pass
            self._send_json({"profiles": profiles})

        elif path == "/api/recovery":
            from ..core.database import ProjectStore
            from ..core.recovery import CrashRecovery
            store = ProjectStore()
            with _RUNS_LOCK:
                live = set(_ACTIVE_RUNS)
            report = CrashRecovery(store).find_incomplete_jobs(
                exclude_job_ids=live)
            store.close()
            self._send_json({
                "count": report.count,
                "jobs": [
                    {
                        "job_id": item.job.id,
                        "project_id": item.job.project_id,
                        "last_completed_stage": item.last_completed_stage,
                        "stages_completed": item.stages_completed,
                        "stages_total": item.stages_total,
                    }
                    for item in report.incomplete
                ],
            })

        elif path == "/api/jobs":
            from ..core.database import ProjectStore
            store = ProjectStore()
            project_id = params.get("project", [None])[0]
            if project_id:
                jobs = store.list_jobs(project_id)
                store.close()
                self._send_json({"jobs": [vars(j) for j in jobs]})
            else:
                store.close()
                self._send_json({"jobs": []})

        elif path.startswith("/api/job/"):
            job_id = path.split("/")[-1]
            from ..core.database import ProjectStore
            store = ProjectStore()
            job = store.get_job(job_id)
            stages = store.list_stages(job_id) if job else []
            qa_issues = store.list_qa_issues(job_id) if job else []
            store.close()
            if job:
                self._send_json({"job": vars(job), "stages": stages,
                                 "qa_issues": qa_issues,
                                 "queue_position": _queue_position(job_id)})
            else:
                self._send_json({"error": "Job not found"}, 404)

        elif path == "/api/glossary":
            from ..core.database import ProjectStore
            store = ProjectStore()
            project_id = params.get("project", [None])[0]
            entries = store.list_glossary(project_id)
            store.close()
            self._send_json({"glossary": entries})

        elif path == "/api/cache/stats":
            from ..core.cache import StageCache
            cache = StageCache()
            self._send_json({
                "size_bytes": cache.total_size_bytes(),
                "cache_dir": str(cache.cache_dir),
            })

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        reason = self._foreign_request(state_changing=True)
        if reason:
            self._refuse(reason)
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # File upload is handled separately (multipart, not JSON)
        if path == "/api/upload":
            self._handle_upload()
            return

        body = self._read_body()
        if body is None:
            self._send_json({"error": "Request body too large"}, 413)
            return

        if path == "/api/run":
            result = self._start_pipeline(body)
            self._send_json(result, 500 if result.get("error") else 200)

        elif path == "/api/glossary/add":
            from ..core.database import ProjectStore
            store = ProjectStore()
            store.add_glossary_entry(
                term=body.get("term", ""),
                correct_form=body.get("correct_form"),
                context=body.get("context"),
                project_id=body.get("project_id"),
            )
            store.close()
            self._send_json({"ok": True})

        elif path == "/api/cache/clear":
            from ..core.cache import StageCache
            cache = StageCache()
            deleted = cache.clear()
            self._send_json({"deleted": deleted})

        elif path == "/api/run/cancel":
            job_id = body.get("job_id")
            dequeued = None
            with _RUNS_LOCK:
                state = _ACTIVE_RUNS.get(job_id)
                if state is None:
                    self._send_json({"error": "Job not found"}, 404)
                    return
                state["cancel_requested"] = True
                runner = state.get("runner")
                if state.get("status") == "queued":
                    for item in list(_QUEUE):
                        if item[0] == job_id:
                            _QUEUE.remove(item)
                            dequeued = item
                            break
            if dequeued is not None:
                _cancel_queued_job(*dequeued)
            if runner is not None:
                runner.cancel()
            self._send_json({"ok": True, "job_id": job_id})

        else:
            self._send_json({"error": "Not found"}, 404)

    def _start_pipeline(self, body: dict) -> dict:
        """Start a pipeline run and return a job ID immediately."""
        if not (body.get("audio_temp_path") or body.get("audio_path")):
            # Without this a job was queued that could only fail.
            return {"error": "No audio file given."}
        job_id = str(uuid.uuid4())[:8]
        upload_id = body.get("upload_id")
        owned_upload_dir = None
        run_params = dict(body)
        run_params.pop("_owned_upload_dir", None)
        if upload_id:
            with _RUNS_LOCK:
                owned_upload_dir = _UPLOAD_DIRS.pop(str(upload_id), None)
            if owned_upload_dir is None:
                return {"error": "Upload session is invalid or expired."}
            owned_root = os.path.abspath(owned_upload_dir)
            for source in (body.get("audio_temp_path"),
                           body.get("transcript_temp_path")):
                if not source:
                    continue
                try:
                    inside = os.path.commonpath(
                        [owned_root, os.path.abspath(source)]) == owned_root
                except ValueError:
                    inside = False
                if not inside:
                    shutil.rmtree(owned_upload_dir, ignore_errors=True)
                    with _RUNS_LOCK:
                        _UPLOAD_DIRS.pop(str(upload_id), None)
                    return {"error": "Upload path is outside its session."}
            run_params["_owned_upload_dir"] = owned_upload_dir

        # Store job record
        try:
            from ..core.database import ProjectStore, JobRecord
            store = ProjectStore()
            project_id = body.get("project_id", job_id)
            audio_source = body.get("audio_temp_path", body.get("audio_path", ""))
            transcript_source = body.get("transcript_temp_path", body.get("transcript_path"))
            # Create project if it doesn't exist
            if not store.get_project(project_id):
                from ..core.database import ProjectRecord
                store.create_project(ProjectRecord(
                    id=project_id,
                    name=body.get("name", Path(audio_source or "unknown").stem),
                    audio_path=audio_source,
                    transcript_path=transcript_source,
                    language=body.get("language"),
                    mode="reference" if transcript_source else "ensemble",
                ))
            profile = body.get("profile")
            profile_data = {"name": profile} if isinstance(profile, str) else (profile or {})
            store.create_job(JobRecord(
                id=job_id, project_id=project_id, status="queued",
                profile=profile_data,
            ))
            store.close()
        except Exception as exc:
            try:
                store.close()
            except Exception:
                pass
            if owned_upload_dir:
                shutil.rmtree(owned_upload_dir, ignore_errors=True)
                with _RUNS_LOCK:
                    if upload_id:
                        _UPLOAD_DIRS.pop(str(upload_id), None)
            return {"error": f"Could not initialize job: {exc}"}

        position = _submit_job(job_id, run_params)
        return {"job_id": job_id, "status": "queued",
                "queue_position": position}

    def _run_pipeline_thread(self, job_id: str, params: dict):
        from ..config import PipelineConfig
        from ..core.config import PipelineProfile
        from ..core.database import ProjectStore
        from ..core.pipeline import PipelineRunner

        store = None
        sink = None
        runner = None
        try:
            store = ProjectStore()
            sink = _JobSink(job_id, store)
            audio_path = params.get("audio_temp_path", params.get("audio_path", ""))
            transcript_path = params.get("transcript_temp_path", params.get("transcript_path"))
            from ..models.paths import ModelPaths
            mp_dict = params.get("model_paths") or {}
            model_paths = ModelPaths(**{k: v for k, v in mp_dict.items() if v}) \
                          if mp_dict else ModelPaths()
            model_paths._fill_auto()
            # The GUI sends its engine checkboxes as use_<engine>. They are
            # explicit choices, so they must override the preset's engine
            # list (bare <engine> keys are accepted for API callers).
            engine_overrides = {}
            for name in ("vosk", "qwen", "whisperx", "mfa"):
                for key in (name, f"use_{name}"):
                    if key in params:
                        engine_overrides[name] = bool(params[key])
            profile_value = params.get("profile")
            profile = (PipelineProfile.from_dict(profile_value)
                       if isinstance(profile_value, dict) else None)
            profile_name = profile_value if isinstance(profile_value, str) else None
            profile_overrides = {}
            if "max_cpl" in params:
                profile_overrides["max_cpl"] = int(params["max_cpl"])
            if "use_tags" in params:
                profile_overrides["tags"] = bool(params["use_tags"])
            formats = tuple(f for f in (params.get("formats") or ["srt"])
                            if f in ("srt", "vtt", "ass")) or ("srt",)
            if "formats" in params:
                profile_overrides["formats"] = formats
            output_dir = Path.home() / "WordAlign" / "outputs" / job_id
            cfg = PipelineConfig(
                audio_path=audio_path,
                transcript_path=transcript_path,
                allow_transcript_mismatch=params.get("allow_transcript_mismatch", False),
                language=params.get("language"),
                output_dir=str(output_dir),
                model_paths=model_paths,
                use_vosk=params.get("use_vosk", True),
                use_qwen=params.get("use_qwen", True),
                use_whisperx=params.get("use_whisperx", True),
                # Same default as the CLI: MFA runs when it is installed and
                # has a model for the language, and is skipped otherwise.
                use_mfa=params.get("use_mfa", True),
                use_legacy_ensemble=params.get("use_legacy_ensemble", False),
                max_cpl=int(params.get("max_cpl", 42)),
                max_lines=int(params.get("max_lines", 2)),
                subtitle_formats=formats,
                diarize=bool(params.get("diarize", False)),
                num_speakers=(int(params["num_speakers"])
                              if params.get("num_speakers") else None),
                use_tags=params.get("use_tags", False),
                punctuation=params.get("punctuation", False),
                qa=params.get("qa", False),
                profile_name=profile_name,
                engine_overrides=engine_overrides,
                profile_overrides=profile_overrides,
                llm_engine=params.get("llm_engine") or _llama_cpp_dir(),
                llm_model=params.get("llm_model") or _default_llm_model_path(),
                llm_mtp_model=params.get("llm_mtp_model") or _default_llm_mtp_model_path(),
            )
            runner = PipelineRunner(cfg, profile=profile, sink=sink)
            with _RUNS_LOCK:
                state = _ACTIVE_RUNS.get(job_id)
                if state is not None:
                    state["runner"] = runner
                    cancel_requested = state.get("cancel_requested", False)
                else:
                    cancel_requested = False
            if cancel_requested:
                runner.cancel()
            result = runner.run()

            if store is not None:
                for issue in result.qa_issues:
                    data = issue.to_dict() if hasattr(issue, "to_dict") else dict(issue)
                    store.save_qa_issue(job_id, data)
            output_files = result.output_files
            status = "cancelled" if result.cancelled else "completed"
            if status == "cancelled":
                shutil.rmtree(output_dir, ignore_errors=True)
                output_files = []
            if status == "completed" and not output_files:
                status = "failed"
            if store is not None:
                store.update_job(
                    job_id, status=status, completed_at=time.time(),
                    output_files=output_files,
                    warnings=result.warnings,
                    word_count=len(result.aligned_words),
                    segment_count=len(result.segments),
                    error=None if status == "completed" else
                    ("Cancelled by user" if status == "cancelled"
                     else (result.error or "Pipeline produced no output")),
                )
        except Exception as exc:
            try:
                if sink is not None and sink.active_stage_id is not None:
                    sink.store.update_stage(
                        sink.active_stage_id, status="failed", completed_at=time.time())
                status = "cancelled" if runner is not None and runner.cancelled else "failed"
                if store is not None:
                    store.update_job(job_id, status=status, error=str(exc),
                                     completed_at=time.time())
            except Exception:
                pass
        finally:
            if store is not None:
                store.close()
            owned_upload_dir = params.get("_owned_upload_dir")
            if owned_upload_dir:
                shutil.rmtree(owned_upload_dir, ignore_errors=True)
                with _RUNS_LOCK:
                    upload_id = params.get("upload_id")
                    if upload_id:
                        _UPLOAD_DIRS.pop(str(upload_id), None)
            with _RUNS_LOCK:
                _ACTIVE_RUNS.pop(job_id, None)

    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


_UPLOAD_CHUNK = 1 << 20           # bytes read from the socket at a time
_MAX_PART_HEADER = 16 * 1024


# filename="..." (quotes may hold ';' and \" escapes) or a bare token.
_FILENAME_PARAM = re.compile(
    r'(?:^|;)\s*filename\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^;]*))', re.I)


def _upload_filename(header: str) -> Optional[str]:
    """Safe file name from a part's Content-Disposition, or None (no file)."""
    for line in header.split("\r\n"):
        if not line.lower().startswith("content-disposition"):
            continue
        match = _FILENAME_PARAM.search(line)
        if match is None:
            return None
        quoted, bare = match.groups()
        name = quoted.replace('\\"', '"') if quoted is not None else bare
        # Browsers may send a full client path (C:\fakepath\x.wav).
        name = os.path.basename(name.strip().replace("\\", "/")).strip()
        # Characters Windows cannot store in a file name.
        name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name)
        return name if name not in ("", ".", "..") else "uploaded_file"
    return None


def _stream_multipart(rfile, length: int, boundary: bytes,
                      upload_dir: str) -> list:
    """Parse a multipart/form-data body from *rfile*, writing file parts to
    *upload_dir* as they arrive. Returns the saved paths.

    The previous parser read the whole request into memory (a long video is
    gigabytes) and then split it; this one only ever holds one chunk plus
    a boundary's worth of look-behind.
    """
    delimiter = b"--" + boundary
    separator = b"\r\n" + delimiter       # what ends every part's body
    remaining = length
    buf = b""
    saved: list = []

    def fill() -> bool:
        nonlocal remaining, buf
        if remaining <= 0:
            return False
        chunk = rfile.read(min(_UPLOAD_CHUNK, remaining))
        if not chunk:
            remaining = 0
            return False
        remaining -= len(chunk)
        buf += chunk
        return True

    while delimiter not in buf:                 # skip any preamble
        if not fill():
            raise ValueError("no multipart boundary in the body")
        buf = buf[-(len(delimiter) + _UPLOAD_CHUNK):]
    buf = buf[buf.index(delimiter) + len(delimiter):]

    while True:
        while len(buf) < 2 and fill():
            pass
        if buf.startswith(b"--"):               # closing delimiter
            break
        if buf.startswith(b"\r\n"):
            buf = buf[2:]
        while b"\r\n\r\n" not in buf:
            if len(buf) > _MAX_PART_HEADER or not fill():
                raise ValueError("incomplete part header")
        header, buf = buf.split(b"\r\n\r\n", 1)
        name = _upload_filename(header.decode("utf-8", errors="replace"))
        out = None
        if name is not None:
            dest = os.path.join(upload_dir, name)
            if os.path.exists(dest):
                stem, ext = os.path.splitext(name)
                dest = os.path.join(upload_dir, f"{stem}_{len(saved)}{ext}")
            out = open(dest, "wb")
        try:
            while True:
                idx = buf.find(separator)
                if idx != -1:
                    if out:
                        out.write(buf[:idx])
                    buf = buf[idx + len(separator):]
                    break
                # Keep a possible partial separator for the next chunk.
                keep = len(separator) - 1
                if len(buf) > keep:
                    if out:
                        out.write(buf[:-keep])
                    buf = buf[-keep:]
                if not fill():
                    raise ValueError("upload ended inside a file")
        finally:
            if out:
                out.close()
        if out:
            saved.append(dest)
    # Drain anything after the closing delimiter so the socket stays usable.
    while remaining > 0 and fill():
        buf = b""
    return saved


def run_server(port: int = 5575, host: str = "127.0.0.1", open_browser: bool = True):
    """Start the GUI API server, optionally opening the browser."""
    server = ThreadingHTTPServer((host, port), PipelineAPIHandler)
    url = f"http://{host}:{port}"
    print(f"WordAlign GUI server: {url}")
    try:
        from ..core.database import ProjectStore
        store = ProjectStore()
        stale = store.mark_interrupted_jobs()
        store.close()
        if stale:
            print(f"Marked {stale} unfinished job(s) from the last session "
                  "as interrupted.")
    except Exception as exc:
        print(f"[warn] Could not check for interrupted jobs: {exc}")

    if open_browser:
        import webbrowser
        import threading as _threading
        _threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        print("Opening GUI in your browser...")

    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    port = 5575
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])
    run_server(port=port)
