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

import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

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


class PipelineAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the GUI API — no external dependencies."""

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

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _handle_upload(self):
        """Receive multipart file upload and save to temp dir."""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json({"error": "Expected multipart upload"}, 400)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        boundary = content_type.split("boundary=")[1].strip() if "boundary=" in content_type else ""
        if not boundary:
            self._send_json({"error": "No boundary"}, 400)
            return
        boundary_bytes = ("--" + boundary).encode()
        parts = raw.split(boundary_bytes)
        import tempfile
        upload_dir = tempfile.mkdtemp(prefix="wordalign_upload_")
        saved_files = []
        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            header_end = part.find(b"\r\n\r\n")
            if header_end == -1:
                continue
            header = part[:header_end].decode(errors="replace")
            file_data = part[header_end + 4:]
            # Each part ends with exactly one CRLF before the next
            # "--boundary". Strip only that: rstrip(b"\r\n-") also ate any
            # trailing CR/LF/'-' bytes that belong to the file itself.
            if file_data.endswith(b"\r\n"):
                file_data = file_data[:-2]
            fname = "uploaded_file"
            for line in header.split("\r\n"):
                for h in line.split(";"):
                    h = h.strip()
                    if h.lower().startswith("filename="):
                        fname = h.split("=", 1)[1].strip().strip('"').strip("'")
                        break
                if fname != "uploaded_file":
                    break
            safe_name = os.path.basename(fname).strip() or "uploaded_file"
            dest = os.path.join(upload_dir, safe_name)
            if os.path.exists(dest):
                stem, ext = os.path.splitext(safe_name)
                dest = os.path.join(upload_dir, f"{stem}_{len(saved_files)}{ext}")
            with open(dest, "wb") as f:
                f.write(file_data)
            saved_files.append(dest)
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
            report = CrashRecovery(store).find_incomplete_jobs()
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
                self._send_json({"job": vars(job), "stages": stages, "qa_issues": qa_issues})
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
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # File upload is handled separately (multipart, not JSON)
        if path == "/api/upload":
            self._handle_upload()
            return

        body = self._read_body()

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
            with _RUNS_LOCK:
                state = _ACTIVE_RUNS.get(job_id)
                if state is None:
                    self._send_json({"error": "Job not found"}, 404)
                    return
                state["cancel_requested"] = True
                runner = state.get("runner")
            if runner is not None:
                runner.cancel()
            self._send_json({"ok": True, "job_id": job_id})

        else:
            self._send_json({"error": "Not found"}, 404)

    def _start_pipeline(self, body: dict) -> dict:
        """Start a pipeline run and return a job ID immediately."""
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
                id=job_id, project_id=project_id, status="running",
                started_at=time.time(),
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

        with _RUNS_LOCK:
            _ACTIVE_RUNS[job_id] = {
                "runner": None,
                "cancel_requested": False,
            }

        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(job_id, run_params),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "status": "running"}

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
                use_mfa=params.get("use_mfa", False),
                use_legacy_ensemble=params.get("use_legacy_ensemble", False),
                max_cpl=int(params.get("max_cpl", 42)),
                max_lines=int(params.get("max_lines", 2)),
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
            output_files = [p for p in [
                result.word_level_srt_path,
                result.sentence_level_srt_path,
                result.transcript_txt_path,
                result.transcript_docx_path,
                result.audio_tags_srt_path,
                result.combined_srt_path,
                result.job_manifest_path,
            ] if p]
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


def run_server(port: int = 5575, host: str = "127.0.0.1", open_browser: bool = True):
    """Start the GUI API server, optionally opening the browser."""
    server = ThreadingHTTPServer((host, port), PipelineAPIHandler)
    url = f"http://{host}:{port}"
    print(f"WordAlign GUI server: {url}")

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
