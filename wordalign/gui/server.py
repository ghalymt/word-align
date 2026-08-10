"""Local HTTP API server for the WordAlign GUI.

Runs a Flask/FastAPI server that the web GUI can talk to.
Provides endpoints for: file selection, pipeline execution,
progress events, QA issues, model management, and hardware info.

Usage:
    python -m wordalign.gui.server                  # default port 5575
    python -m wordalign.gui.server --port 8080      # custom port

Then open wordalign/gui/app.html in a browser — it connects automatically.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


class PipelineAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the GUI API — no external dependencies."""

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
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

    def do_OPTIONS(self):
        self._send_json({"ok": True})

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        if path == "/api/health":
            self._send_json({"status": "ok", "version": "2.0"})

        elif path == "/api/hardware":
            from ..runtimes.detector import HardwareDetector
            detector = HardwareDetector()
            info = detector.detect()
            self._send_json({
                "os": info.os,
                "cpu_count": info.cpu_count,
                "ram_gb": info.ram_gb,
                "gpu": info.gpu_name,
                "gpu_vram_gb": info.gpu_vram_gb,
                "cuda_available": info.cuda_available,
                "ffmpeg_available": info.ffmpeg_available,
                "python_version": info.python_version,
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
            for plugin_id in reg.list_ids():
                plugin = reg.get(plugin_id)
                desc = plugin.descriptor()
                health = plugin.health_check()
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
                    "name": entry.name,
                    "plugin_id": entry.plugin_id,
                    "size_gb": entry.size_gb,
                    "languages": entry.languages,
                    "recommended": entry.recommended,
                    "url": entry.url,
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
                    with open(f) as fh:
                        data = json.load(fh)
                    profiles.append({"name": data.get("name", f.stem), "file": f.name, "data": data})
                except Exception:
                    pass
            self._send_json({"profiles": profiles})

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
        body = self._read_body()

        if path == "/api/run":
            # Start a pipeline run in a background thread
            result = self._start_pipeline(body)
            self._send_json(result)

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

        elif path == "/api/project/create":
            from ..core.database import ProjectStore, ProjectRecord
            project_id = body.get("id", str(uuid.uuid4())[:8])
            store = ProjectStore()
            store.create_project(ProjectRecord(
                id=project_id,
                name=body.get("name", "Untitled"),
                audio_path=body.get("audio_path", ""),
                transcript_path=body.get("transcript_path"),
                language=body.get("language"),
                mode=body.get("mode", "reference"),
            ))
            store.close()
            self._send_json({"id": project_id, "ok": True})

        else:
            self._send_json({"error": "Not found"}, 404)

    def _start_pipeline(self, body: dict) -> dict:
        """Start a pipeline run and return a job ID immediately."""
        job_id = str(uuid.uuid4())[:8]

        # Store job record
        try:
            from ..core.database import ProjectStore, JobRecord
            store = ProjectStore()
            project_id = body.get("project_id", job_id)
            # Create project if it doesn't exist
            if not store.get_project(project_id):
                from ..core.database import ProjectRecord
                store.create_project(ProjectRecord(
                    id=project_id,
                    name=body.get("name", Path(body.get("audio_path", "unknown")).stem),
                    audio_path=body.get("audio_path", ""),
                    transcript_path=body.get("transcript_path"),
                    language=body.get("language"),
                    mode="reference" if body.get("transcript_path") else "ensemble",
                ))
            store.create_job(JobRecord(
                id=job_id, project_id=project_id, status="running",
                started_at=time.time(),
                profile=body.get("profile", {}),
            ))
            store.close()
        except Exception:
            pass  # DB is optional; pipeline can run without it

        # Run pipeline in background thread
        thread = threading.Thread(
            target=self._run_pipeline_thread,
            args=(job_id, body),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "status": "running"}

    def _run_pipeline_thread(self, job_id: str, params: dict):
        """Run the pipeline and update job status."""
        try:
            from ..config import PipelineConfig
            from ..core.pipeline import PipelineRunner
            from ..core.events import CollectSink

            sink = CollectSink()
            cfg = PipelineConfig(
                audio_path=params["audio_path"],
                transcript_path=params.get("transcript_path"),
                language=params.get("language"),
                vosk_models_dir=params.get("vosk_models_dir") or os.environ.get("WORDALIGN_VOSK_MODELS"),
                use_vosk=params.get("use_vosk", True),
                use_mfa=params.get("use_mfa", False),
                max_cpl=params.get("max_cpl", 42),
                max_lines=params.get("max_lines", 2),
            )
            runner = PipelineRunner(cfg, sink=sink)
            result = runner.run()

            # Update job record
            try:
                from ..core.database import ProjectStore
                store = ProjectStore()
                output_files = [p for p in [
                    result.word_level_srt_path,
                    result.sentence_level_srt_path,
                    result.transcript_txt_path,
                    result.transcript_docx_path,
                ] if p]
                store.update_job(job_id,
                    status="completed",
                    completed_at=time.time(),
                    output_files=output_files,
                    word_count=len(result.aligned_words),
                )
                store.close()
            except Exception:
                pass

        except Exception as exc:
            try:
                from ..core.database import ProjectStore
                store = ProjectStore()
                store.update_job(job_id, status="failed", error=str(exc),
                                 completed_at=time.time())
                store.close()
            except Exception:
                pass

    def log_message(self, format, *args):
        # Suppress default HTTP logging
        pass


def run_server(port: int = 5575, host: str = "127.0.0.1"):
    """Start the GUI API server."""
    server = HTTPServer((host, port), PipelineAPIHandler)
    print(f"WordAlign GUI API server running on http://{host}:{port}")
    print(f"Open wordalign/gui/app.html in your browser to use the GUI.")
    print(f"Press Ctrl+C to stop.")
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
