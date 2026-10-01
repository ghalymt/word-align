"""WhisperX adapter."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from ...models.paths import ModelPaths
from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability


class WhisperXAdapter(EnginePlugin):

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="whisperx",
            display_name="WhisperX",
            version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.WORD_TIMING | Capability.VOTE,
            runtime_type="subprocess",   # now runs out-of-process via worker venv
            supported_languages=["*"],
            models=["whisper-large-v3", "whisper-medium", "whisper-small"],
            hardware={"cpu": True, "cuda": True},
            help_text="Whisper large-v3 + wav2vec2 forced alignment. "
                      "Best multilingual transcription. Runs in a separate "
                      "venv via WORDALIGN_WHISPERX_PYTHON.",
        )

    def health_check(self) -> HealthStatus:
        whisperx_python = os.environ.get("WORDALIGN_WHISPERX_PYTHON")
        missing = []
        if whisperx_python and not Path(whisperx_python).is_file():
            missing.append("whisperx_python")
        elif whisperx_python:
            try:
                probe = subprocess.run(
                    [whisperx_python, "-c", "import whisperx, torch"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30, check=False)
                if probe.returncode != 0:
                    missing.append("whisperx")
            except (OSError, subprocess.TimeoutExpired):
                missing.append("whisperx")
        elif not whisperx_python:
            if getattr(sys, "frozen", False):
                missing.append("whisperx_python")
            else:
                try:
                    import whisperx  # noqa: F401
                except Exception:
                    missing.append("whisperx")
                try:
                    import torch  # noqa: F401
                except Exception:
                    missing.append("torch")
        if not ModelPaths().resolve("whisperx"):
            missing.append("whisperx_model")
        if missing:
            return HealthStatus(ready=False, runtime_status="missing",
                                missing_components=missing,
                                message="Configure a WhisperX venv and local models")
        return HealthStatus(ready=True, runtime_status="ready",
                            message=f"venv: {whisperx_python}" if whisperx_python else "in-process")

    def transcribe(self, request) -> dict:
        from ..whisperx_engine import run_whisperx
        words, events = run_whisperx(
            request.audio_path,
            request.language or "en",
            request.options.get("model", "large-v3"),
            request.options.get("device"),
            models_dir=request.options.get("models_dir"),
        )
        return {"words": words, "events": events,
                "engine_id": "whisperx"}
