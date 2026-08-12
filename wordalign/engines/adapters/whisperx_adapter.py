"""WhisperX adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus, ValidationResult
from ...plugins.capabilities import Capability


class WhisperXAdapter(EnginePlugin):

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="whisperx",
            display_name="WhisperX",
            version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.WORD_TIMING | Capability.VOTE,
            runtime_type="in_process",
            supported_languages=["*"],
            models=["whisper-large-v3", "whisper-medium", "whisper-small"],
            hardware={"cpu": True, "cuda": True},
            help_text="Whisper large-v3 + wav2vec2 forced alignment. "
                      "Best multilingual transcription.",
        )

    def health_check(self) -> HealthStatus:
        missing = []
        try:
            import whisperx  # noqa: F401
        except ImportError:
            missing.append("whisperx")
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch")
        if missing:
            return HealthStatus(ready=False, runtime_status="missing",
                                missing_components=missing,
                                message="Install: pip install whisperx torch")
        return HealthStatus(ready=True, runtime_status="ready")

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
