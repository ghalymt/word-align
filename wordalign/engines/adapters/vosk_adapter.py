"""Vosk adapter."""
from __future__ import annotations

from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability


class VoskAdapter(EnginePlugin):

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="vosk",
            display_name="Vosk",
            version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.WORD_TIMING | Capability.VOTE,
            runtime_type="in_process",
            supported_languages=["en", "fr", "es", "de", "it", "ru", "zh",
                                 "ja", "fa", "pt", "nl", "tr", "sv", "ca",
                                 "uk", "ar", "tl", "vi", "hi", "hr", "sr", "bs"],
            models=["vosk-model-en-us-0.22"],
            hardware={"cpu": True, "cuda": False},
            help_text="Lightweight CPU engine. Fast, best timestamps.",
        )

    def health_check(self) -> HealthStatus:
        missing = []
        try:
            from vosk import Model  # noqa: F401
        except Exception:
            missing.append("vosk")
        from ...models.paths import ModelPaths
        from ...models.validator import ModelValidator
        model_root = ModelPaths().resolve("vosk")
        valid_model = False
        if model_root:
            validator = ModelValidator()
            valid_model = any(
                child.is_dir() and validator.validate(child, "vosk")["valid"]
                for child in model_root.iterdir()
            )
        if not valid_model:
            missing.append("vosk_model")
        if missing:
            return HealthStatus(ready=False, runtime_status="missing",
                                missing_components=missing,
                                message="Install Vosk and unpack a model under models/vosk")
        return HealthStatus(ready=True, runtime_status="ready")

    def transcribe(self, request) -> dict:
        from ..vosk_engine import run_vosk_parallel
        model_path = request.options.get("model_path")
        if not model_path:
            return {"words": [], "events": [], "engine_id": "vosk",
                    "warnings": ["No Vosk model path provided"]}
        words = run_vosk_parallel(request.audio_path, model_path)
        return {"words": words, "events": [], "engine_id": "vosk"}
