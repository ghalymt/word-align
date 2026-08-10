"""NeMo Parakeet/Canary adapter."""
from __future__ import annotations

from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability


class NeMoAdapter(EnginePlugin):
    """Combined adapter for NeMo Parakeet and Canary-Qwen."""

    def __init__(self, variant: str = "parakeet"):
        self.variant = variant

    def descriptor(self) -> EngineDescriptor:
        if self.variant == "parakeet":
            return EngineDescriptor(
                engine_id="parakeet",
                display_name="NeMo Parakeet TDT",
                version="1.0",
                capabilities=Capability.TRANSCRIBE | Capability.WORD_TIMING | Capability.VOTE,
                runtime_type="in_process",
                supported_languages=["en", "fr", "de", "es", "it", "pt", "nl",
                                     "pl", "ru", "uk", "sv", "da", "no", "fi",
                                     "cs", "sk", "sl", "hr", "bg", "ro", "hu",
                                     "el", "lt", "lv", "et"],
                models=["nvidia/parakeet-tdt-0.6b-v3"],
                hardware={"cpu": False, "cuda": True},
                help_text="Fast, high-accuracy English/European transcription.",
            )
        else:
            return EngineDescriptor(
                engine_id="canary",
                display_name="NeMo Canary-Qwen",
                version="1.0",
                capabilities=Capability.TRANSCRIBE | Capability.VOTE,
                runtime_type="in_process",
                supported_languages=["en"],
                models=["nvidia/canary-qwen-2.5b"],
                hardware={"cpu": False, "cuda": True},
                help_text="Highest WER accuracy for English. Text-only voter.",
            )

    def health_check(self) -> HealthStatus:
        try:
            import nemo.collections.asr  # noqa: F401
            return HealthStatus(ready=True, runtime_status="ready")
        except ImportError:
            return HealthStatus(
                ready=False, runtime_status="missing",
                missing_components=["nemo_toolkit"],
                message="Install: pip install 'nemo_toolkit[asr]'")

    def transcribe(self, request) -> dict:
        if self.variant == "parakeet":
            from ..nemo_engine import run_parakeet
            words = run_parakeet(request.audio_path, request.language or "en")
        else:
            from ..nemo_engine import run_canary_qwen
            words = run_canary_qwen(request.audio_path, request.language or "en")
        return {"words": words, "events": [],
                "engine_id": self.descriptor().engine_id}
