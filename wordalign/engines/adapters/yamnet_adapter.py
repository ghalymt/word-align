"""YAMNet audio-event tagging adapter."""
from __future__ import annotations

from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability


class YAMNetAdapter(EnginePlugin):

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="yamnet",
            display_name="YAMNet Audio Events",
            version="1.0",
            capabilities=Capability.AUDIO_EVENTS,
            runtime_type="in_process",
            supported_languages=["*"],
            models=["yamnet"],
            hardware={"cpu": True, "cuda": False},
            help_text="Experimental audio-event detection (laughs, music, etc.).",
        )

    def health_check(self) -> HealthStatus:
        missing = []
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            missing.append("tensorflow")
        try:
            import tensorflow_hub  # noqa: F401
        except ImportError:
            missing.append("tensorflow-hub")
        if missing:
            return HealthStatus(ready=False, runtime_status="missing",
                                missing_components=missing)
        return HealthStatus(ready=True, runtime_status="ready")

    def detect_events(self, request) -> dict:
        from ..tags import detect_audio_tags_yamnet
        events = detect_audio_tags_yamnet(
            request.audio_path,
            request.options.get("human_tags", set()),
            request.options.get("whisperx_events", []),
            request.options.get("confidence", 0.9),
        )
        return {"events": events, "engine_id": "yamnet"}
