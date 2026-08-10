"""Qwen3-ASR adapter."""
from __future__ import annotations

from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability


class QwenAdapter(EnginePlugin):

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="qwen_asr",
            display_name="Qwen ASR",
            version="1.0",
            capabilities=Capability.TRANSCRIBE | Capability.WORD_TIMING | Capability.VOTE,
            runtime_type="subprocess",
            supported_languages=["en", "zh", "ar", "fr", "de", "es", "pt",
                                 "it", "ko", "ru", "th", "vi", "ja", "tr",
                                 "hi", "ms", "nl", "sv", "da", "fi", "pl",
                                 "cs", "tl", "fa", "el", "ro", "hu", "mk"],
            models=["qwen3-asr-1.7b", "qwen3-forcedaligner-0.6b"],
            hardware={"cpu": True, "cuda": True},
            help_text="Qwen3-ASR with forced aligner. Timed challenger voter.",
        )

    def health_check(self) -> HealthStatus:
        import os
        qwen_python = os.environ.get("WORDALIGN_QWEN_PYTHON")
        if qwen_python:
            return HealthStatus(ready=True, runtime_status="ready",
                                message=f"venv: {qwen_python}")
        try:
            import qwen_asr  # noqa: F401
            return HealthStatus(ready=True, runtime_status="ready")
        except ImportError:
            return HealthStatus(
                ready=False, runtime_status="missing",
                missing_components=["qwen_asr"],
                message="Set WORDALIGN_QWEN_PYTHON to the venv with qwen_asr, "
                        "or pip install qwen-asr")

    def transcribe(self, request) -> dict:
        from ..qwen_engine import run_qwen
        words = run_qwen(
            request.audio_path,
            request.language,
            qwen_python=request.options.get("qwen_python"),
            models_dir=request.options.get("models_dir"),
            chunk_seconds=request.options.get("chunk_seconds", 60.0),
            asr_model=request.options.get("asr_model", "Qwen/Qwen3-ASR-1.7B"),
            aligner_model=request.options.get("aligner_model",
                                               "Qwen/Qwen3-ForcedAligner-0.6B"),
        )
        return {"words": words, "events": [], "engine_id": "qwen_asr"}
