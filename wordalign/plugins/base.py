"""Plugin base interface and supporting types.

Each engine adapter implements ``EnginePlugin``. Capabilities declared in
the manifest determine which methods the pipeline calls.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .capabilities import Capability


@dataclass
class EngineDescriptor:
    """Static metadata about an engine."""
    engine_id: str                     # "whisperx", "qwen_asr", etc.
    display_name: str                  # "WhisperX", "Qwen ASR"
    version: str                       # "1.0"
    capabilities: Capability
    runtime_type: str                  # "in_process" | "subprocess" | "external"
    supported_languages: list[str]     # ["*"] or ["en", "fr", ...]
    models: list[str]                  # model IDs from catalog
    hardware: dict = field(default_factory=dict)  # {"cpu": True, "cuda": True}
    help_text: str = ""
    description: str = ""


@dataclass
class HealthStatus:
    """Result of checking whether an engine is ready to run."""
    ready: bool
    runtime_status: str = "missing"    # "ready" | "missing" | "broken" | "incompatible"
    missing_components: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class ValidationResult:
    """Result of validating a model directory."""
    valid: bool
    reason: str = ""
    expected_files: list[str] = field(default_factory=list)
    detected_files: list[str] = field(default_factory=list)


class EnginePlugin(ABC):
    """Base interface for all engine plugins.

    Not every method must be implemented — capabilities declared in the
    manifest determine which methods are called by the pipeline.
    """

    @abstractmethod
    def descriptor(self) -> EngineDescriptor:
        """Return static metadata about this engine."""
        ...

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Check whether the engine is ready to run."""
        ...

    def validate_model(self, model_path: Path) -> ValidationResult:
        """Check if a directory contains a valid model for this engine."""
        return ValidationResult(valid=False, reason="not implemented")

    def transcribe(self, request) -> dict:
        """Transcribe audio → word list with timestamps."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support TRANSCRIBE")

    def align(self, request) -> dict:
        """Force-align given text against audio."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support FORCED_ALIGN")

    def detect_events(self, request) -> dict:
        """Detect audio events (non-speech)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support AUDIO_EVENTS")

    def review(self, request) -> dict:
        """Review transcript text for issues."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support TEXT_QA")

    def close(self) -> None:
        """Release resources (models, processes, GPU memory)."""
        pass
