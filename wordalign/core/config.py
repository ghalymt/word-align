"""Configuration objects for the pipeline.

Replaces the module-level globals in segment.py with explicit config objects
that are passed through the pipeline. This makes multi-job operation safe.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SegmentationConfig:
    """Subtitle layout constraints — replaces segment.py module globals."""
    max_cpl: int = 42              # characters per line
    max_lines: int = 2             # lines per cue
    max_duration_ms: int = 7000    # max on-screen duration per cue
    min_cue_ms: int = 700          # minimum on-screen duration per cue

    @property
    def cue_char_budget(self) -> int:
        """Total characters a cue may hold: per-line limit × line count."""
        return self.max_cpl * self.max_lines

    @property
    def iteration_start(self) -> int:
        return 1

    @property
    def iteration_end(self) -> int:
        return self.cue_char_budget


@dataclass
class TranscriptionConfig:
    """Which engines to use for ensemble transcription."""
    engines: list[dict] = field(default_factory=list)
    # Each dict: {"plugin": str, "enabled": bool, "model": str,
    #             "weight": float, "backbone": bool, "device": str}


@dataclass
class TimingConfig:
    """Timing waterfall configuration."""
    strategy: str = "fill_only"
    order: list[str] = field(default_factory=lambda: [
        "vosk", "rough_srt", "qwen", "whisperx", "mfa", "interpolation"
    ])


@dataclass
class ExportConfig:
    """Output file configuration."""
    word_srt: bool = True
    sentence_srt: bool = True
    transcript_format: str = "txt"      # "none" | "txt" | "docx" | "both"
    transcript_timestamps: bool = True
    tags: bool = False
    vtt: bool = False                   # sentence-level WebVTT (.vtt)
    ass: bool = False                   # sentence-level ASS (.ass)


@dataclass
class QAConfig:
    """Quality-review settings loaded from a profile."""
    enabled: bool = False
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineProfile:
    """A complete pipeline configuration — loaded from JSON or built programmatically."""
    schema_version: int = 1
    name: str = "Default"
    description: str = ""
    mode: str = "auto"          # "auto" | "reference" | "ensemble"
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    qa: QAConfig = field(default_factory=QAConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineProfile":
        """Parse a profile dict (e.g. from JSON)."""
        seg = d.get("segmentation", {})
        trans = d.get("transcription", {})
        timing = d.get("timing", {})
        export = d.get("export", {})
        qa = d.get("qa", {})
        return cls(
            schema_version=d.get("schema_version", 1),
            name=d.get("name", "Default"),
            description=d.get("description", ""),
            mode=d.get("mode", "auto"),
            transcription=TranscriptionConfig(
                engines=trans.get("engines", [])),
            timing=TimingConfig(
                strategy=timing.get("strategy", "fill_only"),
                order=timing.get("order", [
                    "vosk", "rough_srt", "qwen", "whisperx", "mfa", "interpolation"])),
            segmentation=SegmentationConfig(
                max_cpl=seg.get("max_cpl", 42),
                max_lines=seg.get("max_lines", 2),
                max_duration_ms=seg.get("max_duration_ms", 7000),
                min_cue_ms=seg.get("min_cue_ms", 700)),
            export=ExportConfig(
                word_srt=export.get("word_srt", True),
                sentence_srt=export.get("sentence_srt", True),
                transcript_format=export.get("transcript_format", "txt"),
                transcript_timestamps=export.get("transcript_timestamps", True),
                tags=export.get("tags", False),
                vtt=export.get("vtt", False),
                ass=export.get("ass", False)),
            qa=QAConfig(
                enabled=qa.get("enabled", False),
                signals=qa.get("signals", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineProfile":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def preset(cls, name: str) -> "PipelineProfile":
        """Load a built-in preset."""
        presets = {
            "fast": cls(
                name="Fast",
                description="Vosk + WhisperX, no MFA",
                transcription=TranscriptionConfig(engines=[
                    {"plugin": "whisperx", "enabled": True, "model": "large-v3",
                     "weight": 1.0, "backbone": True},
                    {"plugin": "vosk", "enabled": True, "weight": 0.55},
                ]),
                timing=TimingConfig(order=["vosk", "whisperx", "interpolation"]),
            ),
            "balanced": cls(
                name="Balanced",
                description="WhisperX + Qwen + Vosk, full waterfall",
                qa=QAConfig(enabled=True),
                transcription=TranscriptionConfig(engines=[
                    {"plugin": "whisperx", "enabled": True, "model": "large-v3",
                     "weight": 1.0, "backbone": True},
                    {"plugin": "qwen_asr", "enabled": True, "weight": 0.85},
                    {"plugin": "vosk", "enabled": True, "weight": 0.55},
                ]),
            ),
            "maximum_quality": cls(
                name="Maximum Quality",
                description="All engines, full waterfall, QA",
                qa=QAConfig(enabled=True),
                transcription=TranscriptionConfig(engines=[
                    {"plugin": "whisperx", "enabled": True, "model": "large-v3",
                     "weight": 1.0, "backbone": True},
                    {"plugin": "qwen_asr", "enabled": True, "weight": 0.85},
                    {"plugin": "vosk", "enabled": True, "weight": 0.55},
                ]),
            ),
            "cpu_only": cls(
                name="CPU Only",
                description="Vosk only, no GPU required",
                qa=QAConfig(enabled=True),
                transcription=TranscriptionConfig(engines=[
                    {"plugin": "vosk", "enabled": True, "weight": 1.0,
                     "backbone": True},
                ]),
                timing=TimingConfig(order=["vosk", "interpolation"]),
            ),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. "
                             f"Available: {', '.join(presets)}")
        return presets[name]


@dataclass
class JobContext:
    """Per-run context: cancellation, cache, timing."""
    cancel_requested: bool = False
    cache_dir: Optional[str] = None
    output_base: Optional[str] = None
    profile: PipelineProfile = field(default_factory=PipelineProfile)
    started_at: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def check_cancel(self, stage: str = "") -> None:
        """Raise PipelineCancelled if cancellation was requested."""
        if self.cancel_requested:
            self.cancel_event.set()
            from .errors import PipelineCancelled
            raise PipelineCancelled(f"Cancelled at: {stage}")
