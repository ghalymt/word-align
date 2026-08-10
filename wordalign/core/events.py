"""Structured pipeline events.

The PipelineRunner emits events instead of printing to stdout. Both the CLI
and GUI subscribe to these events and translate them into their respective
output formats.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class EventSink(Protocol):
    """Protocol for objects that receive pipeline events."""
    def emit(self, event: "PipelineEvent") -> None: ...


@dataclass
class PipelineEvent:
    """Base event type."""
    timestamp: float = 0.0  # seconds since pipeline start


@dataclass
class PipelineStarted(PipelineEvent):
    audio_path: str = ""
    mode: str = ""             # "reference" | "ensemble"
    language: str = ""


@dataclass
class StageStarted(PipelineEvent):
    stage: str = ""            # "vosk", "whisperx", "segmentation", etc.
    message: str = ""


@dataclass
class StageProgress(PipelineEvent):
    stage: str = ""
    progress: float = 0.0      # 0.0–1.0
    message: str = ""


@dataclass
class StageMessage(PipelineEvent):
    """Informational message from a stage (replaces print())."""
    stage: str = ""
    message: str = ""
    level: str = "info"        # "info" | "warn" | "error"


@dataclass
class StageCompleted(PipelineEvent):
    stage: str = ""
    duration_seconds: float = 0.0
    message: str = ""


@dataclass
class WarningEvent(PipelineEvent):
    stage: str = ""
    message: str = ""


@dataclass
class ErrorEvent(PipelineEvent):
    stage: str = ""
    message: str = ""
    recoverable: bool = False
    suggestions: list[str] = field(default_factory=list)


@dataclass
class PipelineCompleted(PipelineEvent):
    word_count: int = 0
    segment_count: int = 0
    output_files: list[str] = field(default_factory=list)


@dataclass
class CancelledEvent(PipelineEvent):
    stage: str = ""


class PrintSink:
    """Event sink that prints to stdout — used by the CLI."""
    def emit(self, event: PipelineEvent) -> None:
        if isinstance(event, PipelineStarted):
            print("=" * 60)
            print(f"WORD-ALIGN PIPELINE")
            print(f"  mode: {event.mode}")
            print(f"  language: {event.language}")
            print("=" * 60)
        elif isinstance(event, StageStarted):
            print(f"\n[{event.stage}] {'—' * (40 - len(event.stage))}")
            if event.message:
                print(f"  {event.message}")
        elif isinstance(event, StageProgress):
            pct = int(event.progress * 100)
            print(f"  [{event.stage}] {pct}% — {event.message}")
        elif isinstance(event, StageMessage):
            prefix = {"info": "", "warn": "[warn] ", "error": "[error] "}
            print(f"  {prefix.get(event.level, '')}{event.message}")
        elif isinstance(event, StageCompleted):
            print(f"  [{event.stage}] done ({event.duration_seconds:.1f}s)")
        elif isinstance(event, WarningEvent):
            print(f"  [warn] {event.message}")
        elif isinstance(event, ErrorEvent):
            print(f"  [error] {event.message}")
            if event.suggestions:
                for s in event.suggestions:
                    print(f"    → {s}")
        elif isinstance(event, PipelineCompleted):
            print(f"\n{'=' * 60}")
            print(f"COMPLETE: {event.word_count} words, {event.segment_count} segments")
            for f in event.output_files:
                print(f"  → {f}")
            print("=" * 60)
        elif isinstance(event, CancelledEvent):
            print(f"\n[cancelled] Pipeline stopped at: {event.stage}")


class CollectSink:
    """Event sink that collects events into a list — for testing."""
    def __init__(self):
        self.events: list[PipelineEvent] = []

    def emit(self, event: PipelineEvent) -> None:
        self.events.append(event)

    def by_type(self, event_type: type) -> list[PipelineEvent]:
        return [e for e in self.events if isinstance(e, event_type)]

    @property
    def messages(self) -> list[str]:
        return [e.message for e in self.events
                if isinstance(e, (StageMessage, WarningEvent, ErrorEvent))]
