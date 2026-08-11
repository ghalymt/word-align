"""Background worker thread for the native GUI.

Runs the PipelineRunner in a QThread so the UI stays responsive.
Emits signals for events, completion, and failure.
"""
from __future__ import annotations

from typing import Optional

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    QThread = None  # type: ignore
    Signal = None   # type: ignore

from ...core.config import PipelineProfile
from ...core.events import (
    CollectSink, PipelineCompleted, PipelineEvent, PipelineStarted,
    StageCompleted, StageMessage, WarningEvent,
)
from ...core.pipeline import PipelineRunner
from ...config import PipelineConfig


if QThread is not None:
    class PipelineWorker(QThread):
        """Runs a pipeline in the background, forwarding events as text lines.

        Only usable when PySide6 is installed.
        """

        event_received = Signal(str)      # formatted event line
        finished_run = Signal(list, list)  # output files, raw events
        failed = Signal(str)              # error message

        def __init__(self, config: PipelineConfig,
                     profile: Optional[PipelineProfile] = None,
                     sink: Optional[CollectSink] = None):
            super().__init__()
            self._config = config
            self._profile = profile
            self._sink = sink or CollectSink()
            self._runner: PipelineRunner | None = None

        def cancel(self):
            if self._runner:
                self._runner.cancel()

        def run(self):
            try:
                self._runner = PipelineRunner(self._config,
                                              profile=self._profile,
                                              sink=self._sink)
                result = self._runner.run()

                # Replay events as formatted text
                for event in self._sink.events:
                    line = _format_event(event)
                    if line:
                        self.event_received.emit(line)

                output_files = [
                    p for p in (
                        result.word_level_srt_path,
                        result.sentence_level_srt_path,
                        result.transcript_txt_path,
                        result.transcript_docx_path,
                        result.audio_tags_srt_path,
                        result.combined_srt_path,
                        result.job_manifest_path,
                    ) if p
                ]
                self.finished_run.emit(output_files, self._sink.events)
            except Exception as exc:
                self.failed.emit(str(exc))
else:
    class PipelineWorker:  # type: ignore
        """Stub when PySide6 is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("PySide6 is required for the native GUI. "
                              "Install with: pip install PySide6")


def _format_event(event: PipelineEvent) -> str:
    """Render an event to a single log line."""
    if isinstance(event, PipelineStarted):
        return f"[start] mode={event.mode} language={event.language}"
    if isinstance(event, StageMessage):
        prefix = {"warn": "[warn] ", "error": "[error] "}.get(event.level, "")
        return f"[{event.stage}] {prefix}{event.message}"
    if isinstance(event, StageCompleted):
        return f"[{event.stage}] done ({event.duration_seconds:.1f}s)"
    if isinstance(event, WarningEvent):
        return f"[warn] {event.message}"
    if isinstance(event, PipelineCompleted):
        return f"[done] {event.word_count} words, {event.segment_count} segments"
    return ""
