"""Typed data structures for the WordAlign pipeline.

These replace the anonymous ``Dict[str, Any]`` objects passed through the
v1.0 pipeline. Conversion helpers in ``converters.py`` bridge the new typed
structures to the legacy dict format expected by ``align.py``, ``ensemble.py``,
and ``segment.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WordResult:
    """A single word with optional timing, confidence, and provenance."""
    id: str                        # stable ID: "w000001", "w000002", ...
    text: str                      # original surface form
    normalized_text: str           # lowercased, punctuation stripped
    start: Optional[float] = None  # seconds
    end: Optional[float] = None    # seconds
    confidence: float = 1.0        # fused agreement / engine confidence
    engine_id: Optional[str] = None
    model_id: Optional[str] = None
    timing_source: Optional[str] = None   # "Vosk", "WhisperX", "MFA", etc.
    matched: bool = False
    speaker: Optional[str] = None          # for future diarization
    language: Optional[str] = None         # for future code-switching
    alternatives: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentResult:
    """A subtitle cue (one or more lines)."""
    index: int
    start: float                          # seconds
    end: float                            # seconds
    text: str                             # may contain \n for multi-line
    word_ids: list[str] = field(default_factory=list)
    start_source: Optional[str] = None    # timing source of start boundary
    end_source: Optional[str] = None      # timing source of end boundary
    issues: list[str] = field(default_factory=list)  # issue IDs overlapping
    speaker: Optional[str] = None         # "Speaker 1"... when diarized


@dataclass
class EngineResult:
    """Output from a single engine invocation."""
    engine_id: str
    model_id: Optional[str] = None
    language: Optional[str] = None
    words: list[WordResult] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class TranscriptionRequest:
    """Request to transcribe audio into words."""
    audio_path: str
    language: Optional[str] = None
    model_path: Optional[str] = None
    device: Optional[str] = None        # "cuda" | "cpu"
    chunk_seconds: Optional[float] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentRequest:
    """Request to force-align text against audio."""
    audio_path: str
    words: list[str]                     # word texts to align
    language: Optional[str] = None
    model_path: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Final output of a pipeline run."""
    word_level_srt_path: Optional[str] = None
    sentence_level_srt_path: Optional[str] = None
    transcript_txt_path: Optional[str] = None
    transcript_docx_path: Optional[str] = None
    audio_tags_srt_path: Optional[str] = None
    combined_srt_path: Optional[str] = None
    aligned_words: list[WordResult] = field(default_factory=list)
    segments: list[SegmentResult] = field(default_factory=list)
    consensus: Optional[list[WordResult]] = None
    statistics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    qa_issues: list[Any] = field(default_factory=list)
    job_manifest_path: Optional[str] = None
    sentence_level_vtt_path: Optional[str] = None
    sentence_level_ass_path: Optional[str] = None
    error: Optional[str] = None
    cancelled: bool = False

    @property
    def output_files(self) -> list[str]:
        """Every file this run wrote, in a stable order."""
        return [p for p in (
            self.word_level_srt_path,
            self.sentence_level_srt_path,
            self.sentence_level_vtt_path,
            self.sentence_level_ass_path,
            self.transcript_txt_path,
            self.transcript_docx_path,
            self.audio_tags_srt_path,
            self.combined_srt_path,
            self.job_manifest_path,
        ) if p]
