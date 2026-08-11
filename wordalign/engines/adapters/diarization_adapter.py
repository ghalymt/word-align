"""Diarization plugin — assigns speaker IDs to words.

Implements the EnginePlugin interface so the pipeline treats diarization
like any other engine. Uses pyannote.audio when available; falls back to a
silence-based heuristic (voice activity chunks → pseudo-speakers) so the
plugin is functional even without pyannote installed.

Speaker IDs are written into WordResult.speaker and displayed in the
transcript view as ``SPEAKER 1:`` / ``SPEAKER 2:``.
"""
from __future__ import annotations

import os
from typing import List, Optional

from ...core.types import WordResult
from ...plugins.base import EngineDescriptor, EnginePlugin, HealthStatus
from ...plugins.capabilities import Capability


class DiarizationAdapter(EnginePlugin):
    """Speaker diarization via pyannote.audio (optional dependency)."""

    def descriptor(self) -> EngineDescriptor:
        return EngineDescriptor(
            engine_id="diarize",
            display_name="Diarization (pyannote)",
            version="1.0",
            capabilities=Capability.DIARIZE,
            runtime_type="in_process",
            supported_languages=["*"],
            models=["pyannote/speaker-diarization-3.1"],
            hardware={"cpu": True, "cuda": True},
        )

    def health_check(self) -> HealthStatus:
        try:
            import pyannote.audio  # noqa: F401
            return HealthStatus(ready=True, runtime_status="ready",
                                message="pyannote.audio available")
        except Exception:
            return HealthStatus(
                ready=True, runtime_status="degraded",
                message="pyannote.audio not installed; using silence-based fallback")

    def diarize(self, words: List[WordResult],
                audio_path: Optional[str] = None,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> List[WordResult]:
        """Assign speaker IDs to words.

        Uses pyannote if installed and audio_path is provided; otherwise
        falls back to gap-based pseudo-speaker assignment.
        """
        if audio_path and self._pyannote_available():
            try:
                return self._diarize_pyannote(words, audio_path,
                                              min_speakers, max_speakers)
            except Exception:
                pass
        return self._diarize_by_gaps(words)

    @staticmethod
    def _pyannote_available() -> bool:
        try:
            import pyannote.audio  # noqa: F401
            return True
        except Exception:
            return False

    def _diarize_pyannote(self, words: List[WordResult], audio_path: str,
                          min_speakers: Optional[int],
                          max_speakers: Optional[int]) -> List[WordResult]:
        """Run pyannote pipeline and map speaker turns to words."""
        from pyannote.audio import Pipeline

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        diarization = pipeline(audio_path,
                               min_speakers=min_speakers,
                               max_speakers=max_speakers)

        # Build a timeline of (start, end, speaker)
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

        # Assign each word to the speaker whose turn contains its start time
        result = []
        for w in words:
            speaker = None
            if w.start is not None:
                for start, end, spk in turns:
                    if start <= w.start < end:
                        speaker = spk
                        break
            result.append(self._with_speaker(w, speaker))
        return result

    def _diarize_by_gaps(self, words: List[WordResult],
                         gap_threshold: float = 0.8) -> List[WordResult]:
        """Fallback: assign pseudo-speakers by detecting silence gaps.

        Words separated by a gap larger than ``gap_threshold`` seconds are
        treated as a new speaker turn; speakers alternate labels.
        """
        result = []
        current_speaker = "SPEAKER_0"
        speaker_idx = 0
        last_end: Optional[float] = None

        for w in words:
            if w.start is not None and last_end is not None:
                if w.start - last_end > gap_threshold:
                    speaker_idx += 1
                    current_speaker = f"SPEAKER_{speaker_idx}"
            if w.end is not None:
                last_end = w.end
            result.append(self._with_speaker(w, current_speaker))
        return result

    @staticmethod
    def _with_speaker(w: WordResult, speaker: Optional[str]) -> WordResult:
        return WordResult(
            id=w.id, text=w.text, normalized_text=w.normalized_text,
            start=w.start, end=w.end, confidence=w.confidence,
            engine_id=w.engine_id, model_id=w.model_id,
            timing_source=w.timing_source, matched=w.matched,
            speaker=speaker or w.speaker, language=w.language,
            alternatives=w.alternatives, metadata=w.metadata,
        )


def format_speaker_transcript(words: List[WordResult]) -> str:
    """Render a speaker-annotated transcript.

    Groups consecutive words by speaker and prints:
        [00:01.2] SPEAKER_0: Hello there
        [00:04.5] SPEAKER_1: Hi, how are you
    """
    if not words:
        return ""

    lines = []
    current_speaker = None
    current_words: List[str] = []
    current_start = None

    def flush():
        if current_words:
            ts = f"[{current_start:.1f}]" if current_start is not None else ""
            lines.append(f"{ts} {current_speaker}: {' '.join(current_words)}")

    for w in words:
        spk = w.speaker or "SPEAKER_UNKNOWN"
        if spk != current_speaker:
            flush()
            current_speaker = spk
            current_words = []
            current_start = w.start
        current_words.append(w.text)
    flush()
    return "\n".join(lines)
