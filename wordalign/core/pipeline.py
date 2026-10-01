"""PipelineRunner — the single orchestration entry point.

Both the CLI and GUI call this. It emits structured events instead of
printing to stdout, supports cancellation, and uses typed result objects.

The existing algorithm files (align.py, ensemble.py, segment.py, etc.) are
called directly — they are NOT rewritten. The runner wraps them.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

try:
    import srt
except ImportError:
    srt = None

from ..align import (interpolate_timestamps, make_surgical_mfa,
                     match_timestamps, print_alignment_statistics)
from ..config import PipelineConfig
from ..ensemble import build_consensus, consensus_to_structured_text
from ..segment import (enforce_min_duration,
                       get_config, parse_human_transcript_to_srt_segments,
                       resolve_overlaps, run_iterative_merging,
                       set_config, validate_srt_output,
                       balance_block_enhanced)

from ..utils import (detect_language, extract_tags_from_transcript,
                     get_audio_duration, normalize_word, strip_tags,
                     time_to_ms)
from .config import JobContext, PipelineProfile, SegmentationConfig
from .converters import dict_segment_to_typed, dicts_to_words
from .errors import PipelineCancelled
from .events import (CancelledEvent, ErrorEvent, PipelineCompleted,
                     PipelineEvent, PipelineStarted, StageCompleted,
                     StageMessage, StageStarted, EventSink)
from .types import PipelineResult


_SEGMENTATION_LOCK = threading.RLock()


class PipelineRunner:
    """Orchestrates the full word-align pipeline.

    Parameters
    ----------
    config : PipelineConfig
        The v1.0 config dataclass (audio path, transcript path, engine flags, etc.)
    profile : PipelineProfile, optional
        The v2.0 profile (segmentation config, export config). If None,
        defaults are derived from ``config``.
    sink : EventSink, optional
        Event sink. If None, events are collected internally.
    """

    def __init__(self,
                 config: PipelineConfig,
                 profile: Optional[PipelineProfile] = None,
                 sink: Optional[EventSink] = None):
        self.config = config
        self._profile_explicit = profile is not None or bool(config.profile_name)
        self.profile = profile or self._profile_for_config(config)
        self.sink = sink or _NullSink()
        self.ctx = JobContext(profile=self.profile)
        self._start_time = 0.0
        self._cancel_stage = ""
        self.cancelled = False
        self._stage_durations: Dict[str, float] = {}
        self._warnings: List[str] = []
        self._primary_attempted: set = set()
        self._apply_profile()

    def _profile_for_config(self, cfg: PipelineConfig) -> PipelineProfile:
        if cfg.profile_name:
            from .config import PipelineProfile as Profile
            if os.path.isfile(cfg.profile_name):
                return Profile.from_json(cfg.profile_name)
            return Profile.preset(cfg.profile_name)
        return self._profile_from_config(cfg)

    def _apply_profile(self) -> None:
        overrides = self.config.profile_overrides
        if self._profile_explicit and overrides:
            segmentation = replace(
                self.profile.segmentation,
                **{key: int(overrides[key]) for key in (
                    "max_cpl", "max_lines", "max_duration_ms", "min_cue_ms"
                ) if key in overrides},
            )
            export_overrides = {}
            if "doc_format" in overrides:
                export_overrides["transcript_format"] = overrides["doc_format"]
            if "doc_timestamps" in overrides:
                export_overrides["transcript_timestamps"] = bool(overrides["doc_timestamps"])
            if "tags" in overrides:
                export_overrides["tags"] = bool(overrides["tags"])
            export = replace(self.profile.export, **export_overrides)
            self.profile = replace(self.profile, segmentation=segmentation,
                                   export=export)
        engines = {
            str(item.get("plugin", "")).lower(): bool(item.get("enabled", True))
            for item in self.profile.transcription.engines
            if isinstance(item, dict)
        }
        if engines:
            self.config.use_vosk = engines.get("vosk", False)
            self.config.use_qwen = engines.get(
                "qwen", engines.get("qwen_asr", False))
            self.config.use_whisperx = engines.get("whisperx", False)
        if self._profile_explicit and self.profile.timing.order:
            self.config.use_mfa = "mfa" in self.profile.timing.order
        if "tags" in overrides:
            self.config.use_tags = bool(overrides["tags"])
        else:
            self.config.use_tags = self.config.use_tags or self.profile.export.tags
        self.config.doc_format = self.profile.export.transcript_format
        self.config.doc_timestamps = self.profile.export.transcript_timestamps
        self.config.qa = self.config.qa or self.profile.qa.enabled
        for name, value in self.config.engine_overrides.items():
            if name in {"vosk", "qwen", "whisperx"}:
                setattr(self.config, f"use_{name}", bool(value))
            elif name == "mfa":
                self.config.use_mfa = bool(value)

    @staticmethod
    def _has_content(payload) -> bool:
        """True if *payload* holds at least one real value.

        Engines return ``[]`` / ``([], [])`` on failure or cancel, so an
        empty list, an empty dict, or a container of only empty containers
        all count as "nothing worth caching".
        """
        if payload is None:
            return False
        if isinstance(payload, (list, tuple)):
            return any(PipelineRunner._has_content(item) for item in payload)
        if isinstance(payload, dict):
            return any(PipelineRunner._has_content(v) for v in payload.values())
        return True

    def _cancelled(self) -> bool:
        return self.ctx.cancel_requested or self.ctx.cancel_event.is_set()

    def _cached_engine(self, plugin_id: str, model_id: Optional[str],
                       language: str, settings: dict, loader):
        try:
            from .cache import StageCache
            from .fingerprint import engine_cache_key, fingerprint_audio
            cache = StageCache()
            key = engine_cache_key(
                fingerprint_audio(self.config.audio_path), plugin_id, "2.0",
                model_id, None, language, settings)
            cached = cache.get(key)
            # An empty entry is a failed/cancelled run written by an older
            # version; treat it as a miss so the engine actually runs.
            if self._has_content(cached):
                return cached, True
        except Exception:
            key = None
            cache = None
        output = loader()
        # Never cache a failed run (empty) or a cancelled one (possibly
        # partial): later runs would replay it instead of calling the engine.
        if (cache is not None and key is not None
                and self._has_content(output) and not self._cancelled()):
            try:
                cache.put(key, output)
            except Exception:
                pass
        return output, False

    def _profile_from_config(self, cfg: PipelineConfig) -> PipelineProfile:
        """Derive a v2.0 profile from the v1.0 PipelineConfig."""
        from .config import ExportConfig
        return PipelineProfile(
            segmentation=SegmentationConfig(
                max_cpl=cfg.max_cpl,
                max_lines=cfg.max_lines,
                max_duration_ms=cfg.max_duration_ms,
                min_cue_ms=cfg.min_cue_ms,
            ),
            export=ExportConfig(
                transcript_format=cfg.doc_format,
                transcript_timestamps=cfg.doc_timestamps,
                tags=cfg.use_tags,
            ),
        )

    def _emit(self, event: PipelineEvent) -> None:
        event.timestamp = time.time() - self._start_time
        if isinstance(event, StageCompleted):
            self._stage_durations[event.stage] = event.duration_seconds
        elif isinstance(event, StageMessage) and event.level in {"warn", "error"}:
            self._warnings.append(f"{event.stage}: {event.message}")
        elif isinstance(event, ErrorEvent):
            self._warnings.append(f"{event.stage}: {event.message}")
        self.sink.emit(event)

    def _check_cancel(self, stage: str = "") -> None:
        self._cancel_stage = stage
        self.ctx.check_cancel(stage)

    def run(self) -> PipelineResult:
        self._start_time = time.time()
        self._stage_durations.clear()
        self._warnings.clear()
        self.cancelled = False
        if not self.ctx.cancel_requested:
            self.ctx.cancel_event.clear()
        if self.ctx.cancel_requested:
            self.cancelled = True
            self._emit(CancelledEvent(stage="init"))
            result = PipelineResult(cancelled=True)
            return result
        try:
            return self._run()
        except PipelineCancelled:
            self.cancelled = True
            self._emit(CancelledEvent(stage=self._cancel_stage))
            return PipelineResult(cancelled=True)

    @staticmethod
    @contextmanager
    def _job_layout(seg_cfg: SegmentationConfig):
        """Apply this job's CPL/lines/durations to the segment module.

        segment.py keeps its limits in module globals, and they are reset to
        the defaults after each job. Everything that reads them -- the
        segmenter, the punctuation re-layout and the SRT validator -- must
        run inside this block, or it silently uses the defaults instead.
        """
        with _SEGMENTATION_LOCK:
            previous_layout = get_config()
            try:
                set_config(seg_cfg)
                yield
            finally:
                set_config(SegmentationConfig(**previous_layout))

    def _segment_transcript(self, original_text: str, aligned_words: List[Dict],
                            cue_boundaries: List[Tuple[int, int]],
                            seg_cfg: SegmentationConfig) -> List[Dict]:
        initial_segments = parse_human_transcript_to_srt_segments(
            original_text, aligned_words, cue_boundaries=cue_boundaries)
        if cue_boundaries:
            # Keep the input's cue membership (no merging), but still lay
            # each cue out within the CPL budget and remove overlaps between
            # neighbouring cues -- the validator checks both.
            segments = [dict(seg, text=balance_block_enhanced(seg["text"]))
                        for seg in initial_segments]
            segments = resolve_overlaps(segments)
            self._emit(StageMessage(
                stage="segmentation",
                message=f"Preserving {len(segments)} cues from input SRT (no re-merging)."))
        else:
            segments = run_iterative_merging(initial_segments)
            segments = resolve_overlaps(segments)
            segments = enforce_min_duration(segments, seg_cfg.min_cue_ms)
            segments = resolve_overlaps(segments)
        return segments

    def _stage_enabled(self, stage: str) -> bool:
        if stage in self.config.engine_overrides:
            return bool(self.config.engine_overrides[stage])
        if stage == "rough_srt" and self.config.srt_path:
            return True
        if stage == "interpolation" or not self._profile_explicit:
            return True
        return not self.profile.timing.order or stage in self.profile.timing.order

    def _transcript_mismatch(self, original_text: str,
                            audio_path: str) -> Optional[str]:
        words = strip_tags(original_text).split()
        if len(words) < 200:
            return None
        duration = get_audio_duration(audio_path)
        if duration <= 0:
            return None
        words_per_second = len(words) / duration
        expected_duration = len(words) / 3.0
        if words_per_second > 8.0 or expected_duration > duration * 3.0 + 15.0:
            return (
                f"Transcript/media mismatch: transcript has {len(words)} words "
                f"for {duration:.1f}s of media ({words_per_second:.1f} words/s). "
                "Use a transcript for this exact media or continue in ensemble mode."
            )
        return None

    def _run(self) -> PipelineResult:
        """Execute the full pipeline and return typed results."""
        self._start_time = time.time()
        result = PipelineResult()
        cfg = self.config

        if not os.path.exists(cfg.audio_path):
            self._emit(ErrorEvent(stage="init",
                                  message=f"Audio not found: {cfg.audio_path}"))
            return result

        # Determine mode
        ensemble_conf = None
        wx_events: List[Dict] = []
        primary_transcript_words: List[Dict] = []
        primary_engine: Optional[str] = None
        primary_was_run: bool = False
        legacy_mode = bool(cfg.use_legacy_ensemble and not cfg.transcript_path)
        srt_cue_boundaries: List[Tuple[int, int]] = []

        if cfg.transcript_path:
            try:
                with open(cfg.transcript_path, "r", encoding="utf-8") as f:
                    original_text = f.read()
            except OSError as exc:
                self._emit(ErrorEvent(stage="init",
                                      message=f"Failed to read transcript: {exc}"))
                return result
            if cfg.transcript_path.lower().endswith(".srt") and srt is not None:
                try:
                    cues = list(srt.parse(original_text))
                    flat_lines: List[str] = []
                    cursor = 0
                    for c in cues:
                        content = c.content.strip()
                        if not content:
                            continue
                        # Count words exactly as the aligner will see them:
                        # [tags] are stripped from the word list, so they
                        # must not occupy word slots here or every later
                        # cue boundary shifts. Tag-only cues keep their
                        # text (for --tags) but get no word span.
                        flat_lines.append(content)
                        n_words = len(strip_tags(content).split())
                        if n_words:
                            srt_cue_boundaries.append((cursor, cursor + n_words))
                            cursor += n_words
                    original_text = "\n".join(flat_lines)
                    self._emit(StageMessage(
                        stage="init",
                        message=f"Loaded {len(flat_lines)} cues from SRT transcript; "
                                f"output will preserve their boundaries."))
                except Exception as exc:
                    self._emit(StageMessage(
                        stage="init",
                        message=f"Failed to parse SRT transcript; falling back to raw text ({exc}).",
                        level="warn"))
                    srt_cue_boundaries = []
            language = cfg.language or detect_language(original_text)
            self._emit(PipelineStarted(
                audio_path=cfg.audio_path, mode="reference", language=language))
        else:
            language = cfg.language or "en"
            self._emit(PipelineStarted(
                audio_path=cfg.audio_path, mode="ensemble",
                language=language))
            primary_t0 = time.time()
            self._emit(StageStarted(stage="primary", message="Producing canonical transcript"))
            if legacy_mode:
                (ensemble_conf, original_text, wx_events,
                 _) = self._run_ensemble(cfg, language)
                primary_transcript_words = ensemble_conf or []
                primary_engine = "legacy"
            else:
                primary_transcript_words, original_text, wx_events, primary_engine = \
                    self._run_primary_transcript(cfg, language)
                ensemble_conf = primary_transcript_words
            primary_was_run = True
            self._check_cancel("primary")
            if not original_text:
                self._emit(StageCompleted(stage="primary",
                                          duration_seconds=time.time() - primary_t0))
                self._emit(ErrorEvent(
                    stage="init",
                    message="ASR engines produced no transcript; cannot continue. "
                            "Re-run with --transcript to provide one."))
                return result
            self._emit(StageCompleted(stage="primary",
                                      duration_seconds=time.time() - primary_t0))

        self._check_cancel("primary")

        # Prepare words
        human_tags = extract_tags_from_transcript(original_text)
        text_no_tags = strip_tags(original_text)
        human_words = text_no_tags.strip().split()
        human_words_norm = [normalize_word(w) for w in human_words]
        aligned_words: List[Dict] = [{"word": w} for w in human_words]
        mismatch = (self._transcript_mismatch(original_text, cfg.audio_path)
                    if cfg.transcript_path else None)
        if mismatch and not cfg.allow_transcript_mismatch:
            result.error = mismatch
            self._warnings.append(f"init: {mismatch}")
            self._emit(StageMessage(stage="init", message=mismatch,
                                   level="error"))
            self._emit(ErrorEvent(
                stage="init", message=mismatch,
                suggestions=[
                    "Use a transcript matching the uploaded media.",
                    "Remove the transcript to run transcript-free ensemble mode.",
                    "Set allow_transcript_mismatch=true only after verifying the input.",
                ]))
            return result
        if mismatch:
            self._warnings.append(f"init: {mismatch}; continuing by request")
            self._emit(StageMessage(
                stage="init",
                message=f"{mismatch} Continuing because the mismatch override is enabled.",
                level="warn"))
        # In ensemble mode the primary engine times every word, so the
        # waterfall engines below would have nothing left to fill. Mark the
        # primary's timings as provisional so Vosk/Qwen can refine them.
        refinable_sources: set = set()
        if primary_transcript_words:
            primary_source = {
                "legacy": "Legacy ensemble",
                "whisperx": "WhisperX",
                "qwen": "Qwen3-ASR",
                "vosk": "Vosk",
            }.get(primary_engine or "", "Primary ASR")
            match_timestamps(aligned_words, primary_transcript_words,
                             primary_source, human_words_norm)
            if not legacy_mode:
                refinable_sources = {primary_source}
        self._emit(StageMessage(stage="init",
                                message=f"Prepared {len(aligned_words)} words for alignment"))
        result.aligned_words = dicts_to_words(aligned_words)

        # --- Waterfall (timing refinement) ---
        # Order per design:
        #   1. Vosk       — best per-word timing, lower recall on text
        #   2. (rough SRT — only if --srt was given; carries over from CLI)
        #   3. Qwen3-ASR  — good at both jobs; can fix Vosk's mis-boundaries
        #   4. WhisperX   — strongest diarization + alignment
        #   5. MFA        — surgical per-word timing (last resort)
        # In ensemble mode, the primary WhisperX run already seeded `aligned_words`
        # with timings, so we skip the duplicate WhisperX pass.

        if (cfg.use_vosk and not legacy_mode and primary_engine != "vosk"
                and self._stage_enabled("vosk")):
            self._emit(StageStarted(stage="vosk", message="Running Vosk (parallel)"))
            t0 = time.time()
            from ..engines.vosk_engine import VOSK_AVAILABLE, run_vosk_parallel
            model_path = cfg.vosk_model_path(language)
            if VOSK_AVAILABLE and model_path:
                vosk_words, cache_hit = self._cached_engine(
                    "vosk", model_path, language, {"workers": 4},
                    lambda: run_vosk_parallel(
                        cfg.audio_path, model_path,
                        cancel_event=self.ctx.cancel_event))
                if cache_hit:
                    self._emit(StageMessage(stage="vosk",
                                            message="Loaded Vosk output from cache."))
                match_timestamps(aligned_words, vosk_words, "Vosk",
                                 human_words_norm,
                                 replace_sources=refinable_sources)
                del vosk_words
                gc.collect()
            else:
                self._emit(StageMessage(stage="vosk",
                                        message="Vosk stage skipped (library or model missing).",
                                        level="warn"))
            self._emit(StageCompleted(stage="vosk",
                                       duration_seconds=time.time() - t0))
        self._check_cancel("vosk")

        if cfg.srt_path and self._stage_enabled("rough_srt"):
            self._emit(StageStarted(stage="rough_srt", message="Loading rough SRT"))
            t0 = time.time()
            try:
                with open(cfg.srt_path, "r", encoding="utf-8") as f:
                    subs = list(srt.parse(f.read()))
                srt_words = []
                for s in subs:
                    words_in_sub = s.content.split()
                    duration = (s.end - s.start).total_seconds()
                    per_word = duration / len(words_in_sub) if words_in_sub else 0
                    for i, w in enumerate(words_in_sub):
                        srt_words.append({
                            "word": w,
                            "start": s.start.total_seconds() + i * per_word,
                            "end": s.start.total_seconds() + (i + 1) * per_word})
                match_timestamps(aligned_words, srt_words, "SRT",
                                 human_words_norm)
                del srt_words
                gc.collect()
            except Exception as exc:
                self._emit(StageMessage(stage="rough_srt",
                                        message=f"Error loading SRT: {exc}",
                                        level="warn"))
            self._emit(StageCompleted(stage="rough_srt",
                                       duration_seconds=time.time() - t0))
        self._check_cancel("rough_srt")

        # Qwen already ran (and produced nothing) if the primary step fell
        # through it to Vosk; running the same model again cannot help.
        if (cfg.use_qwen and not legacy_mode and primary_engine != "qwen"
                and "qwen" not in self._primary_attempted
                and self._stage_enabled("qwen")):
            self._emit(StageStarted(stage="qwen", message="Running Qwen3-ASR (timing)"))
            t0 = time.time()
            try:
                from ..engines.qwen_engine import run_qwen
                qw_words, cache_hit = self._cached_engine(
                    "qwen", cfg.qwen_models_dir, language,
                    {"chunk_seconds": cfg.qwen_chunk_seconds,
                     "asr_model": cfg.qwen_asr_model,
                     "aligner_model": cfg.qwen_aligner_model},
                    lambda: run_qwen(
                        cfg.audio_path, language,
                        qwen_python=cfg.qwen_python,
                        models_dir=cfg.qwen_models_dir,
                         chunk_seconds=cfg.qwen_chunk_seconds,
                         asr_model=cfg.qwen_asr_model,
                         aligner_model=cfg.qwen_aligner_model,
                         cancel_event=self.ctx.cancel_event))

                if cache_hit:
                    self._emit(StageMessage(stage="qwen",
                                            message="Loaded Qwen output from cache."))
                if qw_words:
                    match_timestamps(aligned_words, qw_words, "Qwen3-ASR",
                                     human_words_norm,
                                     replace_sources=refinable_sources)
                else:
                    self._emit(StageMessage(stage="qwen",
                                            message="Qwen3-ASR returned no words; skipping.",
                                            level="warn"))
                del qw_words
                gc.collect()
            except Exception as exc:
                self._emit(StageMessage(stage="qwen",
                                        message=f"Qwen stage failed: {exc}",
                                        level="warn"))
            self._emit(StageCompleted(stage="qwen",
                                       duration_seconds=time.time() - t0))
        self._check_cancel("qwen")

        # Skip the second WhisperX pass in ensemble mode — the primary run
        # already produced the timings. Reference mode still needs it.
        # Also skip if --no-whisperx was passed.
        if (not primary_was_run and cfg.use_whisperx
                and self._stage_enabled("whisperx")):
            self._emit(StageStarted(stage="whisperx", message="Running WhisperX"))
            t0 = time.time()
            from ..engines.whisperx_engine import run_whisperx
            wx_output, cache_hit = self._cached_engine(
                "whisperx", cfg.whisper_model, language,
                {"device": cfg.device, "models_dir": cfg.model_paths.whisper_models_dir},
                lambda: run_whisperx(
                    cfg.audio_path, language, cfg.whisper_model, cfg.device,
                    cfg.model_paths.whisper_models_dir,
                    cancel_event=self.ctx.cancel_event))
            wx_words, wx_events = wx_output
            if cache_hit:
                self._emit(StageMessage(stage="whisperx",
                                        message="Loaded WhisperX output from cache."))
            match_timestamps(aligned_words, wx_words, "WhisperX",
                             human_words_norm)
            del wx_words
            gc.collect()
            self._emit(StageCompleted(stage="whisperx",
                                       duration_seconds=time.time() - t0))
        elif primary_was_run:
            self._emit(StageMessage(stage="whisperx",
                                    message="Skipped — already ran as the primary transcript engine.",
                                    level="info"))
        elif not cfg.use_whisperx:
            self._emit(StageMessage(stage="whisperx",
                                    message="Skipped (--no-whisperx).",
                                    level="info"))
        self._check_cancel("whisperx")

        if cfg.use_mfa and self._stage_enabled("mfa"):
            self._emit(StageStarted(stage="mfa", message="Running MFA surgical mode"))
            t0 = time.time()
            make_surgical_mfa(aligned_words, cfg.audio_path, cfg.mfa_cmd,
                              language)
            self._emit(StageCompleted(stage="mfa",
                                       duration_seconds=time.time() - t0))
        self._check_cancel("mfa")

        self._emit(StageStarted(stage="interpolation", message="Interpolating remaining timestamps"))
        t0 = time.time()
        interpolate_timestamps(aligned_words)
        print_alignment_statistics(aligned_words)
        self._emit(StageCompleted(stage="interpolation",
                                   duration_seconds=time.time() - t0))
        self._check_cancel("interpolation")

        # Update typed results
        result.aligned_words = dicts_to_words(aligned_words, 0)

        # --- Segmentation ---
        seg_cfg = self.profile.segmentation
        self._emit(StageStarted(stage="segmentation",
                                message=f"CPL={seg_cfg.max_cpl}, lines={seg_cfg.max_lines}"))
        t0 = time.time()
        with self._job_layout(seg_cfg):
            segments = self._segment_transcript(
                original_text, aligned_words, srt_cue_boundaries, seg_cfg)
        self._emit(StageCompleted(stage="segmentation",
                                   duration_seconds=time.time() - t0))
        result.segments = [dict_segment_to_typed(s) for s in segments]
        self._check_cancel("segmentation")

        # --- Smart punctuation / capitalization ---
        # Only enabled in ensemble mode: a reviewed transcript doesn't need
        # the LLM to second-guess it (per the design contract).
        run_llm = bool(cfg.punctuation) and not bool(cfg.transcript_path)
        if run_llm:
            self._emit(StageStarted(stage="punctuation",
                                    message="Restoring punctuation & capitalization"))
            t0 = time.time()
            from ..qa.punctuation import create_restorer, restore_segments
            restorer = create_restorer(cfg)
            with self._job_layout(seg_cfg):
                segments = restore_segments(segments, restorer,
                                            layout=balance_block_enhanced)
            result.segments = [dict_segment_to_typed(s) for s in segments]
            self._emit(StageCompleted(stage="punctuation",
                                       duration_seconds=time.time() - t0))
            self._check_cancel("punctuation")

        if cfg.qa:
            self._emit(StageStarted(stage="qa", message="Running deterministic quality checks"))
            t0 = time.time()
            try:
                from ..qa.engine import QAEngine
                issues = QAEngine(enable_llm=False).review(result.aligned_words)
                result.qa_issues = list(issues)
                result.statistics["qa_issue_count"] = len(issues)
                self._emit(StageMessage(stage="qa",
                                        message=f"QA found {len(issues)} issue(s)."))
            except Exception as exc:
                self._emit(StageMessage(stage="qa",
                                        message=f"QA stage failed: {exc}",
                                        level="warn"))
            self._emit(StageCompleted(stage="qa",
                                       duration_seconds=time.time() - t0))
            self._check_cancel("qa")

        # --- Outputs ---
        self._emit(StageStarted(stage="output", message="Writing output files"))
        output_t0 = time.time()
        self._check_cancel("output")
        base = cfg.resolve_output_base()
        sentence_entries = []

        if self.profile.export.word_srt:
            if srt is None:
                self._emit(StageMessage(stage="output", message="srt package not installed; skipping word-level SRT", level="warn"))
            else:
                word_entries = [
                    srt.Subtitle(i + 1, timedelta(seconds=w["start"]),
                                 timedelta(seconds=w["end"]), w["word"])
                    for i, w in enumerate(aligned_words) if w.get("start") is not None]
                path = f"{base}_word_level.srt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(srt.compose(word_entries))
                result.word_level_srt_path = path
                self._emit(StageMessage(stage="output",
                                        message=f"Word-level SRT: {len(word_entries)} entries -> {path}"))

        if self.profile.export.sentence_srt:
            if srt is None:
                self._emit(StageMessage(stage="output", message="srt package not installed; skipping sentence-level SRT", level="warn"))
            else:
                sentence_entries = [
                    srt.Subtitle(i + 1,
                                 timedelta(seconds=time_to_ms(s["start"]) / 1000),
                                 timedelta(seconds=time_to_ms(s["end"]) / 1000),
                                 s["text"])
                    for i, s in enumerate(segments)]
                path = f"{base}_sentence_level.srt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(srt.compose(sentence_entries))
                result.sentence_level_srt_path = path
                self._emit(StageMessage(stage="output",
                                        message=f"Sentence-level SRT: {len(sentence_entries)} entries -> {path}"))
                with self._job_layout(seg_cfg):
                    validate_srt_output(sentence_entries, "Sentence-level SRT")

        self._check_cancel("output")
        if ensemble_conf is not None:
            from ..document import export_transcript
            transcript_paths = export_transcript(
                base, os.path.basename(cfg.audio_path), ensemble_conf,
                self.profile.export.transcript_format,
                self.profile.export.transcript_timestamps)
            result.transcript_txt_path = transcript_paths.get("transcript_txt_path")
            result.transcript_docx_path = transcript_paths.get("transcript_docx_path")

        # Tags
        if cfg.use_tags and human_tags and srt is not None:
            from ..tags import (combine_and_sort_srt, detect_audio_tags_yamnet,
                                match_human_tags_to_detections, optimize_tags)
            all_detected = list(wx_events)
            all_detected.extend(detect_audio_tags_yamnet(
                cfg.audio_path, human_tags, wx_events, cfg.yamnet_confidence))
            final_tags = match_human_tags_to_detections(original_text,
                                                        all_detected)
            if final_tags:
                raw_tag_entries = [
                    srt.Subtitle(index=0, start=timedelta(seconds=e["start"]),
                                 end=timedelta(seconds=e["end"]),
                                 content=f"[{e['tag']}]")
                    for e in final_tags]
                optimized = optimize_tags(raw_tag_entries, sentence_entries)
                for i, t in enumerate(optimized):
                    t.index = i + 1
                path = f"{base}_audio_tags.srt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(srt.compose(optimized))
                result.audio_tags_srt_path = path
                combined = combine_and_sort_srt(sentence_entries, raw_tag_entries)
                path2 = f"{base}_combined.srt"
                with open(path2, "w", encoding="utf-8") as f:
                    f.write(srt.compose(combined))
                result.combined_srt_path = path2

        result.aligned_words = dicts_to_words(aligned_words, 0)
        result.consensus = dicts_to_words(ensemble_conf) if ensemble_conf else None
        source_counts: Dict[str, int] = {}
        for word in aligned_words:
            source = word.get("source") or "Unmatched"
            source_counts[source] = source_counts.get(source, 0) + 1
        result.statistics["sources"] = source_counts
        result.warnings = list(self._warnings)
        self._check_cancel("output")
        try:
            from .manifest import write_job_manifest
            result.job_manifest_path = write_job_manifest(
                cfg.audio_path, base,
                profile_name=self.profile.name,
                engines_used=[
                    name for name, enabled in (
                        ("whisperx", cfg.use_whisperx),
                        ("qwen", cfg.use_qwen),
                        ("vosk", cfg.use_vosk),
                        ("mfa", cfg.use_mfa),
                    ) if enabled
                ],
                models_used=[
                    value for value in (
                        cfg.whisper_model if cfg.use_whisperx else None,
                        cfg.vosk_model_path(language) if cfg.use_vosk else None,
                        cfg.qwen_models_dir if cfg.use_qwen else None,
                    ) if value
                ],
                stage_durations=self._stage_durations,
                warnings=self._warnings,
                word_stats={"total": len(aligned_words),
                            "sources": source_counts},
                output_files=[
                    p for p in (
                        result.word_level_srt_path,
                        result.sentence_level_srt_path,
                        result.transcript_txt_path,
                        result.transcript_docx_path,
                        result.audio_tags_srt_path,
                        result.combined_srt_path,
                    ) if p
                ],
            )
        except Exception as exc:
            self._warnings.append(f"manifest: {exc}")

        self._check_cancel("output")
        self._emit(StageCompleted(stage="output",
                                  duration_seconds=time.time() - output_t0))
        result.warnings = list(self._warnings)
        output_files = [p for p in [result.word_level_srt_path,
                                    result.sentence_level_srt_path,
                                    result.transcript_txt_path,
                                    result.transcript_docx_path,
                                    result.audio_tags_srt_path,
                                    result.combined_srt_path,
                                    result.job_manifest_path] if p]

        self._check_cancel("output")
        self._emit(PipelineCompleted(
            word_count=len(aligned_words),
            segment_count=len(segments),
            output_files=output_files))

        return result

    def cancel(self) -> None:
        """Request cancellation. The pipeline stops at the next stage boundary."""
        self.ctx.cancel_requested = True
        self.ctx.cancel_event.set()

    def _run_ensemble(self, cfg: PipelineConfig, language: str):
        """Legacy ensemble path — kept for compatibility (--ensemble flag).

        The new default flow does NOT use this; it calls
        ``_run_primary_transcript`` instead, which runs WhisperX once to get
        the canonical text and timings. Every other engine then refines
        the timing only (see run()).

        This path is invoked only when ``cfg.use_legacy_ensemble`` is True.
        """
        from ..ensemble import consensus_to_structured_text
        engine_words: Dict[str, List[Dict]] = {}
        wx_events: List[Dict] = []

        if "whisperx" in cfg.ensemble_engines:
            from ..engines.whisperx_engine import run_whisperx
            wx_output, _ = self._cached_engine(
                "whisperx", cfg.whisper_model, language,
                {"device": cfg.device, "models_dir": cfg.model_paths.whisper_models_dir},
                lambda: run_whisperx(
                    cfg.audio_path, language, cfg.whisper_model, cfg.device,
                    cfg.model_paths.whisper_models_dir,
                    cancel_event=self.ctx.cancel_event))
            wx_words, wx_events = wx_output
            if wx_words:
                engine_words["whisperx"] = wx_words
        if "qwen" in cfg.ensemble_engines:
            from ..engines.qwen_engine import run_qwen
            qw, _ = self._cached_engine(
                "qwen", cfg.qwen_models_dir, language,
                {"chunk_seconds": cfg.qwen_chunk_seconds,
                 "asr_model": cfg.qwen_asr_model,
                 "aligner_model": cfg.qwen_aligner_model},
                lambda: run_qwen(
                    cfg.audio_path, language,
                    qwen_python=cfg.qwen_python,
                    models_dir=cfg.qwen_models_dir,
                    chunk_seconds=cfg.qwen_chunk_seconds,
                    asr_model=cfg.qwen_asr_model,
                    aligner_model=cfg.qwen_aligner_model,
                    cancel_event=self.ctx.cancel_event))
            if qw:
                engine_words["qwen"] = qw
        if "parakeet" in cfg.ensemble_engines:
            from ..engines.nemo_engine import run_parakeet
            pk = run_parakeet(cfg.audio_path, language)
            if pk:
                engine_words["parakeet"] = pk
        if "canary" in cfg.ensemble_engines:
            from ..engines.nemo_engine import run_canary_qwen
            cq = run_canary_qwen(cfg.audio_path, language)
            if cq:
                engine_words["canary"] = cq
        if "vosk" in cfg.ensemble_engines and cfg.use_vosk:
            from ..engines.vosk_engine import VOSK_AVAILABLE, run_vosk_parallel
            model_path = cfg.vosk_model_path(language)
            if VOSK_AVAILABLE and model_path:
                vk, _ = self._cached_engine(
                    "vosk", model_path, language, {"workers": 4},
                    lambda: run_vosk_parallel(
                        cfg.audio_path, model_path,
                        cancel_event=self.ctx.cancel_event))
                if vk:
                    engine_words["vosk"] = vk
            else:
                self._emit(StageMessage(stage="ensemble",
                                        message="Vosk voter skipped (library or model missing).",
                                        level="warn"))

        try:
            consensus = build_consensus(engine_words)
        except ValueError as exc:
            # No engine produced timed words (all failed or unavailable).
            # Return an empty transcript so run() reports it cleanly instead
            # of crashing with a traceback.
            self._emit(StageMessage(stage="primary", message=str(exc),
                                    level="warn"))
            return [], "", wx_events, []
        return (consensus, consensus_to_structured_text(consensus), wx_events,
                engine_words.get("vosk", []))

    def _primary_text(self, words: List[Dict]) -> str:
        """Render ASR words as subtitle-sized lines for the segmenter.

        The segmenter maps transcript *lines* to cues and only ever merges
        lines, never splits them. Joining the words into one line would turn
        the whole file into a single cue, so break at sentence ends, pauses
        and the line budget -- as the legacy ensemble path already does.
        """
        return consensus_to_structured_text(
            words, max_chars=self.profile.segmentation.max_cpl)

    def _run_primary_transcript(self, cfg: PipelineConfig, language: str):
        """Ensemble-mode step 1: produce the canonical transcript.

        Per the design contract, Whisper is the best ASR in the toolbox, so
        it produces the canonical text and the initial word-level timings.
        Every subsequent engine (Vosk → Qwen → MFA) is only allowed to
        *replace* the timing of words they recognise — never the text.

        Returns: (timed_words, plain_text, wx_events, engine_id)
            timed_words: list of {word,start,end,conf} dicts (the primary timings)
            plain_text : flat text the rest of the pipeline will treat as the
                         "human transcript" — joined with single spaces, tags
                         preserved
            wx_events  : WhisperX audio events (used later for tag detection)
        """
        wx_events: List[Dict] = []
        self._primary_attempted = set()
        if cfg.use_whisperx:
            self._primary_attempted.add("whisperx")
            from ..engines.whisperx_engine import run_whisperx
            wx_output, cache_hit = self._cached_engine(
                "whisperx", cfg.whisper_model, language,
                {"device": cfg.device, "models_dir": cfg.model_paths.whisper_models_dir},
                lambda: run_whisperx(
                    cfg.audio_path, language, cfg.whisper_model, cfg.device,
                    cfg.model_paths.whisper_models_dir,
                    cancel_event=self.ctx.cancel_event))
            wx_words, wx_events = wx_output
            if cache_hit:
                self._emit(StageMessage(stage="whisperx",
                                        message="Loaded WhisperX output from cache."))
            if wx_words:
                return wx_words, self._primary_text(wx_words), wx_events, "whisperx"
            if self.ctx.cancel_event.is_set():
                return [], "", wx_events, None

        if cfg.use_qwen:
            self._primary_attempted.add("qwen")
            from ..engines.qwen_engine import run_qwen
            # Same cache key as the Qwen refinement stage, so either run
            # serves the other.
            qw_words, _ = self._cached_engine(
                "qwen", cfg.qwen_models_dir, language,
                {"chunk_seconds": cfg.qwen_chunk_seconds,
                 "asr_model": cfg.qwen_asr_model,
                 "aligner_model": cfg.qwen_aligner_model},
                lambda: run_qwen(
                    cfg.audio_path, language,
                    qwen_python=cfg.qwen_python,
                    models_dir=cfg.qwen_models_dir,
                    chunk_seconds=cfg.qwen_chunk_seconds,
                    asr_model=cfg.qwen_asr_model,
                    aligner_model=cfg.qwen_aligner_model,
                    cancel_event=self.ctx.cancel_event))
            if qw_words:
                return qw_words, self._primary_text(qw_words), wx_events, "qwen"
            if self.ctx.cancel_event.is_set():
                return [], "", wx_events, None

        if cfg.use_vosk:
            if self.ctx.cancel_event.is_set():
                return [], "", wx_events, None
            from ..engines.vosk_engine import VOSK_AVAILABLE, run_vosk_parallel
            model_path = cfg.vosk_model_path(language)
            if VOSK_AVAILABLE and model_path:
                self._primary_attempted.add("vosk")
                vosk_words, _ = self._cached_engine(
                    "vosk", model_path, language, {"workers": 4},
                    lambda: run_vosk_parallel(
                        cfg.audio_path, model_path,
                        cancel_event=self.ctx.cancel_event))
                if vosk_words:
                    return (vosk_words, self._primary_text(vosk_words),
                            wx_events, "vosk")
        return [], "", wx_events, None


class _NullSink:
    """Default sink that discards events."""
    def emit(self, event: PipelineEvent) -> None:
        pass
