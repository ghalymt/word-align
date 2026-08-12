"""PipelineRunner — the single orchestration entry point.

Both the CLI and GUI call this. It emits structured events instead of
printing to stdout, supports cancellation, and uses typed result objects.

The existing algorithm files (align.py, ensemble.py, segment.py, etc.) are
called directly — they are NOT rewritten. The runner wraps them.
"""
from __future__ import annotations

import gc
import os
import time
from datetime import timedelta
from typing import Dict, List, Optional

try:
    import srt
except ImportError:
    srt = None

from ..align import (interpolate_timestamps, make_surgical_mfa,
                     match_timestamps, print_alignment_statistics)
from ..config import PipelineConfig
from ..ensemble import (build_consensus, consensus_to_structured_text)
from ..segment import (enforce_min_duration,
                       parse_human_transcript_to_srt_segments,
                       resolve_overlaps, run_iterative_merging,
                       set_config, set_layout, validate_srt_output)
from ..utils import (detect_language, extract_tags_from_transcript,
                     normalize_word, strip_tags, time_to_ms)
from .config import JobContext, PipelineProfile, SegmentationConfig
from .converters import dicts_to_words, words_to_dicts
from .errors import PipelineCancelled
from .events import (CancelledEvent, ErrorEvent, PipelineCompleted,
                     PipelineEvent, PipelineStarted, StageCompleted,
                     StageMessage, StageStarted, EventSink)
from .types import PipelineResult, WordResult


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
        self.profile = profile or self._profile_from_config(config)
        self.sink = sink or _NullSink()
        self.ctx = JobContext(profile=self.profile)
        self._start_time = 0.0

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
        self.sink.emit(event)

    def _check_cancel(self, stage: str = "") -> None:
        self.ctx.check_cancel(stage)

    def run(self) -> PipelineResult:
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
        vosk_words_cache: List[Dict] = []

        if cfg.transcript_path:
            # Reference mode
            try:
                with open(cfg.transcript_path, "r", encoding="utf-8") as f:
                    original_text = f.read()
            except OSError as exc:
                self._emit(ErrorEvent(stage="init",
                                      message=f"Failed to read transcript: {exc}"))
                return result
            language = cfg.language or detect_language(original_text)
            wx_pending = True
            self._emit(PipelineStarted(
                audio_path=cfg.audio_path, mode="reference", language=language))
        else:
            # Ensemble mode
            self._emit(PipelineStarted(
                audio_path=cfg.audio_path, mode="ensemble",
                language=cfg.language or "en"))
            language = cfg.language or "en"
            (consensus, original_text, wx_events_cache,
             vosk_words_cache) = self._run_ensemble(cfg, language)
            ensemble_conf = consensus
            wx_pending = False

        # Prepare words
        human_tags = extract_tags_from_transcript(original_text)
        text_no_tags = strip_tags(original_text)
        human_words = text_no_tags.strip().split()
        human_words_norm = [normalize_word(w) for w in human_words]
        aligned_words: List[Dict] = [{"word": w} for w in human_words]
        self._emit(StageMessage(stage="init",
                                message=f"Prepared {len(aligned_words)} words for alignment"))
        result.aligned_words = dicts_to_words(aligned_words)

        # --- Waterfall ---
        wx_events: List[Dict] = []

        if ensemble_conf is not None:
            self._emit(StageStarted(stage="waterfall", message="Ensemble timing waterfall"))
            if vosk_words_cache:
                match_timestamps(aligned_words, vosk_words_cache, "Vosk",
                                 human_words_norm)
            match_timestamps(aligned_words, ensemble_conf, "Ensemble",
                             human_words_norm)
            wx_events = wx_events_cache
            self._emit(StageCompleted(stage="waterfall",
                                       duration_seconds=time.time() - self._start_time))
        self._check_cancel("waterfall")

        if cfg.use_vosk and ensemble_conf is None:
            self._emit(StageStarted(stage="vosk", message="Running Vosk (parallel)"))
            t0 = time.time()
            from ..engines.vosk_engine import VOSK_AVAILABLE, run_vosk_parallel
            model_path = cfg.vosk_model_path(language)
            if VOSK_AVAILABLE and model_path:
                vosk_words = run_vosk_parallel(cfg.audio_path, model_path)
                match_timestamps(aligned_words, vosk_words, "Vosk",
                                 human_words_norm)
                del vosk_words
                gc.collect()
            else:
                self._emit(StageMessage(stage="vosk",
                                        message="Vosk stage skipped (library or model missing).",
                                        level="warn"))
            self._emit(StageCompleted(stage="vosk",
                                       duration_seconds=time.time() - t0))
        self._check_cancel("vosk")

        if cfg.srt_path:
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

        if wx_pending:
            self._emit(StageStarted(stage="whisperx", message="Running WhisperX"))
            t0 = time.time()
            from ..engines.whisperx_engine import run_whisperx
            wx_words, wx_events = run_whisperx(cfg.audio_path, language,
                                               cfg.whisper_model, cfg.device,
                                               cfg.model_paths.whisper_models_dir)
            match_timestamps(aligned_words, wx_words, "WhisperX",
                             human_words_norm)
            del wx_words
            gc.collect()
            self._emit(StageCompleted(stage="whisperx",
                                       duration_seconds=time.time() - t0))
        self._check_cancel("whisperx")

        if cfg.use_mfa:
            self._emit(StageStarted(stage="mfa", message="Running MFA surgical mode"))
            t0 = time.time()
            make_surgical_mfa(aligned_words, cfg.audio_path, cfg.mfa_cmd)
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
        set_config(seg_cfg)
        set_layout(seg_cfg.max_cpl, seg_cfg.max_lines, seg_cfg.max_duration_ms)
        initial_segments = parse_human_transcript_to_srt_segments(
            original_text, aligned_words)
        segments = run_iterative_merging(initial_segments)
        segments = resolve_overlaps(segments)
        segments = enforce_min_duration(segments, seg_cfg.min_cue_ms)
        segments = resolve_overlaps(segments)
        self._emit(StageCompleted(stage="segmentation",
                                   duration_seconds=time.time() - t0))
        self._check_cancel("segmentation")

        # --- Smart punctuation / capitalization ---
        if cfg.punctuation:
            self._emit(StageStarted(stage="punctuation",
                                    message="Restoring punctuation & capitalization"))
            t0 = time.time()
            from ..qa.punctuation import create_restorer, restore_segments
            restorer = create_restorer(cfg)
            segments = restore_segments(segments, restorer)
            self._emit(StageCompleted(stage="punctuation",
                                       duration_seconds=time.time() - t0))
            self._check_cancel("punctuation")

        # --- Outputs ---
        base = cfg.resolve_output_base()

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
                validate_srt_output(sentence_entries, "Sentence-level SRT")

        if ensemble_conf is not None:
            from ..document import export_transcript
            export_transcript(base, os.path.basename(cfg.audio_path),
                              ensemble_conf,
                              self.profile.export.transcript_format,
                              self.profile.export.transcript_timestamps)

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

        output_files = [p for p in [result.word_level_srt_path,
                                    result.sentence_level_srt_path,
                                    result.transcript_txt_path,
                                    result.transcript_docx_path,
                                    result.audio_tags_srt_path,
                                    result.combined_srt_path] if p]

        self._emit(PipelineCompleted(
            word_count=len(aligned_words),
            segment_count=len(segments),
            output_files=output_files))

        return result

    def cancel(self) -> None:
        """Request cancellation. The pipeline stops at the next stage boundary."""
        self.ctx.cancel_requested = True

    def _run_ensemble(self, cfg: PipelineConfig, language: str):
        """Run ensemble transcription — extracted from cli.py._run_ensemble."""
        from ..ensemble import consensus_to_structured_text
        engine_words: Dict[str, List[Dict]] = {}
        wx_events: List[Dict] = []

        if "whisperx" in cfg.ensemble_engines:
            from ..engines.whisperx_engine import run_whisperx
            wx_words, wx_events = run_whisperx(cfg.audio_path, language,
                                               cfg.whisper_model, cfg.device,
                                               cfg.model_paths.whisper_models_dir)
            if wx_words:
                engine_words["whisperx"] = wx_words
        if "qwen" in cfg.ensemble_engines:
            from ..engines.qwen_engine import run_qwen
            qw = run_qwen(cfg.audio_path, language,
                          qwen_python=cfg.qwen_python,
                          models_dir=cfg.qwen_models_dir,
                          chunk_seconds=cfg.qwen_chunk_seconds,
                          asr_model=cfg.qwen_asr_model,
                          aligner_model=cfg.qwen_aligner_model)
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
                vk = run_vosk_parallel(cfg.audio_path, model_path)
                if vk:
                    engine_words["vosk"] = vk
            else:
                self._emit(StageMessage(stage="ensemble",
                                        message="Vosk voter skipped (library or model missing).",
                                        level="warn"))

        consensus = build_consensus(engine_words)
        return (consensus, consensus_to_structured_text(consensus), wx_events,
                engine_words.get("vosk", []))


class _NullSink:
    """Default sink that discards events."""
    def emit(self, event: PipelineEvent) -> None:
        pass
