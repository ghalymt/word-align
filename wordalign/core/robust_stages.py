"""The bad-audio stages of the pipeline, mixed into ``PipelineRunner``.

Stage numbers follow the design (docs/robust_flow.md):

1. analysis  -- VAD, sound events, SNR/bandwidth/clipping (audio.analysis)
2. cleanup   -- gated separation / partial denoising into ``asr_audio``
3. voting    -- chunk-based ASR voters fused with ROVER (transcript-free)
4. crosstalk -- diarization overlap regions; SE-DiCoW on overlapped chunks
5. timing    -- anchors, forced alignment on the original audio, fallbacks
7/8. triage  -- per-cue risk tiers written next to the subtitles

Every stage is optional and degrades to the previous behaviour when its
model or venv is missing, with a message saying what was skipped.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .events import StageCompleted, StageMessage, StageStarted

Chunk = Tuple[float, float]

VOTER_SOURCE_NAMES = {
    "qwen": "Qwen3-ASR", "whisperx": "WhisperX", "parakeet": "parakeet-ultra",
    "cohere": "Cohere", "vosk": "Vosk", "sedicow": "SE-DiCoW",
}


def _as_qwen_payload(out) -> Dict:
    """Normalise ``run_qwen`` output to the payload dict.

    ``run_qwen`` returns the worker payload with ``return_payload=True``, a
    plain word list from older callers and test doubles, or None/[] on
    failure. Failure becomes ``{}`` (never cached)."""
    if isinstance(out, dict):
        return out
    if isinstance(out, list) and out:
        return {"words": out}
    return {}


class RobustStagesMixin:
    """Methods shared by PipelineRunner for the robust flow."""

    # -- helpers -----------------------------------------------------------
    def _models_root(self) -> Optional[str]:
        from ..models.paths import _project_models_root
        root = _project_models_root()
        return str(root) if root.exists() else None

    def _hub_dir(self) -> Optional[str]:
        mp = self.config.model_paths
        hub = mp.huggingface_cache_dir or mp.qwen_models_dir
        if hub and os.path.isdir(hub):
            return hub
        root = self._models_root()
        if root and os.path.isdir(os.path.join(root, "huggingface", "hub")):
            return os.path.join(root, "huggingface", "hub")
        return None

    def _workdir(self) -> str:
        if not getattr(self, "_job_workdir", None):
            self._job_workdir = tempfile.mkdtemp(prefix="wordalign_job_")
        return self._job_workdir

    def _cleanup_workdir(self) -> None:
        wd = getattr(self, "_job_workdir", None)
        if wd:
            shutil.rmtree(wd, ignore_errors=True)
            self._job_workdir = None

    def _msg(self, stage: str, message: str, level: str = "info") -> None:
        self._emit(StageMessage(stage=stage, message=message, level=level))

    def _speech_fns(self):
        q = getattr(self, "_qmap", None)
        if q is None or not q.has_vad:
            return None, None
        return q.speech_mean, q.speech_max

    def _filter_engine_words(self, words: List[Dict], engine: str) -> List[Dict]:
        """Hallucination filter on one engine's timed words."""
        if not words or not self.config.hallucination_filter:
            return words
        from ..qa.hallucination import filter_hallucinations
        mean_fn, max_fn = self._speech_fns()
        kept, removed = filter_hallucinations(words, mean_fn, max_fn, engine=engine)
        if removed:
            self._removals.extend(removed)
            self._msg("hallucination",
                      f"{engine}: removed {len(removed)} unsupported word(s) "
                      f"({', '.join(sorted({r['reason'] for r in removed}))})")
        return kept

    # -- stage 1: analysis -----------------------------------------------------
    def _analysis_stage(self):
        cfg = self.config
        self._qmap = None
        self._removals: List[Dict] = []
        self._analysis_audio = None
        if not cfg.analysis:
            return None
        from ..audio.analysis import QualityMap, analyze_audio
        from ..audio.io import AudioDecodeError, load_audio
        self._emit(StageStarted(stage="analysis",
                                message="Measuring the audio (speech, noise, music)"))
        t0 = time.time()
        try:
            audio = load_audio(cfg.audio_path)
        except AudioDecodeError as exc:
            self._msg("analysis", f"Audio analysis skipped: {exc}", "warn")
            self._emit(StageCompleted(stage="analysis", duration_seconds=time.time() - t0))
            return None
        self._analysis_audio = audio
        settings = {"vad_threshold": cfg.vad_threshold, "sed": cfg.sound_events,
                    "version": 1}

        def compute():
            q = analyze_audio(cfg.audio_path, models_dir=self._models_root(),
                              use_sed=cfg.sound_events, vad_threshold=cfg.vad_threshold,
                              workdir=self._workdir(), cancel_event=self.ctx.cancel_event,
                              log=lambda m: self._msg("analysis", m), audio=audio)
            return q.to_cache()
        try:
            cached, hit = self._cached_engine("analysis", "silero+pretrainedsed",
                                              "*", settings, compute)
            qmap = QualityMap.from_cache(cached)
            if hit:
                self._msg("analysis", "Loaded the audio quality map from cache.")
        except InterruptedError:
            from .errors import PipelineCancelled
            raise PipelineCancelled("Cancelled during audio analysis")
        except Exception as exc:
            self._msg("analysis", f"Audio analysis failed: {exc}", "warn")
            self._emit(StageCompleted(stage="analysis", duration_seconds=time.time() - t0))
            return None
        for note in qmap.notes:
            self._msg("analysis", note, "warn" if "failed" in note else "info")
        summary = qmap.summary()
        conds = ", ".join(f"{k} {v:.0f}s" for k, v in summary["condition_seconds"].items())
        self._msg("analysis",
                  f"{summary['speech_seconds']:.0f}s speech in {summary['chunks']} chunks; "
                  f"median SNR {summary['median_snr_db']} dB"
                  + (f"; conditions: {conds}" if conds else ""))
        self._qmap = qmap
        self._emit(StageCompleted(stage="analysis", duration_seconds=time.time() - t0))
        return qmap

    def _chunks(self) -> Optional[List[Chunk]]:
        q = getattr(self, "_qmap", None)
        if q is None or not q.chunks:
            return None
        return [(c.start, c.end) for c in q.chunks]

    # -- stage 4a: early diarization (overlap map) ----------------------------
    def _diarize_early(self):
        """Run diarization before ASR when overlaps matter; returns turns."""
        cfg = self.config
        self._turns = None
        if not (cfg.diarize or cfg.overlap_asr or cfg.overlap_map):
            return None
        from ..diarize import DiarizationUnavailable, diarize_turns, overlap_regions
        self._emit(StageStarted(stage="diarization", message="Detecting speakers and overlaps"))
        t0 = time.time()
        try:
            with self._gpu_slot("diarize"):
                turns = diarize_turns(cfg.audio_path, num_speakers=cfg.num_speakers,
                                      cache_dir=self._hub_dir(),
                                      cancel_event=self.ctx.cancel_event)
        except DiarizationUnavailable as exc:
            self._msg("diarization", f"Speaker labels skipped: {exc}.", "warn")
            self._emit(StageCompleted(stage="diarization", duration_seconds=time.time() - t0))
            return None
        self._turns = turns
        overlaps = overlap_regions(turns)
        if self._qmap is not None:
            self._qmap.overlaps = overlaps
            for c in self._qmap.chunks:
                c.overlap = self._qmap.overlap_share(c.start, c.end)
        speakers = len({t[2] for t in turns})
        self._msg("diarization",
                  f"{speakers} speaker(s); {sum(e - s for s, e in overlaps):.1f}s of overlapped speech")
        self._emit(StageCompleted(stage="diarization", duration_seconds=time.time() - t0))
        return turns

    # -- stage 2: cleanup ------------------------------------------------------------
    def _cleanup_stage(self) -> str:
        cfg = self.config
        self._asr_audio = cfg.audio_path
        self._cleanup_signature = "none"
        q = getattr(self, "_qmap", None)
        if cfg.cleanup == "off" or q is None or getattr(self, "_analysis_audio", None) is None:
            return self._asr_audio
        from ..audio.prep import plan_cleanup, prepare_asr_audio
        plan = plan_cleanup(q, mode=cfg.cleanup)
        if not plan:
            self._msg("cleanup", "No chunk needs cleanup; ASR reads the original audio.")
            return self._asr_audio
        self._emit(StageStarted(stage="cleanup",
                                message=f"Cleaning {len(plan)} chunk(s) for ASR"))
        t0 = time.time()
        try:
            with self._gpu_slot("enhance"):
                path, applied = prepare_asr_audio(
                    self._analysis_audio, q, plan, self._workdir(),
                    models_dir=self._models_root(), denoise_mix=cfg.denoise_mix,
                    cancel_event=self.ctx.cancel_event,
                    log=lambda m: self._msg("cleanup", m))
        except Exception as exc:
            self._msg("cleanup", f"Cleanup failed; ASR reads the original audio ({exc})", "warn")
            self._emit(StageCompleted(stage="cleanup", duration_seconds=time.time() - t0))
            return self._asr_audio
        if applied:
            self._asr_audio = path
            sig = json.dumps(sorted((round(s, 2), round(e, 2), op) for s, e, op in applied))
            self._cleanup_signature = hashlib.sha256(sig.encode()).hexdigest()[:12]
            kinds: Dict[str, int] = {}
            for _, _, op in applied:
                kinds[op] = kinds.get(op, 0) + 1
            self._msg("cleanup", "Cleaned for ASR: " + ", ".join(
                f"{n} chunk(s) {op}" for op, n in sorted(kinds.items()))
                + "; timing still reads the original audio.")
        self._emit(StageCompleted(stage="cleanup", duration_seconds=time.time() - t0))
        return self._asr_audio

    # -- engine runs on chunks -----------------------------------------------------
    def _chunk_settings(self, chunks: Optional[Sequence[Chunk]]) -> Dict:
        key = json.dumps([[round(s, 2), round(e, 2)] for s, e in chunks]) if chunks else ""
        return {"chunks": hashlib.sha256(key.encode()).hexdigest()[:12],
                "asr_audio": getattr(self, "_cleanup_signature", "none")}

    def _run_qwen_chunks(self, language: str, chunks: Optional[Sequence[Chunk]]):
        cfg = self.config
        from ..engines.qwen_engine import run_qwen
        settings = {"chunk_seconds": cfg.qwen_chunk_seconds,
                    "asr_model": cfg.qwen_asr_model,
                    "aligner_model": cfg.qwen_aligner_model,
                    **self._chunk_settings(chunks), "payload": 2}
        def load():
            out = run_qwen(self._asr_audio, language, qwen_python=cfg.qwen_python,
                           models_dir=cfg.qwen_models_dir,
                           chunk_seconds=cfg.qwen_chunk_seconds,
                           asr_model=cfg.qwen_asr_model,
                           aligner_model=cfg.qwen_aligner_model,
                           cancel_event=self.ctx.cancel_event,
                           chunks=chunks, return_payload=True)
            return _as_qwen_payload(out)

        payload, hit = self._cached_engine("qwen", cfg.qwen_models_dir, language,
                                           settings, load)
        payload = _as_qwen_payload(payload)
        if hit:
            self._msg("qwen", "Loaded Qwen output from cache.")
        if payload and payload.get("failed"):
            self._msg("qwen", f"{len(payload['failed'])} piece(s) ran out of GPU "
                              "memory and were not transcribed", "warn")
        return payload or None

    def _run_chunk_voter(self, name: str, language: str, chunks: Sequence[Chunk]):
        from ..engines import voters
        ok, why = voters.voter_available(name, language)
        if not ok:
            self._msg("voting", f"{name} skipped: {why}")
            return None
        hub = self._hub_dir()

        def run():
            if name == "cohere":
                out = voters.run_cohere(self._asr_audio, chunks, language, hub,
                                        cancel_event=self.ctx.cancel_event,
                                        log=lambda m: self._msg("cohere", m, "warn"))
            else:
                out = voters.run_parakeet(self._asr_audio, chunks, hub,
                                          cancel_event=self.ctx.cancel_event,
                                          log=lambda m: self._msg("parakeet", m, "warn"))
            return {"chunks": {str(k): v for k, v in out.items()}} if out is not None else {}

        payload, hit = self._cached_engine(name, name, language,
                                           self._chunk_settings(chunks), run)
        if hit:
            self._msg(name, f"Loaded {name} output from cache.")
        if not payload:
            return None
        return {int(k): v for k, v in payload.get("chunks", {}).items()}

    # -- stage 3: voting -----------------------------------------------------------------
    def _voter_list(self) -> List[str]:
        cfg = self.config
        if cfg.voters:
            return [v for v in cfg.voters if v in VOTER_SOURCE_NAMES]
        out = []
        if cfg.use_whisperx:
            out.append("whisperx")
        if cfg.use_qwen:
            out.append("qwen")
        if cfg.use_vosk:
            out.append("vosk")
        return out

    def _run_voting(self, language: str):
        """Transcript-free mode: every voter on the same chunks, then ROVER.

        Returns ``(voted_words, wx_events, timed_sources, primary_engine)``.
        ``timed_sources`` maps a timing-source name to that engine's timed
        words for the timing stage.
        """
        from ..audio.analysis import fixed_chunks
        from ..engines.voters import bucket_timed_words, chunks_from_qwen_payload
        from ..rover import vote
        from ..utils import get_audio_duration
        cfg = self.config
        voters = self._voter_list()
        chunks = self._chunks()
        if chunks is None:
            duration = get_audio_duration(cfg.audio_path)
            chunks = fixed_chunks(duration) if duration > 0 else [(0.0, 1e9)]
            chunk_capable = duration > 0
        else:
            chunk_capable = True
        n = len(chunks)
        self._vote_chunks = list(chunks)
        per_engine: Dict[str, Dict[int, List[Dict]]] = {}
        timed_sources: Dict[str, List[Dict]] = {}
        wx_events: List[Dict] = []
        self._primary_attempted = set()

        for name in voters:
            self._check_cancel("voting")
            if name == "whisperx":
                self._primary_attempted.add("whisperx")
                from ..engines.whisperx_engine import run_whisperx
                wx_output, hit = self._cached_engine(
                    "whisperx", cfg.whisper_model, language,
                    {"device": cfg.device, "models_dir": cfg.model_paths.whisper_models_dir,
                     "asr_audio": getattr(self, "_cleanup_signature", "none")},
                    lambda: run_whisperx(self._asr_audio, language, cfg.whisper_model,
                                         cfg.device, cfg.model_paths.whisper_models_dir,
                                         cancel_event=self.ctx.cancel_event))
                if hit:
                    self._msg("whisperx", "Loaded WhisperX output from cache.")
                words, wx_events = wx_output
                words = self._filter_engine_words(words, "WhisperX")
                if words:
                    timed_sources["WhisperX"] = words
                    per_engine["whisperx"], orphans = bucket_timed_words(words, chunks)
                    self._orphans_note("WhisperX", orphans)
            elif name == "qwen":
                self._primary_attempted.add("qwen")
                use_chunks = chunk_capable and self._qmap is not None
                payload = self._run_qwen_chunks(language, chunks if use_chunks else None)
                if payload:
                    by_chunk = None
                    if use_chunks and len(payload.get("chunks") or []) == n:
                        by_chunk = self._refilter_chunks(
                            chunks_from_qwen_payload(payload, n), "Qwen3-ASR")
                        words = [w for i in sorted(by_chunk) for w in by_chunk[i]]
                    else:
                        words = self._filter_engine_words(payload.get("words", []),
                                                          "Qwen3-ASR")
                        by_chunk, _ = bucket_timed_words(words, chunks)
                    if words:
                        timed_sources["Qwen3-ASR"] = words
                    per_engine["qwen"] = by_chunk
            elif name == "vosk":
                words = self._run_vosk(language)
                if words:
                    timed_sources["Vosk"] = words
                    per_engine["vosk"], _ = bucket_timed_words(words, chunks)
            elif name in ("cohere", "parakeet"):
                if not (chunk_capable and self._qmap is not None):
                    self._msg("voting", f"{name} skipped: needs the audio analysis stage "
                                        "(VAD chunks)", "warn")
                    continue
                self._emit(StageStarted(stage=name, message=f"Running {name}"))
                t0 = time.time()
                by_chunk = self._run_chunk_voter(name, language, chunks)
                self._emit(StageCompleted(stage=name, duration_seconds=time.time() - t0))
                if by_chunk is None:
                    continue
                if name == "parakeet":
                    by_chunk = self._refilter_chunks(by_chunk, "parakeet-ultra")
                    timed_sources["parakeet-ultra"] = [w for i in sorted(by_chunk)
                                                       for w in by_chunk[i]]
                else:
                    by_chunk = self._filter_untimed_chunks(by_chunk, chunks, "Cohere")
                per_engine[name] = by_chunk

        ran = [e for e in per_engine if per_engine[e] is not None]
        if not ran:
            return [], wx_events, timed_sources, None
        voted: List[Dict] = []
        from ..rover import DEFAULT_PRIORS
        priors = {**DEFAULT_PRIORS, **(cfg.voter_priors or {})}
        text_voters = [e for e in ran if e != "vosk" or len(ran) <= 3]
        for i in range(n):
            hyps = {e: per_engine[e][i] for e in text_voters if i in per_engine[e]}
            if not hyps:
                continue
            chunk_words = vote(hyps, priors)
            for w in chunk_words:
                w["chunk"] = i
                w["text_conf"] = w.get("agreement", 1.0) if len(hyps) > 1 else 0.8
            voted.extend(chunk_words)
        self._msg("voting", f"ROVER over {', '.join(text_voters)}: {len(voted)} words; "
                            f"{sum(1 for w in voted if w.get('agreement', 1) < 0.6)} disputed")
        primary = "vote" if len(text_voters) > 1 else text_voters[0]
        return voted, wx_events, timed_sources, primary

    def _orphans_note(self, engine: str, orphans: List[Dict]) -> None:
        if orphans and self._qmap is not None and self._qmap.has_vad:
            self._msg("voting", f"{engine}: {len(orphans)} word(s) outside the VAD "
                                "speech chunks were not used for voting")

    def _refilter_chunks(self, by_chunk: Dict[int, List[Dict]], engine: str):
        return {i: self._filter_engine_words(ws, engine) if ws else ws
                for i, ws in by_chunk.items()}

    def _filter_untimed_chunks(self, by_chunk, chunks, engine):
        if not self.config.hallucination_filter:
            return by_chunk
        from ..qa.hallucination import filter_text_chunk
        out = {}
        for i, ws in by_chunk.items():
            text = " ".join(w["word"] for w in ws)
            sp = self._qmap.speech_mean(*chunks[i]) if self._qmap is not None else None
            new_text, reason = filter_text_chunk(text, sp)
            if reason:
                s, e = chunks[i]
                self._removals.append({"word": text, "start": s, "end": e,
                                       "reason": reason, "engine": engine})
            out[i] = [{"word": w} for w in new_text.split()]
        return out

    def _run_vosk(self, language: str) -> List[Dict]:
        cfg = self.config
        from ..engines.vosk_engine import VOSK_AVAILABLE, run_vosk_parallel
        model_path = cfg.vosk_model_path(language)
        if not (VOSK_AVAILABLE and model_path):
            self._msg("vosk", "Vosk skipped (library or model missing).", "warn")
            return []
        self._emit(StageStarted(stage="vosk", message="Running Vosk (parallel)"))
        t0 = time.time()
        words, hit = self._cached_engine(
            "vosk", model_path, language, {"workers": 4},
            lambda: run_vosk_parallel(cfg.audio_path, model_path,
                                      cancel_event=self.ctx.cancel_event))
        if hit:
            self._msg("vosk", "Loaded Vosk output from cache.")
        self._emit(StageCompleted(stage="vosk", duration_seconds=time.time() - t0))
        return self._filter_engine_words(words or [], "Vosk")

    def _carry_text_conf(self, aligned_words: List[Dict], voted: List[Dict]) -> None:
        """Copy each voted word's agreement onto the prepared word list."""
        from difflib import SequenceMatcher
        from ..utils import normalize_word
        a = [normalize_word(w["word"]) for w in aligned_words]
        b = [normalize_word(w["word"]) for w in voted]
        sm = SequenceMatcher(None, a, b, autojunk=False)
        for block in sm.get_matching_blocks():
            for k in range(block.size):
                src = voted[block.b + k]
                dst = aligned_words[block.a + k]
                for key in ("text_conf", "voters", "alternatives", "speaker", "chunk"):
                    if key in src:
                        dst[key] = src[key]

    # -- stage 4b: overlapped speech ---------------------------------------------
    def _overlap_asr(self, language: str, voted: List[Dict],
                     timed_sources: Dict[str, List[Dict]]) -> List[Dict]:
        """SE-DiCoW on chunks with overlapping speakers (transcript-free).

        The voted words stay; words SE-DiCoW gives the other speaker(s) in
        the overlapped stretch are inserted by time, tagged with their
        speaker and a modest text confidence, so triage flags the cue.
        """
        q = getattr(self, "_qmap", None)
        turns = getattr(self, "_turns", None)
        if q is None or not turns or not q.overlaps:
            return voted
        from ..engines.sedicow_engine import merge_overlap_words, run_sedicow
        targets = [i for i, c in enumerate(q.chunks) if c.overlap >= 0.15]
        if not targets:
            return voted
        self._emit(StageStarted(stage="sedicow",
                                message=f"SE-DiCoW on {len(targets)} overlapped chunk(s)"))
        t0 = time.time()
        windows = [(q.chunks[i].start, q.chunks[i].end) for i in targets]
        try:
            with self._gpu_slot("sedicow"):
                result = run_sedicow(self.config.audio_path, windows, turns,
                                     language=language, hub=self._hub_dir(),
                                     cancel_event=self.ctx.cancel_event,
                                     log=lambda m: self._msg("sedicow", m))
        except Exception as exc:
            self._msg("sedicow", f"SE-DiCoW skipped: {exc}", "warn")
            result = None
        self._emit(StageCompleted(stage="sedicow", duration_seconds=time.time() - t0))
        if not result:
            return voted
        merged, added = merge_overlap_words(voted, result, q.overlaps)
        if added:
            timed_sources.setdefault("SE-DiCoW", []).extend(
                w for w in merged if w.get("overlap_added"))
            self._msg("sedicow", f"Added {added} word(s) from overlapping speakers")
        return merged

    # -- reference mode: timed sources ---------------------------------------
    def _reference_sources(self, language: str) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
        """Timed engine outputs used as anchors when the text is given."""
        cfg = self.config
        sources: Dict[str, List[Dict]] = {}
        wx_events: List[Dict] = []
        if cfg.use_vosk and self._stage_enabled("vosk"):
            words = self._run_vosk(language)
            if words:
                sources["Vosk"] = words
        self._check_cancel("vosk")
        if cfg.srt_path and self._stage_enabled("rough_srt"):
            sources["SRT"] = self._rough_srt_words()
        if cfg.use_qwen and self._stage_enabled("qwen"):
            self._emit(StageStarted(stage="qwen", message="Running Qwen3-ASR (anchors)"))
            t0 = time.time()
            payload = self._run_qwen_chunks(language, self._chunks())
            if payload and payload.get("words"):
                sources["Qwen3-ASR"] = self._filter_engine_words(payload["words"], "Qwen3-ASR")
            self._emit(StageCompleted(stage="qwen", duration_seconds=time.time() - t0))
        self._check_cancel("qwen")
        if cfg.use_whisperx and self._stage_enabled("whisperx"):
            self._emit(StageStarted(stage="whisperx", message="Running WhisperX"))
            t0 = time.time()
            from ..engines.whisperx_engine import run_whisperx
            wx_output, hit = self._cached_engine(
                "whisperx", cfg.whisper_model, language,
                {"device": cfg.device, "models_dir": cfg.model_paths.whisper_models_dir},
                lambda: run_whisperx(cfg.audio_path, language, cfg.whisper_model,
                                     cfg.device, cfg.model_paths.whisper_models_dir,
                                     cancel_event=self.ctx.cancel_event))
            if hit:
                self._msg("whisperx", "Loaded WhisperX output from cache.")
            words, wx_events = wx_output
            words = self._filter_engine_words(words, "WhisperX")
            if words:
                sources["WhisperX"] = words
            self._emit(StageCompleted(stage="whisperx", duration_seconds=time.time() - t0))
        self._check_cancel("whisperx")
        return sources, wx_events

    def _rough_srt_words(self) -> List[Dict]:
        try:
            import srt as _srt
            with open(self.config.srt_path, "r", encoding="utf-8") as f:
                subs = list(_srt.parse(f.read()))
        except Exception as exc:
            self._msg("rough_srt", f"Error loading SRT: {exc}", "warn")
            return []
        out = []
        for s in subs:
            ws = s.content.split()
            dur = (s.end - s.start).total_seconds()
            per = dur / len(ws) if ws else 0
            for i, w in enumerate(ws):
                out.append({"word": w, "start": s.start.total_seconds() + i * per,
                            "end": s.start.total_seconds() + (i + 1) * per})
        return out

    # -- stage 5: timing -------------------------------------------------------------
    def _forced_aligner(self, language: str, audio_path: str):
        """A callable aligning windows on *audio_path*, or None."""
        cfg = self.config
        if not cfg.forced_align:
            return None, ""
        from ..engines.qwen_engine import ALIGNER_LANGUAGES, qwen_align
        lang = (language or "en").split("-")[0].lower()
        if cfg.use_qwen and lang in ALIGNER_LANGUAGES:
            def align(jobs):
                with self._gpu_slot("qwen"):
                    return qwen_align(audio_path, jobs, lang, qwen_python=cfg.qwen_python,
                                      models_dir=cfg.qwen_models_dir,
                                      aligner_model=cfg.qwen_aligner_model,
                                      cancel_event=self.ctx.cancel_event)
            return align, "Qwen3-FA"
        from ..engines.whisperx_engine import wav2vec2_available, wav2vec2_align
        if cfg.use_whisperx and wav2vec2_available():
            def align(jobs):
                with self._gpu_slot("whisperx"):
                    return wav2vec2_align(audio_path, jobs, lang, device=cfg.device,
                                          models_dir=cfg.model_paths.whisper_models_dir,
                                          cancel_event=self.ctx.cancel_event)
            return align, "wav2vec2"
        return None, ""

    def _timing_anchor(self, language: str, aligned_words: List[Dict],
                       sources: Dict[str, List[Dict]]) -> Dict:
        from ..timing import enforce_order, interpolate, run_anchor_alignment
        from ..utils import get_audio_duration
        cfg = self.config
        self._emit(StageStarted(stage="timing",
                                message="Anchors + forced alignment on the original audio"))
        t0 = time.time()
        q = getattr(self, "_qmap", None)
        duration = q.duration if q is not None else get_audio_duration(cfg.audio_path)
        first_speech = q.speech[0][0] if q is not None and q.speech else 0.0
        aligner, name = self._forced_aligner(language, cfg.audio_path)
        retry = None
        if aligner is not None and getattr(self, "_asr_audio", cfg.audio_path) != cfg.audio_path:
            retry, _ = self._forced_aligner(language, self._asr_audio)
        windows = None
        vote_chunks = getattr(self, "_vote_chunks", None)
        if (aligner is not None and not cfg.transcript_path and vote_chunks
                and any(w.get("chunk") is not None for w in aligned_words)):
            from ..timing import windows_from_chunks
            windows = windows_from_chunks(aligned_words, vote_chunks) or None
        stats = run_anchor_alignment(aligned_words, sources, duration or 0.0,
                                     aligner=aligner, retry_aligner=retry,
                                     aligner_name=name or "aligner",
                                     first_speech=first_speech, windows=windows,
                                     log=lambda m: self._msg("timing", m))
        # "Heard by": how many engines matched each word (reference cross-check).
        from ..timing import match_source
        from ..utils import normalize_word
        known = [normalize_word(w["word"]) for w in aligned_words]
        heard = [0] * len(aligned_words)
        for src_name, ws in sources.items():
            if src_name == "SRT" or not ws:
                continue
            m = match_source(known, ws, src_name, min_run=1)
            for i in list(m.runs) + list(m.short):
                heard[i] += 1
        if sources:
            for w, h in zip(aligned_words, heard):
                w["heard_by"] = h
        self._check_cancel("timing")
        self._emit(StageCompleted(stage="timing", duration_seconds=time.time() - t0))

        from ..engines.mfa_engine import mfa_supports_language
        if cfg.use_mfa and self._stage_enabled("mfa") and any(
                w.get("start") is None for w in aligned_words):
            if not mfa_supports_language(language):
                self._msg("mfa", f"MFA skipped: no pretrained MFA model for '{language}'.", "warn")
            else:
                self._emit(StageStarted(stage="mfa", message="Running MFA on the remaining gaps"))
                t1 = time.time()
                from ..align import make_surgical_mfa
                make_surgical_mfa(aligned_words, cfg.audio_path, cfg.mfa_cmd, language)
                for w in aligned_words:
                    if w.get("source") == "MFA" and "timing_conf" not in w:
                        w["timing_conf"] = 0.7
                self._emit(StageCompleted(stage="mfa", duration_seconds=time.time() - t1))
        self._check_cancel("mfa")
        self._emit(StageStarted(stage="interpolation", message="Interpolating remaining timestamps"))
        t2 = time.time()
        n_interp = interpolate(aligned_words, duration or None,
                               speech=q.speech if q is not None and q.has_vad else None)
        fixed = enforce_order(aligned_words)
        stats["interpolated"] = n_interp
        stats["reordered"] = fixed
        self._msg("interpolation", f"{n_interp} word(s) interpolated (flagged for review)"
                  + (f"; {fixed} overlapping word time(s) corrected" if fixed else ""))
        self._emit(StageCompleted(stage="interpolation", duration_seconds=time.time() - t2))
        return stats

    # -- stage 6: SDH tags from the sound-event model --------------------------
    def _write_sed_tags(self, base: str, original_text: str, human_tags, sed_events,
                        sentence_entries, result, proposed: bool) -> None:
        import srt as _srt
        from datetime import timedelta
        from ..tags import (combine_and_sort_srt, match_human_tags_to_detections,
                            optimize_tags, sed_detections, sed_proposals)
        if human_tags and not proposed:
            final = match_human_tags_to_detections(original_text, sed_detections(sed_events))
            note = f"{len(final)} of your [tags] placed from sound events"
        else:
            final = sed_proposals(sed_events)
            note = f"{len(final)} proposed tag(s) from sound events (review them)"
        self._msg("tags", note)
        if not final:
            return
        raw = [_srt.Subtitle(index=0, start=timedelta(seconds=e["start"]),
                             end=timedelta(seconds=e["end"]), content=f"[{e['tag']}]")
               for e in final]
        optimized = optimize_tags(raw, sentence_entries)
        for i, t in enumerate(optimized):
            t.index = i + 1
        path = f"{base}_audio_tags.srt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(_srt.compose(optimized))
        result.audio_tags_srt_path = path
        combined = combine_and_sort_srt(list(sentence_entries), raw)
        path2 = f"{base}_combined.srt"
        with open(path2, "w", encoding="utf-8") as f:
            f.write(_srt.compose(combined))
        result.combined_srt_path = path2

    # -- stage 7/8: triage -------------------------------------------------------------
    def _write_triage(self, base: str, segments: List[Dict], aligned_words: List[Dict],
                      mode: str) -> Optional[str]:
        if not self.config.triage or not segments:
            return None
        from ..qa.triage import review_summary, triage_segments, write_review
        from ..utils import time_to_ms
        segs = [{"start": time_to_ms(s["start"]) / 1000.0, "end": time_to_ms(s["end"]) / 1000.0,
                 "text": s.get("text", ""), "speaker": s.get("speaker")} for s in segments]
        q = getattr(self, "_qmap", None)
        cues = triage_segments(segs, aligned_words, mode,
                               conditions_at=q.conditions_at if q is not None else None,
                               removals=getattr(self, "_removals", []))
        meta = {"audio": os.path.basename(self.config.audio_path), "mode": mode,
                "analysis": q.summary() if q is not None else None,
                "removed_words": getattr(self, "_removals", [])[:500],
                "timing": getattr(self, "_timing_stats", None)}
        path = f"{base}_review.json"
        write_review(path, cues, meta)
        s = review_summary(cues)
        c = s["counts"]
        self._msg("triage", f"Review queue: {c['accept']} accept, {c['quick_check']} quick check, "
                            f"{c['redo']} redo ({s['share_to_review'] * 100:.0f}% of cue time to review)")
        self._review_cues = cues
        return path
