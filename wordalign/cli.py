"""Command-line entry point.

Two modes:
    reference mode   audio + human transcript  -> word/sentence SRT
    ensemble mode    audio only                -> consensus transcript
                                                  + word/sentence SRT
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from datetime import timedelta
from typing import Dict, List

import srt

from . import __version__
from .align import (interpolate_timestamps, make_surgical_mfa,
                    match_timestamps, print_alignment_statistics)
from .config import PipelineConfig
from .ensemble import build_consensus, consensus_to_text
from .segment import (parse_human_transcript_to_srt_segments,
                      resolve_overlaps, run_iterative_merging,
                      validate_srt_output)
from .utils import (detect_language, extract_tags_from_transcript,
                    normalize_word, strip_tags, time_to_ms)


def _parse_args(argv=None) -> PipelineConfig:
    p = argparse.ArgumentParser(
        prog="wordalign",
        description="Word-accurate SRT generation via multi-engine "
                    "alignment (Vosk -> WhisperX -> MFA) with an optional "
                    "transcript-free ensemble mode.")
    p.add_argument("audio", help="audio or video file")
    p.add_argument("-t", "--transcript",
                   help="verbatim human transcript (.txt). Omit to let the "
                        "ensemble build one from the audio.")
    p.add_argument("--srt", dest="srt_path",
                   help="optional rough SRT used as an extra timing source")
    p.add_argument("-l", "--language",
                   help="ISO 639-1 code; auto-detected when omitted")
    p.add_argument("-o", "--output-dir", help="output directory "
                                              "(default: next to the audio)")
    p.add_argument("--vosk-models",
                   help="directory of unpacked Vosk models "
                        "(or set WORDALIGN_VOSK_MODELS)")
    p.add_argument("--mfa", dest="mfa_cmd",
                   help="path to the mfa executable or its conda env "
                        "(or set WORDALIGN_MFA)")
    p.add_argument("--no-vosk", action="store_true")
    p.add_argument("--no-mfa", action="store_true")
    p.add_argument("--tags", action="store_true",
                   help="enable experimental audio-event tagging")
    p.add_argument("--engines", default="whisperx,parakeet,vosk",
                   help="comma list for ensemble mode "
                        "(whisperx,parakeet,canary,vosk)")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--device", choices=["cuda", "cpu"],
                   help="force compute device (default: cuda when available)")
    p.add_argument("--yamnet-confidence", type=float, default=0.9,
                   metavar="0.0-1.0",
                   help="YAMNet detection threshold for --tags (default: 0.9)")
    p.add_argument("--doc", choices=["none", "txt", "docx", "both"],
                   default="txt",
                   help="transcript document format in ensemble mode "
                        "(default: txt; docx highlights low-agreement words)")
    p.add_argument("--no-doc-timestamps", action="store_true",
                   help="omit [HH:MM:SS] paragraph timestamps in the "
                        "transcript document")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    a = p.parse_args(argv)
    return PipelineConfig(
        audio_path=a.audio,
        transcript_path=a.transcript,
        srt_path=a.srt_path,
        language=a.language,
        output_dir=a.output_dir,
        vosk_models_dir=a.vosk_models or os.environ.get("WORDALIGN_VOSK_MODELS"),
        mfa_cmd=a.mfa_cmd or os.environ.get("WORDALIGN_MFA"),
        use_vosk=not a.no_vosk,
        use_mfa=not a.no_mfa,
        use_tags=a.tags,
        yamnet_confidence=a.yamnet_confidence,
        whisper_model=a.whisper_model,
        device=a.device,
        ensemble_engines=tuple(e.strip() for e in a.engines.split(",") if e.strip()),
        doc_format=a.doc,
        doc_timestamps=not a.no_doc_timestamps,
    )


def _run_ensemble(cfg: PipelineConfig, language: str):
    """Return (consensus_words, transcript_text, whisperx_events)."""
    from .engines.whisperx_engine import run_whisperx
    engine_words: Dict[str, List[Dict]] = {}
    wx_events: List[Dict] = []

    if "whisperx" in cfg.ensemble_engines:
        wx_words, wx_events = run_whisperx(cfg.audio_path, language,
                                           cfg.whisper_model, cfg.device)
        if wx_words:
            engine_words["whisperx"] = wx_words
    if "parakeet" in cfg.ensemble_engines:
        from .engines.nemo_engine import run_parakeet
        pk = run_parakeet(cfg.audio_path, language)
        if pk:
            engine_words["parakeet"] = pk
    if "canary" in cfg.ensemble_engines:
        from .engines.nemo_engine import run_canary_qwen
        cq = run_canary_qwen(cfg.audio_path, language)
        if cq:
            engine_words["canary"] = cq
    if "vosk" in cfg.ensemble_engines and cfg.use_vosk:
        from .engines.vosk_engine import VOSK_AVAILABLE, run_vosk_parallel
        model_path = cfg.vosk_model_path(language)
        if VOSK_AVAILABLE and model_path:
            vk = run_vosk_parallel(cfg.audio_path, model_path)
            if vk:
                engine_words["vosk"] = vk
        else:
            print("[info] Vosk voter skipped (library or model missing).")

    consensus = build_consensus(engine_words)
    return consensus, consensus_to_text(consensus), wx_events


def main(argv=None) -> int:
    cfg = _parse_args(argv)
    print("=" * 60 + f"\nWORD-ALIGN v{__version__}\n" + "=" * 60)
    if not os.path.exists(cfg.audio_path):
        print(f"[error] Audio not found: {cfg.audio_path}")
        return 1

    # ---------------------------------------------------------------- input
    ensemble_conf = None
    if cfg.transcript_path:
        try:
            with open(cfg.transcript_path, "r", encoding="utf-8") as f:
                original_text = f.read()
        except OSError as exc:
            print(f"[error] Failed to read transcript: {exc}")
            return 1
        language = cfg.language or detect_language(original_text)
        wx_pending = True   # WhisperX still to run inside the waterfall
    else:
        print("[mode] No transcript given -> ensemble transcription mode")
        language = cfg.language or "en"
        if not cfg.language:
            print("[info] No --language given; assuming 'en' for ensemble "
                  "voter selection.")
        consensus, original_text, wx_events_cache = _run_ensemble(cfg, language)
        ensemble_conf = consensus
        wx_pending = False  # WhisperX already ran as the backbone

    human_tags = extract_tags_from_transcript(original_text)
    text_no_tags = strip_tags(original_text)
    human_words = text_no_tags.strip().split()
    human_words_norm = [normalize_word(w) for w in human_words]
    aligned_words: List[Dict] = [{"word": w} for w in human_words]
    print(f"[ok] Prepared {len(aligned_words)} words for alignment")

    # ------------------------------------------------------------ waterfall
    wx_events: List[Dict] = []
    if ensemble_conf is not None:
        # Consensus already carries backbone timestamps: seed them directly.
        match_timestamps(aligned_words, ensemble_conf, "Ensemble",
                         human_words_norm)
        wx_events = wx_events_cache
    if cfg.use_vosk and ensemble_conf is None:
        from .engines.vosk_engine import VOSK_AVAILABLE, run_vosk_parallel
        model_path = cfg.vosk_model_path(language)
        if VOSK_AVAILABLE and model_path:
            vosk_words = run_vosk_parallel(cfg.audio_path, model_path)
            match_timestamps(aligned_words, vosk_words, "Vosk",
                             human_words_norm)
            del vosk_words
            gc.collect()
        else:
            print("[info] Vosk stage skipped (library or model missing).")
    if cfg.srt_path:
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
            print(f"[warn] Error loading SRT: {exc}")
    if wx_pending:
        from .engines.whisperx_engine import run_whisperx
        wx_words, wx_events = run_whisperx(cfg.audio_path, language,
                                           cfg.whisper_model, cfg.device)
        match_timestamps(aligned_words, wx_words, "WhisperX",
                         human_words_norm)
        del wx_words
        gc.collect()
    if cfg.use_mfa:
        make_surgical_mfa(aligned_words, cfg.audio_path, cfg.mfa_cmd)
    interpolate_timestamps(aligned_words)
    print_alignment_statistics(aligned_words)

    # ---------------------------------------------------------- segmenting
    initial_segments = parse_human_transcript_to_srt_segments(
        original_text, aligned_words)
    segments = run_iterative_merging(initial_segments)
    segments = resolve_overlaps(segments)

    # ------------------------------------------------------------- outputs
    base = cfg.resolve_output_base()

    word_entries = [
        srt.Subtitle(i + 1, timedelta(seconds=w["start"]),
                     timedelta(seconds=w["end"]), w["word"])
        for i, w in enumerate(aligned_words) if w.get("start") is not None]
    with open(f"{base}_word_level.srt", "w", encoding="utf-8") as f:
        f.write(srt.compose(word_entries))
    print(f"\n[ok] Word-level SRT: {len(word_entries)} entries")

    sentence_entries = [
        srt.Subtitle(i + 1,
                     timedelta(seconds=time_to_ms(s["start"]) / 1000),
                     timedelta(seconds=time_to_ms(s["end"]) / 1000),
                     s["text"])
        for i, s in enumerate(segments)]
    with open(f"{base}_sentence_level.srt", "w", encoding="utf-8") as f:
        f.write(srt.compose(sentence_entries))
    print(f"[ok] Sentence-level SRT: {len(sentence_entries)} entries")
    validate_srt_output(sentence_entries, "Sentence-level SRT")

    if ensemble_conf is not None:
        from .document import export_transcript
        export_transcript(base, os.path.basename(cfg.audio_path),
                          ensemble_conf, cfg.doc_format, cfg.doc_timestamps)

    if cfg.use_tags and human_tags:
        from .tags import (combine_and_sort_srt, detect_audio_tags_yamnet,
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
            with open(f"{base}_audio_tags.srt", "w", encoding="utf-8") as f:
                f.write(srt.compose(optimized))
            print(f"[ok] Audio-tags SRT: {len(optimized)} merged entries")
            combined = combine_and_sort_srt(sentence_entries, raw_tag_entries)
            with open(f"{base}_combined.srt", "w", encoding="utf-8") as f:
                f.write(srt.compose(combined))
            print(f"[ok] Combined SRT: {len(combined)} entries")
            validate_srt_output(combined, "Combined SRT")

    print("\n" + "=" * 60 + "\n[ok] PROCESSING COMPLETE\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
