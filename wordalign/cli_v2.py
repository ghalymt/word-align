"""Thin CLI — delegates to PipelineRunner with PrintSink.

This is the new CLI entry point that uses the v2.0 PipelineRunner instead
of duplicating the pipeline flow. The original cli.py is preserved for
backward compatibility; this module is used by __main__.py when available.

Usage:
    python -m wordalign audio.mp4 -t transcript.txt --vosk-models /path/to/models
    python -m wordalign audio.mp4 --engines whisperx,qwen,vosk -l en

The CLI translates PipelineEvent%s to terminal output via PrintSink,
producing the same user-visible output as the original CLI.
"""
from __future__ import annotations

import argparse
import os
import sys
import multiprocessing

from . import __version__
from .config import PipelineConfig
from .core.config import PipelineProfile
from .core.events import PrintSink
from .core.pipeline import PipelineRunner
from .models.paths import (
    _llama_cpp_dir,
    _default_llm_model_path,
    _default_llm_mtp_model_path,
)


def _load_profile(name):
    if not name:
        return None
    if os.path.isfile(name):
        return PipelineProfile.from_json(name)
    return PipelineProfile.preset(name)


def _parse_args(argv=None) -> PipelineConfig:
    p = argparse.ArgumentParser(
        prog="wordalign",
        description="Word-accurate SRT generation via multi-engine "
                    "alignment (Vosk -> WhisperX -> MFA) with an optional "
                    "transcript-free ensemble mode.")
    p.add_argument("audio", nargs="?", help="audio or video file")
    p.add_argument("-t", "--transcript",
                   help="verbatim human transcript (.txt). Omit to let the "
                        "ensemble build one from the audio.")
    p.add_argument("--allow-transcript-mismatch", action="store_true",
                   help="continue when the transcript word count is incompatible "
                        "with the media duration")
    p.add_argument("--srt", dest="srt_path",
                   help="optional rough SRT used as an extra timing source")
    p.add_argument("-l", "--language",
                   help="ISO 639-1 code; auto-detected when omitted")
    p.add_argument("-o", "--out", "--output-dir", dest="output_dir",
                   help="output directory "
                        "(default: next to the audio)")

    p.add_argument("--vosk-models",
                   help="directory of unpacked Vosk models "
                        "(or set WORDALIGN_VOSK_MODELS)")
    p.add_argument("--whisper-models",
                   help="directory with Whisper/faster-whisper models "
                        "(or set WORDALIGN_WHISPER_MODELS)")
    p.add_argument("--hf-cache",
                   help="HuggingFace cache directory "
                        "(or set WORDALIGN_HF_CACHE)")
    p.add_argument("--qwen-models",
                   help="HuggingFace cache dir with the Qwen models "
                        "(or set WORDALIGN_QWEN_MODELS)")
    p.add_argument("--no-vosk", action="store_true", default=None,
                   help="skip the Vosk timing pass")
    p.add_argument("--no-qwen", action="store_true", default=None,
                   help="skip the Qwen3-ASR timing pass")
    p.add_argument("--no-whisperx", action="store_true", default=None,
                   help="skip the WhisperX timing pass "
                        "(auto-skipped in ensemble mode since it ran as the primary)")
    p.add_argument("--use-mfa", dest="use_mfa", action="store_true",
                   default=None, help="enable the MFA surgical timing pass")
    p.add_argument("--no-mfa", dest="use_mfa", action="store_false",
                   help="skip the MFA surgical timing pass")
    p.add_argument("--legacy-ensemble", action="store_true",
                   help="use the legacy parallel-consensus ensemble (all "
                        "voters at once) instead of the v2 sequential flow")
    p.add_argument("--tags", action="store_true",
                   help="enable experimental audio-event tagging")
    p.add_argument("--engines", default=None,
                   help="comma-separated engines: whisperx,qwen,vosk,mfa; "
                        "also accepts parakeet,canary for legacy consensus. "
                        "Explicit --no-<engine> / --use-mfa / --no-mfa flags "
                        "take precedence")
    p.add_argument("--qwen-python",
                   help="python.exe of the venv holding qwen_asr "
                        "(or set WORDALIGN_QWEN_PYTHON)")
    p.add_argument("--mfa", "--mfa-cmd", dest="mfa_cmd",
                   help="path to the mfa executable or its conda env "
                        "(or set WORDALIGN_MFA)")
    p.add_argument("--mfa-models",
                   help="MFA acoustic model / dictionary directory "
                        "(or set WORDALIGN_MFA_MODELS)")
    p.add_argument("--qwen-chunk-seconds", type=float, default=60.0,
                   help="Qwen audio chunk length; lower it if the GPU OOMs")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--device", choices=["cuda", "cpu"],
                   help="force compute device (default: cuda when available)")
    p.add_argument("--yamnet-confidence", type=float, default=0.9,
                   metavar="0.0-1.0",
                   help="YAMNet detection threshold for --tags (default: 0.9)")
    p.add_argument("--doc", "--format", dest="doc",
                   choices=["none", "txt", "docx", "both"], default=None,
                   help="transcript document format in ensemble mode "
                        "(default: txt; docx highlights low-agreement words)")
    p.add_argument("--formats", default=None, metavar="srt,vtt,ass",
                   help="sentence-level subtitle formats to write, comma-"
                        "separated: srt (default), vtt (WebVTT), ass "
                        "(Advanced SubStation Alpha)")
    p.add_argument("--no-doc-timestamps", action="store_true",
                   help="omit [HH:MM:SS] paragraph timestamps in the "
                        "transcript document")
    p.add_argument("--cpl", "--max-cpl", dest="max_cpl", type=int, default=None,
                   help="max characters per subtitle line "
                        "(42 = regular video, 32 = vertical/social; default 42)")
    p.add_argument("--max-lines", type=int, default=None,
                   help="max lines per cue (default 2)")
    p.add_argument("--max-duration-ms", type=int, default=None,
                   help="max on-screen duration per cue, ms (default 7000)")
    p.add_argument("--min-cue-ms", type=int, default=None,
                   help="minimum on-screen duration per cue, ms (default 700)")
    p.add_argument("--profile", default=None,
                   help="pipeline profile name (fast, balanced, maximum_quality, cpu_only)")
    p.add_argument("--punctuation", action="store_true",
                   help="restore punctuation & capitalization on subtitle cues "
                        "(uses local LLM if configured, rules otherwise)")
    p.add_argument("--qa", action="store_true",
                   help="run deterministic transcript quality checks")
    p.add_argument("--llm-engine",
                   help="path to llama.cpp dir or llama-cli.exe "
                        "(default: <project>/models/llama.cpp, or set WORDALIGN_LLM_ENGINE)")
    p.add_argument("--llm-model",
                   help="path to the main GGUF model "
                        "(default: largest .gguf in <project>/models/llm/, or set WORDALIGN_LLM_MODEL)")
    p.add_argument("--llm-mtp-model",
                   help="path to the MTP draft GGUF (enables multi-token prediction) "
                        "(default: mtp-*.gguf in <project>/models/llm/, or set WORDALIGN_LLM_MTP_MODEL)")
    p.add_argument("--no-mtp", action="store_true",
                   help="disable multi-token prediction draft model")
    p.add_argument("--batch", metavar="FOLDER",
                   help="process every media file in FOLDER (transcripts are "
                        "paired by name: talk.mp4 + talk.txt/.srt); all other "
                        "flags apply to every file")
    p.add_argument("--recursive", action="store_true",
                   help="with --batch: include subfolders")
    p.add_argument("--skip-existing", action="store_true",
                   help="with --batch: skip files whose subtitles already exist")
    p.add_argument("--gui", action="store_true",
                   help="start the GUI backend server instead of running CLI")
    p.add_argument("--port", type=int, default=5575,
                   help="port for the GUI backend server (default 5575)")
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {__version__}")
    a = p.parse_args(argv)

    if a.gui:
        from .gui.server import run_server
        run_server(port=a.port)
        sys.exit(0)
    if not a.audio:
        p.error("audio is required unless --gui or --batch is used")

    from .models.paths import ModelPaths
    # Build the ModelPaths only with the values the user explicitly
    # supplied via CLI flags. Anything left None will fall through to the
    # field's default_factory, which honors the env var and then the
    # local <project>/models/ folder. Passing `None` directly here would
    # bypass the factory and silently disable auto-detect.
    mp_overrides = {}
    if a.vosk_models:
        mp_overrides["vosk_models_dir"] = a.vosk_models
    if a.whisper_models:
        mp_overrides["whisper_models_dir"] = a.whisper_models
    if a.qwen_models:
        mp_overrides["qwen_models_dir"] = a.qwen_models
    if a.hf_cache:
        mp_overrides["huggingface_cache_dir"] = a.hf_cache
    if a.mfa_models:
        mp_overrides["mfa_models_dir"] = a.mfa_models
    model_paths = ModelPaths(**mp_overrides) if mp_overrides else ModelPaths()
    # Force auto-detect fill so the local <project>/models/ subdirs are
    # visible even when both CLI and env-var are unset.
    model_paths._fill_auto()

    engines = tuple(e.strip() for e in (a.engines or "whisperx,qwen,vosk").split(",")
                    if e.strip())
    engine_overrides = {}
    # --engines is a bulk selection; the per-engine flags are more specific,
    # so they are applied afterwards and win (e.g. --engines ...,vosk
    # --no-vosk, or --engines whisperx,qwen,vosk --use-mfa).
    if a.engines is not None:
        selected = set(engines)
        engine_overrides.update({
            "vosk": "vosk" in selected,
            "qwen": "qwen" in selected or "qwen_asr" in selected,
            "whisperx": "whisperx" in selected,
            "mfa": "mfa" in selected,
        })
    if a.no_vosk is not None:
        engine_overrides["vosk"] = not a.no_vosk
    if a.no_qwen is not None:
        engine_overrides["qwen"] = not a.no_qwen
    if a.no_whisperx is not None:
        engine_overrides["whisperx"] = not a.no_whisperx
    if a.use_mfa is not None:
        engine_overrides["mfa"] = a.use_mfa

    formats = ("srt",)
    if a.formats is not None:
        formats = tuple(f.strip().lower() for f in a.formats.split(",")
                        if f.strip())
        unknown = sorted(set(formats) - {"srt", "vtt", "ass"})
        if unknown or not formats:
            p.error(f"--formats: unknown format(s) {', '.join(unknown)}; "
                    "choose from srt, vtt, ass")

    profile_overrides = {}
    if a.formats is not None:
        profile_overrides["formats"] = formats
    for name in ("max_cpl", "max_lines", "max_duration_ms", "min_cue_ms"):
        value = getattr(a, name)
        if value is not None:
            profile_overrides[name] = value
    if a.doc is not None:
        profile_overrides["doc_format"] = a.doc
    if a.no_doc_timestamps:
        profile_overrides["doc_timestamps"] = False
    if a.tags:
        profile_overrides["tags"] = True

    return PipelineConfig(
        audio_path=a.audio,
        transcript_path=a.transcript,
        allow_transcript_mismatch=a.allow_transcript_mismatch,
        srt_path=a.srt_path,
        language=a.language,
        output_dir=a.output_dir,
        model_paths=model_paths,
        mfa_cmd=a.mfa_cmd or os.environ.get("WORDALIGN_MFA"),
        use_vosk=a.no_vosk is None or not a.no_vosk,
        use_qwen=a.no_qwen is None or not a.no_qwen,
        use_whisperx=a.no_whisperx is None or not a.no_whisperx,
        use_mfa=a.use_mfa is None or a.use_mfa,
        use_legacy_ensemble=a.legacy_ensemble,
        use_tags=a.tags,
        yamnet_confidence=a.yamnet_confidence,
        whisper_model=a.whisper_model,
        device=a.device,
        ensemble_engines=engines,
        qwen_python=a.qwen_python or os.environ.get("WORDALIGN_QWEN_PYTHON"),
        qwen_chunk_seconds=a.qwen_chunk_seconds,
        doc_format=a.doc or "txt",
        doc_timestamps=not a.no_doc_timestamps,
        subtitle_formats=formats,
        max_cpl=a.max_cpl if a.max_cpl is not None else 42,
        max_lines=a.max_lines if a.max_lines is not None else 2,
        max_duration_ms=a.max_duration_ms if a.max_duration_ms is not None else 7000,
        min_cue_ms=a.min_cue_ms if a.min_cue_ms is not None else 700,
        punctuation=a.punctuation,
        qa=a.qa,
        llm_engine=a.llm_engine or _llama_cpp_dir(),
        llm_model=a.llm_model or _default_llm_model_path(),
        llm_mtp_model=a.llm_mtp_model or _default_llm_mtp_model_path(),
        llm_mtp=not a.no_mtp,
        profile_name=a.profile,
        engine_overrides=engine_overrides,
        profile_overrides=profile_overrides,
    )


def _harden_stdio() -> None:
    """Never crash on a character the console/pipe encoding lacks.

    Redirected to a file or pipe on Windows, stdout uses the ANSI code page
    (cp1252), which has no "→". Progress output printed one for every
    output file, so `wordalign a.mp4 > log.txt` wrote all its outputs and
    then died with UnicodeEncodeError (exit 1). Transcript text in other
    scripts hits the same wall.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass   # no stream (windowed build) or not reconfigurable


def main(argv=None) -> int:
    # When the Vosk parallel pool forks a worker inside a frozen
    # PyInstaller bundle, the worker process re-executes the entry
    # point script (this same ``main`` function) with the synthetic
    # argument ``--multiprocessing-fork pipe_handle=N``. ``freeze_support``
    # recognises that arg and runs the worker's target function instead
    # of re-entering ``main``.
    multiprocessing.freeze_support()
    _harden_stdio()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from .gui.server import run_server
        run_server(port=int(os.environ.get("WORDALIGN_PORT", "5575")))
        return 0
    if "--gui" not in args:
        from .batch import batch_requested, run_batch
        if batch_requested(args):
            print(f"WORD-ALIGN v{__version__}")
            return run_batch(args, _run_one, _parse_args)
    cfg = _parse_args(args)
    print(f"WORD-ALIGN v{__version__}")
    result = _run_one(cfg)
    return 0 if result.output_files else 1


def _run_one(cfg: PipelineConfig):
    """Run the pipeline for one file, printing progress and its outputs."""
    print(f"Audio: {cfg.audio_path}")
    if cfg.transcript_path:
        print(f"Transcript: {cfg.transcript_path}")
    print()

    sink = PrintSink()
    profile = _load_profile(cfg.profile_name)
    runner = PipelineRunner(cfg, profile=profile, sink=sink)
    result = runner.run()

    output_files = result.output_files
    if output_files:
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"  Words: {len(result.aligned_words)}")
        for p in output_files:
            if p:
                print(f"  → {p}")
        print("=" * 60)
    else:
        print(f"\n[error] {result.error or 'Pipeline produced no output.'}")
    return result


if __name__ == "__main__":
    sys.exit(main())
