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
from typing import Optional

from . import __version__
from .config import PipelineConfig
from .core.events import PrintSink
from .core.pipeline import PipelineRunner


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
    p.add_argument("--engines", default="whisperx,qwen,vosk",
                   help="comma list of ensemble voters "
                        "(whisperx,qwen,vosk; legacy: parakeet,canary)")
    p.add_argument("--qwen-python",
                   help="python.exe of the venv holding qwen_asr "
                        "(or set WORDALIGN_QWEN_PYTHON)")
    p.add_argument("--qwen-models",
                   help="HuggingFace cache dir with the Qwen models "
                        "(or set WORDALIGN_QWEN_MODELS)")
    p.add_argument("--qwen-chunk-seconds", type=float, default=60.0,
                   help="Qwen audio chunk length; lower it if the GPU OOMs")
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
    p.add_argument("--max-cpl", type=int, default=42,
                   help="max characters per subtitle line "
                        "(42 = regular video, 32 = vertical/social; default 42)")
    p.add_argument("--max-lines", type=int, default=2,
                   help="max lines per cue (default 2)")
    p.add_argument("--max-duration-ms", type=int, default=7000,
                   help="max on-screen duration per cue, ms (default 7000)")
    p.add_argument("--min-cue-ms", type=int, default=700,
                   help="minimum on-screen duration per cue, ms (default 700)")
    p.add_argument("--profile", default=None,
                   help="pipeline profile name (fast, balanced, maximum_quality, cpu_only)")
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
        qwen_python=a.qwen_python or os.environ.get("WORDALIGN_QWEN_PYTHON"),
        qwen_models_dir=a.qwen_models or os.environ.get("WORDALIGN_QWEN_MODELS"),
        qwen_chunk_seconds=a.qwen_chunk_seconds,
        doc_format=a.doc,
        doc_timestamps=not a.no_doc_timestamps,
        max_cpl=a.max_cpl,
        max_lines=a.max_lines,
        max_duration_ms=a.max_duration_ms,
        min_cue_ms=a.min_cue_ms,
    )


def main(argv=None) -> int:
    cfg = _parse_args(argv)
    print(f"WORD-ALIGN v{__version__}")
    print(f"Audio: {cfg.audio_path}")
    if cfg.transcript_path:
        print(f"Transcript: {cfg.transcript_path}")
    print()

    sink = PrintSink()
    runner = PipelineRunner(cfg, sink=sink)
    result = runner.run()

    if result.word_level_srt_path or result.sentence_level_srt_path:
        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"  Words: {len(result.aligned_words)}")
        for p in [result.word_level_srt_path, result.sentence_level_srt_path,
                  result.transcript_txt_path, result.transcript_docx_path,
                  result.audio_tags_srt_path, result.combined_srt_path]:
            if p:
                print(f"  → {p}")
        print("=" * 60)
        return 0
    else:
        print("\n[error] Pipeline produced no output.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
