"""Batch mode: run the pipeline over every media file in a folder.

    wordalign --batch FOLDER [--recursive] [--skip-existing] [any other flags]

Each media file is paired with a transcript of the same name when one
exists (``talk.mp4`` + ``talk.txt``, else ``talk.srt``); files without one
run in transcript-free ensemble mode. Every other flag applies to every
file. With ``-o`` and ``--recursive`` the subfolders are mirrored under the
output folder, so equally named files in different subfolders keep apart. One file failing does not stop the batch: a summary is printed, a
JSON report is written next to the outputs, and the exit code is non-zero
if any file failed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

MEDIA_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg",
}
REPORT_NAME = "wordalign_batch_report.json"


def find_media(folder: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in folder.glob(pattern)
                  if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)


def pair_transcript(media: Path) -> Optional[Path]:
    """``name.txt`` (preferred) or ``name.srt`` next to the media file."""
    for ext in (".txt", ".srt"):
        candidate = media.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


def _split_batch_args(argv: List[str]):
    """Pull the batch-only options out of *argv*; the rest is per-file."""
    folder, recursive, skip_existing, rest = None, False, False, []
    it = iter(range(len(argv)))
    for i in it:
        arg = argv[i]
        if arg == "--batch":
            if i + 1 >= len(argv):
                raise ValueError("--batch needs a folder")
            folder = argv[i + 1]
            next(it, None)
        elif arg.startswith("--batch="):
            folder = arg.split("=", 1)[1]
        elif arg == "--recursive":
            recursive = True
        elif arg == "--skip-existing":
            skip_existing = True
        else:
            rest.append(arg)
    return folder, recursive, skip_existing, rest


def batch_requested(argv: List[str]) -> bool:
    return any(a == "--batch" or a.startswith("--batch=") for a in argv)


def _already_done(cfg) -> bool:
    """True if this file's selected sentence-level outputs all exist."""
    from .core.pipeline import PipelineRunner
    export = PipelineRunner(cfg).profile.export
    base = cfg.resolve_output_base()
    wanted = [ext for ext, on in (("srt", export.sentence_srt),
                                  ("vtt", export.vtt), ("ass", export.ass)) if on]
    return bool(wanted) and all(
        Path(f"{base}_sentence_level.{ext}").exists() for ext in wanted)


def run_batch(argv: List[str], run_one: Callable, parse_args: Callable) -> int:
    """Run *run_one(cfg) -> PipelineResult* for every media file in the batch."""
    try:
        folder_arg, recursive, skip_existing, rest = _split_batch_args(argv)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2
    if any(a in ("-t", "--transcript") or a.startswith("--transcript=")
           for a in rest):
        print("[error] -t/--transcript cannot be used with --batch: "
              "transcripts are paired with media files by name.")
        return 2
    folder = Path(folder_arg)
    if not folder.is_dir():
        print(f"[error] --batch: not a folder: {folder}")
        return 2
    media = find_media(folder, recursive)
    if not media:
        print(f"[error] --batch: no media files in {folder}"
              f"{' (or its subfolders)' if recursive else ''}.")
        return 1

    print(f"Batch: {len(media)} media file(s) in {folder}\n")
    out_root = _output_dir(rest)
    claimed = {}                  # output base -> media file that owns it
    entries = []
    for n, path in enumerate(media, 1):
        transcript = pair_transcript(path)
        file_argv = [*rest, str(path)]
        if transcript:
            file_argv += ["-t", str(transcript)]
        out_dir = path.parent
        if out_root:
            out_dir = Path(out_root)
            if recursive:
                # Mirror the subfolder; argparse keeps the last -o given.
                out_dir = out_dir / path.parent.relative_to(folder)
                file_argv += ["-o", str(out_dir)]
        entry = {"media": str(path),
                 "transcript": str(transcript) if transcript else None}
        print(f"{'#' * 60}\n[{n}/{len(media)}] {path.name}"
              f"{' + ' + transcript.name if transcript else ' (ensemble mode)'}")
        started = time.time()
        base = os.path.normcase(os.path.abspath(out_dir / path.stem))
        if base in claimed:
            # talk.mp4 next to talk.wav: both would write talk_*.srt.
            entry.update(status="failed", outputs=[], seconds=0.0,
                         error=f"same output name as {claimed[base].name}; "
                               "its subtitles would be overwritten "
                               "(rename one of the two files)")
            entries.append(entry)
            print(f"  FAILED: {entry['error']}\n")
            continue
        claimed[base] = path
        try:
            cfg = parse_args(file_argv)
            if skip_existing and _already_done(cfg):
                print("  skipped: outputs already exist (--skip-existing)\n")
                entry.update(status="skipped", outputs=[])
                entries.append(entry)
                continue
            result = run_one(cfg)
            outputs = list(result.output_files)
            status = ("cancelled" if result.cancelled
                      else "ok" if outputs else "failed")
            entry.update(status=status, outputs=outputs,
                         error=None if status == "ok"
                         else (result.error or "no output produced"),
                         warnings=list(result.warnings))
        except SystemExit as exc:           # argparse rejected the flags
            entry.update(status="failed", outputs=[],
                         error=f"invalid arguments (exit {exc.code})")
        except Exception as exc:            # keep going with the next file
            entry.update(status="failed", outputs=[], error=f"{type(exc).__name__}: {exc}")
        entry["seconds"] = round(time.time() - started, 1)
        entries.append(entry)
        if entry["status"] == "failed":
            print(f"  FAILED: {entry['error']}\n")

    counts = {s: sum(1 for e in entries if e["status"] == s)
              for s in ("ok", "skipped", "failed", "cancelled")}
    report_dir = Path(_output_dir(rest) or folder)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / REPORT_NAME
    report_path.write_text(json.dumps({
        "folder": str(folder), "recursive": recursive,
        "counts": counts, "files": entries}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print(f"{'=' * 60}\nBATCH COMPLETE: {counts['ok']} ok, "
          f"{counts['skipped']} skipped, {counts['failed']} failed"
          f"{', ' + str(counts['cancelled']) + ' cancelled' if counts['cancelled'] else ''}")
    for e in entries:
        if e["status"] in ("failed", "cancelled"):
            print(f"  {e['status'].upper()}: {Path(e['media']).name}: {e['error']}")
    print(f"  Report: {report_path}\n{'=' * 60}")
    return 1 if counts["failed"] or counts["cancelled"] else 0


def _output_dir(argv: List[str]) -> Optional[str]:
    for i, arg in enumerate(argv):
        for flag in ("-o", "--out", "--output-dir"):
            if arg == flag and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
    return None
