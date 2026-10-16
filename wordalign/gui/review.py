"""Review queue support for the GUI (stages 7-8 of the robust flow).

A job's output folder holds ``<name>_review.json`` (cue tiers, reasons,
word confidences). The GUI lists the cues that need a look, loops each
one's audio, and saves what the reviewer did:

* ``<name>_review_edits.json`` -- the current state of every touched cue
  (accepted as is, or edited); reloaded when the review is reopened.
* ``<name>_review_log.jsonl`` -- an append-only log of every save, with the
  text before and after, for later analysis.
* ``<name>_corrected.srt`` -- the final subtitles with the edits applied. The
  pipeline's own files are never changed. This is also the file
  ``wordalign-bench fit-triage`` learns from.

The review audio is a small Opus copy of the job's audio
(``review_audio.webm``) made when the job finishes, because GUI uploads are
deleted after the run.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_TEXT_CHARS = 2000
MAX_TIME_S = 48 * 3600.0
REVIEW_AUDIO_NAMES = ("review_audio.webm", "review_audio.m4a", "review_audio.wav")
AUDIO_TYPES = {".webm": "audio/webm", ".m4a": "audio/mp4", ".wav": "audio/wav"}


def outputs_root() -> Path:
    return Path.home() / "WordAlign" / "outputs"


def job_output_dir(job_id: str) -> Optional[Path]:
    """The job's output folder, or None for an invalid id."""
    if not job_id or not JOB_ID_RE.match(job_id):
        return None
    return outputs_root() / job_id


def find_review(folder: Path) -> Optional[Path]:
    if not folder or not folder.is_dir():
        return None
    found = sorted(folder.glob("*_review.json"))
    return found[0] if found else None


def _base(review_path: Path) -> Path:
    return review_path.with_name(review_path.name[:-len("_review.json")])


def find_review_audio(folder: Path) -> Optional[Path]:
    for name in REVIEW_AUDIO_NAMES:
        p = folder / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def make_review_audio(src: str, folder: Path, ffmpeg: Optional[str] = None) -> Optional[Path]:
    """Write a small, seekable mono copy of *src* for review playback."""
    if ffmpeg is None:
        from ..audio.io import ffmpeg_binary
        ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    attempts = [
        ("review_audio.webm", ["-c:a", "libopus", "-b:a", "32k"]),
        ("review_audio.m4a", ["-c:a", "aac", "-b:a", "64k"]),
        ("review_audio.wav", ["-c:a", "pcm_s16le", "-ar", "16000"]),
    ]
    for name, codec in attempts:
        out = folder / name
        tmp = folder / f".{name}.part{out.suffix}"
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-i", src, "-vn", "-ac", "1", *codec, str(tmp)]
        try:
            done = subprocess.run(cmd, capture_output=True, timeout=1800)
        except (OSError, subprocess.TimeoutExpired):
            done = None
        if done is not None and done.returncode == 0 and tmp.is_file() \
                and tmp.stat().st_size > 0:
            os.replace(tmp, out)
            return out
        try:
            tmp.unlink()
        except OSError:
            pass
    return None


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_review(folder: Path) -> Optional[Dict]:
    review_path = find_review(folder)
    if review_path is None:
        return None
    review = _read_json(review_path, None)
    if not isinstance(review, dict):
        return None
    base = _base(review_path)
    edits = _read_json(Path(f"{base}_review_edits.json"), {})
    audio = find_review_audio(folder)
    corrected = Path(f"{base}_corrected.srt")
    return {"review": review, "review_file": review_path.name,
            "edits": edits if isinstance(edits, dict) else {},
            "audio": audio.name if audio else None,
            "corrected_file": corrected.name if corrected.is_file() else None,
            "corrected_path": str(corrected) if corrected.is_file() else None}


def _clean_edit(raw, cue: Dict) -> Optional[Dict]:
    if not isinstance(raw, dict):
        return None
    status = raw.get("status")
    if status not in ("accepted", "edited"):
        return None
    out: Dict = {"status": status}
    if status == "edited":
        text = raw.get("text")
        if not isinstance(text, str):
            return None
        text = text.replace("\r\n", "\n").strip()[:MAX_TEXT_CHARS]
        out["text"] = text
        for key in ("start", "end"):
            val = raw.get(key)
            if val is None:
                continue
            try:
                val = round(float(val), 3)
            except (TypeError, ValueError, OverflowError):
                return None
            # NaN/inf pass a "< 0" test and break the SRT writer later.
            if not math.isfinite(val) or val < 0 or val > MAX_TIME_S:
                return None
            out[key] = val
        start = out.get("start", cue["start"])
        end = out.get("end", cue["end"])
        if end <= start:
            return None
    return out


def _source_srt(base: Path) -> Optional[Path]:
    # The review covers the final cues (speech and tags), so corrections
    # apply to the final SRT; older jobs only have the sentence level.
    for name in (f"{base.name}_final.srt", f"{base.name}_sentence_level.srt",
                 f"{base.name}.srt"):
        p = base.with_name(name)
        if p.is_file():
            return p
    return None


def _strip_speaker(content: str, speaker: Optional[str]) -> tuple:
    """(prefix, text) when the SRT names the speaker ("Speaker 2: ...")."""
    if speaker:
        flat = " ".join(content.split())
        prefix = f"{speaker}: "
        if flat.startswith(prefix):
            return prefix, content
    return "", content


def write_corrected(base: Path, cues: List[Dict], edits: Dict[str, Dict]) -> Path:
    """``<name>_corrected.srt``: the job's subtitles with *edits* applied."""
    import srt
    from datetime import timedelta
    src = _source_srt(base)
    entries = list(srt.parse(src.read_text(encoding="utf-8-sig"))) if src else []
    if len(entries) != len(cues):
        # No usable source file: rebuild from the review's own cues.
        entries = [srt.Subtitle(c["index"], timedelta(seconds=c["start"]),
                                timedelta(seconds=c["end"]), c.get("text", ""))
                   for c in cues]
    out = []
    for entry, cue in zip(entries, cues):
        edit = edits.get(str(cue["index"]))
        if edit and edit.get("status") == "edited":
            prefix, _ = _strip_speaker(entry.content, cue.get("speaker"))
            text = edit["text"]
            if prefix and not text.startswith(prefix.strip()):
                text = prefix + text
            entry = srt.Subtitle(entry.index,
                                 timedelta(seconds=edit.get("start", cue["start"])),
                                 timedelta(seconds=edit.get("end", cue["end"])), text)
        if entry.content.strip():
            out.append(entry)
    for k, e in enumerate(out, 1):
        e.index = k
    path = Path(f"{base}_corrected.srt")
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(srt.compose(out, reindex=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def save_edits(folder: Path, raw_edits) -> Dict:
    """Validate and store the reviewer's edits; returns the written files."""
    review_path = find_review(folder)
    if review_path is None:
        raise FileNotFoundError("this job has no review file")
    review = _read_json(review_path, None)
    if not isinstance(review, dict) or not isinstance(raw_edits, dict):
        raise ValueError("bad review data")
    cues = review.get("cues") or []
    by_index = {str(c["index"]): c for c in cues}
    base = _base(review_path)
    edits_path = Path(f"{base}_review_edits.json")
    current = _read_json(edits_path, {})
    current = current if isinstance(current, dict) else {}
    log_lines = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for key, raw in raw_edits.items():
        cue = by_index.get(str(key))
        if cue is None:
            continue
        if raw is None:                      # undo: back to unreviewed
            if str(key) in current:
                log_lines.append({"time": now, "cue": cue["index"], "action": "reset"})
                current.pop(str(key))
            continue
        edit = _clean_edit(raw, cue)
        if edit is None:
            raise ValueError(f"cue {key}: invalid edit")
        before = current.get(str(key))
        if before == edit:
            continue
        current[str(key)] = edit
        log_lines.append({"time": now, "cue": cue["index"], "tier": cue.get("tier_label"),
                          "risk": cue.get("risk"), "action": edit["status"],
                          "before": cue.get("text"), "after": edit.get("text", cue.get("text")),
                          "start": edit.get("start", cue["start"]),
                          "end": edit.get("end", cue["end"])})
    tmp = edits_path.with_name(edits_path.name + ".part")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, edits_path)
    if log_lines:
        with open(f"{base}_review_log.jsonl", "a", encoding="utf-8") as f:
            for line in log_lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    corrected = write_corrected(base, cues, current)
    return {"edits": current, "corrected_file": corrected.name,
            "corrected_path": str(corrected), "changed": len(log_lines)}
