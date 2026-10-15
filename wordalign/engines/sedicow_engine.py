"""SE-DiCoW for overlapped speech (transcript-free mode).

Overlap is about 30% of NOTSOFAR-1 meeting audio but up to 90% of its
errors, and modular diarize-then-transcribe systems stay robust as the
speaker count grows while single-stream ASR does not. On the chunks where
diarization finds two or more speakers at once, SE-DiCoW transcribes each
speaker separately; the words it gives the *other* speaker, which the
single-stream voters merged away or dropped, are inserted into the voted
transcript with that speaker's label and a modest text confidence (so the
review queue still flags the cue).
"""
from __future__ import annotations

import json
import os
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..backends import backend_python
from ..utils import normalize_word
from ._worker import WorkerCancelled, WorkerError, run_worker

OVERLAP_TEXT_CONF = 0.55
_WORKER = Path(__file__).with_name("_sedicow_worker.py")


def sedicow_model_present(hub: Optional[str]) -> bool:
    return bool(hub) and os.path.isdir(os.path.join(hub, "models--BUT-FIT--SE-DiCoW"))


def run_sedicow(audio_path: str, windows: Sequence[Tuple[float, float]],
                turns: Sequence[Tuple[float, float, str]], language: str = "en",
                hub: Optional[str] = None, cancel_event=None,
                log=print) -> Optional[List[Dict]]:
    python = backend_python("sedicow")
    if not python:
        raise RuntimeError("no venv for SE-DiCoW (it runs in the WhisperX venv)")
    if not sedicow_model_present(hub):
        raise RuntimeError("SE-DiCoW model not downloaded "
                           "(scripts/install_robust.ps1 -Components sedicow)")
    jobs = []
    for k, (s, e) in enumerate(windows):
        wt = [[round(ts, 3), round(te, 3), lbl] for ts, te, lbl in turns
              if te > s and ts < e]
        if len({t[2] for t in wt}) >= 2:
            jobs.append({"id": k, "start": s, "end": e, "turns": wt})
    if not jobs:
        return []
    fd, path = tempfile.mkstemp(prefix="wa_sedicow_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(jobs, f)
        args = ["--audio", audio_path, "--windows-json", path,
                "--language", (language or "en").split("-")[0]]
        if hub:
            args += ["--cache-dir", hub]
        payload = run_worker(python, _WORKER, args, "__WA_SEDICOW_JSON__",
                             cancel_event=cancel_event, name="SE-DiCoW")
    except WorkerCancelled:
        raise
    except WorkerError as exc:
        log(f"SE-DiCoW failed: {exc}")
        return None
    finally:
        os.remove(path)
    return payload.get("windows", [])


def _segment_words(seg: Dict) -> List[Dict]:
    toks = seg["text"].split()
    if not toks:
        return []
    span = max(0.05, float(seg["end"]) - float(seg["start"]))
    step = span / len(toks)
    return [{"word": t, "start": round(seg["start"] + i * step, 3),
             "end": round(seg["start"] + (i + 1) * step, 3)} for i, t in enumerate(toks)]


def _provisional_times(words: List[Dict]) -> List[float]:
    """A start time for every word (voted words may lack one)."""
    times: List[Optional[float]] = [w.get("start") for w in words]
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        return [float(i) for i in range(len(words))]
    out: List[float] = []
    for i, t in enumerate(times):
        if t is not None:
            out.append(float(t))
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        if prev is not None and nxt is not None:
            frac = (i - prev) / (nxt - prev)
            out.append(float(times[prev]) + frac * (float(times[nxt]) - float(times[prev])))
        elif prev is not None:
            out.append(float(times[prev]) + 0.3 * (i - prev))
        else:
            out.append(float(times[nxt]) - 0.3 * (nxt - i))
    return out


def merge_overlap_words(voted: List[Dict], windows: Sequence[Dict],
                        overlaps: Sequence[Tuple[float, float]]) -> Tuple[List[Dict], int]:
    """Insert the non-dominant speakers' overlapped words; returns (words, added)."""
    from ..diarize import speaker_names
    if not windows:
        return voted, 0
    times = _provisional_times(voted)
    all_turns = [(0.0, 0.0, sp["label"]) for w in windows for sp in w.get("speakers", [])]
    names = speaker_names(all_turns)
    inserts: List[Dict] = []
    for win in windows:
        s, e = float(win["start"]), float(win["end"])
        idx = [i for i, t in enumerate(times) if s - 0.2 <= t <= e + 0.2]
        voted_norm = [normalize_word(voted[i]["word"]) for i in idx]
        speakers = []
        for sp in win.get("speakers", []):
            words = [w for seg in sp.get("segments", []) for w in _segment_words(seg)]
            norm = [normalize_word(w["word"]) for w in words]
            sm = SequenceMatcher(None, voted_norm, norm, autojunk=False)
            matched = sum(b.size for b in sm.get_matching_blocks())
            speakers.append((matched, sp["label"], words, sm))
        if len(speakers) < 2:
            continue
        speakers.sort(key=lambda x: -x[0])
        for matched, label, words, sm in speakers[1:]:
            covered = set()
            for b in sm.get_matching_blocks():
                covered.update(range(b.b, b.b + b.size))
            for k, w in enumerate(words):
                if k in covered:
                    continue
                mid = (w["start"] + w["end"]) / 2
                if not any(os_ - 0.5 <= mid <= oe + 0.5 for os_, oe in overlaps):
                    continue
                inserts.append(dict(w, speaker=names.get(label, label),
                                    text_conf=OVERLAP_TEXT_CONF, overlap_added=True,
                                    agreement=OVERLAP_TEXT_CONF))
    if not inserts:
        return voted, 0
    inserts.sort(key=lambda w: w["start"])
    merged: List[Dict] = []
    j = 0
    for i, w in enumerate(voted):
        while j < len(inserts) and inserts[j]["start"] < times[i]:
            merged.append(inserts[j])
            j += 1
        merged.append(w)
    merged.extend(inserts[j:])
    return merged, len(inserts)
