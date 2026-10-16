"""Re-fit the triage weights on your own corrected jobs.

Each job leaves ``<name>_review.json`` beside its subtitles. After you
correct the subtitles, save the corrected file next to it as
``<name>_corrected.srt`` (or ``<name>.corrected.srt``). This module labels
every reviewed cue "needed a fix" when the corrected file changed its words,
moved its start by more than 150 ms or ended it more than 150 ms sooner (a
cue left on screen longer is a style choice, not a fix), then fits the logistic
weights (shrunk towards the defaults, so a few jobs cannot overfit) and
saves them to ``~/.wordalign/triage_weights.json``, which every later run
loads.

    wordalign-bench fit-triage --jobs D:/reviewed-jobs
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .triage import DEFAULT_WEIGHTS, fit_weights, save_weights

MOVE_TOLERANCE_S = 0.15


def _norm(text: str) -> List[str]:
    from ..utils import normalize_word
    return [t for t in (normalize_word(w) for w in text.split()) if t]


def _read_srt(path: Path) -> List[Dict]:
    import srt
    return [{"start": s.start.total_seconds(), "end": s.end.total_seconds(),
             "text": s.content} for s in srt.parse(path.read_text(encoding="utf-8-sig"))]


def corrected_for(review_path: Path) -> Optional[Path]:
    base = review_path.name[:-len("_review.json")]
    for name in (f"{base}_corrected.srt", f"{base}.corrected.srt",
                 f"{base}_sentence_level.corrected.srt"):
        p = review_path.with_name(name)
        if p.is_file():
            return p
    # A file you finished yourself, e.g. "<name>.txt_final_reviewed.srt".
    found = sorted(review_path.parent.glob(f"{base}*reviewed*.srt"))
    return found[0] if found else None


def label_cues(cues: Sequence[Dict], corrected: Sequence[Dict]) -> List[int]:
    """1 if the corrected file changed the cue's words or timing, else 0."""
    labels = []
    for c in cues:
        s0, s1 = float(c["start"]), float(c["end"])
        best, best_ov = None, 0.0
        for k in corrected:
            ov = min(s1, k["end"]) - max(s0, k["start"])
            if ov > best_ov:
                best, best_ov = k, ov
        if best is None:
            labels.append(1)            # the cue was deleted or moved away
            continue
        text = best["text"]
        speaker = c.get("speaker")
        if speaker and " ".join(text.split()).startswith(f"{speaker}:"):
            # The sentence-level SRT names the speaker; the review cue does not.
            text = " ".join(text.split())[len(speaker) + 1:]
        changed = _norm(c.get("text", "")) != _norm(text)
        moved = (abs(best["start"] - s0) > MOVE_TOLERANCE_S
                 or s1 - best["end"] > MOVE_TOLERANCE_S)
        labels.append(1 if (changed or moved) else 0)
    return labels


def collect_examples(folder: str) -> Tuple[List[Tuple[Dict[str, float], int]], List[str]]:
    examples: List[Tuple[Dict[str, float], int]] = []
    used: List[str] = []
    for review_path in sorted(Path(folder).rglob("*_review.json")):
        corrected_path = corrected_for(review_path)
        if corrected_path is None:
            continue
        review = json.loads(review_path.read_text(encoding="utf-8"))
        cues = review.get("cues", [])
        labels = label_cues(cues, _read_srt(corrected_path))
        for c, y in zip(cues, labels):
            examples.append(({k: float(v) for k, v in (c.get("features") or {}).items()}, y))
        used.append(review_path.name)
    return examples, used


def fit_from_jobs(folder: str, dry_run: bool = False) -> int:
    examples, used = collect_examples(folder)
    if not examples:
        print("No <name>_review.json with a <name>_corrected.srt beside it was found.")
        return 1
    positives = sum(y for _, y in examples)
    print(f"{len(used)} job(s), {len(examples)} cues, {positives} corrected.")
    if positives == 0 or positives == len(examples):
        print("Need both corrected and untouched cues to fit; nothing changed.")
        return 1
    weights = fit_weights(examples)
    for k in DEFAULT_WEIGHTS:
        print(f"  {k:<14} {DEFAULT_WEIGHTS[k]:7.3f} -> {weights.get(k, 0):7.3f}")
    if dry_run:
        return 0
    print(f"Saved: {save_weights(weights, len(examples))}")
    return 0
