"""Re-fit the triage weights on your own corrected jobs.

Each job leaves ``<name>_review.json`` beside its subtitles. After you
correct the subtitles, save the corrected file next to it as
``<name>_corrected.srt`` (or ``<name>.corrected.srt``). This module labels
every reviewed cue "needed a fix" when the corrected file changed its words,
moved its start by more than 150 ms or ended it more than 150 ms before its
last word was over (how long a cue stays up, or how the words are cut into
cues, is a style choice, not a fix), then fits the logistic
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
    """1 if the corrected file fixed the cue, else 0.

    A fix is a changed, removed or added word inside the cue, a start moved
    by more than ``MOVE_TOLERANCE_S``, or an end cut more than that before
    the cue's last word was over. The words of both files are aligned as one stream, so
    cutting the same words into different cues (a style choice) is not a
    fix, and neither is a cue left on screen longer.
    """
    from difflib import SequenceMatcher
    from ..utils import strip_tags
    speakers = {c.get("speaker") for c in cues if c.get("speaker")}

    def words_of(text: str) -> List[str]:
        flat = " ".join((text or "").split())
        for sp in speakers:
            if flat.startswith(f"{sp}:"):
                flat = flat[len(sp) + 1:]
                break
        return _norm(strip_tags(flat))

    ours = [(k, w) for k, c in enumerate(cues) for w in words_of(c.get("text", ""))]
    theirs = [(j, w) for j, c in enumerate(corrected) for w in words_of(c.get("text", ""))]
    labels = [0] * len(cues)
    if not theirs:
        return [1] * len(cues)
    sm = SequenceMatcher(None, [w for _, w in ours], [w for _, w in theirs], autojunk=False)
    mapped: Dict[int, int] = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for d in range(i2 - i1):
                mapped[i1 + d] = j1 + d
            continue
        for i in range(i1, i2):
            labels[ours[i][0]] = 1                     # word changed or removed
        if i1 == i2 and ours:                          # words added here
            labels[ours[min(i1, len(ours) - 1)][0]] = 1
    span: Dict[int, List[int]] = {}
    for i, (k, _) in enumerate(ours):
        span.setdefault(k, []).append(i)
    for k, idx in span.items():
        c = cues[k]
        j = mapped.get(idx[0])
        if j is not None and (j == 0 or theirs[j - 1][0] != theirs[j][0]):
            if abs(corrected[theirs[j][0]]["start"] - float(c["start"])) > MOVE_TOLERANCE_S:
                labels[k] = 1
        j = mapped.get(idx[-1])
        if j is not None and (j == len(theirs) - 1 or theirs[j + 1][0] != theirs[j][0]):
            # Only an end cut before the last word was over counts: how long
            # a cue stays up after its words is a style choice.
            spoken = [w.get("end") for w in c.get("words") or [] if w.get("end") is not None]
            last_end = float(spoken[-1]) if spoken else float(c["end"])
            if last_end - corrected[theirs[j][0]]["end"] > MOVE_TOLERANCE_S:
                labels[k] = 1
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
