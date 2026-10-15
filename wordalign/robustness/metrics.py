"""Benchmark metrics.

* WER with a light normalizer in the spirit of the Open ASR Leaderboard's
  (lowercase, no punctuation, a few contractions and spelled numbers);
* word-time shift: mean absolute start error in ms and the share of words
  within 50 / 100 ms, over words matched to the truth;
* invented words per 10 minutes of non-speech audio;
* triage recall (share of erroneous cues placed in Quick check or Redo)
  and safety (share of Accept cues that contain an error).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence

_CONTRACTIONS = {"won't": "will not", "can't": "can not", "n't": " not",
                 "'re": " are", "'ll": " will", "'ve": " have", "'m": " am",
                 "'d": " would"}
_NUMBERS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
            "thirty": "30", "forty": "40", "fifty": "50", "hundred": "100"}
_FILLERS = {"uh", "um", "uhm", "hmm", "mm", "mhm", "er", "erm", "ah"}


def normalize_text(text: str) -> List[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"\[.*?\]|\(.*?\)", " ", t)
    for k, v in _CONTRACTIONS.items():
        t = t.replace(k, v)
    t = re.sub(r"[^\w\s']", " ", t)
    t = t.replace("'", "")
    out = []
    for w in t.split():
        w = _NUMBERS.get(w, w)
        if w not in _FILLERS:
            out.append(w)
    return out


def edit_ops(ref: Sequence[str], hyp: Sequence[str]) -> Dict[str, int]:
    n, m = len(ref), len(hyp)
    # Keep (subs, dels, ins) along the best path.
    rows = [[(0, 0, j) for j in range(m + 1)]]
    for i in range(1, n + 1):
        row = [(0, i, 0)]
        for j in range(1, m + 1):
            s, d, ins = rows[i - 1][j - 1]
            cand = [(s + (ref[i - 1] != hyp[j - 1]), d, ins)]
            s, d, ins = rows[i - 1][j]
            cand.append((s, d + 1, ins))
            s, d, ins = row[j - 1]
            cand.append((s, d, ins + 1))
            row.append(min(cand, key=lambda c: (sum(c), c[0])))
        rows.append(row)
    s, d, ins = rows[n][m]
    return {"substitutions": s, "deletions": d, "insertions": ins, "reference": n}


def wer(ref_text: str, hyp_text: str) -> Dict[str, float]:
    ref, hyp = normalize_text(ref_text), normalize_text(hyp_text)
    if len(ref) * len(hyp) > 4_000_000:          # very long files: blockwise
        return _wer_blocks(ref, hyp)
    ops = edit_ops(ref, hyp)
    errors = ops["substitutions"] + ops["deletions"] + ops["insertions"]
    return {**ops, "errors": errors, "wer": errors / max(1, len(ref))}


def _wer_blocks(ref: List[str], hyp: List[str]) -> Dict[str, float]:
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    tot = {"substitutions": 0, "deletions": 0, "insertions": 0}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        o = edit_ops(ref[i1:i2], hyp[j1:j2])
        for k in tot:
            tot[k] += o[k]
    errors = sum(tot.values())
    return {**tot, "reference": len(ref), "errors": errors,
            "wer": errors / max(1, len(ref))}


def timing_errors(truth: Sequence[Dict], hyp: Sequence[Dict]) -> Dict[str, Optional[float]]:
    """Start-time error over words matched between truth and hypothesis."""
    t_norm = [" ".join(normalize_text(w["word"])) for w in truth]
    h_norm = [" ".join(normalize_text(w["word"])) for w in hyp]
    sm = SequenceMatcher(None, t_norm, h_norm, autojunk=False)
    errs = []
    for b in sm.get_matching_blocks():
        for k in range(b.size):
            tw, hw = truth[b.a + k], hyp[b.b + k]
            if tw.get("start") is None or hw.get("start") is None or not t_norm[b.a + k]:
                continue
            errs.append(abs(float(tw["start"]) - float(hw["start"])))
    if not errs:
        return {"matched": 0, "mean_ms": None, "within_50ms": None, "within_100ms": None}
    return {"matched": len(errs), "mean_ms": round(1000 * sum(errs) / len(errs), 1),
            "within_50ms": round(sum(e <= 0.05 for e in errs) / len(errs), 3),
            "within_100ms": round(sum(e <= 0.10 for e in errs) / len(errs), 3)}


def invented_per_10min(hyp_words: Sequence[Dict], seconds: float) -> float:
    n = sum(1 for w in hyp_words if normalize_text(w.get("word", "")))
    return round(n / max(seconds, 1e-6) * 600.0, 2)


def cue_has_error(cue: Dict, truth: Sequence[Dict], hyp_words: Sequence[Dict],
                  timed_truth: bool, tol_s: float = 0.25) -> bool:
    """Does this output cue contain a text or timing error?"""
    s0, s1 = float(cue["start"]), float(cue["end"])
    inside = [w for w in hyp_words if w.get("start") is not None
              and s0 - 0.01 <= (float(w["start"]) + float(w["end"])) / 2 <= s1 + 0.01]
    if not inside:
        return False
    if timed_truth:
        ref = [w for w in truth if w.get("start") is not None
               and s0 - tol_s <= float(w["start"]) <= s1 + tol_s]
    else:
        ref = list(truth)
    r = normalize_text(" ".join(w["word"] for w in ref))
    h = normalize_text(" ".join(w["word"] for w in inside))
    sm = SequenceMatcher(None, r, h, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    if matched < len(h):
        return True                       # a word in the cue is not in the truth
    if timed_truth:
        te = timing_errors(ref, inside)
        if te["mean_ms"] is not None and te["mean_ms"] > tol_s * 1000:
            return True
    return False


def triage_scores(review: Dict, truth: Sequence[Dict], hyp_words: Sequence[Dict],
                  timed_truth: bool) -> Dict[str, Optional[float]]:
    cues = review.get("cues", [])
    if not cues:
        return {"recall": None, "accept_error_rate": None, "cues": 0}
    errors = [cue_has_error(c, truth, hyp_words, timed_truth) for c in cues]
    flagged = [c["tier"] != "accept" for c in cues]
    n_err = sum(errors)
    accepted = [e for e, f in zip(errors, flagged) if not f]
    return {"cues": len(cues), "erroneous": n_err,
            "recall": round(sum(e and f for e, f in zip(errors, flagged)) / n_err, 3) if n_err else None,
            "accept_error_rate": round(sum(accepted) / len(accepted), 3) if accepted else None,
            "share_flagged": round(sum(flagged) / len(cues), 3)}
