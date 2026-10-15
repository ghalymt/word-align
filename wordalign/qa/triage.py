"""Cue-level review triage: Accept / Quick check / Redo.

Review time falls when the reviewer can skip good cues safely and retype
hopeless ones without wading through them. The evidence points at cue
tiers rather than coloured words:

* highlighting low-confidence words neither sped up correction nor felt
  helpful in a 36-participant study (CHI 2025);
* editing ASR output beat typing from scratch only below about 30% WER;
  above that it was slower, and workers noticed only around 45% (Gaur et
  al.).

So each cue gets a risk score from its words' text and timing confidence
and the audio conditions under it, a predicted error share, a tier, and
one-line reasons the reviewer can act on. The weights below are priors;
``fit_weights`` re-fits them on your own corrected jobs.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ACCEPT = "accept"
QUICK = "quick_check"
REDO = "redo"
TIER_LABELS = {ACCEPT: "Accept", QUICK: "Quick check", REDO: "Redo"}

REDO_ERROR_SHARE = 0.30          # edit-vs-retype line (Gaur et al.)
ACCEPT_RISK = 0.25

# Logistic weights: risk = sigmoid(bias + sum(w * feature)).
DEFAULT_WEIGHTS: Dict[str, float] = {
    "bias": -3.2,
    "text_error": 6.0,        # mean(1 - text confidence)
    "low_text": 2.0,          # share of words with text confidence < 0.6
    "disputed_any": 2.6,      # at least one such word: never "Accept"
    "weak_timing": 2.2,       # share of words with timing confidence < 0.5
    "weak_timing_any": 1.5,   # at least one such word
    "interpolated": 3.0,      # share of words interpolated
    "unheard": 2.5,           # reference words no engine heard
    "overlap": 1.4,
    "music": 0.7,
    "crowd": 0.7,
    "very_low_snr": 1.1,
    "low_snr": 0.45,
    "noise": 0.3,
    "narrowband": 0.5,
    "clipping": 0.4,
    "hallucination": 0.8,     # words the filter removed inside the cue
    "implausible": 0.8,       # share of words with impossible durations
}
CONDITIONS = ("overlap", "music", "crowd", "very_low_snr", "low_snr", "noise",
              "narrowband", "clipping")
_REASON = {
    "overlap": "overlapping speakers",
    "music": "music under speech",
    "crowd": "crowd noise under speech",
    "very_low_snr": "very noisy audio",
    "low_snr": "noisy audio",
    "noise": "background noise",
    "narrowband": "telephone-band audio",
    "clipping": "clipped (distorted) audio",
}


def weights_path() -> Path:
    base = os.environ.get("WORDALIGN_HOME") or os.path.join(os.path.expanduser("~"), ".wordalign")
    return Path(base) / "triage_weights.json"


def load_weights() -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    path = weights_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            weights.update({k: float(v) for k, v in data.get("weights", {}).items()
                            if k in DEFAULT_WEIGHTS})
    except (OSError, ValueError):
        pass
    return weights


@dataclass
class CueReview:
    index: int
    start: float
    end: float
    text: str
    tier: str
    risk: float
    est_error: float
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    speaker: Optional[str] = None
    words: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tier_label"] = TIER_LABELS[self.tier]
        return d


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def cue_features(words: Sequence[Dict], conditions: Iterable[str],
                 removed: int, mode: str) -> Dict[str, float]:
    n = max(1, len(words))
    text_conf = [float(w.get("text_conf", 1.0)) for w in words]
    timing_conf = [float(w.get("timing_conf", 0.75)) for w in words]
    low_text = sum(1 for c in text_conf if c < 0.6)
    weak = sum(1 for c in timing_conf if c < 0.5)
    feats: Dict[str, float] = {
        "text_error": sum(1.0 - c for c in text_conf) / n,
        "low_text": low_text / n,
        "disputed_any": 1.0 if low_text else 0.0,
        "weak_timing": weak / n,
        "weak_timing_any": 1.0 if weak else 0.0,
        "interpolated": sum(1 for w in words if w.get("interpolated")) / n,
        "unheard": (sum(1 for w in words if w.get("heard_by", 1) == 0) / n
                    if mode == "reference" else 0.0),
        "hallucination": min(1.0, removed / 3.0),
        "implausible": sum(
            1 for w in words if w.get("start") is not None and w.get("end") is not None
            and not (0.03 <= float(w["end"]) - float(w["start"]) <= 2.5)) / n,
    }
    conds = set(conditions)
    for c in CONDITIONS:
        feats[c] = 1.0 if c in conds else 0.0
    return feats


def _reasons(words: Sequence[Dict], feats: Dict[str, float], removed: int,
             mode: str) -> List[str]:
    out: List[str] = []
    for c in CONDITIONS:
        if feats.get(c):
            out.append(_REASON[c])
    if mode == "ensemble":
        disputed = [w for w in words if float(w.get("text_conf", 1.0)) < 0.6]
        if disputed:
            out.append(f"voters disagree on {len(disputed)} word"
                       f"{'s' if len(disputed) != 1 else ''}")
    unheard = [w for w in words if w.get("heard_by", 1) == 0]
    if mode == "reference" and unheard:
        out.append(f"no engine heard {len(unheard)} word"
                   f"{'s' if len(unheard) != 1 else ''} (check the transcript)")
    interp = [w for w in words if w.get("interpolated")]
    if interp:
        secs = sum(float(w["end"]) - float(w["start"]) for w in interp
                   if w.get("start") is not None and w.get("end") is not None)
        out.append(f"{secs:.1f} s interpolated ({len(interp)} word"
                   f"{'s' if len(interp) != 1 else ''})")
    weak = [w for w in words if float(w.get("timing_conf", 0.75)) < 0.5
            and not w.get("interpolated")]
    if weak:
        out.append(f"uncertain timing on {len(weak)} word{'s' if len(weak) != 1 else ''}")
    if removed:
        out.append(f"hallucination filter removed {removed} word"
                   f"{'s' if removed != 1 else ''} here")
    if feats.get("implausible"):
        out.append("implausible word durations")
    return out


def score_cue(words: Sequence[Dict], conditions: Iterable[str], removed: int = 0,
              mode: str = "reference",
              weights: Optional[Dict[str, float]] = None) -> Tuple[str, float, float, List[str], Dict[str, float]]:
    """Return (tier, risk, est_error, reasons, features) for one cue."""
    w = weights or DEFAULT_WEIGHTS
    feats = cue_features(words, conditions, removed, mode)
    z = w.get("bias", 0.0) + sum(w.get(k, 0.0) * v for k, v in feats.items())
    risk = _sigmoid(z)
    est_error = min(1.0, feats["text_error"] * 1.25 + 0.5 * feats["unheard"]) \
        if mode == "ensemble" else min(1.0, feats["unheard"])
    # Risk decides whether a cue needs a look; the predicted error share
    # decides whether editing still beats retyping (Gaur et al.: ~30%).
    if est_error >= REDO_ERROR_SHARE:
        tier = REDO
    elif risk < ACCEPT_RISK:
        tier = ACCEPT
    else:
        tier = QUICK
    return tier, round(risk, 3), round(est_error, 3), _reasons(words, feats, removed, mode), feats


def _overlaps(a0: float, a1: float, b0: Optional[float], b1: Optional[float]) -> bool:
    return b0 is not None and b1 is not None and b0 < a1 and b1 > a0


def triage_segments(segments: Sequence[Dict], words: Sequence[Dict],
                    mode: str, conditions_at=None,
                    removals: Sequence[Dict] = (),
                    weights: Optional[Dict[str, float]] = None) -> List[CueReview]:
    """Score every cue. *segments* carry ``start``/``end`` in seconds,
    ``text`` and optionally ``word_range`` (first, last word index)."""
    weights = weights or load_weights()
    out: List[CueReview] = []
    for k, seg in enumerate(segments):
        s0, s1 = float(seg["start"]), float(seg["end"])
        if seg.get("word_range"):
            a, b = seg["word_range"]
            cue_words = list(words[a:b + 1])
        else:
            cue_words = [w for w in words if _overlaps(s0, s1, w.get("start"), w.get("end"))
                         and s0 - 0.01 <= (float(w["start"]) + float(w["end"])) / 2 <= s1 + 0.01]
        conds = list(conditions_at(s0, s1)) if conditions_at else []
        removed = sum(1 for r in removals if r.get("start") is not None
                      and s0 - 0.25 <= float(r["start"]) <= s1 + 0.25)
        tier, risk, est_error, reasons, feats = score_cue(
            cue_words, conds, removed, mode, weights)
        out.append(CueReview(
            index=k + 1, start=round(s0, 3), end=round(s1, 3), text=seg.get("text", ""),
            tier=tier, risk=risk, est_error=est_error, reasons=reasons,
            features={kk: round(v, 3) for kk, v in feats.items() if v},
            speaker=seg.get("speaker"),
            words=[_review_word(w) for w in cue_words]))
    return out


def _review_word(w: Dict) -> Dict:
    out = {"word": w.get("word"), "start": w.get("start"), "end": w.get("end"),
           "text_conf": w.get("text_conf"), "timing_conf": w.get("timing_conf"),
           "source": w.get("source")}
    alts = w.get("alternatives")
    if alts:
        # The readings the losing voters heard ("" = they heard nothing),
        # so a reviewer can pick one instead of retyping.
        out["alternatives"] = alts
    return out


def review_summary(cues: Sequence[CueReview]) -> Dict[str, Any]:
    counts = {ACCEPT: 0, QUICK: 0, REDO: 0}
    seconds = {ACCEPT: 0.0, QUICK: 0.0, REDO: 0.0}
    for c in cues:
        counts[c.tier] += 1
        seconds[c.tier] += c.end - c.start
    total = sum(seconds.values()) or 1.0
    return {"cues": len(cues), "counts": counts,
            "seconds": {k: round(v, 1) for k, v in seconds.items()},
            "share_to_review": round((seconds[QUICK] + seconds[REDO]) / total, 3)}


def write_review(path: str, cues: Sequence[CueReview], meta: Dict[str, Any]) -> str:
    queue = sorted(cues, key=lambda c: (-c.risk, c.start))
    data = {"version": 1, **meta, "summary": review_summary(cues),
            "queue": [c.index for c in queue if c.tier != ACCEPT],
            "cues": [c.to_dict() for c in cues]}
    tmp = f"{path}.part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


# -- calibration -----------------------------------------------------------
def fit_weights(examples: Sequence[Tuple[Dict[str, float], int]],
                l2: float = 0.05, steps: int = 3000, lr: float = 0.2) -> Dict[str, float]:
    """Logistic regression on (features, cue_was_corrected) pairs.

    Starts from the default priors and shrinks towards them (L2), so a
    handful of corrected jobs adjusts the weights without overfitting.
    """
    import numpy as np
    keys = [k for k in DEFAULT_WEIGHTS if k != "bias"]
    if not examples:
        return dict(DEFAULT_WEIGHTS)
    X = np.array([[f.get(k, 0.0) for k in keys] for f, _ in examples], dtype=float)
    y = np.array([float(lbl) for _, lbl in examples])
    prior = np.array([DEFAULT_WEIGHTS[k] for k in keys])
    w = prior.copy()
    b = DEFAULT_WEIGHTS["bias"]
    n = len(y)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        grad_w = X.T @ (p - y) / n + l2 * (w - prior)
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    out = {k: round(float(v), 4) for k, v in zip(keys, w)}
    out["bias"] = round(float(b), 4)
    return out


def save_weights(weights: Dict[str, float], n_examples: int) -> str:
    path = weights_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"weights": weights, "examples": n_examples}, indent=1),
                    encoding="utf-8")
    return str(path)
