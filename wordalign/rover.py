"""ROVER voting over engine hypotheses for one stretch of audio.

Transcript-free mode runs several ASR engines on the same speech chunks and
fuses their outputs with ROVER (Fiscus 1997): the hypotheses are aligned
into a word transition network by edit distance, then each slot is decided
by a weighted vote in which "no word here" is also a candidate. Unlike the
time-anchored voting in :mod:`wordalign.ensemble`, this needs no word
timestamps, so engines that return text only (Cohere Transcribe) vote as
full members and can correct the others.

Each output word carries ``agreement``: the share of the participating
engines' weight that voted for it. Triage uses it as the word's text
confidence.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from .utils import normalize_word

# Vote weights. Set from published robustness results (FFASR far-field,
# Appen private accents, the Open ASR Leaderboard) and meant to be re-fitted
# on your own corrected jobs (``wordalign-bench fit-priors``).
DEFAULT_PRIORS: Dict[str, float] = {
    "cohere": 1.00,     # most robust on far-field noise (FFASR #1)
    "qwen": 0.95,       # most robust on accents (Appen private track #1)
    "parakeet": 0.85,   # independent transducer architecture
    "whisperx": 0.80,   # widest language coverage; hallucinates on non-speech
    "sedicow": 0.90,    # only used on overlapped stretches
    "vosk": 0.40,       # weak on words; kept as a tie-breaker when enabled
}

# Timing quality of each engine's own word times (used to pick a word's
# provisional time from the engines that voted for it).
TIMING_PRIORITY = ("qwen", "parakeet", "vosk", "whisperx", "cohere", "sedicow")

Slot = Dict[str, Optional[Dict]]


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - 0.5 * SequenceMatcher(None, a, b).ratio()


def _align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Edit-distance alignment; pairs of (ref index | None, hyp index | None)."""
    n, m = len(ref), len(hyp)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
    for j in range(1, m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1.0, row[j - 1] + 1.0,
                         prev[j - 1] + _sub_cost(ri, hyp[j - 1]))
    pairs: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(dp[i][j] - (dp[i - 1][j - 1]
                                               + _sub_cost(ref[i - 1], hyp[j - 1]))) < 1e-9:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(dp[i][j] - (dp[i - 1][j] + 1.0)) < 1e-9:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def _representative(slot: Slot, priors: Dict[str, float]) -> str:
    score: Dict[str, float] = {}
    for eng, w in slot.items():
        if w is not None:
            tok = normalize_word(w["word"])
            score[tok] = score.get(tok, 0.0) + priors.get(eng, 0.5)
    if not score:
        return ""
    return max(sorted(score), key=score.get)


def build_network(hyps: Dict[str, List[Dict]], priors: Dict[str, float]) -> Tuple[List[Slot], List[str]]:
    """Align every hypothesis into one word transition network."""
    order = sorted(hyps, key=lambda e: (-priors.get(e, 0.5), e))
    slots: List[Slot] = []
    placed: List[str] = []
    for eng in order:
        words = [w for w in hyps[eng] if normalize_word(w.get("word", ""))]
        if not placed:
            slots = [{eng: w} for w in words]
            placed.append(eng)
            continue
        ref = [_representative(s, priors) for s in slots]
        hyp = [normalize_word(w["word"]) for w in words]
        new_slots: List[Slot] = []
        for ri, hj in _align(ref, hyp):
            if ri is None:
                slot: Slot = {e: None for e in placed}
                slot[eng] = words[hj]
            else:
                slot = slots[ri]
                slot[eng] = words[hj] if hj is not None else None
            new_slots.append(slot)
        slots = new_slots
        placed.append(eng)
    return slots, placed


def vote(hyps: Dict[str, List[Dict]], priors: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Fuse the hypotheses of one chunk into a voted word list.

    *hyps* maps engine -> word dicts (``word`` and optionally ``start``,
    ``end``) for the same audio. An engine that ran and heard nothing must
    be present with an empty list (it votes "no word" everywhere); an
    engine that failed must be left out.
    """
    priors = {**DEFAULT_PRIORS, **(priors or {})}
    hyps = {e: list(ws) for e, ws in hyps.items() if ws is not None}
    if not hyps:
        return []
    slots, placed = build_network(hyps, priors)
    total = sum(priors.get(e, 0.5) for e in hyps) or 1.0
    backbone = placed[0] if placed else None
    out: List[Dict] = []
    for slot in slots:
        score: Dict[Optional[str], float] = {}
        for eng in hyps:
            w = slot.get(eng)
            tok = normalize_word(w["word"]) if w is not None else None
            score[tok] = score.get(tok, 0.0) + priors.get(eng, 0.5)
        best = max(score.values())
        winners = [t for t, s in score.items() if abs(s - best) < 1e-9]
        if len(winners) > 1:
            bw = slot.get(backbone) if backbone else None
            btok = normalize_word(bw["word"]) if bw is not None else None
            winner = btok if btok in winners else sorted(
                winners, key=lambda t: (t is None, t or ""))[0]
        else:
            winner = winners[0]
        if winner is None:
            continue
        voters = [e for e in hyps if slot.get(e) is not None
                  and normalize_word(slot[e]["word"]) == winner]
        display_eng = max(voters, key=lambda e: (priors.get(e, 0.5), e))
        word = dict(word=slot[display_eng]["word"])
        timed = [e for e in TIMING_PRIORITY if e in voters
                 and slot[e].get("start") is not None and slot[e].get("end") is not None]
        timed += [e for e in voters if e not in timed
                  and slot[e].get("start") is not None and slot[e].get("end") is not None]
        if timed:
            word["start"] = float(slot[timed[0]]["start"])
            word["end"] = float(slot[timed[0]]["end"])
        word["agreement"] = round(score[winner] / total, 3)
        word["voters"] = sorted(voters)
        alternatives = {(t if t is not None else ""): round(s / total, 3)
                        for t, s in score.items() if t != winner}
        if alternatives:
            word["alternatives"] = alternatives
        out.append(word)
    return out
