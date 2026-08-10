"""Transcript-free mode: build a consensus transcript from multiple engines.

Rationale
---------
No single model wins on every axis, so word-align treats transcription as
a *voting problem* -- a lightweight, time-anchored variant of the classic
ROVER technique (Fiscus, 1997). The default lineup:

  * WhisperX (Whisper large-v3 + wav2vec2) -- strongest transcription here
    and 99-language coverage, so it is the backbone and top-weighted voter.
  * Qwen3-ASR (+ Qwen3-ForcedAligner) -- a close second on transcription and,
    crucially, a *timed* voter: real word timestamps let it overturn a
    backbone error, not merely reinforce it.
  * Vosk -- an architecturally independent voter; weakest on words but the
    most accurate on timestamps (which is why the waterfall leans on it).

(NVIDIA NeMo's Parakeet/Canary remain available as legacy voters.)

Method
------
1. Run every requested engine; collect word streams with timestamps and
   per-word confidence where available.
2. Use the highest-weight *timed* engine (WhisperX by default) as the
   backbone sequence.
3. For each backbone word, gather overlapping-in-time candidates from
   the other timed engines, plus lexically-aligned candidates from
   untimed engines (e.g. Canary-Qwen) via difflib matching.
4. Each candidate votes for its normalized token with weight
   ``engine_prior * word_confidence``. The winning token replaces the
   backbone token only if it beats it by a clear margin, which keeps the
   backbone's casing/punctuation when the engines merely disagree on
   formatting.
5. The consensus word list is rendered to plain text and fed into the
   standard alignment waterfall -- so the transcript-free path reuses
   exactly the same battle-tested timing code as the human-transcript
   path.

The fused per-word confidence is retained (``conf``) so downstream
consumers can flag low-agreement regions for human review.
"""
from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .utils import normalize_word

# Vote-weight priors reflect *transcription* accuracy (voting is about which
# WORD is right), not timestamp quality. Field-tuned ranking on this setup:
# WhisperX > Qwen > Vosk for words. WhisperX is therefore both the backbone and
# the highest-weighted voter; a challenger only overturns it when others agree
# (Qwen + Vosk = 0.85 + 0.55 clears the backbone's 1.00 + margin). Timestamp
# quality is handled separately in the waterfall, where Vosk leads (see cli).
DEFAULT_PRIORS = {
    "whisperx": 1.00,   # best transcription here -> backbone + top vote weight
    "qwen": 0.85,       # strong transcription, just behind WhisperX; word-timed
    "vosk": 0.55,       # weakest transcription, but the best timestamps
    "canary": 1.20,     # legacy (NeMo), text-only; opt in via --engines
    "parakeet": 1.10,   # legacy (NeMo); opt in via --engines
}

# Backbone selection is deliberately SEPARATE from vote weight. The
# backbone supplies the word timestamps every downstream stage depends
# on, so it is chosen for timestamp quality and language coverage --
# not for raw WER. WhisperX wins that job even though Canary/Parakeet
# carry more voting weight on lexical accuracy.
BACKBONE_PREFERENCE = ("whisperx", "qwen", "vosk")

REPLACE_MARGIN = 0.25   # challenger must beat backbone vote by this margin
OVERLAP_WINDOW = 0.40   # seconds of midpoint tolerance for time clustering

# Votes decay linearly with temporal distance from the backbone word.
# Without this, a fixed window wider than the inter-word gap lets the
# NEIGHBOURING word vote in the current slot with full strength -- at
# normal speech rates (~3 words/sec) that produced ties in which a
# neighbour could displace the correct word. Proximity weighting makes
# the result stable regardless of how the window compares to speech rate.
def _proximity_weight(delta: float) -> float:
    return max(0.0, 1.0 - (abs(delta) / OVERLAP_WINDOW))


def _midpoint(w: Dict) -> Optional[float]:
    if w.get("start") is None or w.get("end") is None:
        return None
    return (float(w["start"]) + float(w["end"])) / 2.0


def _index_by_time(words: List[Dict]) -> List[Dict]:
    return sorted((w for w in words if _midpoint(w) is not None),
                  key=lambda w: _midpoint(w))


def _lexical_votes(backbone: List[Dict], untimed: List[Dict],
                   engine: str, prior: float,
                   votes: List[defaultdict],
                   display: List[Dict[str, str]]) -> None:
    """Project an untimed word stream onto backbone positions via difflib.

    Note the structural consequence of matching on *equal* tokens: a
    lexical vote can only ever land on the backbone's own token. Untimed
    engines therefore **reinforce** the backbone -- defending a slot against
    a timed challenger -- but cannot themselves introduce a correction.
    Canary-Qwen is a confidence source here, not a proofreader; see KNOWN
    LIMITATIONS in the README.

    ``display`` is populated in step with ``votes`` so the surface form is
    already correct if this projection is ever made fuzzy (which is what
    would let an untimed engine propose a genuine substitution).
    """
    a = [normalize_word(w["word"]) for w in backbone]
    b = [normalize_word(w["word"]) for w in untimed]
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    for block in matcher.get_matching_blocks():
        for off in range(block.size):
            i = block.a + off
            w = untimed[block.b + off]
            tok = normalize_word(w["word"])
            votes[i][tok] += prior * float(w.get("conf", 0.9))
            display[i].setdefault(tok, w["word"])


def build_consensus(engine_words: Dict[str, List[Dict]],
                    priors: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Fuse per-engine word streams into one consensus word list.

    Parameters
    ----------
    engine_words:
        Mapping of engine name -> word dicts. Engines whose words carry
        timestamps participate in time-anchored voting; untimed engines
        contribute lexical votes only.
    """
    priors = {**DEFAULT_PRIORS, **(priors or {})}
    timed = {k: _index_by_time(v) for k, v in engine_words.items()
             if any(_midpoint(w) is not None for w in v)}
    untimed = {k: v for k, v in engine_words.items() if k not in timed and v}

    if not timed:
        raise ValueError("Ensemble needs at least one engine with word "
                         "timestamps (e.g. whisperx).")

    backbone_name = next((n for n in BACKBONE_PREFERENCE if n in timed),
                         max(timed, key=lambda k: priors.get(k, 0.5)))
    backbone = timed[backbone_name]
    print(f"[ensemble] Backbone engine: {backbone_name} "
          f"({len(backbone)} words)")

    # votes[i] : normalized token -> accumulated weight, for backbone slot i
    votes: List[defaultdict] = [defaultdict(float) for _ in backbone]
    display: List[Dict[str, str]] = [{} for _ in backbone]

    for i, w in enumerate(backbone):
        tok = normalize_word(w["word"])
        votes[i][tok] += priors.get(backbone_name, 1.0) * float(w.get("conf", 0.9))
        display[i].setdefault(tok, w["word"])

    # Time-anchored votes from the other timed engines.
    for name, words in timed.items():
        if name == backbone_name:
            continue
        prior = priors.get(name, 0.5)
        j = 0
        for i, bw in enumerate(backbone):
            mid = _midpoint(bw)
            while j < len(words) and _midpoint(words[j]) < mid - OVERLAP_WINDOW:
                j += 1
            k = j
            while k < len(words) and _midpoint(words[k]) <= mid + OVERLAP_WINDOW:
                cw = words[k]
                tok = normalize_word(cw["word"])
                weight = (prior * float(cw.get("conf", 0.9))
                          * _proximity_weight(_midpoint(cw) - mid))
                if weight > 0:
                    votes[i][tok] += weight
                    display[i].setdefault(tok, cw["word"])
                k += 1

    # Lexical votes from untimed engines (e.g. Canary-Qwen text output).
    for name, words in untimed.items():
        _lexical_votes(backbone, words, name, priors.get(name, 0.5),
                       votes, display)

    consensus: List[Dict] = []
    replaced = 0
    for i, bw in enumerate(backbone):
        slot = votes[i]
        backbone_tok = normalize_word(bw["word"])
        best_tok = max(sorted(slot), key=slot.get)
        total = sum(slot.values()) or 1.0
        if (best_tok != backbone_tok
                and slot[best_tok] - slot.get(backbone_tok, 0.0) > REPLACE_MARGIN):
            word_out = display[i].get(best_tok, best_tok)
            replaced += 1
        else:
            best_tok = backbone_tok
            word_out = bw["word"]
        consensus.append({
            "word": word_out,
            "start": bw.get("start"),
            "end": bw.get("end"),
            "conf": round(slot[best_tok] / total, 3),
        })
    agree = sum(1 for c in consensus if c["conf"] >= 0.66)
    print(f"[ensemble] Consensus built: {len(consensus)} words, "
          f"{replaced} backbone corrections, "
          f"{agree}/{len(consensus)} high-agreement.")
    return consensus


def consensus_to_text(consensus: List[Dict]) -> str:
    """Render the consensus word list as a plain transcript."""
    return " ".join(w["word"] for w in consensus)


def consensus_to_structured_text(consensus: List[Dict],
                                 max_chars: int = 42,
                                 max_gap: float = 0.6) -> str:
    """Render the consensus as newline-separated, roughly subtitle-sized lines.

    A human transcript arrives pre-broken into caption lines, which is what the
    segmenter merges and balances. The raw consensus is one unbroken word
    stream, so feeding it in directly collapses the whole file into a single
    cue. Here we insert line breaks at sentence ends, at speech pauses, and
    before a line would grow too long -- giving the segmenter the same kind of
    short initial segments it gets in reference mode.
    """
    from .utils import has_terminal_punctuation

    lines: List[str] = []
    cur: List[str] = []
    cur_chars = 0
    prev_end: Optional[float] = None

    def flush():
        nonlocal cur, cur_chars
        if cur:
            lines.append(" ".join(cur))
            cur, cur_chars = [], 0

    for w in consensus:
        word = str(w["word"])
        start = w.get("start")
        gap = (start - prev_end
               if start is not None and prev_end is not None else 0.0)
        # Break before this word on a pause or if the line would get unwieldy.
        if cur and (gap > max_gap or cur_chars + 1 + len(word) > max_chars):
            flush()
        cur.append(word)
        cur_chars += len(word) + (1 if cur_chars else 0)
        if has_terminal_punctuation(word):   # break after a sentence end
            flush()
        if w.get("end") is not None:
            prev_end = w["end"]
    flush()
    return "\n".join(lines)
