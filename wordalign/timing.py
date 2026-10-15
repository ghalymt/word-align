"""Anchor-then-align word timing (stage 5 of the robust flow).

The legacy waterfall copied the first engine's timestamp onto every word it
matched lexically, even single-word matches, and later engines could only
fill gaps. On bad audio that locks in wrong matches ("the" found 2 s away)
and leaves long stretches to interpolation. This module replaces it:

1. **Anchors.** A timed engine's word is trusted only inside a run of at
   least 3 matching words, with a plausible duration and times that keep
   increasing. Words two engines place within 150 ms of each other are
   *strong* anchors.
2. **Forced alignment between anchors.** The known text (the human
   transcript, or the voted transcript) is cut into windows whose ends are
   anchors, and each window is aligned on the *original* audio by
   Qwen3-ForcedAligner (or wav2vec2 for languages it does not cover). A
   window whose result collapses or contradicts its anchors is retried on
   the cleaned ASR track, then discarded.
3. **Fallbacks.** Anchors and short consistent matches keep their engine
   times; MFA and interpolation fill the rest, and every interpolated word
   is flagged.

Each word ends with ``source`` (where its time came from) and
``timing_conf`` (0-1), which triage rolls up per cue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .utils import normalize_word

STRONG_AGREE_S = 0.15        # two sources within this -> strong anchor
MIN_RUN = 3                  # words in a row before a match is an anchor
MIN_WORD_S = 0.02
MAX_WORD_S = 3.0
WINDOW_TARGET_S = 30.0
WINDOW_MAX_S = 60.0
WINDOW_HARD_MAX_S = 170.0    # Qwen3-ForcedAligner takes at most 180 s per call
WINDOW_PAD_S = 0.3

# Which source's own time to trust when no forced alignment is available.
SOURCE_PRIORITY = ("Vosk", "Qwen3-ASR", "parakeet-ultra", "WhisperX",
                   "SE-DiCoW", "SRT")

CONF_FA_CONFIRMED = 0.95
CONF_FA = 0.85
CONF_FA_DISPUTED = 0.6
CONF_STRONG = 0.9
CONF_RUN = 0.75
CONF_MFA = 0.7
CONF_SHORT = 0.5
CONF_INTERPOLATED = 0.2

Span = Tuple[float, float]


def _plausible(w: Dict) -> bool:
    s, e = w.get("start"), w.get("end")
    if s is None or e is None:
        return False
    try:
        s, e = float(s), float(e)
    except (TypeError, ValueError):
        return False
    return s >= 0 and MIN_WORD_S <= e - s <= MAX_WORD_S


def _increasing(pairs: List[Tuple[int, float]]) -> set:
    """Indices (into *pairs*) of the longest run of non-decreasing times."""
    n = len(pairs)
    if n == 0:
        return set()
    import bisect
    tails: List[float] = []
    tails_idx: List[int] = []
    prev = [-1] * n
    for i, (_, t) in enumerate(pairs):
        k = bisect.bisect_right(tails, t)
        if k == len(tails):
            tails.append(t)
            tails_idx.append(i)
        else:
            tails[k] = t
            tails_idx[k] = i
        prev[i] = tails_idx[k - 1] if k > 0 else -1
    keep, i = set(), tails_idx[-1]
    while i != -1:
        keep.add(i)
        i = prev[i]
    return keep


@dataclass
class SourceMatches:
    """Where one timed source places the known words."""
    name: str
    runs: Dict[int, Span] = field(default_factory=dict)    # in runs >= MIN_RUN
    short: Dict[int, Span] = field(default_factory=dict)   # in shorter runs

    def get(self, i: int) -> Optional[Span]:
        return self.runs.get(i) or self.short.get(i)


def match_source(known_norm: Sequence[str], source_words: Sequence[Dict],
                 name: str, min_run: int = MIN_RUN) -> SourceMatches:
    """Lexically match *source_words* onto the known words.

    Implausible words are ignored, and matches whose times run backwards
    (a block matched against the wrong occurrence) are dropped.
    """
    out = SourceMatches(name)
    src = [w for w in source_words if _plausible(w)]
    if not src or not known_norm:
        return out
    src_norm = [normalize_word(w.get("word", "")) for w in src]
    matcher = SequenceMatcher(None, list(known_norm), src_norm, autojunk=False)
    pairs: List[Tuple[int, int, int]] = []          # (known idx, src idx, block size)
    for block in matcher.get_matching_blocks():
        for off in range(block.size):
            if known_norm[block.a + off]:
                pairs.append((block.a + off, block.b + off, block.size))
    if not pairs:
        return out
    keep = _increasing([(a, float(src[b]["start"])) for a, b, _ in pairs])
    kept = [p for k, p in enumerate(pairs) if k in keep]
    for a, b, size in kept:
        span = (float(src[b]["start"]), float(src[b]["end"]))
        if size >= min_run:
            out.runs[a] = span
        else:
            out.short[a] = span
    # A short match must sit, in time, between its neighbouring run anchors.
    if out.short and out.runs:
        import bisect
        run_idx = sorted(out.runs)
        for a in list(out.short):
            k = bisect.bisect_left(run_idx, a)
            left = out.runs[run_idx[k - 1]] if k > 0 else None
            right = out.runs[run_idx[k]] if k < len(run_idx) else None
            s, e = out.short[a]
            if ((left and s < left[1] - STRONG_AGREE_S)
                    or (right and e > right[0] + STRONG_AGREE_S)):
                del out.short[a]
    return out


@dataclass
class Anchor:
    index: int
    span: Span
    sources: List[str]
    strong: bool


def build_anchors(n_words: int, matches: Sequence[SourceMatches]) -> Dict[int, Anchor]:
    """Anchors from run matches; strong where two sources agree."""
    anchors: Dict[int, Anchor] = {}
    order = {name: k for k, name in enumerate(SOURCE_PRIORITY)}
    ranked = sorted(matches, key=lambda m: order.get(m.name, len(order)))
    for i in range(n_words):
        spans = [(m.name, m.runs[i]) for m in ranked if i in m.runs]
        if not spans:
            continue
        name, span = spans[0]
        agree = [n for n, sp in spans if abs(sp[0] - span[0]) <= STRONG_AGREE_S]
        # Short matches from another source also confirm a run anchor.
        for m in ranked:
            if m.name not in agree and i in m.short and abs(m.short[i][0] - span[0]) <= STRONG_AGREE_S:
                agree.append(m.name)
        anchors[i] = Anchor(i, span, agree, strong=len(agree) >= 2)
    # Drop anchors whose times run backwards against their neighbours.
    idx = sorted(anchors)
    keep = _increasing([(i, anchors[i].span[0]) for i in idx])
    return {idx[k]: anchors[idx[k]] for k in keep}


@dataclass
class AlignWindow:
    id: int
    first: int            # first word index (inclusive)
    last: int             # last word index (inclusive)
    start: float
    end: float

    def to_job(self, words: Sequence[str]) -> Dict:
        return {"id": self.id, "start": round(self.start, 3),
                "end": round(self.end, 3),
                "words": list(words[self.first:self.last + 1])}


def plan_windows(n_words: int, anchors: Dict[int, Anchor], duration: float,
                 first_speech: float = 0.0, target: float = WINDOW_TARGET_S,
                 max_len: float = WINDOW_MAX_S,
                 hard_max: float = WINDOW_HARD_MAX_S) -> List[AlignWindow]:
    """Cut the words into alignment windows whose inner ends are anchors.

    Strong anchors are preferred as boundaries; any anchor is used where
    strong ones are too sparse. A window grows to about *target* seconds
    (never past *max_len* when a nearer boundary exists); a stretch longer
    than *hard_max* cannot be force-aligned and gets no window (its words
    fall back to anchors, MFA and interpolation).
    """
    if n_words == 0:
        return []
    duration = max(duration, 0.0)
    t_start = max(0.0, first_speech - 1.0)
    if not anchors:
        if 0 < duration - t_start <= hard_max:
            return [AlignWindow(0, 0, n_words - 1, t_start, duration)]
        return []
    strong = [i for i in sorted(anchors) if anchors[i].strong]
    allb = sorted(anchors)
    bounds = strong if len(strong) >= max(2, len(allb) // 4) else allb

    def end_of(b: int) -> float:
        return anchors[b].span[1] + WINDOW_PAD_S

    windows: List[AlignWindow] = []
    cur, t0, pos = 0, t_start, 0
    while cur < n_words:
        while pos < len(bounds) and bounds[pos] < cur:
            pos += 1
        if pos >= len(bounds):
            break
        best = None
        k = pos
        while k < len(bounds):
            span = end_of(bounds[k]) - t0
            if span > max_len:
                break
            best = k
            if span >= target:
                break
            k += 1
        if best is None:                 # next boundary is far away
            b = bounds[pos]
            if end_of(b) - t0 <= hard_max:
                windows.append(AlignWindow(len(windows), cur, b, t0, end_of(b)))
        else:
            b = bounds[best]
            windows.append(AlignWindow(len(windows), cur, b, t0, end_of(b)))
        cur = b + 1
        t0 = max(0.0, anchors[b].span[1] - 0.05)
    if cur < n_words and 0 < duration - t0 <= hard_max:
        windows.append(AlignWindow(len(windows), cur, n_words - 1, t0, duration))
    return windows


def check_window(result: Dict, window: AlignWindow, anchors: Dict[int, Anchor]) -> Tuple[bool, str]:
    """Is a forced-alignment result usable? Returns (ok, reason)."""
    times = result.get("times") or []
    n = window.last - window.first + 1
    if not times or len(times) != n:
        return False, f"no result ({result.get('status')})"
    timed = [t for t in times if t]
    if len(timed) < max(1, int(0.7 * n)):
        return False, f"only {len(timed)}/{n} words placed"
    tiny = sum(1 for s, e in timed if e - s <= 0.01)
    if tiny > 0.3 * len(timed):
        return False, f"collapsed ({tiny} zero-length words)"
    starts = [s for s, _ in timed]
    if len(timed) >= 5 and len(set(round(s, 2) for s in starts)) < 0.5 * len(timed):
        return False, "collapsed (repeated times)"
    devs = []
    for k, t in enumerate(times):
        a = anchors.get(window.first + k)
        if t and a is not None:
            devs.append(abs(t[0] - a.span[0]))
    if len(devs) >= 3 and median(devs) > 0.3:
        return False, f"disagrees with anchors (median {median(devs):.2f}s)"
    return True, "ok"


def assign_times(words: List[Dict], matches: Sequence[SourceMatches],
                 anchors: Dict[int, Anchor],
                 aligned: Dict[int, Tuple[Span, str]]) -> Dict[str, int]:
    """Give every word its best time, ``source`` and ``timing_conf``.

    *aligned*: word index -> (span, aligner name) from accepted windows.
    Words with no time are left for MFA / interpolation.
    """
    order = {name: k for k, name in enumerate(SOURCE_PRIORITY)}
    ranked = sorted(matches, key=lambda m: order.get(m.name, len(order)))
    counts: Dict[str, int] = {}
    for i, w in enumerate(words):
        anchor = anchors.get(i)
        src_spans = [(m.name, m.get(i)) for m in ranked if m.get(i)]
        chosen: Optional[Tuple[Span, str, float]] = None
        if i in aligned:
            span, aligner = aligned[i]
            if src_spans:
                close = any(abs(sp[0] - span[0]) <= STRONG_AGREE_S for _, sp in src_spans)
                far = (anchor is not None and abs(anchor.span[0] - span[0]) > 0.5)
                conf = CONF_FA_CONFIRMED if close else (CONF_FA_DISPUTED if far else CONF_FA)
            else:
                conf = CONF_FA
            if not (MIN_WORD_S <= span[1] - span[0] <= MAX_WORD_S):
                conf = min(conf, CONF_SHORT)
            chosen = (span, aligner, conf)
        elif anchor is not None:
            chosen = (anchor.span, "+".join(anchor.sources),
                      CONF_STRONG if anchor.strong else CONF_RUN)
        else:
            for name, span in src_spans:
                chosen = (span, name, CONF_SHORT)
                break
        if chosen is None:
            continue
        span, source, conf = chosen
        w.update(start=round(span[0], 3), end=round(span[1], 3), matched=True,
                 source=source, timing_conf=round(conf, 3))
        counts[source] = counts.get(source, 0) + 1
    return counts


def enforce_order(words: List[Dict], min_dur: float = 0.04) -> int:
    """Remove overlaps/back-steps between consecutive timed words.

    A word that starts before its predecessor ends is clipped (or, if it
    would vanish, moved just after it) and loses timing confidence.
    Returns the number of words changed.
    """
    changed = 0
    prev_end = None
    for w in words:
        s, e = w.get("start"), w.get("end")
        if s is None or e is None:
            continue
        if prev_end is not None and s < prev_end - 0.005:
            new_s = prev_end
            new_e = max(e, new_s + min_dur)
            w["start"], w["end"] = round(new_s, 3), round(new_e, 3)
            w["timing_conf"] = round(min(float(w.get("timing_conf", 0.5)), CONF_SHORT), 3)
            changed += 1
        prev_end = w["end"]
    return changed


def estimated_duration(word: str) -> float:
    """Rough spoken length of *word* in seconds (about 15 characters/s)."""
    chars = len(normalize_word(word)) or 1
    return min(0.8, max(0.15, 0.07 * chars + 0.05))


def _ends_sentence(word: str) -> bool:
    from .utils import has_terminal_punctuation
    return has_terminal_punctuation(word)


def _place_block(t0: float, t1: float, need: float, lean: str,
                 speech: Optional[Sequence[Span]]) -> Span:
    """Where in the gap ``[t0, t1]`` a run of untimed words of total length
    *need* most likely sits.

    With VAD speech segments, the run goes into the speech inside the gap
    nearest the side it leans to; otherwise it hugs that side ("left": just
    after the previous word, "right": just before the next one). A gap
    shorter than the estimate is used whole.
    """
    gap = max(0.0, t1 - t0)
    if gap <= need:
        return t0, t1
    if speech:
        inside = [(max(s, t0), min(e, t1)) for s, e in speech if e > t0 and s < t1]
        inside = [(s, e) for s, e in inside if e - s > 0.05]
        if inside:
            lo = min(s for s, _ in inside)
            hi = max(e for _, e in inside)
            if hi - lo <= need:
                pad = (need - (hi - lo)) / 2
                return max(t0, lo - pad), min(t1, hi + pad)
            if lean == "right":
                return max(lo, hi - need), hi
            if lean == "left":
                return lo, min(hi, lo + need)
            return lo, hi
    if lean == "right":
        return t1 - need, t1
    if lean == "left":
        return t0, t0 + need
    mid = (t0 + t1) / 2
    return mid - need / 2, mid + need / 2


def interpolate(words: List[Dict], duration: Optional[float] = None,
                speech: Optional[Sequence[Span]] = None) -> int:
    """Fill untimed words between timed neighbours; flags them interpolated.

    Each run of untimed words gets a length estimated from its text and is
    placed where it most likely sits in the gap: a run that starts a
    sentence leans towards the next timed word, one that ends a sentence
    towards the previous word, and VAD *speech* segments (when given) pull
    the run into the stretch where someone is actually talking. Within the
    run, time is shared in proportion to the word length estimates.
    """
    n = 0
    i = 0
    total = len(words)
    while i < total:
        w = words[i]
        if w.get("start") is not None and w.get("end") is not None:
            i += 1
            continue
        j = i
        while j < total and (words[j].get("start") is None or words[j].get("end") is None):
            j += 1
        run = words[i:j]
        prev = words[i - 1] if i > 0 else None
        nxt = words[j] if j < total else None
        est = [estimated_duration(x.get("word", "")) for x in run]
        need = sum(est)
        starts_sentence = prev is None or _ends_sentence(str(prev.get("word", "")))
        ends_sentence = _ends_sentence(str(run[-1].get("word", "")))
        lean = ("right" if starts_sentence and not ends_sentence
                else "left" if ends_sentence and not starts_sentence else "middle")
        if prev is not None and nxt is not None:
            t0, t1 = float(prev["end"]), float(nxt["start"])
            if t1 < t0:
                t1 = t0
            s0, s1 = _place_block(t0, t1, need, lean, speech)
        elif prev is not None:
            t0 = float(prev["end"]) + 0.05
            limit = duration if duration else t0 + need + 1.0
            s0, s1 = _place_block(t0, max(t0, limit), need, "left", speech)
        elif nxt is not None:
            t1 = float(nxt["start"])
            s0, s1 = _place_block(0.0, max(0.0, t1 - 0.05), need, "right", speech)
        else:
            # Nothing is timed at all: a fixed 0.3 s per word, as before.
            for k, x in enumerate(run):
                s = (i + k) * 0.3
                x.update(start=round(s, 3), end=round(s + 0.25, 3), matched=False,
                         source="Interpolated", interpolated=True,
                         timing_conf=CONF_INTERPOLATED)
                n += 1
            i = j
            continue
        span = max(0.0, s1 - s0)
        scale = span / need if need > 0 else 0.0
        t = s0
        for x, d in zip(run, est):
            s, e = t, t + d * scale
            if duration:
                s, e = min(s, duration), min(e, duration)
            x.update(start=round(s, 3), end=round(max(e, s), 3), matched=False,
                     source="Interpolated", interpolated=True,
                     timing_conf=CONF_INTERPOLATED)
            t += d * scale
            n += 1
        i = j
    return n


def run_anchor_alignment(words: List[Dict], sources: Dict[str, List[Dict]],
                         duration: float,
                         aligner: Optional[Callable[[List[Dict]], Optional[List[Dict]]]] = None,
                         retry_aligner: Optional[Callable[[List[Dict]], Optional[List[Dict]]]] = None,
                         aligner_name: str = "Qwen3-FA",
                         first_speech: float = 0.0,
                         log: Callable[[str], None] = print) -> Dict:
    """Time *words* (dicts with ``word``) in place; returns statistics.

    *sources*: source name -> timed word dicts from engines.
    *aligner*: takes alignment jobs (``AlignWindow.to_job``) and returns the
    worker's window results, or None if unavailable. *retry_aligner* does
    the same on the cleaned ASR track for windows that failed.
    """
    known = [normalize_word(w["word"]) for w in words]
    texts = [w["word"] for w in words]
    matches = [match_source(known, ws, name) for name, ws in sources.items() if ws]
    anchors = build_anchors(len(words), matches)
    strong = sum(1 for a in anchors.values() if a.strong)
    log(f"Anchors: {len(anchors)} ({strong} strong) from "
        f"{', '.join(m.name for m in matches) or 'no timed source'}")
    aligned: Dict[int, Tuple[Span, str]] = {}
    stats: Dict = {"anchors": len(anchors), "strong_anchors": strong,
                   "windows": 0, "windows_ok": 0, "windows_retried": 0,
                   "window_failures": []}
    if aligner is not None and words:
        windows = plan_windows(len(words), anchors, duration, first_speech)
        stats["windows"] = len(windows)
        results = aligner([w.to_job(texts) for w in windows]) if windows else []
        if results is None:
            log(f"{aligner_name} unavailable; keeping engine times")
            stats["aligner"] = "unavailable"
        else:
            by_id = {r.get("id"): r for r in results}
            failed: List[AlignWindow] = []
            for win in windows:
                r = by_id.get(win.id) or {}
                ok, why = check_window(r, win, anchors)
                if ok:
                    stats["windows_ok"] += 1
                    for k, t in enumerate(r["times"]):
                        if t:
                            aligned[win.first + k] = ((float(t[0]), float(t[1])), aligner_name)
                else:
                    failed.append(win)
                    stats["window_failures"].append(
                        {"start": win.start, "end": win.end, "reason": why})
            if failed and retry_aligner is not None:
                retried = retry_aligner([w.to_job(texts) for w in failed]) or []
                by_id = {r.get("id"): r for r in retried}
                for win in failed:
                    r = by_id.get(win.id) or {}
                    ok, _why = check_window(r, win, anchors)
                    if ok:
                        stats["windows_retried"] += 1
                        for k, t in enumerate(r["times"]):
                            if t:
                                aligned[win.first + k] = ((float(t[0]), float(t[1])),
                                                          aligner_name + " (cleaned)")
            log(f"{aligner_name}: {stats['windows_ok']}/{len(windows)} windows "
                f"aligned, {stats['windows_retried']} rescued on the cleaned track")
    stats["sources"] = assign_times(words, matches, anchors, aligned)
    return stats
