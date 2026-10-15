"""Drop ASR words that the audio does not support.

Attention-decoder and LLM-style ASR invents text on silence, music and
noise: in one study Whisper hallucinated on 40.3% of non-speech clips, and
"thank you" alone was a quarter of those. VAD gating removes most of it
(WER on non-speech fell from 104.8% to 8.0%); de-looping plus a stock
phrase list took it to 6.5%. This module does the last two, using the VAD
probabilities from the audio quality map so that real speech is kept:

* **loops** -- the same 1- to 8-word phrase repeated back to back; extra
  copies go when they sit in low-speech frames, or always when the loop is
  long enough that no speaker would say it;
* **stock phrases** ("thank you for watching", "subtitles by ...") that
  stand alone in low-speech frames;
* **words in non-speech** -- a timed word whose whole span has almost no
  speech probability.

Every removal is returned with its reason so triage can show it.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..utils import normalize_word

LOW_SPEECH = 0.35          # mean VAD probability below which a span is "low speech"
NO_SPEECH = 0.12           # max VAD probability below which a span has no speech
ISLAND_GAP = 0.8           # seconds of silence around a stand-alone phrase
MAX_LOOP_N = 8
SURE_LOOP_REPEATS = {1: 7, 2: 5}   # repeats that are a loop whatever the audio
SURE_LOOP_DEFAULT = 4
GATED_LOOP_REPEATS = {1: 4}        # repeats checked against the VAD
GATED_LOOP_DEFAULT = 3

_STOCK = [
    # English (by far the most common in published reports)
    "thank you", "thank you very much", "thank you so much", "thanks",
    "thanks for watching", "thank you for watching", "thanks for listening",
    "thank you for listening", "thank you for your attention",
    "please subscribe", "subscribe", "like and subscribe",
    "please like and subscribe", "dont forget to subscribe",
    "subscribe to my channel", "subscribe to the channel",
    "see you next time", "see you in the next video", "see you later",
    "bye", "bye bye", "goodbye", "you", "the end", "so", "okay", "oh",
    "music", "applause", "laughter", "silence",
    "subtitles by the amaraorg community", "subtitles by", "transcribed by",
    "transcription by", "captions by", "translated by", "amaraorg",
    # A few frequent non-English ones
    "merci", "merci beaucoup", "gracias", "muchas gracias", "danke",
    "vielen dank", "grazie", "obrigado", "obrigada", "شكرا", "شكرا لكم",
    "اشترك في القناة", "谢谢", "谢谢观看", "ご視聴ありがとうございました",
]
STOCK_PHRASES = {tuple(p.split()) for p in (normalize_word(s) for s in _STOCK) if p}
_MAX_STOCK_LEN = max(len(p) for p in STOCK_PHRASES)

SpeechFn = Callable[[float, float], Optional[float]]


def _norm(words: Sequence[Dict]) -> List[str]:
    return [normalize_word(w.get("word", "")) for w in words]


def _span(words: Sequence[Dict], i: int, j: int) -> Tuple[Optional[float], Optional[float]]:
    starts = [w.get("start") for w in words[i:j] if w.get("start") is not None]
    ends = [w.get("end") for w in words[i:j] if w.get("end") is not None]
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def find_loops(tokens: Sequence[str]) -> List[Tuple[int, int, int]]:
    """``(start, n, repeats)`` of maximal back-to-back repeats of an n-gram."""
    loops = []
    i = 0
    L = len(tokens)
    while i < L:
        best = None
        for n in range(1, MAX_LOOP_N + 1):
            if i + 2 * n > L:
                break
            gram = tokens[i:i + n]
            if not any(gram):
                continue
            reps = 1
            while (i + (reps + 1) * n <= L
                   and tokens[i + reps * n:i + (reps + 1) * n] == gram):
                reps += 1
            gated = GATED_LOOP_REPEATS.get(n, GATED_LOOP_DEFAULT)
            if reps >= gated and (best is None or reps * n > best[1] * best[2]):
                best = (i, n, reps)
        if best:
            loops.append(best)
            i = best[0] + best[1] * best[2]
        else:
            i += 1
    return loops


def filter_hallucinations(words: List[Dict], speech_mean: Optional[SpeechFn] = None,
                          speech_max: Optional[SpeechFn] = None,
                          engine: str = "") -> Tuple[List[Dict], List[Dict]]:
    """Return ``(kept_words, removals)``; *words* is left unchanged.

    ``speech_mean``/``speech_max`` give the VAD probability over a time
    span (None when no VAD ran, which disables the audio-gated rules).
    """
    if not words:
        return [], []
    tokens = _norm(words)
    drop: Dict[int, str] = {}

    def low_speech(i: int, j: int) -> bool:
        if speech_mean is None:
            return False
        s, e = _span(words, i, j)
        if s is None:
            return False
        p = speech_mean(s, e)
        return p is not None and p < LOW_SPEECH

    # 1. loops
    for start, n, reps in find_loops(tokens):
        sure = reps >= SURE_LOOP_REPEATS.get(n, SURE_LOOP_DEFAULT)
        for r in range(1, reps):
            i, j = start + r * n, start + (r + 1) * n
            if sure or low_speech(i, j):
                for k in range(i, j):
                    drop.setdefault(k, f"loop x{reps}")

    # 2. stand-alone stock phrases in low-speech frames
    if speech_mean is not None:
        i = 0
        while i < len(words):
            matched = 0
            for n in range(min(_MAX_STOCK_LEN, len(words) - i), 0, -1):
                if tuple(tokens[i:i + n]) in STOCK_PHRASES:
                    matched = n
                    break
            if matched:
                j = i + matched
                s, e = _span(words, i, j)
                prev_end = words[i - 1].get("end") if i > 0 else None
                next_start = words[j].get("start") if j < len(words) else None
                island = (s is not None
                          and (prev_end is None or s - prev_end >= ISLAND_GAP)
                          and (next_start is None or next_start - e >= ISLAND_GAP))
                if island and low_speech(i, j):
                    for k in range(i, j):
                        drop.setdefault(k, "stock phrase in non-speech")
                i = j
            else:
                i += 1

    # 3. words with no speech at all under them
    if speech_max is not None:
        for k, w in enumerate(words):
            if k in drop or w.get("start") is None or w.get("end") is None:
                continue
            p = speech_max(float(w["start"]), float(w["end"]))
            if p is not None and p < NO_SPEECH:
                drop[k] = "no speech under word"

    kept = [w for k, w in enumerate(words) if k not in drop]
    removals = [{"word": words[k].get("word", ""), "start": words[k].get("start"),
                 "end": words[k].get("end"), "reason": reason, "engine": engine}
                for k, reason in sorted(drop.items())]
    return kept, removals


_PUNCT_ONLY = re.compile(r"^\W+$")


def filter_text_chunk(text: str, chunk_speech: Optional[float]) -> Tuple[str, Optional[str]]:
    """For untimed engines: drop a whole chunk's text when it is a stock
    phrase and the chunk is mostly non-speech. Returns (text, reason)."""
    tokens = tuple(t for t in (normalize_word(x) for x in text.split()) if t)
    if not tokens:
        return "", None
    if (chunk_speech is not None and chunk_speech < LOW_SPEECH
            and tokens in STOCK_PHRASES):
        return "", "stock phrase in non-speech"
    # De-loop untimed text with the "sure" thresholds only.
    words = [{"word": w} for w in text.split()]
    kept, removed = filter_hallucinations(words)
    if removed:
        return " ".join(w["word"] for w in kept), f"loop ({len(removed)} words)"
    return text, None
