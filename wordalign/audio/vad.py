"""Silero VAD (ONNX) without torch: speech probabilities and ASR chunk plans.

Silero VAD (MIT, https://github.com/snakers4/silero-vad) is a 2 MB model
that scores every 32 ms of 16 kHz audio. word-align uses it to keep ASR
engines away from silence, music and noise -- the inputs on which
attention-decoder and LLM-style ASR invent text. In one study, gating
Whisper with Silero VAD cut WER on non-speech audio from 104.8% to 8.0%.

The loop below mirrors the reference ``OnnxWrapper``: 512-sample windows,
a 64-sample context carried between calls, and a recurrent state.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .io import SAMPLE_RATE

WINDOW = 512                      # samples per probability (32 ms at 16 kHz)
CONTEXT = 64                      # samples of left context the model expects
HOP_S = WINDOW / SAMPLE_RATE      # 0.032 s

Segment = Tuple[float, float]


class VADUnavailable(RuntimeError):
    """The VAD model or onnxruntime is missing."""


def find_vad_model(models_dir: Optional[str]) -> Optional[str]:
    """Return the Silero ONNX file under ``<models>/vad`` if present."""
    env = os.environ.get("WORDALIGN_VAD_MODEL")
    if env and os.path.isfile(env):
        return env
    if models_dir:
        candidate = os.path.join(models_dir, "vad", "silero_vad.onnx")
        if os.path.isfile(candidate):
            return candidate
    return None


class SileroVAD:
    """Frame-level speech probabilities from the Silero ONNX model."""

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise VADUnavailable("onnxruntime is not installed") from exc
        if not os.path.isfile(model_path):
            raise VADUnavailable(f"VAD model not found: {model_path}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"])

    def probabilities(self, audio: np.ndarray, cancel_event=None) -> np.ndarray:
        """Speech probability for each 32 ms window of 16 kHz *audio*."""
        audio = np.asarray(audio, dtype=np.float32)
        frames = int(np.ceil(len(audio) / WINDOW)) if len(audio) else 0
        if frames == 0:
            return np.zeros(0, dtype=np.float32)
        padded = np.zeros(frames * WINDOW, dtype=np.float32)
        padded[:len(audio)] = audio
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(CONTEXT, dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        out = np.empty(frames, dtype=np.float32)
        run = self.session.run
        for i in range(frames):
            if cancel_event is not None and i % 2000 == 0 and cancel_event.is_set():
                raise InterruptedError("VAD cancelled")
            x = np.concatenate([context, padded[i * WINDOW:(i + 1) * WINDOW]])
            prob, state = run(None, {"input": x[None, :], "state": state, "sr": sr})
            out[i] = float(prob[0, 0])
            context = x[-CONTEXT:]
        return out


def speech_segments(probs: np.ndarray, threshold: float = 0.5,
                    neg_threshold: Optional[float] = None,
                    min_speech_s: float = 0.25, min_silence_s: float = 0.10,
                    pad_s: float = 0.03, duration: Optional[float] = None,
                    hop: float = HOP_S) -> List[Segment]:
    """Turn frame probabilities into ``(start, end)`` speech segments.

    Same hysteresis as Silero's ``get_speech_timestamps``: speech starts at
    ``threshold``, ends after ``min_silence_s`` below ``neg_threshold``.
    """
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    total = duration if duration is not None else len(probs) * hop
    segments: List[List[float]] = []
    triggered = False
    start = 0.0
    temp_end: Optional[float] = None
    for i, p in enumerate(probs):
        t = i * hop
        if p >= threshold:
            temp_end = None
            if not triggered:
                triggered = True
                start = t
            continue
        if triggered and p < neg_threshold:
            if temp_end is None:
                temp_end = t
            if t + hop - temp_end >= min_silence_s:
                segments.append([start, temp_end])
                triggered = False
                temp_end = None
    if triggered:
        segments.append([start, total])
    segments = [s for s in segments if s[1] - s[0] >= min_speech_s]
    # Pad, without letting neighbours overlap (split the gap between them).
    out: List[Segment] = []
    for i, (s, e) in enumerate(segments):
        lo = 0.0 if i == 0 else (segments[i - 1][1] + s) / 2
        hi = total if i == len(segments) - 1 else (e + segments[i + 1][0]) / 2
        out.append((max(lo, s - pad_s, 0.0), min(hi, e + pad_s, total)))
    return out


def _cut_long(start: float, end: float, probs: np.ndarray, max_len: float,
              hop: float) -> List[Segment]:
    """Split one long span at its quietest frames (pauses), each <= max_len."""
    pieces: List[Segment] = []
    cur = start
    while end - cur > max_len:
        lo = cur + max_len * 0.5
        hi = cur + max_len
        i0, i1 = int(lo / hop), max(int(hi / hop), int(lo / hop) + 1)
        window = probs[i0:i1]
        if len(window):
            # Lowest probability; prefer the latest of equally quiet frames.
            k = len(window) - 1 - int(np.argmin(window[::-1]))
            cut = (i0 + k) * hop + hop / 2
        else:
            cut = hi
        cut = min(max(cut, lo), hi)
        pieces.append((cur, cut))
        cur = cut
    pieces.append((cur, end))
    return pieces


def plan_asr_chunks(segments: Sequence[Segment], probs: np.ndarray,
                    duration: float, max_len: float = 28.0,
                    max_gap: float = 2.0, pad_s: float = 0.2,
                    hop: float = HOP_S) -> List[Segment]:
    """Group speech segments into ASR chunks of at most *max_len* seconds.

    Chunks never span a silence longer than *max_gap* (so no engine is fed
    long non-speech stretches) and long speech is cut at pauses. 28 s keeps
    every chunk inside Whisper-style 30 s windows after padding.
    """
    chunks: List[List[float]] = []
    for s, e in segments:
        if chunks and s - chunks[-1][1] <= max_gap and e - chunks[-1][0] <= max_len:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])
    out: List[Segment] = []
    for s, e in chunks:
        out.extend(_cut_long(s, e, probs, max_len, hop) if e - s > max_len
                   else [(s, e)])
    padded: List[Segment] = []
    for i, (s, e) in enumerate(out):
        lo = 0.0 if i == 0 else max(out[i - 1][1], (out[i - 1][1] + s) / 2)
        hi = duration if i == len(out) - 1 else min(out[i + 1][0], (e + out[i + 1][0]) / 2)
        padded.append((round(max(lo, s - pad_s, 0.0), 3),
                       round(min(hi, e + pad_s, duration), 3)))
    return [c for c in padded if c[1] - c[0] > 0.05]


def mean_prob(probs: np.ndarray, start: float, end: float,
              hop: float = HOP_S) -> float:
    """Average speech probability over ``[start, end]`` (0 if empty)."""
    if probs is None or len(probs) == 0:
        return 0.0
    i0 = max(0, int(start / hop))
    i1 = min(len(probs), max(i0 + 1, int(np.ceil(end / hop))))
    if i0 >= len(probs):
        return 0.0
    return float(np.mean(probs[i0:i1]))


def max_prob(probs: np.ndarray, start: float, end: float,
             hop: float = HOP_S) -> float:
    if probs is None or len(probs) == 0:
        return 0.0
    i0 = max(0, int(start / hop))
    i1 = min(len(probs), max(i0 + 1, int(np.ceil(end / hop))))
    if i0 >= len(probs):
        return 0.0
    return float(np.max(probs[i0:i1]))
