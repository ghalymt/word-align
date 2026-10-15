"""Controlled degradations of 16 kHz mono speech (numpy float32 in [-1, 1]).

Levels are set the way the published tests set them: additive sources are
scaled to a target SNR against the speech's *active* power (samples above
a floor), so silence in the clean clip does not inflate the SNR.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional, Sequence

import numpy as np

SR = 16_000


def active_power(x: np.ndarray, floor_db: float = -45.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    frame = SR // 50
    n = x.size // frame
    if n == 0:
        return float(np.mean(x ** 2))
    p = np.mean(x[:n * frame].reshape(n, frame) ** 2, axis=1)
    peak = p.max() if p.size else 0.0
    if peak <= 0:
        return 0.0
    active = p[p > peak * 10 ** (floor_db / 10)]
    return float(active.mean()) if active.size else float(p.mean())


def fit_length(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float32)
    if noise.size == 0:
        return np.zeros(n, dtype=np.float32)
    if noise.size >= n:
        start = int(rng.integers(0, noise.size - n + 1))
        return noise[start:start + n]
    reps = int(np.ceil(n / noise.size))
    return np.tile(noise, reps)[:n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    noise = fit_length(noise, speech.size, rng)
    ps, pn = active_power(speech), float(np.mean(noise.astype(np.float64) ** 2))
    if ps <= 0 or pn <= 0:
        return speech.copy()
    gain = np.sqrt(ps / (pn * 10 ** (snr_db / 10)))
    return _limit(speech + gain * noise)


def _limit(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    return x.astype(np.float32)


# -- synthetic sources (used when no recordings are supplied) ----------------
def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1] if f.size > 1 else 1.0
    spec /= np.sqrt(f)
    out = np.fft.irfft(spec, n)
    return (out / (np.std(out) + 1e-9)).astype(np.float32)


def synthetic_music(n: int, rng: np.random.Generator) -> np.ndarray:
    """Chords with harmonics, a bass line and percussive noise bursts.

    A stand-in for background music when no music files are given; real
    recordings (``--music-dir``) are preferred for any reported number.
    """
    t = np.arange(n) / SR
    out = np.zeros(n)
    beat = 0.5
    roots = rng.choice([110.0, 130.81, 146.83, 164.81, 196.0], size=int(n / SR / 2) + 2)
    for k, root in enumerate(roots):
        s, e = int(k * 2 * SR), min(n, int((k + 1) * 2 * SR))
        if s >= n:
            break
        seg = t[s:e]
        for ratio in (1.0, 1.25, 1.5, 2.0):
            for h, amp in ((1, 1.0), (2, 0.4), (3, 0.2)):
                out[s:e] += amp * np.sin(2 * np.pi * root * ratio * h * seg)
        out[s:e] += 1.5 * np.sin(2 * np.pi * root / 2 * seg)
    burst = int(0.03 * SR)
    for k in range(int(n / SR / beat)):
        s = int(k * beat * SR)
        if s + burst < n:
            out[s:s + burst] += 3 * rng.standard_normal(burst) * np.linspace(1, 0, burst)
    return (out / (np.std(out) + 1e-9)).astype(np.float32)


def babble(pool: Sequence[np.ndarray], n: int, talkers: int,
           rng: np.random.Generator) -> np.ndarray:
    """Sum of *talkers* random speech excerpts (crowd / babble noise)."""
    out = np.zeros(n, dtype=np.float32)
    if not pool:
        return pink_noise(n, rng)
    for _ in range(talkers):
        src = pool[int(rng.integers(0, len(pool)))]
        piece = fit_length(src, n, rng)
        out += piece / (np.sqrt(active_power(piece)) + 1e-6)
    return out


def reverb(x: np.ndarray, rt60: float, rng: np.random.Generator,
           direct_ratio: float = 0.5) -> np.ndarray:
    """Convolve with an exponentially decaying noise impulse response."""
    length = int(rt60 * SR)
    if length < 2:
        return x.copy()
    t = np.arange(length) / SR
    ir = rng.standard_normal(length) * np.exp(-6.9 * t / rt60)
    ir[0] = 0.0
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9) * (1 - direct_ratio)
    ir[0] += direct_ratio
    wet = np.convolve(x, ir)[:x.size]
    return _limit(wet)


def overlap(a: np.ndarray, b: np.ndarray, sir_db: float, offset_s: float) -> np.ndarray:
    """Second talker *b* starting at *offset_s*, *sir_db* below *a*."""
    out = a.astype(np.float32).copy()
    start = int(offset_s * SR)
    if start >= out.size:
        return out
    piece = b[:out.size - start].astype(np.float32)
    pa, pb = active_power(a), active_power(piece)
    if pa <= 0 or pb <= 0:
        return out
    gain = np.sqrt(pa / (pb * 10 ** (sir_db / 10)))
    out[start:start + piece.size] += gain * piece
    return _limit(out)


def clip(x: np.ndarray, drive_db: float = 12.0) -> np.ndarray:
    """Hard clipping after *drive_db* of gain."""
    y = x * (10 ** (drive_db / 20))
    peak = float(np.max(np.abs(x))) or 1.0
    return np.clip(y, -peak, peak).astype(np.float32)


def codec(x: np.ndarray, kind: str) -> np.ndarray:
    """Round-trip through a lossy codec with ffmpeg; returns 16 kHz audio.

    kind: ``mulaw8k`` (telephone), ``opus12k`` (low-bitrate VoIP),
    ``mp3_32k``.
    """
    from ..audio.io import load_audio, write_wav
    ffmpeg = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for codec degradations")
    opts = {"mulaw8k": (["-ar", "8000", "-c:a", "pcm_mulaw"], ".wav"),
            "opus12k": (["-c:a", "libopus", "-b:a", "12k"], ".ogg"),
            "mp3_32k": (["-c:a", "libmp3lame", "-b:a", "32k"], ".mp3")}[kind]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        dst = os.path.join(tmp, "out" + opts[1])
        write_wav(src, x)
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", src, *opts[0], dst],
                       check=True)
        y = load_audio(dst)
    if y.size < x.size:
        y = np.pad(y, (0, x.size - y.size))
    return y[:x.size].astype(np.float32)
