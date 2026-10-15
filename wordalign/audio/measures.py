"""Signal measurements for the audio quality map.

* WADA-SNR (Kim & Stern, Interspeech 2008): blind SNR of a speech segment,
  the estimator the TV background-music study used to decide where music
  separation helps. The lookup table below was computed numerically from
  the method's own model (Gamma-distributed speech amplitude, alpha = 0.4,
  plus Gaussian noise), so it needs no external data.
* Bandwidth: share of speech energy above 4 kHz, which is near zero on
  telephone (8 kHz) or heavily compressed audio.
* Clipping: share of samples in flat runs at the waveform's peak.
"""
from __future__ import annotations

import numpy as np

from .io import SAMPLE_RATE

# G(SNR) = log(E|z|) - E[log|z|] for SNR = -20 ... 100 dB.
_WADA_DB = np.arange(-20, 101, dtype=np.float64)
_WADA_G = np.array([
    0.40946, 0.40949, 0.40949, 0.409548, 0.409666, 0.409685, 0.409828,
    0.410097, 0.410534, 0.411296, 0.412217, 0.413329, 0.414968, 0.417135,
    0.420208, 0.423841, 0.428652, 0.43456, 0.442188, 0.451141, 0.461979,
    0.474447, 0.488942, 0.505365, 0.523571, 0.543423, 0.565313, 0.588872,
    0.613666, 0.63984, 0.667161, 0.69521, 0.723972, 0.75327, 0.782705,
    0.812073, 0.841358, 0.870657, 0.89954, 0.927795, 0.955561, 0.982808,
    1.009487, 1.035291, 1.060269, 1.084534, 1.108039, 1.130575, 1.152373,
    1.173462, 1.193817, 1.21356, 1.232362, 1.250442, 1.267754, 1.284476,
    1.300263, 1.315666, 1.330248, 1.344301, 1.357675, 1.370523, 1.382822,
    1.394617, 1.405921, 1.416568, 1.426927, 1.436731, 1.44609, 1.455027,
    1.463596, 1.471795, 1.479594, 1.487069, 1.494159, 1.500923, 1.507461,
    1.513741, 1.519566, 1.525245, 1.530632, 1.535769, 1.540708, 1.545431,
    1.549926, 1.55422, 1.558343, 1.562282, 1.566066, 1.569648, 1.573015,
    1.576341, 1.579515, 1.582502, 1.585379, 1.58811, 1.59073, 1.593203,
    1.595581, 1.597826, 1.599948, 1.602027, 1.604002, 1.605855, 1.60763,
    1.609353, 1.610986, 1.612541, 1.614048, 1.615476, 1.616844, 1.618168,
    1.619424, 1.620618, 1.621763, 1.622827, 1.623845, 1.624832, 1.625767,
    1.626663, 1.627513,
])
_EPS = 1e-10


def wada_snr(audio: np.ndarray) -> float:
    """Blind SNR estimate in dB (clamped to -20..100); NaN for silence."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size < SAMPLE_RATE // 10:
        return float("nan")
    peak = np.max(np.abs(x))
    if peak < 1e-6:
        return float("nan")
    a = np.abs(x / peak)
    a[a < _EPS] = _EPS
    g = np.log(max(_EPS, a.mean())) - np.log(a).mean()
    if g <= _WADA_G[0]:
        return float(_WADA_DB[0])
    if g >= _WADA_G[-1]:
        return float(_WADA_DB[-1])
    return float(np.interp(g, _WADA_G, _WADA_DB))


def high_band_ratio(audio: np.ndarray, split_hz: float = 4000.0,
                    n_fft: int = 512) -> float:
    """Share of spectral energy above *split_hz* (0..1)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.size < n_fft:
        return float("nan")
    frames = x.size // n_fft
    spec = np.abs(np.fft.rfft(
        x[:frames * n_fft].reshape(frames, n_fft) * np.hanning(n_fft), axis=1)) ** 2
    power = spec.sum(axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    total = power.sum()
    if total <= 0:
        return float("nan")
    return float(power[freqs >= split_hz].sum() / total)


def is_narrowband(audio: np.ndarray, threshold: float = 0.0006) -> bool:
    """True when almost no energy reaches above 4 kHz (telephone-like)."""
    ratio = high_band_ratio(audio)
    return bool(ratio == ratio and ratio < threshold)


def clipping_ratio(audio: np.ndarray, min_run: int = 3,
                   tolerance: float = 1e-3) -> float:
    """Share of samples in runs of >= *min_run* equal values at the peak."""
    x = np.asarray(audio, dtype=np.float32)
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    if peak < 0.3:     # quiet files are not hard-clipped
        return 0.0
    at_peak = np.abs(np.abs(x) - peak) <= tolerance * max(peak, 1e-6)
    if not at_peak.any():
        return 0.0
    # Lengths of consecutive True runs.
    edges = np.diff(np.concatenate(([0], at_peak.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    lengths = ends - starts
    clipped = lengths[lengths >= min_run].sum()
    return float(clipped / x.size)


def rms_db(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    if x.size == 0:
        return float("-inf")
    return float(10 * np.log10(np.mean(x ** 2) + 1e-12))
