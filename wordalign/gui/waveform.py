"""Waveform peak generation for the GUI.

Generates cached min/max peak data from audio files for fast waveform rendering.
The output is a JSON file with downsampled peak pairs that can be rendered
on an HTML5 canvas.
"""
from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Optional


def generate_peaks(audio_path: str | Path,
                   samples_per_pixel: int = 441,
                   max_pixels: int = 2000) -> dict:
    """Generate min/max peak data from an audio file.

    Parameters
    ----------
    audio_path : str
        Path to a WAV file (other formats are decoded via ffmpeg).
    samples_per_pixel : int
        Audio samples per peak pair. 44100/100 ≈ 441 for 100px/sec.
    max_pixels : int
        Maximum number of peak pairs to generate.

    Returns
    -------
    dict with keys:
        - peaks: list of [min, max] pairs (-1.0 to 1.0)
        - duration_seconds: float
        - sample_rate: int
        - channels: int
    """
    path = Path(audio_path)

    # Try WAV first (fast, no ffmpeg needed)
    if path.suffix.lower() == ".wav":
        return _peaks_from_wav(str(path), samples_per_pixel, max_pixels)

    # For other formats, decode to WAV via ffmpeg
    wav_path = _decode_to_wav(path)
    try:
        return _peaks_from_wav(wav_path, samples_per_pixel, max_pixels)
    finally:
        if wav_path != str(path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def _peaks_from_wav(wav_path: str,
                    samples_per_pixel: int,
                    max_pixels: int) -> dict:
    """Read peaks from a WAV file."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        sample_rate = wf.getframerate()

        # Read all frames
        raw = wf.readframes(n_frames)

    # Convert to normalized samples
    if sample_width == 2:
        import struct
        # 16-bit signed
        samples = struct.unpack(f"<{n_frames * n_channels}h", raw)
        # Normalize to -1.0..1.0
        samples = [s / 32768.0 for s in samples]
    elif sample_width == 1:
        # 8-bit unsigned
        samples = [b / 128.0 - 1.0 for b in raw]
    else:
        # Unsupported bit depth, return empty
        return {"peaks": [], "duration_seconds": n_frames / sample_rate,
                "sample_rate": sample_rate, "channels": n_channels}

    # If stereo, average the channels
    if n_channels > 1:
        mono = []
        for i in range(0, len(samples), n_channels):
            mono.append(sum(samples[i:i+n_channels]) / n_channels)
        samples = mono

    # Downsample to peaks
    peaks = []
    for i in range(0, len(samples), samples_per_pixel):
        chunk = samples[i:i+samples_per_pixel]
        if not chunk:
            break
        peaks.append([min(chunk), max(chunk)])
        if len(peaks) >= max_pixels:
            break

    duration = n_frames / sample_rate

    return {
        "peaks": peaks,
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "channels": n_channels,
    }


def _decode_to_wav(path: Path) -> str:
    """Decode any audio file to 16-bit mono WAV via ffmpeg."""
    import subprocess
    import tempfile

    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(path), "-ac", "1", "-ar", "16000",
             "-sample_fmt", "s16", "-y", wav_path],
            capture_output=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return wav_path


def save_peaks(audio_path: str, output_path: Optional[str] = None) -> str:
    """Generate and save peaks to a JSON file. Returns the output path."""
    if output_path is None:
        base = os.path.splitext(audio_path)[0]
        output_path = f"{base}_peaks.json"

    peaks = generate_peaks(audio_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(peaks, f)
    return output_path
