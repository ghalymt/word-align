"""Decode and write 16 kHz mono audio with ffmpeg and the standard library.

Everything in the bad-audio stages works on 16 kHz mono float32 arrays, the
rate every ASR model here expects. Decoding goes through ffmpeg (already a
hard requirement of word-align) so any media file works; writing uses the
``wave`` module, so no extra dependency is needed in the main app.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import wave
from typing import Optional

import numpy as np

SAMPLE_RATE = 16_000


class AudioDecodeError(RuntimeError):
    """ffmpeg is missing or could not decode the file."""


def ffmpeg_binary() -> Optional[str]:
    path = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    return path if path and os.path.isfile(path) else (path or None)


def load_audio(path: str, sample_rate: int = SAMPLE_RATE,
               start: Optional[float] = None,
               duration: Optional[float] = None) -> np.ndarray:
    """Decode *path* to mono float32 in [-1, 1] at *sample_rate*.

    ``start``/``duration`` (seconds) decode only part of the file.
    """
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise AudioDecodeError("ffmpeg not found (set WORDALIGN_FFMPEG)")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    cmd += ["-i", path]
    if duration is not None:
        cmd += ["-t", f"{max(0.0, duration):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    except OSError as exc:
        raise AudioDecodeError(f"could not run ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise AudioDecodeError(detail[-1] if detail else
                               f"ffmpeg exit {proc.returncode}")
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if audio.size == 0:
        raise AudioDecodeError("no audio samples decoded")
    return audio


def write_wav(path: str, audio: np.ndarray,
              sample_rate: int = SAMPLE_RATE) -> str:
    """Write *audio* (float, [-1, 1]) as 16-bit PCM mono WAV; returns *path*."""
    data = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (data * 32767.0).round().astype("<i2")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.part"
    with wave.open(tmp, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    os.replace(tmp, path)
    return path


def read_wav(path: str) -> np.ndarray:
    """Read a 16-bit PCM mono WAV written by :func:`write_wav`."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            raise AudioDecodeError(f"{path}: expected 16-bit PCM")
        frames = wf.readframes(wf.getnframes())
        channels = wf.getnchannels()
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32767.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm


def seconds_to_samples(t: float, sample_rate: int = SAMPLE_RATE) -> int:
    return int(round(t * sample_rate))
