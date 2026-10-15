"""Stage 1: the audio quality map.

Before any ASR runs, measure the audio so that every later stage -- cleanup
gating, engine input, hallucination filtering, review triage -- knows the
conditions of the stretch it is working on:

* speech probability every 32 ms (Silero VAD) and the speech segments;
* ASR chunks: speech grouped into <= 28 s pieces cut at pauses;
* per chunk: WADA-SNR, narrowband flag, clipping share;
* per frame (40 ms), when the sound-event model is installed: speech,
  music, crowd, noise, laughter and applause probabilities, plus timed
  events for SDH tags;
* overlap regions, filled in later by diarization.

Every part degrades gracefully: without the VAD model the whole file is one
speech region cut into fixed chunks; without the sound-event backend the
music/crowd fields stay ``None`` and cleanup gating relies on SNR alone.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from . import measures
from .io import SAMPLE_RATE, load_audio, write_wav
from .vad import (SileroVAD, VADUnavailable, find_vad_model,
                  max_prob, mean_prob, plan_asr_chunks, speech_segments)

Segment = Tuple[float, float]

# Condition thresholds. Group probabilities are the sound-event model's
# (median-filtered) frame scores averaged over a chunk.
MUSIC_THRESHOLD = 0.35
CROWD_THRESHOLD = 0.35
NOISE_THRESHOLD = 0.45
LOW_SNR_DB = 10.0
VERY_LOW_SNR_DB = 3.0
CLIPPING_THRESHOLD = 0.001
OVERLAP_THRESHOLD = 0.15


@dataclass
class ChunkInfo:
    """One ASR chunk and the conditions measured on it."""
    start: float
    end: float
    speech: float = 1.0                  # mean VAD probability
    snr_db: Optional[float] = None
    narrowband: bool = False
    clipping: float = 0.0
    music: Optional[float] = None
    crowd: Optional[float] = None
    noise: Optional[float] = None
    overlap: float = 0.0                 # share of the chunk with 2+ speakers
    cleanup: Optional[str] = None        # set by audio.prep

    @property
    def conditions(self) -> List[str]:
        out: List[str] = []
        if self.music is not None and self.music >= MUSIC_THRESHOLD:
            out.append("music")
        if self.crowd is not None and self.crowd >= CROWD_THRESHOLD:
            out.append("crowd")
        if self.snr_db is not None and self.snr_db == self.snr_db:
            if self.snr_db < VERY_LOW_SNR_DB:
                out.append("very_low_snr")
            elif self.snr_db < LOW_SNR_DB:
                out.append("low_snr")
        if self.noise is not None and self.noise >= NOISE_THRESHOLD:
            out.append("noise")
        if self.narrowband:
            out.append("narrowband")
        if self.clipping >= CLIPPING_THRESHOLD:
            out.append("clipping")
        if self.overlap >= OVERLAP_THRESHOLD:
            out.append("overlap")
        return out

    def to_dict(self) -> Dict[str, Any]:
        def r(v, n=3):
            return None if v is None or v != v else round(float(v), n)
        return {"start": r(self.start), "end": r(self.end),
                "speech": r(self.speech), "snr_db": r(self.snr_db, 1),
                "narrowband": self.narrowband, "clipping": r(self.clipping, 5),
                "music": r(self.music), "crowd": r(self.crowd),
                "noise": r(self.noise), "overlap": r(self.overlap),
                "cleanup": self.cleanup, "conditions": self.conditions}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChunkInfo":
        return cls(start=d["start"], end=d["end"], speech=d.get("speech", 1.0),
                   snr_db=d.get("snr_db"), narrowband=d.get("narrowband", False),
                   clipping=d.get("clipping", 0.0), music=d.get("music"),
                   crowd=d.get("crowd"), noise=d.get("noise"),
                   overlap=d.get("overlap", 0.0), cleanup=d.get("cleanup"))


@dataclass
class QualityMap:
    """Everything stage 1 learned about the audio."""
    duration: float
    vad_probs: Optional[np.ndarray] = None
    speech: List[Segment] = field(default_factory=list)
    chunks: List[ChunkInfo] = field(default_factory=list)
    sed_hop: float = 0.04
    sed_groups: Dict[str, np.ndarray] = field(default_factory=dict)
    sed_events: List[Dict[str, Any]] = field(default_factory=list)
    overlaps: List[Segment] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- queries -------------------------------------------------------
    @property
    def has_vad(self) -> bool:
        return self.vad_probs is not None and len(self.vad_probs) > 0

    @property
    def has_sed(self) -> bool:
        return bool(self.sed_groups)

    def speech_mean(self, start: float, end: float) -> Optional[float]:
        if not self.has_vad:
            return None
        return mean_prob(self.vad_probs, start, end)

    def speech_max(self, start: float, end: float) -> Optional[float]:
        if not self.has_vad:
            return None
        return max_prob(self.vad_probs, start, end)

    def group_mean(self, name: str, start: float, end: float) -> Optional[float]:
        arr = self.sed_groups.get(name)
        if arr is None or len(arr) == 0:
            return None
        i0 = max(0, int(start / self.sed_hop))
        i1 = min(len(arr), max(i0 + 1, int(np.ceil(end / self.sed_hop))))
        if i0 >= len(arr):
            return None
        return float(np.mean(arr[i0:i1].astype(np.float32)))

    def chunk_at(self, t: float) -> Optional[ChunkInfo]:
        for c in self.chunks:
            if c.start <= t < c.end:
                return c
        return None

    def overlap_share(self, start: float, end: float) -> float:
        span = max(end - start, 1e-6)
        covered = sum(max(0.0, min(end, e) - max(start, s))
                      for s, e in self.overlaps)
        return min(1.0, covered / span)

    def conditions_at(self, start: float, end: float) -> List[str]:
        """Conditions of the chunks overlapping ``[start, end]`` (union)."""
        out: List[str] = []
        for c in self.chunks:
            if c.end > start and c.start < end:
                for cond in c.conditions:
                    if cond not in out:
                        out.append(cond)
        if "overlap" not in out and self.overlap_share(start, end) >= OVERLAP_THRESHOLD:
            out.append("overlap")
        return out

    def summary(self) -> Dict[str, Any]:
        snrs = [c.snr_db for c in self.chunks if c.snr_db is not None and c.snr_db == c.snr_db]
        cond_time: Dict[str, float] = {}
        for c in self.chunks:
            for cond in c.conditions:
                cond_time[cond] = cond_time.get(cond, 0.0) + (c.end - c.start)
        speech_time = sum(e - s for s, e in self.speech)
        return {
            "duration": round(self.duration, 3),
            "speech_seconds": round(speech_time, 1),
            "chunks": len(self.chunks),
            "median_snr_db": round(float(np.median(snrs)), 1) if snrs else None,
            "condition_seconds": {k: round(v, 1) for k, v in sorted(cond_time.items())},
            "vad": self.has_vad,
            "sound_events": self.has_sed,
            "overlap_seconds": round(sum(e - s for s, e in self.overlaps), 1),
            "notes": list(self.notes),
        }

    # -- (de)serialisation for the stage cache ---------------------------
    def to_cache(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "vad_probs": (np.round(self.vad_probs, 3).tolist()
                          if self.has_vad else None),
            "speech": [list(s) for s in self.speech],
            "chunks": [c.to_dict() for c in self.chunks],
            "sed_hop": self.sed_hop,
            "sed_groups": {k: np.round(v.astype(np.float32), 3).tolist()
                           for k, v in self.sed_groups.items()},
            "sed_events": self.sed_events,
            "notes": self.notes,
        }

    @classmethod
    def from_cache(cls, d: Dict[str, Any]) -> "QualityMap":
        probs = d.get("vad_probs")
        return cls(
            duration=float(d["duration"]),
            vad_probs=np.asarray(probs, dtype=np.float32) if probs is not None else None,
            speech=[tuple(s) for s in d.get("speech", [])],
            chunks=[ChunkInfo.from_dict(c) for c in d.get("chunks", [])],
            sed_hop=float(d.get("sed_hop", 0.04)),
            sed_groups={k: np.asarray(v, dtype=np.float16)
                        for k, v in d.get("sed_groups", {}).items()},
            sed_events=list(d.get("sed_events", [])),
            notes=list(d.get("notes", [])),
        )


def fixed_chunks(duration: float, max_len: float = 28.0) -> List[Segment]:
    n = max(1, int(np.ceil(duration / max_len)))
    step = duration / n
    return [(round(i * step, 3), round(min(duration, (i + 1) * step), 3))
            for i in range(n)]


def _measure_chunk(audio: np.ndarray, start: float, end: float) -> Dict[str, Any]:
    i0, i1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
    piece = audio[i0:i1]
    snr = measures.wada_snr(piece)
    return {"snr_db": None if snr != snr else snr,
            "narrowband": measures.is_narrowband(piece),
            "clipping": measures.clipping_ratio(piece)}


def run_sed(wav_path: str, sed_dir: str, python: str, workdir: str,
            cancel_event=None, model: str = "BEATs") -> Tuple[float, Dict[str, np.ndarray], List[Dict]]:
    """Run the sound-event worker; returns (hop, groups, events)."""
    from ..engines._worker import run_worker
    worker = Path(__file__).resolve().parent.parent / "engines" / "_sed_worker.py"
    out = os.path.join(workdir, "sed_groups.npz")
    payload = run_worker(python, worker,
                         ["--audio", wav_path, "--sed-dir", sed_dir,
                          "--model", model, "--out", out],
                         "__WA_SED_JSON__", cancel_event=cancel_event,
                         name="sound-event worker")
    with np.load(payload.get("groups_file", out)) as data:
        hop = float(data["hop"])
        groups = {k: data[k] for k in data.files if k != "hop"}
    return hop, groups, list(payload.get("events", []))


def find_sed_dir(models_dir: Optional[str]) -> Optional[str]:
    env = os.environ.get("WORDALIGN_SED_DIR")
    if env and os.path.isdir(env):
        return env
    if models_dir:
        cand = os.path.join(models_dir, "sed", "PretrainedSED")
        if os.path.isfile(os.path.join(cand, "resources", "BEATs_strong_1.pt")):
            return cand
    return None


def analyze_audio(audio_path: str, models_dir: Optional[str] = None,
                  use_vad: bool = True, use_sed: bool = True,
                  vad_threshold: float = 0.5, max_chunk_s: float = 28.0,
                  workdir: Optional[str] = None, cancel_event=None,
                  log: Callable[[str], None] = print,
                  audio: Optional[np.ndarray] = None) -> QualityMap:
    """Build the :class:`QualityMap` for *audio_path*."""
    if audio is None:
        audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    qmap = QualityMap(duration=duration)

    # -- speech ------------------------------------------------------------
    vad_model = find_vad_model(models_dir) if use_vad else None
    if vad_model:
        try:
            probs = SileroVAD(vad_model).probabilities(audio, cancel_event=cancel_event)
            qmap.vad_probs = probs
            qmap.speech = speech_segments(probs, threshold=vad_threshold,
                                          duration=duration)
            qmap.chunks = [ChunkInfo(s, e) for s, e in plan_asr_chunks(
                qmap.speech, probs, duration, max_len=max_chunk_s)]
            log(f"VAD: {sum(e - s for s, e in qmap.speech):.1f}s of speech in "
                f"{len(qmap.speech)} segments; {len(qmap.chunks)} ASR chunks")
        except VADUnavailable as exc:
            qmap.notes.append(f"vad unavailable: {exc}")
    elif use_vad:
        qmap.notes.append("vad model not installed (models/vad/silero_vad.onnx)")
    if not qmap.has_vad:
        qmap.speech = [(0.0, duration)]
        qmap.chunks = [ChunkInfo(s, e) for s, e in fixed_chunks(duration, max_chunk_s)]

    for c in qmap.chunks:
        m = _measure_chunk(audio, c.start, c.end)
        c.snr_db, c.narrowband, c.clipping = m["snr_db"], m["narrowband"], m["clipping"]
        if qmap.has_vad:
            c.speech = mean_prob(qmap.vad_probs, c.start, c.end)

    # -- sound events ----------------------------------------------------------
    if use_sed:
        from ..backends import backend_python
        sed_dir = find_sed_dir(models_dir)
        python = backend_python("sed")
        if not sed_dir:
            qmap.notes.append("sound-event model not installed (models/sed/PretrainedSED)")
        elif not python:
            qmap.notes.append("no venv for the sound-event model (whisperx venv)")
        else:
            own_dir = workdir is None
            wd = workdir or tempfile.mkdtemp(prefix="wordalign_sed_")
            try:
                wav = os.path.join(wd, "analysis_16k.wav")
                if not os.path.exists(wav):
                    write_wav(wav, audio)
                hop, groups, events = run_sed(wav, sed_dir, python, wd,
                                              cancel_event=cancel_event)
                qmap.sed_hop, qmap.sed_groups, qmap.sed_events = hop, groups, events
                for c in qmap.chunks:
                    c.music = qmap.group_mean("music", c.start, c.end)
                    c.crowd = qmap.group_mean("crowd", c.start, c.end)
                    c.noise = qmap.group_mean("noise", c.start, c.end)
                log(f"Sound events: {len(events)} timed events")
            except Exception as exc:   # worker failure must not stop the job
                qmap.notes.append(f"sound-event detection failed: {exc}")
            finally:
                if own_dir:
                    import shutil
                    shutil.rmtree(wd, ignore_errors=True)
    return qmap
