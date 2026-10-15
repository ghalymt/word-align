"""Stage 2: clean only where it pays, into a second track for the ASR.

Published tests disagree with blanket enhancement: MetricGAN+ raised WER in
all 40 tested set-ups, SAM-Audio separation raised Whisper's WER in every
configuration, and music separation helped only on segments around -5 to
5-8 dB SNR. Of six enhancers, only MossFormerGAN lowered Whisper's WER, and
weak correction beat full strength. So cleanup here is gated per chunk and
under-applied:

* music under speech, SNR below 8 dB  -> RoFormer vocal separation;
* crowd under speech, SNR below 3 dB  -> Mel-Band RoFormer crowd removal
  (no ASR study yet, so only for very loud crowds unless ``mode="all"``);
* low SNR without music or crowd      -> MossFormerGAN, mixed back with the
  original (``denoise_mix``, default 0.5);
* narrowband or clipped audio         -> nothing (generative restorers can
  change words); triage raises the cue's risk instead.

The result is ``asr_audio``: a copy of the original where only the planned
chunks are replaced, with 50 ms crossfades. Untouched stretches stay
sample-identical, and word timing always reads the original audio.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .analysis import CROWD_THRESHOLD, MUSIC_THRESHOLD, QualityMap
from .io import SAMPLE_RATE, read_wav, write_wav

MUSIC_SNR_MAX = 8.0
CROWD_SNR_MAX = 3.0
DENOISE_SNR_MAX = 10.0
CONTEXT_S = 1.0          # extra audio around a chunk for the models
CROSSFADE_S = 0.05

VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
CROWD_MODEL = "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"

Plan = List[Tuple[int, str]]


def _snr(c) -> Optional[float]:
    return c.snr_db if c.snr_db is not None and c.snr_db == c.snr_db else None


def plan_cleanup(qmap: QualityMap, mode: str = "auto") -> Plan:
    """``[(chunk_index, op)]`` with op in vocals / crowd / denoise."""
    if mode == "off":
        return []
    plan: Plan = []
    for i, c in enumerate(qmap.chunks):
        conds = c.conditions
        if "narrowband" in conds or "clipping" in conds:
            continue
        snr = _snr(c)
        music = c.music is not None and c.music >= MUSIC_THRESHOLD
        crowd = c.crowd is not None and c.crowd >= CROWD_THRESHOLD
        if mode == "all":
            if music:
                plan.append((i, "vocals"))
            elif crowd:
                plan.append((i, "crowd"))
            elif snr is not None and snr < DENOISE_SNR_MAX:
                plan.append((i, "denoise"))
            continue
        if music and snr is not None and snr < MUSIC_SNR_MAX:
            plan.append((i, "vocals"))
        elif crowd and snr is not None and snr < CROWD_SNR_MAX:
            plan.append((i, "crowd"))
        elif not music and not crowd and snr is not None and snr < DENOISE_SNR_MAX:
            plan.append((i, "denoise"))
    return plan


def _crossfade_insert(out: np.ndarray, piece: np.ndarray, start: int) -> None:
    """Write *piece* into *out* at *start* with linear crossfades at both ends."""
    end = min(len(out), start + len(piece))
    piece = piece[:end - start]
    fade = min(int(CROSSFADE_S * SAMPLE_RATE), len(piece) // 4)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        piece = piece.copy()
        piece[:fade] = out[start:start + fade] * (1 - ramp) + piece[:fade] * ramp
        piece[-fade:] = out[end - fade:end] * ramp + piece[-fade:] * (1 - ramp)
    out[start:end] = piece


def prepare_asr_audio(audio: np.ndarray, qmap: QualityMap, plan: Plan,
                      workdir: str, models_dir: Optional[str] = None,
                      denoise_mix: float = 0.5, separation_mix: float = 0.9,
                      cancel_event=None,
                      log: Callable[[str], None] = print) -> Tuple[str, List[Tuple[float, float, str]]]:
    """Build ``asr_audio.wav``; returns (path, applied [(start, end, op)])."""
    from ..backends import backend_python
    from ..engines._worker import WorkerCancelled, WorkerError, run_worker
    base = Path(workdir) / "cleanup"
    (base / "in").mkdir(parents=True, exist_ok=True)
    (base / "out").mkdir(parents=True, exist_ok=True)
    jobs: Dict[str, List[Dict]] = {"vocals": [], "crowd": [], "denoise": []}
    spans: Dict[str, Tuple[int, int, int, str]] = {}
    for i, op in plan:
        c = qmap.chunks[i]
        s = max(0, int((c.start - CONTEXT_S) * SAMPLE_RATE))
        e = min(len(audio), int((c.end + CONTEXT_S) * SAMPLE_RATE))
        cs, ce = int(c.start * SAMPLE_RATE), int(c.end * SAMPLE_RATE)
        job_id = f"c{i:05d}_{op}"
        src = str(base / "in" / f"{job_id}.wav")
        write_wav(src, audio[s:e])
        jobs[op].append({"id": job_id, "in": src,
                         "out": str(base / "out" / f"{job_id}.wav")})
        spans[job_id] = (s, cs, ce, op)
    applied: List[Tuple[float, float, str]] = []
    results: Dict[str, str] = {}
    models_dir = models_dir or ""
    sep_jobs = [dict(j, model=VOCALS_MODEL) for j in jobs["vocals"]] + \
               [dict(j, model=CROWD_MODEL) for j in jobs["crowd"]]
    if sep_jobs:
        python = backend_python("separate")
        if not python:
            log("Separation venv not installed; music/crowd chunks left as they are.")
        else:
            spec = base / "separate.json"
            spec.write_text(json.dumps(sep_jobs), encoding="utf-8")
            try:
                payload = run_worker(
                    python, Path(__file__).resolve().parent.parent / "engines" / "_separate_worker.py",
                    ["--jobs-json", str(spec), "--model-dir",
                     os.path.join(models_dir, "separation")],
                    "__WA_SEPARATE_JSON__", cancel_event=cancel_event,
                    name="separation worker")
                results.update({j["id"]: j["out"] for j in payload.get("done", [])})
                for f in payload.get("failed", []):
                    log(f"separation failed on {f.get('id')}: {f.get('error')}")
            except WorkerCancelled:
                raise
            except WorkerError as exc:
                log(f"Separation failed: {exc}")
    if jobs["denoise"]:
        python = backend_python("denoise")
        if not python:
            log("MossFormerGAN venv not installed; noisy chunks left as they are.")
        else:
            spec = base / "denoise.json"
            spec.write_text(json.dumps(jobs["denoise"]), encoding="utf-8")
            try:
                payload = run_worker(
                    python, Path(__file__).resolve().parent.parent / "engines" / "_denoise_worker.py",
                    ["--jobs-json", str(spec), "--model-dir",
                     os.path.join(models_dir, "denoise")],
                    "__WA_DENOISE_JSON__", cancel_event=cancel_event,
                    name="denoise worker")
                results.update({j["id"]: j["out"] for j in payload.get("done", [])})
                for f in payload.get("failed", []):
                    log(f"denoise failed on {f.get('id')}: {f.get('error')}")
            except WorkerCancelled:
                raise
            except WorkerError as exc:
                log(f"Denoising failed: {exc}")
    out = np.array(audio, dtype=np.float32, copy=True)
    for job_id, path in sorted(results.items()):
        s, cs, ce, op = spans[job_id]
        try:
            processed = read_wav(path)
        except Exception as exc:
            log(f"could not read {path}: {exc}")
            continue
        orig = audio[s:s + len(processed)]
        processed = processed[:len(orig)]
        mix = denoise_mix if op == "denoise" else separation_mix
        blended = (mix * processed + (1.0 - mix) * orig).astype(np.float32)
        # Keep only the chunk itself (drop the context padding).
        piece = blended[cs - s:ce - s]
        _crossfade_insert(out, piece, cs)
        applied.append((cs / SAMPLE_RATE, ce / SAMPLE_RATE, op))
        qmap.chunks[int(job_id[1:6])].cleanup = op
    path = str(Path(workdir) / "asr_audio.wav")
    write_wav(path, out)
    return path, applied
