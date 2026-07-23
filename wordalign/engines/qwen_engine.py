"""Qwen3-ASR ensemble voter for word-align.

Qwen3-ASR-1.7B (transcription) paired with Qwen3-ForcedAligner-0.6B gives a
*timed* ensemble voter: real word-level timestamps, not just text. That makes
it a first-class challenger that can overturn a backbone error -- unlike a
text-only voter, which can only reinforce the backbone (see
``wordalign.ensemble``).

Qwen typically lives in its own virtualenv whose torch differs from the host's
(word-align's WhisperX stack), so by default this adapter calls Qwen
out-of-process: the bundled ``_qwen_worker.py`` runs inside that venv and
returns word segments as JSON. If ``qwen_asr`` happens to be importable in the
current interpreter, the worker just runs under it instead.

Configuration (CLI flags override these environment variables):
    WORDALIGN_QWEN_PYTHON   python executable of the venv that has ``qwen_asr``
    WORDALIGN_QWEN_MODELS   HuggingFace cache dir holding the Qwen models
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# The worker tags its JSON payload with this so we can find it amid any model
# or library chatter printed to stdout.
SENTINEL = "__WA_QWEN_JSON__"
_WORKER = Path(__file__).with_name("_qwen_worker.py")


def _parse_worker_output(stdout: str) -> Dict:
    """Pull the sentinel-tagged JSON payload out of worker stdout."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    raise ValueError("Qwen worker produced no JSON payload")


def _words_from_payload(payload: Dict) -> List[Dict]:
    """Normalize the payload into word-align word dicts, dropping untimed ones."""
    words: List[Dict] = []
    for w in payload.get("words", []):
        if w.get("start") is None or w.get("end") is None:
            continue
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        words.append({
            "word": text,
            "start": float(w["start"]),
            "end": float(w["end"]),
            "conf": float(w.get("conf", 0.9)),
        })
    return words


def run_qwen(audio_path: str,
             language: Optional[str] = None,
             qwen_python: Optional[str] = None,
             models_dir: Optional[str] = None,
             chunk_seconds: float = 60.0,
             asr_model: str = "Qwen/Qwen3-ASR-1.7B",
             aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B") -> List[Dict]:
    """Transcribe *audio_path* with Qwen and return timed word dicts.

    Returns an empty list (never raises) if Qwen is unavailable or fails, so
    the ensemble simply proceeds with its other voters.
    """
    print("\n" + "=" * 60 + "\nRUNNING QWEN3-ASR (ENSEMBLE VOTER)")
    qwen_python = qwen_python or os.environ.get("WORDALIGN_QWEN_PYTHON")
    models_dir = models_dir or os.environ.get("WORDALIGN_QWEN_MODELS")

    if not qwen_python:
        # No dedicated venv configured -- only workable if qwen_asr is present
        # in the interpreter we're already running under.
        try:
            import qwen_asr  # noqa: F401
        except Exception:
            print("[info] Qwen voter skipped: set WORDALIGN_QWEN_PYTHON to the "
                  "venv that has qwen_asr (or `pip install qwen-asr` here).")
            return []

    runner = qwen_python or sys.executable
    args = [runner, str(_WORKER), "--audio", str(audio_path),
            "--chunk-seconds", str(chunk_seconds),
            "--asr-model", asr_model, "--aligner-model", aligner_model]
    if language:
        args += ["--language", language]
    if models_dir:
        args += ["--models-dir", models_dir]

    try:
        proc = subprocess.run(args, capture_output=True, text=True)
    except OSError as exc:
        print(f"[warn] Qwen voter could not launch ({runner}): {exc}")
        return []

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        print(f"[warn] Qwen worker failed (exit {proc.returncode}):\n{tail}")
        return []

    try:
        payload = _parse_worker_output(proc.stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[warn] Qwen worker output unparseable: {exc}")
        return []

    words = _words_from_payload(payload)
    print(f"[ok] Qwen: {len(words)} timed words (lang={payload.get('language')}).")
    return words
