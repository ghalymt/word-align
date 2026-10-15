"""Chunk-based ASR voters for transcript-free mode.

Every voter transcribes the same VAD speech chunks, so their outputs can be
fused chunk by chunk with ROVER (:mod:`wordalign.rover`). Each runner here
returns ``{chunk_index: [word dicts]}`` for the chunks it processed (an
empty list means "ran and heard nothing"; a missing index means "failed
there, do not vote"), or None when the voter is not installed or failed
outright.

* ``cohere``   -- Cohere Transcribe 03-2026 (text only; most robust on
                  far-field noise), venv ``cohere``.
* ``parakeet`` -- parakeet-ultra via Moondream Photon (word timestamps),
                  venv ``parakeet``.
* ``qwen``     -- Qwen3-ASR-1.7B + ForcedAligner (word timestamps), venv
                  ``qwen``; see :mod:`wordalign.engines.qwen_engine`.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..backends import backend_python
from ._worker import WorkerCancelled, WorkerError, run_worker

Chunk = Tuple[float, float]
ChunkWords = Dict[int, List[Dict]]

_HERE = Path(__file__).resolve().parent
VOTER_LANGUAGES = {
    "cohere": {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl",
               "pl", "pt", "vi", "zh"},
    # parakeet-tdt-0.6b-v3: 25 European languages.
    "parakeet": {"bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
                 "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
                 "sl", "es", "sv", "ru", "uk"},
}


def voter_available(name: str, language: str) -> Tuple[bool, str]:
    """(available, reason) for voter *name* on *language*."""
    lang = (language or "en").split("-")[0].lower()
    langs = VOTER_LANGUAGES.get(name)
    if langs is not None and lang not in langs:
        return False, f"{name} does not support '{lang}'"
    if name in ("cohere", "parakeet") and not backend_python(name):
        return False, f"{name} venv not installed (scripts/install_robust.ps1)"
    return True, ""


def _chunks_file(chunks: Sequence[Chunk]) -> str:
    fd, path = tempfile.mkstemp(prefix="wa_chunks_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump([[round(float(s), 3), round(float(e), 3)] for s, e in chunks], f)
    return path


def _split_text(text: str) -> List[Dict]:
    return [{"word": w} for w in text.split() if w.strip()]


def run_cohere(audio_path: str, chunks: Sequence[Chunk], language: str,
               models_dir: Optional[str] = None, cancel_event=None,
               log=print) -> Optional[ChunkWords]:
    python = backend_python("cohere")
    if not python:
        return None
    path = _chunks_file(chunks)
    try:
        args = ["--audio", audio_path, "--chunks-json", path,
                "--language", (language or "en").split("-")[0].lower()]
        if models_dir:
            args += ["--models-dir", models_dir]
        payload = run_worker(python, _HERE / "_cohere_worker.py", args,
                             "__WA_COHERE_JSON__", cancel_event=cancel_event,
                             name="Cohere Transcribe")
    except WorkerCancelled:
        raise
    except WorkerError as exc:
        log(f"[warn] {exc}")
        return None
    finally:
        os.remove(path)
    if payload.get("unsupported_language"):
        return None
    out: ChunkWords = {}
    for i, c in enumerate(payload.get("chunks", [])):
        if c.get("status") == "ok":
            out[i] = _split_text(c.get("text", ""))
    return out


def run_parakeet(audio_path: str, chunks: Sequence[Chunk],
                 models_dir: Optional[str] = None, cancel_event=None,
                 log=print) -> Optional[ChunkWords]:
    python = backend_python("parakeet")
    if not python:
        return None
    path = _chunks_file(chunks)
    try:
        args = ["--audio", audio_path, "--chunks-json", path]
        if models_dir:
            args += ["--models-dir", models_dir]
        payload = run_worker(python, _HERE / "_parakeet_worker.py", args,
                             "__WA_PARAKEET_JSON__", cancel_event=cancel_event,
                             name="parakeet-ultra")
    except WorkerCancelled:
        raise
    except WorkerError as exc:
        log(f"[warn] {exc}")
        return None
    finally:
        os.remove(path)
    out: ChunkWords = {}
    for i, c in enumerate(payload.get("chunks", [])):
        if c.get("status") == "ok":
            out[i] = list(c.get("words", []))
    return out


def chunks_from_qwen_payload(payload: Optional[Dict], n_chunks: int) -> Optional[ChunkWords]:
    if not payload:
        return None
    out: ChunkWords = {}
    for i, c in enumerate(payload.get("chunks", [])[:n_chunks]):
        if c.get("status") == "ok":
            out[i] = [w for w in c.get("words", [])
                      if w.get("start") is not None and str(w.get("word", "")).strip()]
    return out


def bucket_timed_words(words: Sequence[Dict], chunks: Sequence[Chunk],
                       tolerance: float = 0.5) -> Tuple[ChunkWords, List[Dict]]:
    """Assign whole-file timed words to chunks by midpoint.

    Returns ``(per_chunk, orphans)``; orphans are words outside every chunk
    (beyond *tolerance*), i.e. where the VAD heard no speech.
    """
    per: ChunkWords = {i: [] for i in range(len(chunks))}
    orphans: List[Dict] = []
    for w in words:
        if w.get("start") is None or w.get("end") is None:
            continue
        mid = (float(w["start"]) + float(w["end"])) / 2
        best, best_d = None, None
        for i, (s, e) in enumerate(chunks):
            d = 0.0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
            if best_d is None or d < best_d:
                best, best_d = i, d
            if d == 0.0:
                break
        if best is not None and best_d <= tolerance:
            per[best].append(w)
        else:
            orphans.append(w)
    return per, orphans
