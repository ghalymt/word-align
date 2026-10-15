"""Qwen3-ASR voter and Qwen3-ForcedAligner timing for word-align.

Qwen3-ASR-1.7B (transcription) paired with Qwen3-ForcedAligner-0.6B gives a
*timed* voter: real word-level timestamps, not just text. The aligner on its
own is word-align's main timing engine: it times known text (a human
transcript, or the voted transcript) about twice as tightly as WhisperX's
wav2vec2 in Qwen's published comparison, and its average shift only rose
from 27.8 to 41.8 ms on noisy human-labelled speech.

Qwen typically lives in its own virtualenv whose torch differs from the host's
(word-align's WhisperX stack), so by default this adapter calls Qwen
out-of-process: the bundled ``_qwen_worker.py`` runs inside that venv and
returns JSON. If ``qwen_asr`` happens to be importable in the current
interpreter, the worker just runs under it instead.

Configuration (CLI flags override these environment variables):
    WORDALIGN_QWEN_PYTHON   python executable of the venv that has ``qwen_asr``
    WORDALIGN_QWEN_MODELS   HuggingFace cache dir holding the Qwen models
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# The worker tags its JSON payload with this so we can find it amid any model
# or library chatter printed to stdout.
SENTINEL = "__WA_QWEN_JSON__"
_WORKER = Path(__file__).with_name("_qwen_worker.py")

# Languages Qwen3-ForcedAligner can time (ISO 639-1 / word-align codes).
ALIGNER_LANGUAGES = {"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt",
                     "ru", "es"}


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


def _resolve_python(qwen_python: Optional[str]) -> Optional[str]:
    """The interpreter to run the worker under, or None (with a message)."""
    qwen_python = qwen_python or os.environ.get("WORDALIGN_QWEN_PYTHON")
    if qwen_python:
        return qwen_python
    from ..backends import backend_python
    found = backend_python("qwen")
    if found:
        return found
    if getattr(sys, "frozen", False):
        print("[info] Qwen skipped: the frozen build has no dedicated "
              "Qwen venv; set WORDALIGN_QWEN_PYTHON to enable it.")
        return None
    try:
        import qwen_asr  # noqa: F401
    except Exception:
        print("[info] Qwen skipped: set WORDALIGN_QWEN_PYTHON to the "
              "venv that has qwen_asr (or `pip install qwen-asr` here).")
        return None
    return sys.executable


def _run(args: List[str], runner: str, cancel_event=None) -> Optional[Dict]:
    try:
        # UTF-8 both ways: on Windows a piped child otherwise writes cp1252
        # and dies on the first non-Latin path or log line, and the parent
        # would mis-decode whatever it did write.
        proc = subprocess.Popen(
            [runner, str(_WORKER), *args], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
            env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        while True:
            try:
                stdout, stderr = proc.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    print("[info] Qwen worker cancelled.")
                    return None
    except OSError as exc:
        print(f"[warn] Qwen could not launch ({runner}): {exc}")
        return None
    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-5:])
        print(f"[warn] Qwen worker failed (exit {proc.returncode}):\n{tail}")
        return None
    try:
        return _parse_worker_output(stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[warn] Qwen worker output unparseable: {exc}")
        return None


def run_qwen(audio_path: str,
             language: Optional[str] = None,
             qwen_python: Optional[str] = None,
             models_dir: Optional[str] = None,
             chunk_seconds: float = 60.0,
             asr_model: str = "Qwen/Qwen3-ASR-1.7B",
             aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
             cancel_event=None,
             chunks: Optional[Sequence[Tuple[float, float]]] = None,
             return_payload: bool = False):
    """Transcribe *audio_path* with Qwen and return timed word dicts.

    With *chunks* (``[(start, end), ...]`` speech chunks from the VAD) only
    those stretches are transcribed. With ``return_payload=True`` the whole
    worker payload is returned (``chunks``, ``failed``, ``language``) or
    None on failure; otherwise a word list that is empty on failure.
    Never raises, so the ensemble simply proceeds with its other voters.
    """
    print("\n" + "=" * 60 + "\nRUNNING QWEN3-ASR (ENSEMBLE VOTER)")
    empty = None if return_payload else []
    if cancel_event is not None and cancel_event.is_set():
        return empty
    runner = _resolve_python(qwen_python)
    if not runner:
        return empty
    models_dir = models_dir or os.environ.get("WORDALIGN_QWEN_MODELS")
    args = ["--audio", str(audio_path), "--chunk-seconds", str(chunk_seconds),
            "--asr-model", asr_model, "--aligner-model", aligner_model]
    if language:
        args += ["--language", language]
    if models_dir:
        args += ["--models-dir", models_dir]
    tmp = None
    try:
        if chunks:
            fd, tmp = tempfile.mkstemp(prefix="wa_qwen_chunks_", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump([[float(s), float(e)] for s, e in chunks], f)
            args += ["--chunks-json", tmp]
        payload = _run(args, runner, cancel_event)
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
    if payload is None:
        return empty
    words = _words_from_payload(payload)
    failed = payload.get("failed") or []
    if failed:
        print(f"[warn] Qwen: {len(failed)} piece(s) ran out of GPU memory: {failed}")
    print(f"[ok] Qwen: {len(words)} timed words (lang={payload.get('language')}).")
    if return_payload:
        payload["words"] = words
        return payload
    return words


def qwen_align(audio_path: str, windows: List[Dict], language: str,
               qwen_python: Optional[str] = None,
               models_dir: Optional[str] = None,
               aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
               cancel_event=None) -> Optional[List[Dict]]:
    """Force-align known words in *windows* with Qwen3-ForcedAligner.

    *windows*: ``[{"id", "start", "end", "words": [str, ...]}, ...]``.
    Returns the worker's windows (each with ``times``: ``[start, end]`` or
    None per word, and ``status``), or None if Qwen is unavailable or the
    language is not one the aligner supports.
    """
    if (language or "en").split("-")[0].lower() not in ALIGNER_LANGUAGES:
        return None
    if not windows or (cancel_event is not None and cancel_event.is_set()):
        return [] if not windows else None
    runner = _resolve_python(qwen_python)
    if not runner:
        return None
    models_dir = models_dir or os.environ.get("WORDALIGN_QWEN_MODELS")
    fd, tmp = tempfile.mkstemp(prefix="wa_qwen_align_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(windows, f, ensure_ascii=False)
        args = ["--mode", "align", "--audio", str(audio_path),
                "--align-json", tmp, "--aligner-model", aligner_model,
                "--language", language or "en"]
        if models_dir:
            args += ["--models-dir", models_dir]
        payload = _run(args, runner, cancel_event)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if payload is None:
        return None
    if payload.get("unsupported_language"):
        return None
    return list(payload.get("windows", []))
