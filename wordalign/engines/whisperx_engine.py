"""WhisperX: Whisper large-v3 transcription + wav2vec2 forced alignment.

Uses language-specific wav2vec2 checkpoints where a stronger one than
WhisperX's default is known, falling back to WhisperX auto-selection.

WhisperX runs out-of-process via :mod:`wordalign.engines._whisperx_worker`,
because the whisperx stack (whisperx, faster-whisper, ctranslate2, optional
pyannote) has version pins that clash with the rest of word-align's
dependencies. The worker is invoked in the venv pointed to by
``WORDALIGN_WHISPERX_PYTHON`` (or it falls back to in-process when
``whisperx`` happens to be importable from the current interpreter).

Returns ``(word_segments, audio_events)`` matching the legacy in-process
contract so the pipeline is agnostic to where WhisperX actually ran.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ALIGNMENT_MODEL_MAP = {
    "en": "facebook/wav2vec2-large-960h-lv60-self",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    "vi": "nguyenvulebinh/wav2vec2-large-vi-vlsp2020",
    "tl": "khaderb/wav2vec2-large-xlsr-53-tagalog",
    "tr": "mpoyraz/wav2vec2-large-xlsr-53-turkish",
    "ca": "softcatala/wav2vec2-large-xlsr-53-catalan",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "ro": "anton-l/wav2vec2-large-xlsr-53-romanian",
    "lt": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "hr": "facebook/wav2vec2-large-xlsr-53-croatian",
    "sr": "facebook/wav2vec2-large-xlsr-53-croatian",
    "bs": "facebook/wav2vec2-large-xlsr-53-croatian",
    "default": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
}

_EVENT_RE = re.compile(r"\[(.*?)\]|\((.*?)\)")

SENTINEL = "__WA_WHISPERX_JSON__"
_WORKER = Path(__file__).with_name("_whisperx_worker.py")


def _parse_worker_output(stdout: str) -> Dict:
    """Pull the sentinel-tagged JSON payload out of worker stdout."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(SENTINEL):
            return json.loads(line[len(SENTINEL):])
    raise ValueError("WhisperX worker produced no JSON payload")


def _words_from_payload(payload: Dict) -> List[Dict]:
    """Normalize the payload into word-align word dicts, dropping untimed ones."""
    out: List[Dict] = []
    for w in payload.get("words", []):
        if w.get("start") is None or w.get("end") is None:
            continue
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        out.append({
            "word": text,
            "start": float(w["start"]),
            "end": float(w["end"]),
            "conf": float(w.get("conf", 0.9)),
        })
    return out


def run_whisperx(audio_path: str,
                 language: str,
                 model_size: str = "large-v3",
                 device: Optional[str] = None,
                 models_dir: Optional[str] = None,
                 cancel_event=None
                 ) -> Tuple[List[Dict], List[Dict]]:
    """Return ``(word_segments, audio_events)``.

    Prefers the out-of-process worker (env ``WORDALIGN_WHISPERX_PYTHON``).
    Falls back to in-process only if ``whisperx`` is importable in the
    current interpreter and no separate venv is configured. Returns
    ``([], [])`` (never raises) on any failure, so the ensemble simply
    proceeds without WhisperX.
    """
    print("\n" + "=" * 60 + "\nRUNNING WHISPERX (HIGH ACCURACY MODE)")
    if cancel_event is not None and cancel_event.is_set():
        return [], []

    whisperx_python = os.environ.get("WORDALIGN_WHISPERX_PYTHON")
    models_dir = models_dir or os.environ.get("WORDALIGN_WHISPERX_MODELS")

    if not whisperx_python:
        if getattr(sys, "frozen", False):
            print("[info] WhisperX skipped: the frozen build has no dedicated "
                  "WhisperX venv; set WORDALIGN_WHISPERX_PYTHON to enable it.")
            return [], []
        # No dedicated venv configured -- only workable if whisperx happens
        # to be importable in the interpreter we're already running under.
        try:
            import whisperx  # noqa: F401
            return _run_whisperx_inproc(audio_path, language, model_size,
                                       device, models_dir, cancel_event)
        except ImportError:
            print("[info] WhisperX skipped: set WORDALIGN_WHISPERX_PYTHON to "
                  "the venv that has whisperx (or `pip install whisperx` here).")
            return [], []

    args = [whisperx_python, str(_WORKER),
            "--audio", str(audio_path),
            "--language", language,
            "--model-size", model_size]
    if device:
        args += ["--device", device]
    if models_dir:
        args += ["--models-dir", models_dir]

    try:
        # UTF-8 both ways (see qwen_engine.run_qwen).
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            encoding="utf-8", errors="replace",
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
                    print("[info] WhisperX worker cancelled.")
                    return [], []
    except OSError as exc:
        print(f"[warn] WhisperX worker could not launch ({whisperx_python}): {exc}")
        return [], []

    if proc.returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-5:])
        print(f"[warn] WhisperX worker failed (exit {proc.returncode}):\n{tail}")
        return [], []

    try:
        payload = _parse_worker_output(stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[warn] WhisperX worker output unparseable: {exc}")
        return [], []

    words = _words_from_payload(payload)
    events = list(payload.get("events", []))
    print(f"[ok] WhisperX: {len(words)} words, {len(events)} events "
          f"(lang={payload.get('language')}).")
    return words, events


def _run_whisperx_inproc(audio_path: str,
                         language: str,
                         model_size: str,
                         device: Optional[str],
                         models_dir: Optional[str],
                         cancel_event=None
                         ) -> Tuple[List[Dict], List[Dict]]:
    """In-process WhisperX fallback (only used when no worker venv is set).

    Returns ``([], [])`` on any failure so the ensemble keeps moving.
    """
    import gc  # noqa: F401  -- used by whisperx/ctranslate2 internals
    try:
        import torch
        import whisperx
    except ImportError:
        print("[info] WhisperX skipped: not importable in this interpreter.")
        return [], []

    # HF_HOME is pointed at models_dir for the duration of this call only;
    # leaving it set would redirect every later HuggingFace lookup in the
    # process (Qwen, the LLM stage, the next job) to this directory.
    previous_hf_home = os.environ.get("HF_HOME")
    hf_home_overridden = False
    try:
        if cancel_event is not None and cancel_event.is_set():
            return [], []
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Device: {device} | compute: {compute_type} | model: {model_size} "
              f"| language: {language}")

        if models_dir:
            os.environ["HF_HOME"] = models_dir
            hf_home_overridden = True

        model = whisperx.load_model(model_size, device,
                                    compute_type=compute_type,
                                    language=language)
        audio = whisperx.load_audio(audio_path)
        print("Transcribing...")
        result = model.transcribe(audio, batch_size=8, language=language)
        if cancel_event is not None and cancel_event.is_set():
            return [], []
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        model_name = ALIGNMENT_MODEL_MAP.get(language,
                                             ALIGNMENT_MODEL_MAP["default"])
        print(f"Loading alignment model: {model_name}")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device, model_name=model_name)
        except Exception as exc:
            print(f"[warn] {model_name} unavailable ({exc}); "
                  f"falling back to WhisperX auto-selection.")
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device)
        print("Aligning timestamps...")
        if cancel_event is not None and cancel_event.is_set():
            return [], []
        aligned_result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False, interpolate_method="nearest")

        events: List[Dict] = []
        for seg in result.get("segments", []):
            for match in _EVENT_RE.findall(seg.get("text", "")):
                tag = (match[0] or match[1]).lower()
                events.append({"start": seg["start"], "end": seg["end"],
                               "tag": tag, "source": "WhisperX"})

        del model_a, metadata, audio
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        word_segments = aligned_result.get("word_segments", [])
        for w in word_segments:
            w["conf"] = float(w.get("score", 0.9))
        print(f"[ok] WhisperX: {len(word_segments)} words, "
              f"{len(events)} potential audio events.")
        return word_segments, events
    except Exception as exc:
        print(f"[error] WhisperX failed: {exc}")
        import traceback
        traceback.print_exc()
        return [], []
    finally:
        if hf_home_overridden:
            if previous_hf_home is None:
                os.environ.pop("HF_HOME", None)
            else:
                os.environ["HF_HOME"] = previous_hf_home
