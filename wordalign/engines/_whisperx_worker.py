"""Standalone WhisperX worker for word-align.

WhisperX pulls in transformers, faster-whisper, ctranslate2, and (optionally)
pyannote-audio — each of which has its own version pins that frequently clash
with the main word-align stack (e.g. transformers' CTC loss API, torch's
pytree registrar). To keep those conflicts from polluting the main bundle,
this worker runs in its own venv and the main exe spawns it as a subprocess
(see :func:`wordalign.engines.whisperx_engine.run_whisperx`).

The worker transcribes one audio file and prints word segments + audio
events as a single JSON line to stdout, tagged with a sentinel so the caller
can pick it out of any library chatter. All human-readable logging goes to
stderr.

It imports only the standard library plus torch / torchaudio / whisperx, so
the worker venv does not need to know anything about the ``wordalign``
package. Every path and model name is a CLI argument.
"""
import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys

SENTINEL = "__WA_WHISPERX_JSON__"


def _resolve_ffmpeg() -> str | None:
    """Locate ffmpeg.exe for the worker.

    Order:
    1. ``WORDALIGN_FFMPEG`` env var (set by the bundled launcher).
    2. ``<bundle>/_internal/ffmpeg.exe`` next to the launcher exe.
    3. ``<venv>/Scripts/ffmpeg.exe`` or ``<venv>/bin/ffmpeg``.
    4. Anything already on PATH.

    WhisperX's audio loader shells out to ``ffmpeg`` by bare name; if PATH
    doesn't carry it, the call fails with ``Failed to load audio:`` and an
    empty stderr. We resolve and either hard-prepend the dir to PATH or
    inject the absolute path via ``WORDALIGN_FFMPEG`` so any downstream
    code that consults the env var (torchaudio, ffmpeg-python) also picks
    it up.
    """
    candidates: list[str] = []
    env_ffmpeg = os.environ.get("WORDALIGN_FFMPEG")
    if env_ffmpeg and os.path.isfile(env_ffmpeg):
        return env_ffmpeg
    # Walk up from this file to find the bundle root (parent of _internal/).
    here = os.path.dirname(os.path.abspath(__file__))
    # here = <bundle>/_internal/wordalign/engines
    parts = here.split(os.sep)
    try:
        idx = parts.index("_internal")
        bundle = os.sep.join(parts[:idx])
        candidates.append(os.path.join(bundle, "_internal", "ffmpeg.exe"))
    except ValueError:
        pass
    candidates += [
        os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),  # next to venv python
        shutil.which("ffmpeg") or "",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            os.environ["WORDALIGN_FFMPEG"] = c
            # Also pre-pend its dir to PATH so ``subprocess.run(["ffmpeg"])``
            # -- which is what whisperx.audio.load_audio uses -- can find it.
            ffmpeg_dir = os.path.dirname(c)
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return c
    return None

# language code -> preferred wav2vec2 alignment model. Mirrors the table
# in wordalign/engines/whisperx_engine.py.
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


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _clean(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--model-size", default="large-v3")
    ap.add_argument("--device", default=None,
                    help="cuda|cpu (auto-detected if omitted)")
    ap.add_argument("--models-dir", default=None,
                    help="HF cache dir holding the WhisperX + alignment models "
                         "(enables offline mode).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--mode", choices=["transcribe", "align"], default="transcribe",
                    help="align: time known text windows (--align-json) with "
                         "the wav2vec2 alignment model only")
    ap.add_argument("--align-json", default=None)
    args = ap.parse_args()

    # Locate ffmpeg BEFORE any audio loading so whisperx.audio.load_audio
    # (which shells out to a bare "ffmpeg") can find it on PATH.
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        log("[whisperx] WARNING: no ffmpeg found; audio loading will likely fail.")

    # transformers on newer torch renamed the pytree registrar; shim the old
    # name so whisperx (which still imports `transformers.pytree_utils` etc.)
    # doesn't crash at import time. Same shim the qwen worker uses.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    import torch
    import torch.utils._pytree as _pt
    if (not hasattr(_pt, "register_pytree_node")
            and hasattr(_pt, "_register_pytree_node")):
        _pt.register_pytree_node = (
            lambda typ, f, u, **k: _pt._register_pytree_node(typ, f, u))

    # Prefer the local model cache and never reach for the network mid-run.
    offline = bool(args.models_dir)
    if offline:
        os.environ["HF_HOME"] = args.models_dir
        os.environ["HF_HUB_CACHE"] = args.models_dir
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import whisperx  # noqa: E402

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "align":
        _align_mode(args, whisperx, torch, device)
        return
    compute_type = "float16" if device == "cuda" else "int8"
    log(f"[whisperx] device={device} model={args.model_size} "
        f"language={args.language} offline={offline}")

    model = whisperx.load_model(args.model_size, device,
                                compute_type=compute_type,
                                language=args.language)
    audio = whisperx.load_audio(args.audio)
    log(f"[whisperx] audio loaded ({len(audio)/16000:.1f}s)")
    result = model.transcribe(audio, batch_size=args.batch_size,
                              language=args.language)
    del model
    _clean(torch)

    model_a, metadata = _load_align_model(whisperx, args.language, device,
                                          args.models_dir)

    log("[whisperx] aligning timestamps...")
    aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False, interpolate_method="nearest")

    events = []
    for seg in result.get("segments", []):
        for match in _EVENT_RE.findall(seg.get("text", "")):
            tag = (match[0] or match[1]).lower()
            events.append({"start": seg["start"], "end": seg["end"],
                           "tag": tag, "source": "WhisperX"})

    del model_a, metadata, audio
    _clean(torch)

    words = []
    for w in aligned.get("word_segments", []):
        words.append({
            "word": str(w.get("word", "")).strip(),
            "start": float(w["start"]) if w.get("start") is not None else None,
            "end": float(w["end"]) if w.get("end") is not None else None,
            "conf": float(w.get("score", 0.9)),
        })

    payload = {"language": args.language, "words": words, "events": events}
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()
    log(f"[whisperx] done: {len(words)} words, {len(events)} events")


def align_model_dirs(models_dir, model_name):
    """Where to look for the alignment model: the models folder itself and
    its ``hub`` subfolder (a Hugging Face ``HF_HOME`` keeps models there),
    the one that holds *model_name* first. ``[None]`` = the default cache."""
    if not models_dir:
        return [None]
    found = [d for d in (models_dir, os.path.join(models_dir, "hub")) if os.path.isdir(d)]
    folder = "models--" + model_name.replace("/", "--")
    found.sort(key=lambda d: not os.path.isdir(os.path.join(d, folder)))
    return found or [None]


def _load_align_model(whisperx, language, device, models_dir=None):
    model_name = ALIGNMENT_MODEL_MAP.get(language, ALIGNMENT_MODEL_MAP["default"])
    log(f"[whisperx] loading alignment model: {model_name}")
    last = None
    for model_dir in align_model_dirs(models_dir, model_name):
        try:
            kwargs = {"model_dir": model_dir} if model_dir else {}
            return whisperx.load_align_model(language_code=language, device=device,
                                             model_name=model_name, **kwargs)
        except Exception as exc:          # not in this folder (or an old whisperx)
            last = exc
    log(f"[whisperx] {model_name} unavailable ({last}); falling back to "
        "WhisperX auto-selection.")
    return whisperx.load_align_model(language_code=language, device=device)


def _align_mode(args, whisperx, torch, device):
    """Time known text: one whisperx.align() segment per window."""
    from difflib import SequenceMatcher

    def norm(w):
        return re.sub(r"[^\w\s]", "", str(w)).lower()

    with open(args.align_json, "r", encoding="utf-8") as f:
        windows = json.load(f)
    audio = whisperx.load_audio(args.audio)
    model_a, metadata = _load_align_model(whisperx, args.language, device,
                                          args.models_dir)
    out = []
    for win in windows:
        words = [str(w) for w in win["words"]]
        res = {"id": win.get("id"), "start": win["start"], "end": win["end"],
               "times": [None] * len(words), "status": "ok"}
        try:
            aligned = whisperx.align(
                [{"start": float(win["start"]), "end": float(win["end"]),
                  "text": " ".join(words)}],
                model_a, metadata, audio, device, return_char_alignments=False,
                interpolate_method="ignore")
            got = [w for w in aligned.get("word_segments", [])
                   if w.get("start") is not None and w.get("end") is not None]
        except Exception as exc:
            res["status"] = f"error: {exc}"
            out.append(res)
            continue
        sm = SequenceMatcher(None, [norm(w) for w in words],
                             [norm(g.get("word", "")) for g in got], autojunk=False)
        for block in sm.get_matching_blocks():
            for k in range(block.size):
                g = got[block.b + k]
                res["times"][block.a + k] = [round(float(g["start"]), 3),
                                             round(float(g["end"]), 3)]
        out.append(res)
    del model_a, metadata, audio
    _clean(torch)
    payload = {"mode": "align", "windows": out}
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()
    log(f"[whisperx] aligned {len(out)} windows")


if __name__ == "__main__":
    main()
