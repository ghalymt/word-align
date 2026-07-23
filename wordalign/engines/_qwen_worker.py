"""Standalone Qwen3-ASR + ForcedAligner worker for word-align.

word-align's ``qwen_engine`` runs this script inside the virtualenv that has
the ``qwen_asr`` package (that venv's torch usually differs from the host's,
so it must stay a separate process). The script transcribes one audio file and
prints its word-level segments as a single JSON line to stdout, tagged with a
sentinel so the caller can pick it out of any library chatter. All human-
readable logging goes to stderr.

It imports only the standard library plus torch / torchaudio / qwen_asr, so it
runs in a venv that knows nothing about the wordalign package. Every path and
model name is a CLI argument -- nothing machine-specific is baked in.

Approach (bounded for ~12 GB GPUs): load the ASR model once with its forced
aligner, split the audio into fixed-length chunks, transcribe each with
word timestamps, offset those times by the chunk start, and stitch. Chunking
is what keeps long files from exhausting VRAM.
"""
import argparse
import json
import os
import sys
import tempfile

SENTINEL = "__WA_QWEN_JSON__"

# Qwen3-ASR validates against full language NAMES, not ISO codes. word-align
# speaks ISO 639-1, so translate; anything unmapped falls back to None, which
# tells Qwen to auto-detect (its default, and what works well in practice).
_QWEN_LANG = {
    "en": "English", "zh": "Chinese", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "tl": "Filipino", "fil": "Filipino", "fa": "Persian", "el": "Greek",
    "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}


def _qwen_language(code):
    """Map an ISO code (or full name) to Qwen's expected name, else None."""
    if not code:
        return None
    c = code.strip().lower()
    if c in _QWEN_LANG:
        return _QWEN_LANG[c]
    if c in {v.lower() for v in _QWEN_LANG.values()}:
        return c.capitalize()          # already a full name
    return None                        # unknown -> let Qwen auto-detect


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _clean(torch):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _transcribe_chunked(model, audio_path, chunk_seconds, language, torch):
    """Return (words, detected_language). words: list of {word,start,end,conf}."""
    import torchaudio

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:                       # force mono
        waveform = waveform.mean(dim=0, keepdim=True)
    total = waveform.shape[1]
    step = max(1, int(chunk_seconds * sr))
    tmp = tempfile.gettempdir()

    words = []
    detected = language
    idx = 0
    for start in range(0, total, step):
        end = min(start + step, total)
        chunk_path = os.path.join(tmp, f"_wa_qwen_{os.getpid()}_{idx:04d}.wav")
        torchaudio.save(chunk_path, waveform[:, start:end], sr)
        offset = start / sr
        try:
            results = model.transcribe(audio=chunk_path, language=language,
                                       return_time_stamps=True)
            if results:
                r = results[0]
                detected = getattr(r, "language", detected) or detected
                for ts in (getattr(r, "time_stamps", None) or []):
                    words.append({
                        "word": ts.text,
                        "start": float(ts.start_time) + offset,
                        "end": float(ts.end_time) + offset,
                        "conf": 0.9,
                    })
            log(f"[qwen] chunk {idx} (+{offset:.1f}s): {len(words)} words total")
        except getattr(torch.cuda, "OutOfMemoryError", Exception):
            log(f"[qwen] OOM on chunk {idx}; skipping "
                f"(lower --chunk-seconds to reduce VRAM)")
            _clean(torch)
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            _clean(torch)
        idx += 1
    return words, detected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--language", default=None)
    ap.add_argument("--models-dir", default=None,
                    help="HF cache dir holding the Qwen models (enables offline)")
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    ap.add_argument("--chunk-seconds", type=float, default=60.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    # Must be set before torch is imported.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

    import torch
    # transformers on newer torch renamed the pytree registrar; shim the old
    # name so importing qwen_asr (which pulls in transformers) doesn't crash.
    import torch.utils._pytree as _pt
    if (not hasattr(_pt, "register_pytree_node")
            and hasattr(_pt, "_register_pytree_node")):
        _pt.register_pytree_node = (
            lambda typ, f, u, **k: _pt._register_pytree_node(typ, f, u))

    # Prefer the local model cache and never reach for the network mid-run.
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    offline = bool(args.models_dir)
    if args.models_dir:
        os.environ["HF_HOME"] = args.models_dir
        os.environ["HF_HUB_CACHE"] = args.models_dir

    from qwen_asr import Qwen3ASRModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else "cpu"
    log(f"[qwen] device={device} asr={args.asr_model} "
        f"aligner={args.aligner_model}")

    load_kwargs = dict(
        dtype=dtype,
        device_map=device_map,
        forced_aligner=args.aligner_model,
        forced_aligner_kwargs=dict(
            dtype=dtype,
            device_map=device_map,
            local_files_only=offline,
            attn_implementation="eager",
        ),
        max_new_tokens=args.max_new_tokens,
        local_files_only=offline,
        attn_implementation="eager",
    )
    if args.models_dir:
        load_kwargs["cache_dir"] = args.models_dir

    model = Qwen3ASRModel.from_pretrained(args.asr_model, **load_kwargs)
    log("[qwen] model loaded")

    qwen_lang = _qwen_language(args.language)
    log(f"[qwen] language: {args.language!r} -> {qwen_lang!r}")
    words, language = _transcribe_chunked(
        model, args.audio, args.chunk_seconds, qwen_lang, torch)

    del model
    _clean(torch)

    payload = {"language": language, "words": words}
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()
    log(f"[qwen] done: {len(words)} words")


if __name__ == "__main__":
    main()
