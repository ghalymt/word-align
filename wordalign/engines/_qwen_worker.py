"""Standalone Qwen3-ASR + ForcedAligner worker for word-align.

word-align's ``qwen_engine`` runs this script inside the virtualenv that has
the ``qwen_asr`` package (that venv's torch usually differs from the host's,
so it must stay a separate process). It prints one sentinel-tagged JSON
line to stdout; all human-readable logging goes to stderr.

Two modes:

``--mode transcribe`` (default)
    Transcribe with word timestamps. With ``--chunks-json`` the audio is
    cut into the given ``[[start, end], ...]`` speech chunks (from the VAD,
    cut at pauses); otherwise into fixed ``--chunk-seconds`` pieces. Each
    chunk's words come back in ``chunks`` (for voting) and flattened in
    ``words``. Chunks that run out of GPU memory are retried in halves and
    reported in ``failed`` instead of vanishing.

``--mode align``
    Load only Qwen3-ForcedAligner-0.6B and time known text: ``--align-json``
    holds ``[{"id", "start", "end", "words": [...]}, ...]`` windows. Each
    window's words come back with times (``null`` for words the aligner
    could not place, e.g. punctuation-only tokens).

It imports only the standard library plus torch / numpy / qwen_asr, so it
runs in a venv that knows nothing about the wordalign package. Every path
and model name is a CLI argument -- nothing machine-specific is baked in.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

SENTINEL = "__WA_QWEN_JSON__"
SAMPLE_RATE = 16000

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
# Languages the forced aligner supports (model card: 11 languages).
ALIGNER_LANGS = {"Chinese", "English", "Cantonese", "French", "German",
                 "Italian", "Japanese", "Korean", "Portuguese", "Russian",
                 "Spanish"}


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


def _load_audio(audio_path: str):
    """Decode any media file to 16 kHz mono float32 with ffmpeg."""
    import numpy as np
    ffmpeg = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg or not os.path.isfile(ffmpeg):
        raise RuntimeError("FFmpeg is required to decode Qwen audio input")
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
               "-i", audio_path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-f", "f32le", "-"]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("FFmpeg audio extraction failed: "
                           + (detail[-1] if detail else str(proc.returncode)))
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if audio.size == 0:
        raise RuntimeError("FFmpeg produced no readable audio")
    return audio


def _oom_types(torch):
    return (getattr(torch.cuda, "OutOfMemoryError", RuntimeError),)


def _transcribe_piece(model, audio, start, end, language, torch, depth=0):
    """Transcribe audio[start:end] (seconds). Returns (words, language, failed)."""
    i0, i1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
    piece = audio[i0:i1]
    if len(piece) < int(0.5 * SAMPLE_RATE):      # Qwen's minimum input
        return [], None, []
    try:
        results = model.transcribe(audio=(piece, SAMPLE_RATE), language=language,
                                   return_time_stamps=True)
    except _oom_types(torch):
        _clean(torch)
        if depth < 2 and end - start > 4.0:
            mid = (start + end) / 2
            log(f"[qwen] OOM on {start:.1f}-{end:.1f}s; retrying in halves")
            w1, l1, f1 = _transcribe_piece(model, audio, start, mid, language, torch, depth + 1)
            w2, l2, f2 = _transcribe_piece(model, audio, mid, end, language, torch, depth + 1)
            return w1 + w2, l1 or l2, f1 + f2
        log(f"[qwen] OOM on {start:.1f}-{end:.1f}s; giving up on this piece")
        return [], None, [[round(start, 3), round(end, 3)]]
    words, detected = [], None
    if results:
        r = results[0]
        detected = getattr(r, "language", None)
        for ts in (getattr(r, "time_stamps", None) or []):
            words.append({"word": ts.text,
                          "start": round(float(ts.start_time) + start, 3),
                          "end": round(float(ts.end_time) + start, 3),
                          "conf": 0.9})
    return words, detected, []


def transcribe(model, audio, chunks, language, torch):
    out_chunks, words, failed = [], [], []
    detected = language
    for idx, (start, end) in enumerate(chunks):
        cw, lang, bad = _transcribe_piece(model, audio, start, end, language, torch)
        detected = lang or detected
        failed += bad
        status = "failed" if bad and not cw else "ok"
        out_chunks.append({"start": start, "end": end, "status": status,
                           "words": cw})
        words += cw
        log(f"[qwen] chunk {idx + 1}/{len(chunks)} ({start:.1f}-{end:.1f}s): "
            f"{len(cw)} words")
        _clean(torch)
    return out_chunks, words, detected, failed


def _word_item_counts(aligner, words, language):
    """How many aligner items each word becomes (same tokenizer as align())."""
    proc = aligner.aligner_processor
    return [len(proc.tokenize_space_lang(w)) for w in words]


def align_windows(aligner, audio, windows, language, torch):
    out = []
    lang_name = _qwen_language(language) or "English"
    for wi, win in enumerate(windows):
        words = [str(w) for w in win["words"]]
        start, end = float(win["start"]), float(win["end"])
        result = {"id": win.get("id", wi), "start": start, "end": end,
                  "times": [None] * len(words), "status": "ok"}
        piece = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        counts = _word_item_counts(aligner, words, lang_name) if lang_name not in (
            "Japanese", "Korean") else None
        if not words or len(piece) < int(0.2 * SAMPLE_RATE) or (
                counts is not None and sum(counts) == 0):
            result["status"] = "empty"
            out.append(result)
            continue
        try:
            res = aligner.align(audio=(piece, SAMPLE_RATE), text=" ".join(words),
                                language=lang_name)
            items = list(res[0]) if res else []
        except _oom_types(torch):
            _clean(torch)
            result["status"] = "oom"
            out.append(result)
            continue
        except Exception as exc:  # tokenizer edge cases
            result["status"] = f"error: {exc}"
            out.append(result)
            continue
        if counts is not None:
            if sum(counts) != len(items):
                result["status"] = f"mismatch {sum(counts)}!={len(items)}"
            k = 0
            for i, c in enumerate(counts):
                if c and k + c <= len(items):
                    first, last = items[k], items[k + c - 1]
                    result["times"][i] = [round(start + float(first.start_time), 3),
                                          round(start + float(last.end_time), 3)]
                k += c
        else:
            # Japanese/Korean: consume items by character length.
            k = 0
            for i, w in enumerate(words):
                need = len("".join(ch for ch in w if ch.isalnum()))
                if not need:
                    continue
                got, first = 0, k
                while k < len(items) and got < need:
                    got += len(items[k].text)
                    k += 1
                if k > first:
                    result["times"][i] = [round(start + float(items[first].start_time), 3),
                                          round(start + float(items[k - 1].end_time), 3)]
        out.append(result)
        if (wi + 1) % 20 == 0:
            log(f"[qwen-align] {wi + 1}/{len(windows)} windows")
        _clean(torch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--mode", choices=["transcribe", "align"], default="transcribe")
    ap.add_argument("--language", default=None)
    ap.add_argument("--models-dir", default=None,
                    help="HF cache dir holding the Qwen models (enables offline). "
                         "Default: <project>/models/huggingface/hub")
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    ap.add_argument("--chunk-seconds", type=float, default=60.0)
    ap.add_argument("--chunks-json", default=None,
                    help="JSON file with [[start, end], ...] speech chunks")
    ap.add_argument("--align-json", default=None,
                    help="JSON file with alignment windows (align mode)")
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
    if not args.models_dir:
        here = os.path.dirname(os.path.abspath(__file__))
        local_hub = os.path.join(os.path.dirname(os.path.dirname(here)),
                                 "models", "huggingface", "hub")
        if os.path.isdir(local_hub):
            args.models_dir = local_hub
    offline = bool(args.models_dir)
    if args.models_dir:
        os.environ["HF_HUB_CACHE"] = args.models_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "auto" if device == "cuda" else "cpu"
    audio = _load_audio(args.audio)
    log(f"[qwen] mode={args.mode} device={device} audio={len(audio) / SAMPLE_RATE:.1f}s")

    if args.mode == "align":
        from qwen_asr import Qwen3ForcedAligner
        kwargs = dict(dtype=dtype, device_map=device_map,
                      local_files_only=offline, attn_implementation="eager")
        if args.models_dir:
            kwargs["cache_dir"] = args.models_dir
        aligner = Qwen3ForcedAligner.from_pretrained(args.aligner_model, **kwargs)
        with open(args.align_json, "r", encoding="utf-8") as f:
            windows = json.load(f)
        lang = _qwen_language(args.language) or "English"
        if lang not in ALIGNER_LANGS:
            payload = {"mode": "align", "windows": [], "unsupported_language": lang}
        else:
            with torch.inference_mode():
                payload = {"mode": "align",
                           "windows": align_windows(aligner, audio, windows,
                                                    args.language, torch)}
        del aligner
        _clean(torch)
    else:
        from qwen_asr import Qwen3ASRModel
        load_kwargs = dict(
            dtype=dtype, device_map=device_map,
            forced_aligner=args.aligner_model,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map,
                                       local_files_only=offline,
                                       attn_implementation="eager"),
            max_new_tokens=args.max_new_tokens,
            local_files_only=offline, attn_implementation="eager")
        if args.models_dir:
            load_kwargs["cache_dir"] = args.models_dir
        model = Qwen3ASRModel.from_pretrained(args.asr_model, **load_kwargs)
        log("[qwen] model loaded")
        if args.chunks_json:
            with open(args.chunks_json, "r", encoding="utf-8") as f:
                chunks = [(float(s), float(e)) for s, e in json.load(f)]
        else:
            total = len(audio) / SAMPLE_RATE
            step = max(1.0, args.chunk_seconds)
            chunks, t = [], 0.0
            while t < total:
                chunks.append((t, min(total, t + step)))
                t += step
        qwen_lang = _qwen_language(args.language)
        log(f"[qwen] language: {args.language!r} -> {qwen_lang!r}; {len(chunks)} chunks")
        out_chunks, words, language, failed = transcribe(model, audio, chunks,
                                                         qwen_lang, torch)
        del model
        _clean(torch)
        payload = {"mode": "transcribe", "language": language, "words": words,
                   "chunks": out_chunks, "failed": failed}

    sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()
    log("[qwen] done")


if __name__ == "__main__":
    main()
