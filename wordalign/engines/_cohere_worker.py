"""Cohere Transcribe 03-2026 worker for word-align.

Runs in its own venv (transformers >= 5.4). Transcribes the given speech
chunks (``--chunks-json``: ``[[start, end], ...]`` from the VAD) and prints
one sentinel-tagged JSON line: ``{"chunks": [{"start", "end", "status",
"text"}], ...}``. Cohere returns punctuated, cased text with no timestamps;
word-align votes on the text and times the result with the forced aligner.

The model card warns that it transcribes silence and noise, which is why
word-align only ever feeds it VAD speech chunks.

Model: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
(Apache-2.0, gated: accept the terms once on Hugging Face).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

SENTINEL = "__WA_COHERE_JSON__"
SAMPLE_RATE = 16000
LANGUAGES = {"ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl",
             "pl", "pt", "vi", "zh"}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_audio(path):
    import numpy as np
    ffmpeg = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found (set WORDALIGN_FFMPEG)")
    proc = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel",
                           "error", "-i", path, "-vn", "-ac", "1", "-ar",
                           str(SAMPLE_RATE), "-f", "f32le", "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode("utf-8", "replace")[-400:])
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--chunks-json", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026")
    ap.add_argument("--models-dir", default=None, help="Hugging Face hub cache")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-punctuation", action="store_true")
    args = ap.parse_args()

    language = (args.language or "en").split("-")[0].lower()
    if language not in LANGUAGES:
        payload = {"unsupported_language": language, "chunks": []}
        sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
        return

    if args.models_dir:
        os.environ["HF_HUB_CACHE"] = args.models_dir
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    kw = {"cache_dir": args.models_dir} if args.models_dir else {}
    processor = AutoProcessor.from_pretrained(args.model, **kw)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, device_map=device, **kw)
    model.eval()
    log(f"[cohere] loaded on {device}; language={language}")

    audio = load_audio(args.audio)
    with open(args.chunks_json, "r", encoding="utf-8") as f:
        chunks = [(float(s), float(e)) for s, e in json.load(f)]

    results = [None] * len(chunks)
    order = sorted(range(len(chunks)), key=lambda i: chunks[i][1] - chunks[i][0])

    def run(batch_idx):
        pieces = [audio[int(chunks[i][0] * SAMPLE_RATE):int(chunks[i][1] * SAMPLE_RATE)]
                  for i in batch_idx]
        inputs = processor(pieces, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                           language=language, punctuation=not args.no_punctuation)
        chunk_index = inputs.pop("audio_chunk_index", None)
        inputs = inputs.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        if chunk_index is not None:
            texts = processor.decode(out, skip_special_tokens=True,
                                     audio_chunk_index=chunk_index, language=language)
        else:
            texts = processor.batch_decode(out, skip_special_tokens=True)
        return [str(t).strip() for t in texts]

    k = 0
    while k < len(order):
        batch_idx = [i for i in order[k:k + args.batch]
                     if chunks[i][1] - chunks[i][0] >= 0.3]
        skipped = [i for i in order[k:k + args.batch] if i not in batch_idx]
        for i in skipped:
            results[i] = {"status": "ok", "text": ""}
        if batch_idx:
            try:
                texts = run(batch_idx)
                for i, t in zip(batch_idx, texts):
                    results[i] = {"status": "ok", "text": t}
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for i in batch_idx:          # retry one at a time
                    try:
                        results[i] = {"status": "ok", "text": run([i])[0]}
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        results[i] = {"status": "failed", "text": ""}
        k += args.batch
        log(f"[cohere] {min(k, len(order))}/{len(order)} chunks")

    payload = {"language": language, "model": args.model,
               "chunks": [{"start": s, "end": e, **r}
                          for (s, e), r in zip(chunks, results)]}
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    log("[cohere] done")


if __name__ == "__main__":
    main()
