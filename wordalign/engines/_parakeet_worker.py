"""parakeet-ultra worker (Moondream Photon) for word-align.

Runs in its own venv (``pip install moondream`` with a CUDA torch). Photon
is Moondream's local inference engine; it runs on NVIDIA GPUs under Linux
and Windows. parakeet-ultra is a post-trained parakeet-tdt-0.6b-v3 (25
European languages, CC-BY-4.0, native word timestamps): an architecturally
independent third voter beside the attention-decoder models.

Transcribes the given speech chunks (``--chunks-json``) and prints one
sentinel-tagged JSON line with per-chunk timed words.

Model: https://huggingface.co/moondream/parakeet-ultra (CC-BY-4.0 --
attribution: NVIDIA parakeet-tdt-0.6b-v3, post-trained by Moondream).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

SENTINEL = "__WA_PARAKEET_JSON__"
SAMPLE_RATE = 16000


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
    ap.add_argument("--model", default="moondream/parakeet-ultra")
    ap.add_argument("--models-dir", default=None, help="Hugging Face hub cache")
    args = ap.parse_args()

    if args.models_dir:
        os.environ["HF_HUB_CACHE"] = args.models_dir
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import numpy as np
    import moondream as md

    audio = load_audio(args.audio)
    with open(args.chunks_json, "r", encoding="utf-8") as f:
        chunks = [(float(s), float(e)) for s, e in json.load(f)]

    out = []
    with md.photon(args.model) as speech:
        log(f"[parakeet] loaded {args.model}")
        for idx, (start, end) in enumerate(chunks):
            piece = np.ascontiguousarray(
                np.clip(audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)], -1.0, 1.0))
            entry = {"start": start, "end": end, "status": "ok", "words": []}
            if len(piece) < int(0.3 * SAMPLE_RATE):
                out.append(entry)
                continue
            try:
                res = speech.transcribe(audio=piece, sample_rate=SAMPLE_RATE,
                                        timestamps="word")
            except Exception as exc:   # one bad chunk must not sink the file
                log(f"[parakeet] chunk {idx} failed: {exc}")
                entry["status"] = "failed"
                out.append(entry)
                continue
            for seg in res.get("segments", []) or []:
                for w in seg.get("words", []) or []:
                    text = str(w.get("word", "")).strip()
                    if not text or w.get("start") is None or w.get("end") is None:
                        continue
                    entry["words"].append({
                        "word": text,
                        "start": round(start + float(w["start"]), 3),
                        "end": round(start + float(w["end"]), 3)})
            entry["text"] = str(res.get("text", "")).strip()
            out.append(entry)
            if (idx + 1) % 20 == 0:
                log(f"[parakeet] {idx + 1}/{len(chunks)} chunks")

    payload = {"model": args.model, "chunks": out}
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    log("[parakeet] done")


if __name__ == "__main__":
    main()
