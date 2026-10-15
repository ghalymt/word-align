"""SE-DiCoW worker: target-speaker ASR on overlapped stretches.

Runs in the WhisperX venv (transformers 4.5x, torch). For each window
(``--windows-json``: ``[{"id", "start", "end", "turns": [[s, e, label]]}]``,
absolute times, each <= 30 s) it transcribes every speaker separately,
conditioned on the diarization (silence / target / non-target / overlap
masks at 50 frames per second), and prints per-speaker segments.

The masks and self-enrollment follow the reference DiCoW pipeline
(https://github.com/BUTSpeechFIT/DiCoW, Apache-2.0); diarization comes from
word-align's pyannote run instead of DiariZen (whose weights are
non-commercial).

Model: https://huggingface.co/BUT-FIT/SE-DiCoW (Whisper large-v3-turbo
based, Apache-2.0 per the model card).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

SENTINEL = "__WA_SEDICOW_JSON__"
SAMPLE_RATE = 16000
FPS = 50                      # diarization mask frames per second
MAX_S = 30.0
_SEG_RE = re.compile(r"<\|(\d+(?:\.\d+)?)\|>(.*?)<\|(\d+(?:\.\d+)?)\|>", re.S)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def load_audio(path):
    import numpy as np
    ffmpeg = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    proc = subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
                           "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
                           "-f", "f32le", "-"], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode("utf-8", "replace")[-400:])
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)


def lower_uppercase_mapping(tokenizer):
    """Same helper the DiCoW inference script installs on the tokenizer."""
    tokenizer.upper_cased_tokens = {}
    vocab = tokenizer.get_vocab()
    for token, index in vocab.items():
        if len(token) < 1:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lower = token[0] + token[1].lower() + (token[2:] if len(token) > 2 else "")
        else:
            lower = token[0].lower() + token[1:]
        if lower != token and vocab.get(lower) is not None:
            tokenizer.upper_cased_tokens[vocab[lower]] = index


def stno_masks(torch, per_speaker, n_frames):
    diar = torch.zeros(len(per_speaker), n_frames)
    for i, spans in enumerate(per_speaker):
        for s, e in spans:
            diar[i, max(0, round(s * FPS)):min(n_frames, round(e * FPS))] = 1
    masks = []
    for idx in range(len(per_speaker)):
        others = torch.ones(len(per_speaker), dtype=torch.bool)
        others[idx] = False
        sil = (1 - diar).prod(axis=0)
        anyone_else = (1 - diar[others]).prod(axis=0)
        target = diar[idx] * anyone_else
        non_target = (1 - diar[idx]) * (1 - anyone_else)
        overlap = diar[idx] - target
        masks.append(torch.stack([sil, target, non_target, overlap], axis=0))
    return torch.stack(masks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--model", default="BUT-FIT/SE-DiCoW")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--language", default="en")
    args = ap.parse_args()

    if args.cache_dir:
        os.environ["HF_HUB_CACHE"] = args.cache_dir
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kw = {"cache_dir": args.cache_dir} if args.cache_dir else {}
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype, **kw).to(device)
    model.eval()
    fe = AutoFeatureExtractor.from_pretrained(args.model, **kw)
    tok = AutoTokenizer.from_pretrained(args.model, **kw)
    lower_uppercase_mapping(tok)
    model.set_tokenizer(tok)
    use_enroll = bool(getattr(model.config, "use_enrollments", False))
    log(f"[sedicow] loaded on {device}; enrollments={use_enroll}")

    audio = load_audio(args.audio)
    with open(args.windows_json, "r", encoding="utf-8") as f:
        windows = json.load(f)
    out = []
    for win in windows:
        start = float(win["start"])
        end = min(float(win["end"]), start + MAX_S)
        piece = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        labels = sorted({t[2] for t in win["turns"]})
        per_speaker = []
        for lbl in labels:
            spans = [(max(0.0, s - start), min(end - start, e - start))
                     for s, e, l in win["turns"] if l == lbl and e > start and s < end]
            per_speaker.append([sp for sp in spans if sp[1] > sp[0]])
        keep = [i for i, sp in enumerate(per_speaker) if sp]
        labels = [labels[i] for i in keep]
        per_speaker = [per_speaker[i] for i in keep]
        entry = {"id": win.get("id"), "start": start, "end": end, "speakers": []}
        if len(labels) < 1 or len(piece) < SAMPLE_RATE // 2:
            out.append(entry)
            continue
        feats = fe(piece, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                   return_attention_mask=True)
        x = feats["input_features"]
        n_frames = x.shape[-1] // 2
        stno = stno_masks(torch, per_speaker, n_frames)
        S = len(labels)
        x = x.repeat(S, 1, 1).to(device, dtype)
        am = torch.ones(S, x.shape[-1], dtype=torch.bool, device=device)
        stno = stno.to(device, dtype)
        gen = {"input_features": x, "attention_mask": am, "stno_mask": stno,
               "return_timestamps": True, "task": "transcribe"}
        if args.language:
            gen["language"] = args.language
        if use_enroll:
            gen["enrollments"] = {"input_features": x.clone(), "attention_mask": am.clone(),
                                  "stno_mask": stno.clone()}
        try:
            with torch.inference_mode():
                tokens = model.generate(**gen)
        except Exception as exc:
            log(f"[sedicow] window {win.get('id')} failed: {exc}")
            entry["error"] = str(exc)[:300]
            out.append(entry)
            continue
        if isinstance(tokens, dict):
            tokens = tokens.get("sequences", tokens)
        texts = tok.batch_decode(tokens, decode_with_timestamps=True,
                                 skip_special_tokens=True)
        for lbl, text in zip(labels, texts):
            segs = []
            for m in _SEG_RE.finditer(text):
                t0, body, t1 = float(m.group(1)), m.group(2).strip(), float(m.group(3))
                if body:
                    segs.append({"start": round(start + t0, 3),
                                 "end": round(start + max(t1, t0), 3), "text": body})
            if not segs:
                plain = re.sub(r"<\|[^|]*\|>", " ", text).strip()
                if plain:
                    segs.append({"start": start, "end": end, "text": plain})
            entry["speakers"].append({"label": lbl, "segments": segs})
        out.append(entry)
        if device == "cuda":
            torch.cuda.empty_cache()
    sys.stdout.write("\n" + SENTINEL + json.dumps({"windows": out}, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    log(f"[sedicow] done: {len(out)} windows")


if __name__ == "__main__":
    main()
