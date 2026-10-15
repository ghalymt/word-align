"""Speaker diarization worker (pyannote.audio 4, Community-1 by default).

Runs in the WhisperX venv (pyannote.audio 4.x). Prints one sentinel-tagged
JSON line: ``{"model", "turns": [[start, end, label], ...], "exclusive":
[...]}``. ``turns`` keeps overlapping speech (two speakers at once), which
word-align turns into overlap regions; ``exclusive`` has one speaker per
instant, which is what word-to-speaker assignment needs.

Model: https://huggingface.co/pyannote/speaker-diarization-community-1
(CC-BY-4.0, gated: accept its terms once on Hugging Face). Falls back to
pyannote/speaker-diarization-3.1 when Community-1 cannot be loaded.
"""
import argparse
import json
import os
import sys
import wave

SENTINEL = "__WA_DIARIZE_JSON__"
FALLBACK = "pyannote/speaker-diarization-3.1"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_wav(path):
    import numpy as np
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def load_pipeline(model, cache_dir):
    from pyannote.audio import Pipeline
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    try:
        return Pipeline.from_pretrained(model, token=token, **kwargs)
    except TypeError:                                   # pyannote 3.x
        return Pipeline.from_pretrained(model, use_auth_token=token, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="16 kHz mono 16-bit WAV")
    ap.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args()

    import torch
    used = args.model
    pipeline = None
    for candidate in (args.model, FALLBACK):
        try:
            pipeline = load_pipeline(candidate, args.cache_dir)
        except Exception as exc:
            log(f"[diarize] could not load {candidate}: {exc}")
            pipeline = None
        if pipeline is not None:
            used = candidate
            break
    if pipeline is None:
        raise SystemExit("no diarization pipeline could be loaded (accept the "
                         "model terms on huggingface.co and log in)")
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    audio, sr = read_wav(args.audio)
    options = {}
    if args.num_speakers:
        options["num_speakers"] = args.num_speakers
    if args.min_speakers:
        options["min_speakers"] = args.min_speakers
    if args.max_speakers:
        options["max_speakers"] = args.max_speakers
    log(f"[diarize] {used} on {len(audio) / sr:.1f}s")
    output = pipeline({"waveform": torch.from_numpy(audio)[None, :], "sample_rate": sr},
                      **options)
    regular = getattr(output, "speaker_diarization", output)
    exclusive = getattr(output, "exclusive_speaker_diarization", None)

    def tracks(annotation):
        return [[round(float(t.start), 3), round(float(t.end), 3), str(label)]
                for t, _, label in annotation.itertracks(yield_label=True)]

    payload = {"model": used, "turns": tracks(regular),
               "exclusive": tracks(exclusive) if exclusive is not None else None}
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()
    log(f"[diarize] done: {len(payload['turns'])} turns")


if __name__ == "__main__":
    main()
