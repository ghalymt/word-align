"""Sound-event detection worker (PretrainedSED, BEATs strong checkpoint).

Runs in a venv with torch + torchaudio (the WhisperX venv works). It scores
the AudioSet-strong ontology (447 classes) every 40 ms and returns:

* ``groups``: per-frame probabilities for the groups word-align gates on
  (speech, music, crowd, noise, laughter, applause), saved as an .npz file
  because an hour of audio is ~90k frames;
* ``events``: timed events above a threshold (label, start, end, score),
  used for SDH tags.

Model: Schmid et al., "Effective Pre-Training of Audio Transformers for
Sound Event Detection" (ICASSP 2025), https://github.com/fschmid56/PretrainedSED
(MIT). Imports only the standard library, numpy, torch and the
PretrainedSED code under ``--sed-dir``.
"""
import argparse
import json
import os
import sys
import wave

SENTINEL = "__WA_SED_JSON__"
SAMPLE_RATE = 16000
CHUNK_S = 10                      # the models were trained on 10 s pieces
FRAMES_PER_CHUNK = 250            # 40 ms resolution
HOP_S = CHUNK_S / FRAMES_PER_CHUNK

GROUPS = {
    "speech": ["Speech", "Male speech, man speaking",
               "Female speech, woman speaking", "Child speech, kid speaking",
               "Conversation", "Narration, monologue", "Human voice",
               "Whispering"],
    "music": ["Music", "Singing", "Male singing", "Female singing",
              "Child singing", "Choir", "Rapping", "Synthetic singing",
              "Chant", "Yodeling"],
    "crowd": ["Crowd", "Hubbub, speech noise, speech babble", "Cheering",
              "Applause", "Clapping", "Children playing", "Booing",
              "Children shouting", "Cacophony"],
    "noise": ["Noise", "Background noise", "Environmental noise",
              "White noise, pink noise", "Static", "Hum", "Mains hum",
              "Wind noise (microphone)", "Tape hiss", "Hiss",
              "Traffic noise, roadway noise", "Mechanical fan",
              "Air conditioning", "Engine", "Rain", "Wind", "Distortion"],
    "laughter": ["Laughter", "Giggle", "Chuckle, chortle", "Belly laugh",
                 "Snicker", "Baby laughter"],
    "applause": ["Applause", "Clapping", "Cheering"],
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def read_wav(path):
    import numpy as np
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16 kHz 16-bit PCM")
        data = wf.readframes(wf.getnframes())
        ch = wf.getnchannels()
    x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32767.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x


def median_filter(x, width):
    """Median over time (axis 0), odd *width*; edges use a shrunken window."""
    import numpy as np
    if width <= 1 or len(x) < 3:
        return x
    half = width // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
    return np.median(windows, axis=-1).astype(x.dtype)


def events_from_probs(probs, classes, threshold, min_dur, merge_gap):
    import numpy as np
    events = []
    on = probs >= threshold
    for c in np.flatnonzero(on.any(axis=0)):
        col = on[:, c].astype(np.int8)
        edges = np.diff(np.concatenate(([0], col, [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        spans = []
        for s, e in zip(starts, ends):
            if spans and (s - spans[-1][1]) * HOP_S <= merge_gap:
                spans[-1][1] = e
            else:
                spans.append([s, e])
        for s, e in spans:
            if (e - s) * HOP_S >= min_dur:
                events.append({"label": classes[c],
                               "start": round(s * HOP_S, 3),
                               "end": round(e * HOP_S, 3),
                               "score": round(float(probs[s:e, c].mean()), 3)})
    events.sort(key=lambda ev: (ev["start"], ev["label"]))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="16 kHz mono 16-bit WAV")
    ap.add_argument("--sed-dir", required=True, help="PretrainedSED checkout")
    ap.add_argument("--model", default="BEATs")
    ap.add_argument("--out", required=True, help=".npz for the frame groups")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--merge-gap", type=float, default=0.3)
    args = ap.parse_args()

    import numpy as np
    import torch

    sed_dir = os.path.abspath(args.sed_dir)
    sys.path.insert(0, sed_dir)
    from data_util import audioset_classes
    from models.prediction_wrapper import PredictionsWrapper

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cwd = os.getcwd()
    os.chdir(sed_dir)            # checkpoints resolve relative to resources/
    try:
        if args.model == "BEATs":
            from models.beats.BEATs_wrapper import BEATsWrapper
            model = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
        elif args.model.startswith("frame_mn"):
            from models.frame_mn.Frame_MN_wrapper import FrameMNWrapper
            from models.frame_mn.utils import NAME_TO_WIDTH
            net = FrameMNWrapper(NAME_TO_WIDTH(args.model))
            embed_dim = net.state_dict()["frame_mn.features.16.1.bias"].shape[0]
            model = PredictionsWrapper(net, checkpoint=f"{args.model}_strong_1",
                                       embed_dim=embed_dim)
        else:
            raise SystemExit(f"unsupported SED model {args.model}")
    finally:
        os.chdir(cwd)
    model.eval().to(device)
    classes = list(audioset_classes.as_strong_train_classes)
    log(f"[sed] model={args.model} device={device} classes={len(classes)}")

    audio = read_wav(args.audio)
    seg = CHUNK_S * SAMPLE_RATE
    n_chunks = max(1, int(np.ceil(len(audio) / seg)))
    padded = np.zeros(n_chunks * seg, dtype=np.float32)
    padded[:len(audio)] = audio
    out = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for b in range(0, n_chunks, args.batch):
            x = torch.from_numpy(padded[b * seg:min(n_chunks, b + args.batch) * seg]
                                 .reshape(-1, seg)).to(device)
            mel = model.mel_forward(x)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                strong, _ = model(mel)
            out.append(torch.sigmoid(strong.float()).cpu().numpy())   # B x C x T
    probs = np.concatenate(out, axis=0).transpose(0, 2, 1).reshape(-1, len(classes))
    n_frames = int(np.ceil(len(audio) / SAMPLE_RATE / HOP_S))
    probs = probs[:n_frames]
    smooth = median_filter(probs, 9)

    index = {c: i for i, c in enumerate(classes)}
    groups = {}
    for name, members in GROUPS.items():
        cols = [index[m] for m in members if m in index]
        groups[name] = smooth[:, cols].max(axis=1).astype(np.float16)
    np.savez_compressed(args.out, hop=np.float32(HOP_S), **groups)
    events = events_from_probs(smooth, classes, args.threshold,
                               args.min_dur, args.merge_gap)
    payload = {"hop": HOP_S, "frames": int(n_frames), "groups_file": args.out,
               "model": args.model, "events": events}
    sys.stdout.write("\n" + SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()
    log(f"[sed] done: {n_frames} frames, {len(events)} events")


if __name__ == "__main__":
    main()
