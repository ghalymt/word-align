"""RoFormer separation worker (python-audio-separator) for word-align.

Runs in the ``separate`` venv. ``--jobs-json`` lists ``{"id", "in", "out",
"model"}``: for each, separate ``in`` with ``model`` and write the speech
stem (vocals, or everything but the crowd) to ``out`` as 16 kHz mono WAV.

Models (MIT, via audio-separator / UVR):
  model_bs_roformer_ep_317_sdr_12.9755.ckpt            BS-RoFormer vocals (viperx)
  mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt Mel-Band RoFormer crowd
"""
import argparse
import json
import os
import sys
import tempfile

SENTINEL = "__WA_SEPARATE_JSON__"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def pick_speech_stem(paths, model):
    """The stem that holds the speech: Vocals, or the non-crowd stem."""
    names = [(p, os.path.basename(p).lower()) for p in paths]
    if "crowd" in model:
        keep = [p for p, n in names if "crowd" not in n.split("_(")[-1]
                or "no crowd" in n or "no_crowd" in n or "other" in n]
        if keep:
            return keep[0]
    for p, n in names:
        if "(vocals)" in n:
            return p
    return paths[0] if paths else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-json", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    import librosa
    import numpy as np
    import soundfile as sf
    from audio_separator.separator import Separator

    with open(args.jobs_json, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    tmp = tempfile.mkdtemp(prefix="wa_sep_")
    sep = Separator(model_file_dir=args.model_dir, output_dir=tmp, output_format="WAV")
    loaded = None
    done, failed = [], []
    for job in jobs:
        try:
            if loaded != job["model"]:
                sep.load_model(model_filename=job["model"])
                loaded = job["model"]
            outputs = sep.separate(job["in"])
            outputs = [o if os.path.isabs(o) else os.path.join(tmp, o) for o in outputs]
            stem = pick_speech_stem(outputs, job["model"])
            if not stem:
                raise RuntimeError("no output stem")
            y, _ = librosa.load(stem, sr=16000, mono=True)
            sf.write(job["out"], np.clip(y, -1, 1), 16000, subtype="PCM_16")
            for o in outputs:
                try:
                    os.remove(o)
                except OSError:
                    pass
            done.append({"id": job["id"], "out": job["out"], "stem": os.path.basename(stem)})
            log(f"[separate] {job['id']} -> {os.path.basename(stem)}")
        except Exception as exc:
            failed.append({"id": job.get("id"), "error": str(exc)[:300]})
            log(f"[separate] {job.get('id')} failed: {exc}")
    sys.stdout.write("\n" + SENTINEL + json.dumps({"done": done, "failed": failed}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
