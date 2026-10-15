"""MossFormerGAN speech-enhancement worker (ClearerVoice-Studio).

Runs in the ``denoise`` venv. ``--jobs-json`` lists ``{"id", "in", "out"}``
16 kHz WAV files; each is enhanced with MossFormerGAN_SE_16K and written to
``out``. word-align mixes the result back with the original (weak
correction beat full strength in the published test), so this worker
returns the full-strength output only.

Model: alibabasglab/MossFormerGAN_SE_16K (ClearerVoice-Studio, Apache-2.0
code). Checkpoints live in ``<model-dir>/checkpoints/MossFormerGAN_SE_16K``.
"""
import argparse
import json
import os
import sys

SENTINEL = "__WA_DENOISE_JSON__"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-json", required=True)
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    with open(args.jobs_json, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    os.makedirs(args.model_dir, exist_ok=True)
    os.chdir(args.model_dir)        # ClearVoice resolves checkpoints/ relative to CWD
    import numpy as np
    import soundfile as sf
    from clearvoice import ClearVoice

    cv = ClearVoice(task="speech_enhancement", model_names=["MossFormerGAN_SE_16K"])
    done, failed = [], []
    for job in jobs:
        try:
            out = cv(input_path=job["in"], online_write=False)
            y = np.asarray(out, dtype=np.float32).reshape(-1)
            sf.write(job["out"], np.clip(y, -1, 1), 16000, subtype="PCM_16")
            done.append({"id": job["id"], "out": job["out"]})
        except Exception as exc:
            failed.append({"id": job.get("id"), "error": str(exc)[:300]})
            log(f"[denoise] {job.get('id')} failed: {exc}")
    sys.stdout.write("\n" + SENTINEL + json.dumps({"done": done, "failed": failed}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
