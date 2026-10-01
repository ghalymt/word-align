"""Regenerate espeak_en.flac + its ground truth (needs espeak-ng and ffmpeg).

Each word is synthesised on its own and trimmed to its audible part, then
the words are concatenated with fixed gaps. That gives exact ground-truth
word boundaries -- something a recording cannot provide -- so the real
engines' timing accuracy can be measured, not just snapshotted.

    python tests/golden/audio/make_clip.py
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
SR = 16000
SENTENCES = [
    "the weather will be cold and windy tomorrow morning",
    "please bring a warm coat and an umbrella",
    "the meeting starts at nine in the main hall",
]
LEAD_IN, WORD_GAP, SENTENCE_GAP = 0.40, 0.14, 0.70


def synth(word: str, workdir: Path) -> np.ndarray:
    raw = workdir / "raw.wav"
    wav = workdir / "w16k.wav"
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(raw), word],
                   check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw),
                    "-ar", str(SR), "-ac", "1", str(wav)], check=True)
    audio, _ = sf.read(wav, dtype="float32")
    loud = np.flatnonzero(np.abs(audio) > 0.02 * np.abs(audio).max())
    return audio[loud[0]:loud[-1] + 1]


def main() -> None:
    pieces, truth, t = [np.zeros(int(LEAD_IN * SR), np.float32)], [], LEAD_IN
    with tempfile.TemporaryDirectory() as d:
        for s_idx, sentence in enumerate(SENTENCES):
            words = sentence.split()
            for w_idx, word in enumerate(words):
                audio = synth(word, Path(d))
                truth.append({"word": word, "start": round(t, 3),
                              "end": round(t + len(audio) / SR, 3)})
                pieces.append(audio)
                t += len(audio) / SR
                last = w_idx == len(words) - 1
                gap = SENTENCE_GAP if last else WORD_GAP
                pieces.append(np.zeros(int(gap * SR), np.float32))
                t += int(gap * SR) / SR
    sf.write(HERE / "espeak_en.flac", np.concatenate(pieces), SR)
    (HERE / "espeak_en_truth.json").write_text(
        json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    (HERE / "espeak_en.txt").write_text(
        "\n".join(s[0].upper() + s[1:] + "." for s in SENTENCES) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
