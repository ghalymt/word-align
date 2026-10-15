"""Build a bad-audio benchmark suite: one long file of condition blocks.

Each speech block joins one LibriSpeech speaker's utterances (natural
pauses between them) to about ``block_seconds`` and applies one condition;
non-speech blocks hold music, noise, babble or silence with no words. One
long file per suite means every pipeline variant loads its models once,
and the metrics are cut per block afterwards.

Files written to the suite folder:

    bench.wav          the audio (16 kHz mono)
    bench.words.json   truth words with absolute times and their block id
    bench.blocks.json  blocks: id, condition, kind (speech/nonspeech), start, end
    bench.txt          the reference transcript (one utterance per line)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import degrade
from .sources import SR, ensure_sources, load_wavs

# (condition id, kind, parameters). Levels follow the published tests: SNR
# 10 / 5 / 0 / -5 dB for additive sources, RT60 0.6 and 1.0 s, telephone and
# low-bitrate codecs, a second talker 5 and 0 dB below the first.
DEFAULT_CONDITIONS: List[Tuple[str, str, Dict]] = [
    ("clean", "speech", {}),
    ("noise_10dB", "speech", {"noise": 10}),
    ("noise_5dB", "speech", {"noise": 5}),
    ("noise_0dB", "speech", {"noise": 0}),
    ("music_10dB", "speech", {"music": 10}),
    ("music_5dB", "speech", {"music": 5}),
    ("music_0dB", "speech", {"music": 0}),
    ("music_-5dB", "speech", {"music": -5}),
    ("babble_5dB", "speech", {"babble": 5}),
    ("babble_0dB", "speech", {"babble": 0}),
    ("reverb_0.6s", "speech", {"reverb": 0.6}),
    ("reverb_1.0s", "speech", {"reverb": 1.0}),
    ("phone_mulaw8k", "speech", {"codec": "mulaw8k"}),
    ("voip_opus12k", "speech", {"codec": "opus12k"}),
    ("overlap_5dB", "speech", {"overlap": 5}),
    ("overlap_0dB", "speech", {"overlap": 0}),
    ("clipped", "speech", {"clip": 12}),
    ("only_music", "nonspeech", {"music": None}),
    ("only_noise", "nonspeech", {"noise": None}),
    ("only_babble", "nonspeech", {"babble": None}),
    ("silence", "nonspeech", {}),
]
QUICK_CONDITIONS = ["clean", "noise_5dB", "music_0dB", "babble_0dB",
                    "reverb_1.0s", "phone_mulaw8k", "overlap_5dB",
                    "only_music", "only_noise"]


def _speaker_passage(speech_dir: Path, utts: Sequence[Dict], seconds: float,
                     rng: random.Random) -> Tuple[np.ndarray, List[Dict]]:
    """Join utterances with 0.4-1.2 s pauses until *seconds* is reached."""
    from ..audio.io import read_wav
    pieces: List[np.ndarray] = []
    words: List[Dict] = []
    t = 0.0
    for u in utts:
        audio = read_wav(str(speech_dir / f"{u['id']}.wav"))
        uw = json.loads((speech_dir / f"{u['id']}.words.json").read_text(encoding="utf-8"))
        if pieces:
            gap = rng.uniform(0.4, 1.2)
            pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
            t += gap
        for w in uw:
            words.append({"word": w["word"].lower(), "start": round(t + w["start"], 3),
                          "end": round(t + w["end"], 3), "utt": u["id"]})
        pieces.append(audio)
        t += len(audio) / SR
        if t >= seconds:
            break
    return np.concatenate(pieces) if pieces else np.zeros(0, np.float32), words


def build_suite(out_dir: str, sources_dir: str, conditions: Optional[List[str]] = None,
                block_seconds: float = 30.0, nonspeech_seconds: float = 12.0,
                seed: int = 0, log: Callable[[str], None] = print) -> Dict:
    """Write the suite; returns its manifest."""
    from ..audio.io import write_wav
    src = ensure_sources(sources_dir, log=log)
    speech: List[Dict] = src["speech"]  # type: ignore[assignment]
    speech_dir = Path(sources_dir) / "speech"
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    noise_pool = load_wavs(src["noise"])       # type: ignore[arg-type]
    music_pool = load_wavs(src["music"])       # type: ignore[arg-type]
    by_spk: Dict[str, List[Dict]] = {}
    for u in speech:
        by_spk.setdefault(u["speaker"], []).append(u)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    wanted = [c for c in DEFAULT_CONDITIONS if not conditions or c[0] in conditions]
    if not wanted:
        raise ValueError("no known condition selected")

    def pick(pool: List[np.ndarray], n: int) -> np.ndarray:
        if not pool:
            return degrade.pink_noise(n, nrng)
        return degrade.fit_length(pool[int(nrng.integers(0, len(pool)))], n, nrng)

    def babble_noise(n: int, exclude: str) -> np.ndarray:
        others = [u for u in speech if u["speaker"] != exclude]
        rng.shuffle(others)
        from ..audio.io import read_wav
        pool = [read_wav(str(speech_dir / f"{u['id']}.wav")) for u in others[:8]]
        return degrade.babble(pool, n, talkers=6, rng=nrng)

    audio_parts: List[np.ndarray] = []
    truth: List[Dict] = []
    blocks: List[Dict] = []
    t = 0.0
    spk_i = 0
    for cond, kind, params in wanted:
        if kind == "speech":
            spk = speakers[spk_i % len(speakers)]
            spk_i += 1
            x, words = _speaker_passage(speech_dir, by_spk[spk], block_seconds, rng)
            if "noise" in params:
                x = degrade.mix_at_snr(x, pick(noise_pool, x.size), params["noise"], nrng)
            elif "music" in params:
                x = degrade.mix_at_snr(x, pick(music_pool, x.size), params["music"], nrng)
            elif "babble" in params:
                x = degrade.mix_at_snr(x, babble_noise(x.size, spk), params["babble"], nrng)
            elif "reverb" in params:
                x = degrade.reverb(x, params["reverb"], nrng)
            elif "codec" in params:
                x = degrade.codec(x, params["codec"])
            elif "clip" in params:
                x = degrade.clip(x, params["clip"])
            elif "overlap" in params:
                other = speakers[spk_i % len(speakers)]
                spk_i += 1
                y, owords = _speaker_passage(speech_dir, by_spk[other], block_seconds * 0.5, rng)
                offset = round(float(rng.uniform(2.0, max(2.5, x.size / SR - y.size / SR - 1.0))), 2)
                x = degrade.overlap(x, y, params["overlap"], offset)
                for w in owords:
                    if w["start"] + offset < x.size / SR:
                        words.append(dict(w, start=round(w["start"] + offset, 3),
                                          end=round(w["end"] + offset, 3), interferer=True))
                words.sort(key=lambda w: w["start"])
        else:
            n = int(nonspeech_seconds * SR)
            words = []
            if "music" in params:
                x = pick(music_pool, n)
            elif "noise" in params:
                x = pick(noise_pool, n)
            elif "babble" in params:
                x = babble_noise(n, "")
            else:
                x = np.zeros(n, dtype=np.float32)
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            if peak > 0:
                x = (x / peak * 0.3).astype(np.float32)
        gap = np.zeros(int(1.0 * SR), dtype=np.float32)
        start = t
        audio_parts.extend([x.astype(np.float32), gap])
        for w in words:
            truth.append(dict(w, start=round(w["start"] + start, 3),
                              end=round(w["end"] + start, 3), block=len(blocks)))
        end = start + x.size / SR
        blocks.append({"id": len(blocks), "condition": cond, "kind": kind,
                       "start": round(start, 3), "end": round(end, 3),
                       "words": len(words)})
        t = end + gap.size / SR
        log(f"  block {len(blocks):2d} {cond:<14} {end - start:5.1f}s {len(words)} words")
    audio = np.concatenate(audio_parts)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_wav(str(out / "bench.wav"), audio)
    (out / "bench.words.json").write_text(json.dumps(truth), encoding="utf-8")
    (out / "bench.blocks.json").write_text(json.dumps(blocks, indent=1), encoding="utf-8")
    # One line per utterance, ordered by when it starts -- the way a human
    # transcript writes crosstalk: the second talker's line sits between
    # the main speaker's lines instead of word-by-word interleaving.
    utts: Dict = {}
    for w in truth:
        utts.setdefault((w.get("block"), w.get("utt"), bool(w.get("interferer"))), []).append(w)
    lines = [" ".join(w["word"] for w in sorted(ws, key=lambda w: w["start"]))
             for _, ws in sorted(utts.items(), key=lambda kv: min(w["start"] for w in kv[1]))]
    (out / "bench.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"version": 1, "seconds": round(len(audio) / SR, 1), "blocks": len(blocks),
                "words": len(truth), "conditions": [b["condition"] for b in blocks],
                "seed": seed, "sources": sources_dir}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"Suite: {manifest['seconds']:.0f}s, {len(blocks)} blocks, {len(truth)} words -> {out}")
    return manifest
