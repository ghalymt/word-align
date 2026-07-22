"""NVIDIA NeMo engines: Parakeet TDT (speed) and Canary-Qwen (accuracy).

These are optional ensemble voters. As of 2026, Canary-Qwen 2.5B tops the
Hugging Face Open ASR Leaderboard for English WER and Parakeet TDT beats
Whisper large-v3 on English while running orders of magnitude faster --
but both trail Whisper's 99-language coverage, which is why word-align
keeps WhisperX as the multilingual backbone and uses these as
high-precision voters (see ``wordalign.ensemble``).

Requires: ``pip install "nemo_toolkit[asr]"`` (heavy; only needed for
transcript-free ensemble mode).
"""
from __future__ import annotations

import gc
from typing import Dict, List

PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
CANARY_MODEL = "nvidia/canary-qwen-2.5b"

# Languages Parakeet TDT v3 supports (25 European languages).
PARAKEET_LANGS = {
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "uk", "sv", "da",
    "no", "fi", "cs", "sk", "sl", "hr", "bg", "ro", "hu", "el", "lt", "lv",
    "et",
}


def nemo_available() -> bool:
    try:
        import nemo.collections.asr  # noqa: F401
        return True
    except ImportError:
        return False


def run_parakeet(audio_path: str, language: str = "en") -> List[Dict]:
    """Transcribe with Parakeet TDT and return word dicts with timestamps."""
    if language not in PARAKEET_LANGS:
        print(f"[info] Parakeet does not cover '{language}'; skipping.")
        return []
    print("\n" + "=" * 60 + "\nRUNNING PARAKEET TDT (ENSEMBLE VOTER)")
    try:
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(PARAKEET_MODEL)
        out = model.transcribe([audio_path], timestamps=True)
        words: List[Dict] = []
        hyp = out[0]
        for w in getattr(hyp, "timestamp", {}).get("word", []):
            words.append({
                "word": w.get("word", ""),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "conf": 0.9,   # TDT decodes greedily; flat prior
            })
        del model
        gc.collect()
        print(f"[ok] Parakeet: {len(words)} words.")
        return words
    except Exception as exc:
        print(f"[warn] Parakeet unavailable/failed: {exc}")
        return []


def run_canary_qwen(audio_path: str, language: str = "en") -> List[Dict]:
    """Transcribe with Canary-Qwen 2.5B (English; highest-WER-accuracy voter)."""
    if language != "en":
        print(f"[info] Canary-Qwen voter is English-only; skipping "
              f"'{language}'.")
        return []
    print("\n" + "=" * 60 + "\nRUNNING CANARY-QWEN (ENSEMBLE VOTER)")
    try:
        import nemo.collections.speechlm2 as slm
        model = slm.models.SALM.from_pretrained(CANARY_MODEL)
        answer_ids = model.generate(
            prompts=[[{"role": "user",
                       "content": f"Transcribe the following: {model.audio_locator_tag}",
                       "audio": [audio_path]}]],
            max_new_tokens=2048,
        )
        text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
        del model
        gc.collect()
        # Canary-Qwen returns text without word timestamps; emit untimed
        # words -- the ensemble uses them for lexical voting only.
        words = [{"word": w, "start": None, "end": None, "conf": 0.95}
                 for w in text.split()]
        print(f"[ok] Canary-Qwen: {len(words)} words (untimed, lexical votes).")
        return words
    except Exception as exc:
        print(f"[warn] Canary-Qwen unavailable/failed: {exc}")
        return []
