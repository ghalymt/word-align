"""WhisperX: Whisper large-v3 transcription + wav2vec2 forced alignment.

Uses language-specific wav2vec2 checkpoints where a stronger one than
WhisperX's default is known, falling back to WhisperX auto-selection.
"""
from __future__ import annotations

import gc
import re
from typing import Dict, List, Optional, Tuple

ALIGNMENT_MODEL_MAP = {
    "en": "facebook/wav2vec2-large-960h-lv60-self",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "id": "indonesian-nlp/wav2vec2-large-xlsr-indonesian",
    "vi": "nguyenvulebinh/wav2vec2-large-vi-vlsp2020",
    "tl": "khaderb/wav2vec2-large-xlsr-53-tagalog",
    "tr": "mpoyraz/wav2vec2-large-xlsr-53-turkish",
    "ca": "softcatala/wav2vec2-large-xlsr-53-catalan",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "ro": "anton-l/wav2vec2-large-xlsr-53-romanian",
    "lt": "DeividasM/wav2vec2-large-xlsr-53-lithuanian",
    "hr": "facebook/wav2vec2-large-xlsr-53-croatian",
    "sr": "facebook/wav2vec2-large-xlsr-53-croatian",
    "bs": "facebook/wav2vec2-large-xlsr-53-croatian",
    "default": "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
}

_EVENT_RE = re.compile(r"\[(.*?)\]|\((.*?)\)")


def run_whisperx(audio_path: str, language: str,
                 model_size: str = "large-v3",
                 device: Optional[str] = None,
                 models_dir: Optional[str] = None
                 ) -> Tuple[List[Dict], List[Dict]]:
    """Return ``(word_segments, audio_events)``.

    ``word_segments`` items carry ``word``/``start``/``end`` and, when
    WhisperX provides one, an alignment ``score`` (exposed as ``conf``).

    If *models_dir* is provided, sets ``HF_HOME`` temporarily so whisperx
    finds models in the custom directory.
    """
    print("\n" + "=" * 60 + "\nRUNNING WHISPERX (HIGH ACCURACY MODE)")
    try:
        import torch
        import whisperx

        gc.collect()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Device: {device} | compute: {compute_type} | model: {model_size} "
              f"| language: {language}")

        # Allow custom model directory override
        _prev_hf = None
        if models_dir:
            _prev_hf = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = models_dir
            print(f"Using custom model dir: {models_dir}")

        model = whisperx.load_model(model_size, device,
                                    compute_type=compute_type,
                                    language=language)
        audio = whisperx.load_audio(audio_path)
        print("Transcribing...")
        result = model.transcribe(audio, batch_size=8, language=language)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        model_name = ALIGNMENT_MODEL_MAP.get(language,
                                             ALIGNMENT_MODEL_MAP["default"])
        print(f"Loading alignment model: {model_name}")
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device, model_name=model_name)
        except Exception as exc:
            print(f"[warn] {model_name} unavailable ({exc}); "
                  "falling back to WhisperX auto-selection.")
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device)
        print("Aligning timestamps...")
        aligned_result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False, interpolate_method="nearest")

        events: List[Dict] = []
        for seg in result.get("segments", []):
            for match in _EVENT_RE.findall(seg.get("text", "")):
                tag = (match[0] or match[1]).lower()
                events.append({"start": seg["start"], "end": seg["end"],
                               "tag": tag, "source": "WhisperX"})

        del model_a, metadata, audio
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        word_segments = aligned_result.get("word_segments", [])
        for w in word_segments:
            w["conf"] = float(w.get("score", 0.9))
        print(f"[ok] WhisperX: {len(word_segments)} words, "
              f"{len(events)} potential audio events.")
        return word_segments, events
    except Exception as exc:
        print(f"[error] WhisperX failed: {exc}")
        import traceback
        traceback.print_exc()
        return [], []
    finally:
        if models_dir and _prev_hf is not None:
            os.environ["HF_HOME"] = _prev_hf
