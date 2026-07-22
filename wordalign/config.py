"""Central configuration for word-align.

Every machine-specific value lives here and can be overridden by CLI flags
or environment variables -- nothing is hardcoded to a particular computer.

Environment variables:
    WORDALIGN_VOSK_MODELS   directory containing unpacked Vosk models
    WORDALIGN_MFA           path to the ``mfa`` executable (or conda env root)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Subtitle shaping rules (industry-standard captioning constraints)
# ---------------------------------------------------------------------------
MAX_CPL = 32                # max characters per line
MAX_DURATION_MS = 3000      # max on-screen duration for a merged block
MIN_LINE_RATIO = 0.33       # min share of characters a split line must keep
ITERATION_START = 1         # iterative merge sweep: growing char budget ...
ITERATION_END = 32          # ... up to MAX_CPL
VOSK_CHUNK_SECONDS = 600    # chunk size for parallel Vosk decoding
MAX_WORKERS = 4             # process pool size for Vosk

# Terminal punctuation, optionally followed by one closing quote or bracket:
# `He said "Go."` and `Он сказал «Иди.»` both end sentences. An earlier
# version wrote the trailing class as ["\"] , which collapses to a single
# straight double-quote and so missed every curly-quoted line of dialogue.
TERMINAL_PUNCT_PATTERN = re.compile(
    r"[.!?…。！？‥]+[\"'”’»）】]?$", re.UNICODE)

# Words that read better at the start of a subtitle line than at the end.
# Covers EN / FR / ES / DE / NL / IT / PT.
PREFER_NEW_LINE_WORDS = {
    # English
    "a", "an", "the",
    "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "as", "like", "over", "under",
    "and", "but", "or", "so", "yet", "while", "whereas", "though", "although",
    "is", "are", "was", "were", "be", "been",
    "that", "which", "who",
    # French
    "un", "une", "le", "la", "les",
    "de", "à", "dans", "sur", "par", "pour", "avec",
    "comme", "sous",
    "et", "mais", "ou", "donc", "pendant", "tandis", "bien que",
    "est", "sont", "était", "étaient", "être",
    "que", "qui", "lequel",
    # Spanish
    "el", "los", "las",
    "en", "sobre", "por", "para", "con",
    "como", "bajo",
    "y", "pero", "o", "así", "mientras", "aunque",
    "es", "son", "era", "eran", "ser",
    "cual", "quien",
    # German
    "ein", "eine", "der", "die", "das",
    "von", "zu", "auf", "an", "bei", "mit", "für",
    "wie", "über", "unter",
    "und", "aber", "oder", "also", "während", "obwohl",
    "ist", "sind", "war", "waren", "sein",
    "dass", "welche", "wer",
    # Dutch
    "een", "het",
    "van", "naar", "op", "aan", "bij", "met", "voor",
    "zoals", "onder",
    "maar", "of", "dus", "terwijl", "hoewel",
    "zijn", "waren",
    "dat", "welke", "wie",
    # Italian
    "il", "lo", "i", "gli",
    "di", "su", "da", "per",
    "sopra", "sotto",
    "e", "ma", "quindi", "benché",
    "è", "sono", "erano", "essere",
    "che", "quale", "chi",
    # Portuguese
    "um", "uma", "o", "os", "as",
    "em", "sob",
    "então", "enquanto", "embora",
    "são", "eram",
    "qual", "quem",
}

CLAUSE_MARKERS = {
    'conjunctions': {
        "and", "but", "or", "so", "yet", "because", "while", "if",
        "et", "mais", "ou", "donc", "car", "parce", "pendant", "si",
        "y", "pero", "o", "así", "porque", "mientras",
        "und", "aber", "oder", "also", "denn", "weil", "während", "wenn",
        "en", "maar", "of", "dus", "want", "omdat", "terwijl", "als",
        "e", "ma", "quindi", "perché", "mentre", "se",
        "então", "porque", "enquanto",
    },
    'relative_pronouns': {
        "who", "which", "that", "whom", "whose", "where", "when",
        "qui", "que", "lequel", "laquelle", "dont", "où", "quand",
        "quien", "quienes", "cual", "cuales", "cuyo", "donde", "cuando",
        "wer", "wen", "was", "welche", "welcher", "dessen", "wo", "wann",
        "wie", "wat", "wiens", "waar", "wanneer",
        "chi", "che", "quale", "cui", "dove", "quando",
        "quem", "qual", "cujo", "onde",
    },
}

# Human tag text -> AudioSet class name used by YAMNet (experimental tagging)
TAG_MAP = {
    "laughs": "Laughter", "laughter": "Laughter", "chuckles": "Chuckling", "giggles": "Laughter",
    "cries": "Crying, sobbing", "crying": "Crying, sobbing", "weeps": "Crying, sobbing",
    "whimpering": "Whimper", "whimpers": "Whimper", "sobbing": "Crying, sobbing",
    "groans": "Groan", "grunts": "Grunt",
    "screaming": "Screaming", "screams": "Screaming", "shouts": "Shout", "yelling": "Yell",
    "gasps": "Gasp", "sighs": "Sigh", "yawns": "Yawn", "coughs": "Cough", "chokes": "Choke",
    "hiccup": "Hiccup", "snores": "Snoring", "snoring": "Snoring", "gulps": "Gulp",
    "breathing": "Breathing", "vomits": "Vomiting",
    "music": "Music", "eerie music": "Music", "dissonant music": "Music", "background music": "Music",
    "singing": "Singing", "sings": "Singing",
    "cheering": "Cheering", "applause": "Applause", "background chatter": "Chatter",
    "wind": "Wind", "wind blows": "Wind",
    "thunder": "Thunder", "thunderstorm": "Thunderstorm",
    "rain": "Rain", "raining": "Rain", "rain pours": "Rain",
    "heartbeat": "Heartbeat", "beeping": "Beep, bleep", "scanner whirring": "Electronic music",
    "explosion": "Explosion", "crashing sound": "Crash", "banging": "Banging",
    "stairs creaking": "Creak", "snip sound": "Scissors",
    "horses neighing": "Horse", "dog barks": "Dog", "cat meowing": "Cat", "sheep bleating": "Bleat",
    "crows scatter": "Crow", "birds chirping": "Bird",
    "phone ringing": "Telephone ringing", "bell": "Bell", "bell ringing": "Bell", "bell tolls": "Bell",
    "bell tolling": "Bell", "door knock": "Knock", "door squeaking": "Squeak", "car door opens": "Car door",
    "motorcycle engine": "Motorcycle", "scooter engine": "Motorcycle", "car engine": "Vehicle",
    "car brakes": "Skidding", "tires screeching": "Skidding", "car screeching": "Skidding",
    "helicopter": "Helicopter", "helicopter whumping": "Helicopter",
    "train horn": "Train horn",
}

# Vosk model directory names per language (https://alphacephei.com/vosk/models)
VOSK_MODEL_MAP = {
    "en": "vosk-model-en-us-0.22",
    "fr": "vosk-model-fr-0.22",
    "es": "vosk-model-es-0.42",
    "de": "vosk-model-de-0.21",
    "it": "vosk-model-it-0.22",
    "ru": "vosk-model-ru-0.42",
    "zh": "vosk-model-cn-0.22",
    "ja": "vosk-model-ja-0.22",
    "fa": "vosk-model-fa-0.5",
    "pt": "vosk-model-small-pt-0.3",
    "nl": "vosk-model-nl-spraakherkenning-0.6",
    "tr": "vosk-model-small-tr-0.3",
    "sv": "vosk-model-small-sv-rhasspy-0.15",
    "ca": "vosk-model-small-ca-0.4",
    "uk": "vosk-model-uk-v3",
    "ar": "vosk-model-ar-0.22-linto-1.1.0",
    "tl": "vosk-model-tl-ph-generic-0.6",
    "vi": "vosk-model-vn-0.4",
    "hi": "vosk-model-hi-0.22",
    "hr": "vosk-model-small-cnr-0.4",
    "sr": "vosk-model-small-cnr-0.4",
    "bs": "vosk-model-small-cnr-0.4",
}


@dataclass
class PipelineConfig:
    """Runtime options resolved from CLI flags and environment variables."""

    audio_path: str
    transcript_path: Optional[str] = None      # None -> ensemble transcription mode
    srt_path: Optional[str] = None             # optional rough SRT as extra timing source
    language: Optional[str] = None             # None -> auto-detect
    output_dir: Optional[str] = None           # default: alongside the audio file
    vosk_models_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("WORDALIGN_VOSK_MODELS"))
    mfa_cmd: Optional[str] = field(
        default_factory=lambda: os.environ.get("WORDALIGN_MFA"))
    use_vosk: bool = True
    use_mfa: bool = True
    use_tags: bool = False                     # experimental; off by default
    yamnet_confidence: float = 0.9
    whisper_model: str = "large-v3"
    ensemble_engines: tuple = ("whisperx", "parakeet", "vosk")
    device: Optional[str] = None               # None -> cuda if available
    doc_format: str = "txt"                    # ensemble transcript: none/txt/docx/both
    doc_timestamps: bool = True                # [HH:MM:SS] paragraph prefixes

    def vosk_model_path(self, language: str) -> Optional[str]:
        if not self.vosk_models_dir:
            return None
        name = VOSK_MODEL_MAP.get(language)
        if not name:
            return None
        path = Path(self.vosk_models_dir) / name
        return str(path) if path.exists() else None

    def resolve_output_base(self) -> str:
        base = os.path.splitext(os.path.basename(self.audio_path))[0]
        out_dir = self.output_dir or os.path.dirname(os.path.abspath(self.audio_path))
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, base)
