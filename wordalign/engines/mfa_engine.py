"""Montreal Forced Aligner wrapper (MFA CLI v3.0+).

The ``mfa`` executable is resolved in this order:
    1. explicit ``mfa_cmd`` argument
    2. ``WORDALIGN_MFA`` environment variable (exe path *or* conda env root)
    3. ``mfa`` found on PATH
    4. common conda env locations (``aligner``/``aligner_v2`` under the
       active or default conda root)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set


def _candidate_env_roots() -> List[Path]:
    roots: List[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.append(Path(conda_prefix))
    home = Path.home()
    for base in (home / ".conda" / "envs", home / "miniconda3" / "envs",
                 home / "anaconda3" / "envs"):
        for name in ("aligner_v2", "aligner", "mfa"):
            roots.append(base / name)
    return roots


def _exe_in_env(env_root: Path) -> Optional[Path]:
    for rel in (("Scripts", "mfa.exe"), ("bin", "mfa")):
        cand = env_root.joinpath(*rel)
        if cand.exists():
            return cand
    return None


# MFA language model mapping. Maps ISO 639-1 codes to MFA model names.
# Covers every language WordAlign supports; languages without an MFA
# pretrained model fall back to English (get_mfa_models default).
MFA_LANGUAGE_MAP = {
    "en": {"acoustic": "english_us_arpa", "g2p": "english_us_arpa", "dictionary": "english_us_arpa"},
    "ar": {"acoustic": "arabic_mfa", "g2p": "arabic_mfa", "dictionary": "arabic_mfa"},
    "zh": {"acoustic": "mandarin_chinese_mfa", "g2p": "mandarin_chinese_pinyin_g2p", "dictionary": "mandarin_chinese_mfa"},
    "fr": {"acoustic": "french_mfa", "g2p": "french_mfa", "dictionary": "french_mfa"},
    "de": {"acoustic": "german_mfa", "g2p": "german_mfa", "dictionary": "german_mfa"},
    "es": {"acoustic": "spanish_mfa", "g2p": "spanish_mfa", "dictionary": "spanish_mfa"},
    "pt": {"acoustic": "portuguese_brazil_mfa", "g2p": "portuguese_brazil_mfa", "dictionary": "portuguese_brazil_mfa"},
    "it": {"acoustic": "italian_mfa", "g2p": "italian_mfa", "dictionary": "italian_mfa"},
    "nl": {"acoustic": "dutch_mfa", "g2p": "dutch_mfa", "dictionary": "dutch_mfa"},
    "ru": {"acoustic": "russian_mfa", "g2p": "russian_mfa", "dictionary": "russian_mfa"},
    "ja": {"acoustic": "japanese_mfa", "g2p": "japanese_mfa", "dictionary": "japanese_mfa"},
    "ko": {"acoustic": "korean_mfa", "g2p": "korean_mfa", "dictionary": "korean_mfa"},
    "fa": {"acoustic": "persian_mfa", "g2p": "persian_mfa", "dictionary": "persian_mfa"},
    "tr": {"acoustic": "turkish_mfa", "g2p": "turkish_mfa", "dictionary": "turkish_mfa"},
    "sv": {"acoustic": "swedish_mfa", "g2p": "swedish_mfa", "dictionary": "swedish_mfa"},
    "ca": {"acoustic": "catalan_mfa", "g2p": "catalan_mfa", "dictionary": "catalan_mfa"},
    "uk": {"acoustic": "ukrainian_mfa", "g2p": "ukrainian_mfa", "dictionary": "ukrainian_mfa"},
    "vi": {"acoustic": "southern_vietnamese_mfa", "g2p": "southern_vietnamese_mfa", "dictionary": "southern_vietnamese_mfa"},
    "hi": {"acoustic": "hindi_mfa", "g2p": "hindi_mfa", "dictionary": "hindi_mfa"},
    "hr": {"acoustic": "croatian_mfa", "g2p": "croatian_mfa", "dictionary": "croatian_mfa"},
    "sr": {"acoustic": "serbian_mfa", "g2p": "serbian_mfa", "dictionary": "serbian_mfa"},
    "bs": {"acoustic": "bosnian_mfa", "g2p": "bosnian_mfa", "dictionary": "bosnian_mfa"},
    "th": {"acoustic": "thai_mfa", "g2p": "thai_mfa", "dictionary": "thai_mfa"},
    "pl": {"acoustic": "polish_mfa", "g2p": "polish_mfa", "dictionary": "polish_mfa"},
    "cs": {"acoustic": "czech_mfa", "g2p": "czech_mfa", "dictionary": "czech_mfa"},
    "sk": {"acoustic": "slovak_mfa", "g2p": "slovak_mfa", "dictionary": "slovak_mfa"},
    "el": {"acoustic": "greek_mfa", "g2p": "greek_mfa", "dictionary": "greek_mfa"},
    "ro": {"acoustic": "romanian_mfa", "g2p": "romanian_mfa", "dictionary": "romanian_mfa"},
    "hu": {"acoustic": "hungarian_mfa", "g2p": "hungarian_mfa", "dictionary": "hungarian_mfa"},
    "he": {"acoustic": "hebrew_mfa", "g2p": "hebrew_mfa", "dictionary": "hebrew_mfa"},
    "fi": {"acoustic": "finnish_mfa", "g2p": "finnish_mfa", "dictionary": "finnish_mfa"},
    "da": {"acoustic": "danish_mfa", "g2p": "danish_mfa", "dictionary": "danish_mfa"},
    "no": {"acoustic": "norwegian_mfa", "g2p": "norwegian_mfa", "dictionary": "norwegian_mfa"},
    "id": {"acoustic": "indonesian_mfa", "g2p": "indonesian_mfa", "dictionary": "indonesian_mfa"},
    "ms": {"acoustic": "malay_mfa", "g2p": "malay_mfa", "dictionary": "malay_mfa"},
    "tl": {"acoustic": "tagalog_mfa", "g2p": "tagalog_mfa", "dictionary": "tagalog_mfa"},
    "bn": {"acoustic": "bengali_mfa", "g2p": "bengali_mfa", "dictionary": "bengali_mfa"},
    "ur": {"acoustic": "urdu_mfa", "g2p": "urdu_mfa", "dictionary": "urdu_mfa"},
    "ta": {"acoustic": "tamil_mfa", "g2p": "tamil_mfa", "dictionary": "tamil_mfa"},
    "te": {"acoustic": "telugu_mfa", "g2p": "telugu_mfa", "dictionary": "telugu_mfa"},
    "ne": {"acoustic": "nepali_mfa", "g2p": "nepali_mfa", "dictionary": "nepali_mfa"},
    "si": {"acoustic": "sinhala_mfa", "g2p": "sinhala_mfa", "dictionary": "sinhala_mfa"},
    "km": {"acoustic": "khmer_mfa", "g2p": "khmer_mfa", "dictionary": "khmer_mfa"},
    "lo": {"acoustic": "lao_mfa", "g2p": "lao_mfa", "dictionary": "lao_mfa"},
    "my": {"acoustic": "burmese_mfa", "g2p": "burmese_mfa", "dictionary": "burmese_mfa"},
    "hy": {"acoustic": "armenian_mfa", "g2p": "armenian_mfa", "dictionary": "armenian_mfa"},
    "ka": {"acoustic": "georgian_mfa", "g2p": "georgian_mfa", "dictionary": "georgian_mfa"},
    "bg": {"acoustic": "bulgarian_mfa", "g2p": "bulgarian_mfa", "dictionary": "bulgarian_mfa"},
    "mk": {"acoustic": "macedonian_mfa", "g2p": "macedonian_mfa", "dictionary": "macedonian_mfa"},
    "sl": {"acoustic": "slovenian_mfa", "g2p": "slovenian_mfa", "dictionary": "slovenian_mfa"},
    "sq": {"acoustic": "albanian_mfa", "g2p": "albanian_mfa", "dictionary": "albanian_mfa"},
    "gl": {"acoustic": "galician_mfa", "g2p": "galician_mfa", "dictionary": "galician_mfa"},
    "eu": {"acoustic": "basque_mfa", "g2p": "basque_mfa", "dictionary": "basque_mfa"},
    "sw": {"acoustic": "swahili_mfa", "g2p": "swahili_mfa", "dictionary": "swahili_mfa"},
    "am": {"acoustic": "amharic_mfa", "g2p": "amharic_mfa", "dictionary": "amharic_mfa"},
    "zu": {"acoustic": "zulu_mfa", "g2p": "zulu_mfa", "dictionary": "zulu_mfa"},
    "xh": {"acoustic": "xhosa_mfa", "g2p": "xhosa_mfa", "dictionary": "xhosa_mfa"},
    "yo": {"acoustic": "yoruba_mfa", "g2p": "yoruba_mfa", "dictionary": "yoruba_mfa"},
    "ig": {"acoustic": "igbo_mfa", "g2p": "igbo_mfa", "dictionary": "igbo_mfa"},
    "ha": {"acoustic": "hausa_mfa", "g2p": "hausa_mfa", "dictionary": "hausa_mfa"},
    "ceb": {"acoustic": "cebuano_mfa", "g2p": "cebuano_mfa", "dictionary": "cebuano_mfa"},
}


def get_mfa_models(language: str = "en") -> dict:
    """Get MFA acoustic, G2P, and dictionary model names for a language."""
    return MFA_LANGUAGE_MAP.get(language, MFA_LANGUAGE_MAP["en"])


class MFAWrapper:
    """Corpus prep, G2P dictionary extension, and alignment execution."""

    def __init__(self, work_dir: str, mfa_cmd: Optional[str] = None,
                 language: str = "en"):
        self.work_dir = Path(work_dir)
        self.corpus_dir = self.work_dir / "corpus"
        self.output_dir = self.work_dir / "mfa_output"
        self.env = os.environ.copy()
        self.mfa_exe = self._resolve_mfa(mfa_cmd)
        self.language = language
        self.mfa_models = get_mfa_models(language)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- executable resolution ---------------------------------------------
    def _resolve_mfa(self, mfa_cmd: Optional[str]) -> str:
        candidates: List[str] = []
        env_setting = mfa_cmd or os.environ.get("WORDALIGN_MFA")
        if env_setting:
            p = Path(env_setting)
            if p.is_dir():                       # treat as conda env root
                exe = _exe_in_env(p)
                if exe:
                    self._augment_path(p)
                    return str(exe)
            candidates.append(env_setting)
        which = shutil.which("mfa")
        if which:
            candidates.append(which)
        for env_root in _candidate_env_roots():
            exe = _exe_in_env(env_root)
            if exe:
                self._augment_path(env_root)
                candidates.append(str(exe))
                break
        if not candidates:
            raise FileNotFoundError(
                "Could not locate the MFA executable. Install Montreal Forced "
                "Aligner and either add `mfa` to PATH or set WORDALIGN_MFA.")
        return candidates[0]

    def _augment_path(self, env_root: Path) -> None:
        """Simulate conda activation so MFA finds _kalpy and its DLLs."""
        paths = [env_root, env_root / "Scripts", env_root / "bin",
                 env_root / "Library" / "bin",
                 env_root / "Library" / "usr" / "bin",
                 env_root / "Library" / "mingw-w64" / "bin"]
        self.env["PATH"] = (os.pathsep.join(str(p) for p in paths)
                            + os.pathsep + self.env.get("PATH", ""))
        self.env["CONDA_PREFIX"] = str(env_root)

    # -- corpus / dictionary / alignment -----------------------------------
    # NB: corpus construction (paired .wav/.lab files) lives in
    # ``align.run_mfa_on_gaps``, which owns the audio slicing; this class
    # only consumes the corpus directory it is pointed at.

    def detect_oov_words(self) -> Set[str]:
        all_words: Set[str] = set()
        for lab_file in self.corpus_dir.glob("*.lab"):
            try:
                for w in lab_file.read_text(encoding="utf-8").split():
                    clean_w = "".join(filter(str.isalnum, w)).lower()
                    if clean_w:
                        all_words.add(clean_w)
            except OSError as exc:
                print(f"[warn] Could not read {lab_file}: {exc}")
        return all_words

    def _get_model_path(self, model_type: str, model_name: str) -> str:
        # Look in the project's local models/mfa/pretrained_models first
        # (fully portable layout). Fall back to the historical
        # ~/Documents/MFA location only if the explicit env var is set,
        # so existing users on a non-portable install keep working.
        from ..models.paths import _project_models_root
        exts = {"g2p": ".zip", "dictionary": ".dict", "acoustic": ".zip"}
        candidates = [
            _project_models_root() / "mfa" / "pretrained_models",
        ]
        # Allow env-var override via the unified ModelPaths mechanism
        if os.environ.get("WORDALIGN_MFA_MODELS"):
            candidates.append(Path(os.environ["WORDALIGN_MFA_MODELS"]))
        # Last-resort legacy location (kept so previously installed setups still work)
        candidates.append(Path.home() / "Documents" / "MFA" / "pretrained_models")

        for mfa_root in candidates:
            candidate = mfa_root / model_type / (model_name + exts.get(model_type, ""))
            if candidate.exists():
                return str(candidate)
        # Return the local canonical path even if missing — caller can detect.
        return str(candidates[0] / model_type / (model_name + exts.get(model_type, "")))

    def generate_custom_dictionary(self, words: Set[str],
                                   model_name: Optional[str] = None) -> Path:
        if model_name is None:
            model_name = self.mfa_models["g2p"]
        input_words_path = self.work_dir / "words.txt"
        output_dict_path = self.work_dir / "custom_dict.txt"
        input_words_path.write_text(
            "\n".join(sorted(words)) + "\n", encoding="utf-8")
        cmd = [self.mfa_exe, "g2p", str(input_words_path),
               self._get_model_path("g2p", model_name),
               str(output_dict_path), "--clean", "--num_jobs", "4"]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, env=self.env)
        return output_dict_path

    def run_alignment(self, dictionary_path: str,
                      acoustic_model_name: Optional[str] = None) -> None:
        if acoustic_model_name is None:
            acoustic_model_name = self.mfa_models["acoustic"]
        cmd = [self.mfa_exe, "align", str(self.corpus_dir),
               str(dictionary_path),
               self._get_model_path("acoustic", acoustic_model_name),
               str(self.output_dir), "--clean",
               "--output_format", "long_textgrid",
               "--num_jobs", "4", "--single_speaker"]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, env=self.env)

    def cleanup(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)
