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
# English is the default; add more as you install MFA models.
MFA_LANGUAGE_MAP = {
    "en": {
        "acoustic": "english_us_arpa",
        "g2p": "english_us_arpa",
        "dictionary": "english_us_arpa",
    },
    "ar": {
        "acoustic": "arabic_mfa",
        "g2p": "arabic_mfa",
        "dictionary": "arabic_mfa",
    },
    "zh": {
        "acoustic": "mandarin_chinese_mfa",
        "g2p": "mandarin_chinese_pinyin_g2p",
        "dictionary": "mandarin_chinese_mfa",
    },
    "fr": {
        "acoustic": "french_mfa",
        "g2p": "french_mfa",
        "dictionary": "french_mfa",
    },
    "de": {
        "acoustic": "german_mfa",
        "g2p": "german_mfa",
        "dictionary": "german_mfa",
    },
    "es": {
        "acoustic": "spanish_mfa",
        "g2p": "spanish_mfa",
        "dictionary": "spanish_mfa",
    },
    "pt": {
        "acoustic": "portuguese_brazil_mfa",
        "g2p": "portuguese_brazil_mfa",
        "dictionary": "portuguese_brazil_mfa",
    },
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
        mfa_root = Path.home() / "Documents" / "MFA" / "pretrained_models"
        exts = {"g2p": ".zip", "dictionary": ".dict", "acoustic": ".zip"}
        candidate = mfa_root / model_type / (model_name + exts.get(model_type, ""))
        return str(candidate) if candidate.exists() else model_name

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
