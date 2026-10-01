"""Unified model path configuration for all ASR engines.

WordAlign is **fully portable**: the canonical location for every model
lives inside the project, not in the user's ``~/.cache`` or ``%USERPROFILE%``.

Local model layout (resolved at runtime):

    <project_root>/models/
        vosk/                  unpacked Vosk models  (vosk-model-en-us-0.22/, ...)
        whisper/               Whisper .pt / .bin  +  huggingface/hub/ inside
        huggingface/hub/       HuggingFace snapshot cache (Qwen, faster-whisper, ...)
        llama.cpp/             llama.cpp binaries (llama-cli.exe + .dll)
        llm/                   GGUF models for the LLM QA / punctuation stage
        mfa/pretrained_models/ Montreal-Forced-Aligner acoustic + dict + g2p

The project root is detected as follows:

* Source checkout  →  two parents above this file (``wordalign/models/paths.py``)
* PyInstaller exe  →  directory of ``sys.executable`` (``WordAlign.exe``)

Every field on :class:`ModelPaths` defaults to ``<project_root>/models/<key>``.
Environment variables (e.g. ``WORDALIGN_VOSK_MODELS``) override the default;
they are the only way to point the app at a different location. There is
**no** ``~/.cache`` fallback — portability is the whole point.

CLI flags (e.g. ``--vosk-models``) take precedence over environment variables.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Project root resolution (source checkout + PyInstaller frozen bundle)
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    """Absolute path to the WordAlign project root.

    Source: ``<repo>/wordalign/models/paths.py`` → repo is two parents up.
    Frozen: directory containing ``WordAlign.exe`` (sys.executable). If
    ``models/`` is not found there, walk up to the parent so a layout like
    ``<project>/dist/WordAlign.exe`` + ``<project>/models/`` works.
    """
    if getattr(sys, "frozen", False):
        # PyInstaller bundle — sys.executable points to WordAlign.exe
        start = Path(sys.executable).resolve().parent
        # A models/ dir is only useful if it actually contains at least one
        # engine subdir. This skips empty placeholder dirs that may exist
        # next to a partially-built bundle.
        known = {"vosk", "whisper", "huggingface", "llama.cpp", "llm", "mfa", "yamnet"}
        for candidate in (start, *start.parents):
            mp = candidate / "models"
            if mp.is_dir() and any((mp / name).is_dir() for name in known):
                return candidate
        return start
    # Source checkout — this file is wordalign/models/paths.py
    return Path(__file__).resolve().parent.parent.parent


def _project_models_root() -> Path:
    """Canonical ``<project_root>/models`` directory.

    Always returned even if the directory does not exist yet — callers that
    need an existing path should call ``.exists()`` first.
    """
    return _project_root() / "models"


# ---------------------------------------------------------------------------
# Default per-engine subdirectory under models/
# ---------------------------------------------------------------------------
def _models_subdir(name: str) -> Optional[str]:
    """Return ``<models_root>/<name>`` if that directory exists, else None."""
    candidate = _project_models_root() / name
    return str(candidate) if candidate.exists() else None


# ---------------------------------------------------------------------------
# ModelPaths — unified configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelPaths:
    """Resolved paths for all ASR model types.

    Every field is ``Optional[str]``: ``None`` means "not configured, use
    auto-detect" (which in this build is the local ``<project>/models/``).
    Set the field explicitly to override; explicit settings never fall back.
    """

    vosk_models_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("WORDALIGN_VOSK_MODELS")
                                or _models_subdir("vosk"))
    whisper_models_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("WORDALIGN_WHISPER_MODELS")
                                or _models_subdir("whisper"))
    qwen_models_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("WORDALIGN_QWEN_MODELS")
                                or _models_subdir("huggingface/hub"))
    huggingface_cache_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("WORDALIGN_HF_CACHE")
                                or _models_subdir("huggingface/hub"))
    mfa_models_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("WORDALIGN_MFA_MODELS")
                                or _models_subdir("mfa/pretrained_models"))

    # Extra custom paths: {"model_type": "/path/to/models"}
    custom_paths: Dict[str, str] = field(default_factory=dict)

    # -- internal tracking --
    _explicit: set = field(default_factory=set)

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "ModelPaths":
        """Create from environment variables (no auto-detect)."""
        return cls()

    @classmethod
    def auto_detect(cls) -> "ModelPaths":
        """Create with the local project ``models/`` folder filled in."""
        return cls()

    def _fill_auto(self):
        """Fill None fields with the local ``models/`` subdir if it exists."""
        if self.vosk_models_dir is None:
            self.vosk_models_dir = _models_subdir("vosk")
        if self.whisper_models_dir is None:
            self.whisper_models_dir = _models_subdir("whisper")
        if self.qwen_models_dir is None:
            self.qwen_models_dir = _models_subdir("huggingface/hub")
        if self.huggingface_cache_dir is None:
            self.huggingface_cache_dir = _models_subdir("huggingface/hub")
        if self.mfa_models_dir is None:
            self.mfa_models_dir = _models_subdir("mfa/pretrained_models")

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve(self, model_type: str,
                model_name: Optional[str] = None) -> Optional[Path]:
        """Resolve a model directory for *model_type*.

        Priority order:
        1. Explicitly-set field on this instance
        2. Custom path for this model_type
        3. Environment variable default
        4. Local ``<project>/models/`` subdir

        Returns ``None`` only when no path is configured AND the local
        directory does not exist.
        """
        # 1. Custom path
        if model_type in self.custom_paths:
            p = Path(self.custom_paths[model_type])
            if p.exists():
                return p

        # 2. Typed field
        field_map = {
            "vosk": self.vosk_models_dir,
            "whisper": self.whisper_models_dir,
            "whisperx": self.whisper_models_dir,
            "qwen": self.qwen_models_dir,
            "qwen_asr": self.qwen_models_dir,
            "huggingface": self.huggingface_cache_dir,
            "mfa": self.mfa_models_dir,
        }
        typed = field_map.get(model_type, self.huggingface_cache_dir)
        if typed:
            p = Path(typed)
            if p.exists():
                return p

        # 3. HuggingFace cache as fallback for HF-hosted model types
        hf_types = {"whisper", "whisperx", "qwen", "qwen_asr", "huggingface"}
        if model_type in hf_types and self.huggingface_cache_dir:
            p = Path(self.huggingface_cache_dir)
            if p.exists():
                return p

        return None

    def resolve_model(self, model_type: str,
                      model_name: str) -> Optional[Path]:
        """Resolve a *specific* model directory within the model type root.

        e.g. ``resolve_model("whisper", "large-v3")`` looks for:
            ``<whisper_models_dir>/large-v3.pt``
            ``<huggingface_cache>/models--Systran--faster-whisper-large-v3``
        """
        root = self.resolve(model_type)
        if not root:
            return None

        # Direct file (e.g. whisper large-v3.pt)
        candidates: List[Path] = []
        if model_type in ("whisper", "whisperx"):
            candidates = [
                root / f"{model_name}.pt",
                root / f"faster-whisper-{model_name}",
                root / f"models--Systran--faster-whisper-{model_name}",
                root / "snapshots" / model_name,
            ]
        elif model_type in ("qwen", "qwen_asr"):
            # Qwen models are in HF cache format
            org, _, name = model_name.partition("/")
            if org and name:
                # "Qwen/Qwen3-ASR-1.7B" → "models--Qwen--Qwen3-ASR-1.7B"
                safe_name = model_name.replace("/", "--")
                candidates = [
                    root / f"models--{safe_name}",
                    root / f"models--{org}--{name}",
                ]
            candidates.append(root / model_name)
        elif model_type == "vosk":
            candidates = [root / model_name]
        else:
            candidates = [root / model_name]

        for c in candidates:
            if c.exists():
                return c

        # Model not found at custom path — return root only if it exists
        return None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Export all paths as a dict (suitable for JSON/serialization)."""
        return {
            "vosk_models_dir": self.vosk_models_dir,
            "whisper_models_dir": self.whisper_models_dir,
            "qwen_models_dir": self.qwen_models_dir,
            "huggingface_cache_dir": self.huggingface_cache_dir,
            "mfa_models_dir": self.mfa_models_dir,
            "custom_paths": dict(self.custom_paths),
        }

    def summary(self) -> str:
        """Human-readable one-line-per-model summary."""
        lines = [
            f"Project root : {_project_root()}",
            f"Models root  : {_project_models_root()}",
            "Model paths  :",
        ]
        self._fill_auto()
        for label, path in [
            ("Vosk      ", self.vosk_models_dir),
            ("Whisper   ", self.whisper_models_dir),
            ("Qwen      ", self.qwen_models_dir),
            ("HF Cache  ", self.huggingface_cache_dir),
            ("MFA       ", self.mfa_models_dir),
        ]:
            status = path if path else "(not found)"
            lines.append(f"  {label} {status}")
        if self.custom_paths:
            lines.append("  Custom:")
            for k, v in self.custom_paths.items():
                lines.append(f"    {k} → {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM (llama.cpp + GGUF) helpers — also defaulted to <models>/llama.cpp etc.
# ---------------------------------------------------------------------------
def _llama_cpp_dir() -> Optional[str]:
    """Default directory containing ``llama-cli.exe`` (or unix equivalent)."""
    if os.environ.get("WORDALIGN_LLM_ENGINE"):
        return os.environ["WORDALIGN_LLM_ENGINE"]
    candidate = _project_models_root() / "llama.cpp"
    if any((candidate / name).exists() for name in ("llama-cli.exe", "llama-cli")):
        return str(candidate)
    return None


def _default_llm_model_path() -> Optional[str]:
    """Default GGUF file for the LLM QA stage.

    Looks for the first ``*.gguf`` under ``<models>/llm`` and returns the
    largest one (best quality / most parameters).
    """
    if os.environ.get("WORDALIGN_LLM_MODEL"):
        return os.environ["WORDALIGN_LLM_MODEL"]
    llm_dir = _project_models_root() / "llm"
    if not llm_dir.exists():
        return None
    ggufs = sorted(llm_dir.glob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    return str(ggufs[0]) if ggufs else None


def _default_llm_mtp_model_path() -> Optional[str]:
    """Default MTP draft GGUF for speculative decoding.

    Looks for any ``mtp-*.gguf`` under ``<models>/llm``; falls back to a
    smaller ``*.gguf`` so the main model can act as its own draft.
    """
    if os.environ.get("WORDALIGN_LLM_MTP_MODEL"):
        return os.environ["WORDALIGN_LLM_MTP_MODEL"]
    llm_dir = _project_models_root() / "llm"
    if not llm_dir.exists():
        return None
    mtps = sorted(llm_dir.glob("mtp-*.gguf"), key=lambda p: p.stat().st_size, reverse=True)
    if mtps:
        return str(mtps[0])
    # Fallback: smallest non-mmproj GGUF
    ggufs = [p for p in llm_dir.glob("*.gguf") if "mmproj" not in p.name]
    ggufs = sorted(ggufs, key=lambda p: p.stat().st_size)
    return str(ggufs[0]) if ggufs else None
