"""Locate the Python interpreters of the isolated backend venvs.

Heavy models run out-of-process, each in a venv whose torch/transformers
pins would clash with the others (see scripts/build_backends.ps1 and
scripts/install_robust.ps1). This module finds those interpreters:

1. the backend's environment variable (``WORDALIGN_<NAME>_PYTHON``);
2. ``<venvs>/<name>/Scripts/python.exe`` (or ``bin/python``), where
   ``<venvs>`` is ``WORDALIGN_VENVS_DIR``, ``<project>/venvs`` or
   ``<project>/dist/WordAlign/venvs``;
3. for backends that share a venv (sound events, diarization and SE-DiCoW
   run in the WhisperX venv), the venv they share.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

BACKEND_ENV: Dict[str, str] = {
    "qwen": "WORDALIGN_QWEN_PYTHON",
    "whisperx": "WORDALIGN_WHISPERX_PYTHON",
    "cohere": "WORDALIGN_COHERE_PYTHON",
    "parakeet": "WORDALIGN_PARAKEET_PYTHON",
    "separate": "WORDALIGN_SEPARATE_PYTHON",
    "denoise": "WORDALIGN_DENOISE_PYTHON",
    "sed": "WORDALIGN_SED_PYTHON",
    "diarize": "WORDALIGN_DIARIZE_PYTHON",
    "sedicow": "WORDALIGN_SEDICOW_PYTHON",
}

# Backends without a venv of their own run in another backend's venv.
SHARED_VENV: Dict[str, str] = {
    "sed": "whisperx",
    "diarize": "whisperx",
    "sedicow": "whisperx",
}


def venv_roots() -> List[Path]:
    """Directories that may contain ``<backend>/`` venvs, most specific first."""
    from .models.paths import _project_root
    roots: List[Path] = []
    env = os.environ.get("WORDALIGN_VENVS_DIR")
    if env:
        roots.append(Path(env))
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent / "venvs")
    project = _project_root()
    roots += [project / "venvs", project / "dist" / "WordAlign" / "venvs"]
    seen, out = set(), []
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _venv_python(root: Path, name: str) -> Optional[str]:
    for rel in (("Scripts", "python.exe"), ("bin", "python"), ("python.exe",),
                ("python",)):
        candidate = root / name / Path(*rel)
        if candidate.is_file():
            return str(candidate)
    return None


def backend_python(name: str) -> Optional[str]:
    """Interpreter for backend *name*, or None when it is not installed."""
    env_var = BACKEND_ENV.get(name)
    if env_var:
        value = os.environ.get(env_var)
        if value and os.path.isfile(value):
            return value
    for root in venv_roots():
        found = _venv_python(root, name)
        if found:
            return found
    shared = SHARED_VENV.get(name)
    if shared:
        return backend_python(shared)
    return None


def backend_status() -> Dict[str, Optional[str]]:
    """Every known backend -> its interpreter (None if missing)."""
    return {name: backend_python(name) for name in BACKEND_ENV}
