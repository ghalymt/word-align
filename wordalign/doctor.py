"""``wordalign --doctor``: what the robust flow can use on this machine."""
from __future__ import annotations

import os
import shutil
from typing import List, Tuple


def _hub_has(hub: str, repo: str) -> bool:
    return bool(hub) and os.path.isdir(os.path.join(hub, "models--" + repo.replace("/", "--")))


def collect() -> List[Tuple[str, str, bool, str]]:
    """Rows of (stage, component, ok, detail)."""
    from .backends import backend_python
    from .models.paths import ModelPaths, _project_models_root
    root = str(_project_models_root())
    mp = ModelPaths()
    mp._fill_auto()
    hub = mp.huggingface_cache_dir or os.path.join(root, "huggingface", "hub")
    rows: List[Tuple[str, str, bool, str]] = []

    def add(stage, name, ok, detail=""):
        rows.append((stage, name, bool(ok), detail))

    ffmpeg = os.environ.get("WORDALIGN_FFMPEG") or shutil.which("ffmpeg")
    add("core", "ffmpeg", ffmpeg, ffmpeg or "not found (set WORDALIGN_FFMPEG)")
    try:
        import onnxruntime  # noqa: F401
        ort = True
    except ImportError:
        ort = False
    vad = os.path.join(root, "vad", "silero_vad.onnx")
    add("1 analysis", "Silero VAD", ort and os.path.isfile(vad),
        vad if ort else "onnxruntime not installed")
    sed = os.path.join(root, "sed", "PretrainedSED", "resources", "BEATs_strong_1.pt")
    add("1 analysis", "sound events (PretrainedSED BEATs)",
        os.path.isfile(sed) and backend_python("sed"), sed)
    sep = os.path.join(root, "separation")
    add("2 cleanup", "RoFormer separation", backend_python("separate") and os.path.isdir(sep),
        backend_python("separate") or "venv 'separate' missing")
    dn = os.path.join(root, "denoise", "checkpoints", "MossFormerGAN_SE_16K")
    add("2 cleanup", "MossFormerGAN", backend_python("denoise") and os.path.isdir(dn),
        backend_python("denoise") or "venv 'denoise' missing")
    add("3 voting", "Cohere Transcribe",
        backend_python("cohere") and _hub_has(hub, "CohereLabs/cohere-transcribe-03-2026"),
        backend_python("cohere") or "venv 'cohere' missing")
    add("3 voting", "Qwen3-ASR-1.7B",
        backend_python("qwen") and _hub_has(hub, "Qwen/Qwen3-ASR-1.7B"),
        backend_python("qwen") or "venv 'qwen' missing")
    add("3 voting", "parakeet-ultra",
        backend_python("parakeet") and _hub_has(hub, "moondream/parakeet-ultra"),
        backend_python("parakeet") or "venv 'parakeet' missing")
    add("3 voting", "WhisperX", backend_python("whisperx"),
        backend_python("whisperx") or "venv 'whisperx' missing")
    add("4 crosstalk", "pyannote Community-1",
        backend_python("diarize") and _hub_has(hub, "pyannote/speaker-diarization-community-1"),
        "accept the terms on huggingface.co if missing")
    add("4 crosstalk", "SE-DiCoW",
        backend_python("sedicow") and _hub_has(hub, "BUT-FIT/SE-DiCoW"), "")
    add("5 timing", "Qwen3-ForcedAligner-0.6B",
        backend_python("qwen") and _hub_has(hub, "Qwen/Qwen3-ForcedAligner-0.6B"), "")
    add("5 timing", "Vosk models", mp.vosk_models_dir, mp.vosk_models_dir or "")
    add("5 timing", "MFA", os.environ.get("WORDALIGN_MFA") or shutil.which("mfa"),
        os.environ.get("WORDALIGN_MFA") or shutil.which("mfa") or "")
    return rows


def print_report() -> None:
    from .models.paths import _project_models_root
    print(f"Models root: {_project_models_root()}")
    stage = None
    for st, name, ok, detail in collect():
        if st != stage:
            print(f"\n[{st}]")
            stage = st
        mark = "ok " if ok else "-- "
        print(f"  {mark} {name:<36} {detail}")
    print("\nMissing parts are skipped at run time with a message; install them "
          "with scripts/install_robust.ps1.")
