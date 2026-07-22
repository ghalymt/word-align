"""word-align preflight: check everything the pipeline needs, before a real run.

Usage:
    python preflight.py
    python preflight.py --vosk-models D:\\vosk-models --lang en

Exits 0 if the reference-mode path is ready, 1 if a hard requirement is
missing. Optional components are reported but never fail the check.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys

OK, WARN, BAD = "  OK  ", " WARN ", " MISS "
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""))


def check_binary(name: str, required: bool = True) -> bool:
    path = shutil.which(name)
    if path:
        record(OK, name, path)
        return True
    record(BAD if required else WARN, name, "not on PATH")
    return False


def check_import(module: str, label: str = "", required: bool = True) -> bool:
    label = label or module
    try:
        m = importlib.import_module(module)
        ver = getattr(m, "__version__", "")
        record(OK, label, str(ver))
        return True
    except Exception as exc:
        record(BAD if required else WARN, label, f"{type(exc).__name__}: {exc}")
        return False


def check_cuda() -> None:
    try:
        import torch
    except ImportError:
        record(BAD, "torch", "not installed")
        return
    record(OK, "torch", torch.__version__)
    if not torch.cuda.is_available():
        record(WARN, "CUDA", "not available -- the run will fall back to CPU "
                             "and take many times longer")
        return
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    vram = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
    record(OK, "CUDA", f"{name}, {vram:.1f} GB VRAM, "
                       f"torch cuda {torch.version.cuda}")
    if vram < 10:
        record(WARN, "VRAM headroom",
               "under 10 GB -- use --whisper-model medium, or expect OOM on "
               "large-v3 with the default batch size")


def check_ffmpeg_works() -> None:
    """PATH presence is not enough -- confirm ffprobe actually executes."""
    if not shutil.which("ffprobe"):
        return
    try:
        out = subprocess.run(["ffprobe", "-version"], capture_output=True,
                             text=True, timeout=15)
        first = (out.stdout or out.stderr).splitlines()[0]
        record(OK, "ffprobe runs", first[:60])
    except Exception as exc:
        record(BAD, "ffprobe runs", str(exc))


def check_vosk_models(models_dir: str | None, lang: str) -> None:
    if not models_dir:
        models_dir = os.environ.get("WORDALIGN_VOSK_MODELS")
    if not models_dir:
        record(WARN, "Vosk models",
               "no --vosk-models and no WORDALIGN_VOSK_MODELS; stage 1 of the "
               "waterfall will be skipped")
        return
    if not os.path.isdir(models_dir):
        record(WARN, "Vosk models", f"directory not found: {models_dir}")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from wordalign.config import VOSK_MODEL_MAP
    except Exception:
        record(WARN, "Vosk models", "run this from inside the repo to resolve "
                                    "the model name for your language")
        return
    expected = VOSK_MODEL_MAP.get(lang)
    if not expected:
        record(WARN, "Vosk models", f"no model mapped for language '{lang}'")
        return
    full = os.path.join(models_dir, expected)
    if os.path.isdir(full):
        record(OK, "Vosk model", full)
    else:
        record(WARN, "Vosk model",
               f"expected '{expected}' in {models_dir} -- download it from "
               "https://alphacephei.com/vosk/models")


def check_mfa() -> None:
    from pathlib import Path
    env = os.environ.get("WORDALIGN_MFA")
    if env:
        p = Path(env)
        if p.is_file() or (p.is_dir() and (
                (p / "Scripts" / "mfa.exe").exists()
                or (p / "bin" / "mfa").exists())):
            record(OK, "MFA", f"WORDALIGN_MFA -> {env}")
            return
        record(WARN, "MFA", f"WORDALIGN_MFA set but unusable: {env}")
        return
    if shutil.which("mfa"):
        record(OK, "MFA", shutil.which("mfa"))
        return
    record(WARN, "MFA", "not found -- stage 4 will be skipped (the waterfall "
                        "still runs; gaps fall through to interpolation)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vosk-models")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    print("=" * 66)
    print("word-align preflight")
    print("=" * 66)

    print("\n-- required -------------------------------------------------")
    check_binary("ffmpeg")
    check_binary("ffprobe")
    check_ffmpeg_works()
    check_cuda()
    check_import("whisperx")
    check_import("srt")
    check_import("langdetect")

    print("\n-- optional: waterfall stages -------------------------------")
    check_import("vosk", required=False)
    check_vosk_models(args.vosk_models, args.lang)
    check_mfa()
    check_import("textgrid", "textgrid (MFA output parsing)", required=False)

    print("\n-- optional: ensemble voters --------------------------------")
    check_import("nemo", "nemo_toolkit (Parakeet / Canary)", required=False)

    print("\n-- optional: outputs ----------------------------------------")
    check_import("docx", "python-docx (--doc docx)", required=False)

    print("\n" + "=" * 66)
    missing = [n for s, n, _ in _results if s == BAD]
    warned = [n for s, n, _ in _results if s == WARN]
    if missing:
        print(f"BLOCKED: {len(missing)} required component(s) missing:")
        for n in missing:
            print(f"   - {n}")
        print("\nReference mode cannot run until these are installed.")
        return 1
    print("READY: reference mode can run.")
    if warned:
        print(f"\n{len(warned)} optional component(s) unavailable -- these "
              f"degrade quality or skip stages, they do not block:")
        for n in warned:
            print(f"   - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
