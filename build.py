# build.py — one-command Windows portable build
"""Build the portable WordAlign.exe distribution.

Usage:
    python build.py            # builds dist/WordAlign/ (folder)
    python build.py --zip      # also creates dist/WordAlign-portable.zip
                               # (app only; models are downloaded separately)
    python build.py --zip --zip-with-models
                               # full offline bundle (tens of GB; too big
                               # for a GitHub release asset)

Requirements:
    pip install pyinstaller
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def check_requirements() -> None:
    """Verify build prerequisites are present."""
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[error] PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)
    if sys.platform != "win32":
        print("[warn] Portable .exe builds are Windows-only. "
              "You can still run: python -m wordalign")


def run_pyinstaller() -> Path:
    """Run PyInstaller with build.spec. Returns the bundle directory or exe.

    The spec produces a one-folder bundle. This function returns the
    bundle directory, or a one-file executable when a custom spec supplies one.
    """
    spec = ROOT / "build.spec"
    print(f"[1/3] Running PyInstaller with {spec.name} ...")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    "--clean", str(spec)], cwd=str(ROOT), check=True)
    dist_dir = ROOT / "dist" / "WordAlign"          # one-folder layout
    exe_path = ROOT / "dist" / "WordAlign.exe"      # one-file layout
    if dist_dir.exists():
        return dist_dir
    if exe_path.exists():
        return exe_path
    raise FileNotFoundError(
        f"Build output missing: {dist_dir} or {exe_path}")


def copy_extra_files(dist_dir: Path, include_models: bool = True,
                     copy_models: bool = False) -> None:
    """Copy README, LICENSE, docs, and the local ``models/`` folder into
    the bundle.

    For one-file builds, copies them next to the exe instead.

    The ``models/`` folder is the whole point of portability — every
    engine (Vosk, Whisper, Qwen, MFA, llama.cpp, GGUF LLM) reads from
    here. If it's missing from the bundle, end users get a "no backend"
    error when they try to subtitle anything.
    """
    target = dist_dir if dist_dir.is_dir() else dist_dir.parent
    for name in ("README.md", "LICENSE", "TODO.md", "HANDOFF.md"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, target / name)
    docs_dir = target / "docs"
    docs_dir.mkdir(exist_ok=True)
    for doc in (ROOT / "docs").glob("*.md"):
        shutil.copy2(doc, docs_dir / doc.name)

    launcher_src = ROOT / "scripts" / "Launch WordAlign.vbs"
    if launcher_src.exists():
        shutil.copy2(launcher_src, target / "Launch WordAlign.vbs")

    if not include_models:
        print("    [info] Skipping models/ staging (--skip-models).")
        return

    # Include the local models/ folder so the exe is truly portable.
    # The folder is large (~70 GB for the full quality set); use a
    # junction when possible to avoid duplicating tens of GB on the
    # build host. Falls back to a real copy otherwise.
    src_models = ROOT / "models"
    if not src_models.exists():
        print(f"    [warn] {src_models} missing — exe will not be portable. "
              f"Run scripts/copy_models.ps1 to populate it.")
        return
    dst_models = target / "models"
    if dst_models.exists():
        if dst_models.is_dir() and any(dst_models.iterdir()):
            return
        try:
            dst_models.rmdir()
        except OSError:
            print(f"    [warn] Existing models/ is not empty: {dst_models}")
            return
    if copy_models:
        print("    [info] copying models/ for a relocatable release bundle ...")
        shutil.copytree(src_models, dst_models, dirs_exist_ok=True)
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(dst_models), str(src_models)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            raise OSError("directory junctions are only available on Windows")
        if not dst_models.exists():
            raise OSError("mklink failed silently")
        print(f"    [ok] models/ → junction to {src_models}")
        print("    [warn] Junction bundles are for local testing; use --copy-models for release ZIPs.")
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        print(f"    [info] copying models/ into bundle (large; may take a while): {detail.strip()}")
        shutil.copytree(src_models, dst_models, dirs_exist_ok=True)


# GitHub rejects release assets of 2 GiB or more.
GITHUB_ASSET_LIMIT = 2 * 1024 ** 3

MODELS_README = """\
WordAlign -- getting the models
===============================

This download contains the application only. The speech models are large
(hundreds of MB to several GB each) and are installed separately into the
models\\ folder next to WordAlign.exe:

  models\\vosk\\        Vosk models (CPU), e.g. vosk-model-en-us-0.22
                        https://alphacephei.com/vosk/models
  models\\whisper\\     faster-whisper / WhisperX models (GPU)
  models\\huggingface\\ Hugging Face cache (Qwen3-ASR, alignment models)
  models\\mfa\\         Montreal Forced Aligner models
                        (mfa model download acoustic|g2p <name>)
  models\\llm\\         GGUF model for smart punctuation (optional)

Start WordAlign and open Settings -> Model Paths: it shows which models it
found, and every path can point somewhere else (for example a shared drive).
Engines whose models are missing are skipped automatically.
"""


def make_zip(dist_dir: Path, include_models: bool = False) -> Path:
    """Zip the bundle for distribution.

    The models/ tree is left out unless *include_models*: it is tens of GB,
    far over GitHub's 2 GiB release-asset limit, and users install the
    models they need separately (MODELS.txt in the ZIP says how).
    """
    zip_path = ROOT / "dist" / "WordAlign-portable.zip"
    if dist_dir.is_dir():
        models_dir = dist_dir / "models"
        files = [f for f in dist_dir.rglob("*") if f.is_file()
                 and (include_models or models_dir not in f.parents)]
    else:
        files = [dist_dir]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            zf.write(file, file.name if dist_dir.is_file()
                     else file.relative_to(dist_dir.parent))
        if not include_models:
            top = dist_dir.name if dist_dir.is_dir() else ""
            zf.writestr(f"{top}/MODELS.txt" if top else "MODELS.txt",
                        MODELS_README)
    size = zip_path.stat().st_size
    print(f"    -> {zip_path} ({size / 1e6:.1f} MB"
          f"{', models included' if include_models else ', models excluded'})")
    if size >= GITHUB_ASSET_LIMIT:
        print("    [warn] The ZIP is over GitHub's 2 GiB release-asset limit; "
              "build the release without --zip-with-models.")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portable WordAlign")
    parser.add_argument("--zip", action="store_true",
                        help="also create a distributable ZIP")
    parser.add_argument("--skip-models", action="store_true",
                        help="do not stage the local models/ tree")
    parser.add_argument("--copy-models", action="store_true",
                        help="copy models/ instead of creating a junction")
    parser.add_argument("--zip-only", action="store_true",
                        help="create the ZIP from an existing dist/WordAlign bundle")
    parser.add_argument("--zip-with-models", action="store_true",
                        help="put models/ in the ZIP too (offline bundle; "
                             "too large for a GitHub release)")
    args = parser.parse_args()

    if args.zip_only:
        dist_dir = ROOT / "dist" / "WordAlign"
        if not dist_dir.exists():
            raise FileNotFoundError(f"Build output missing: {dist_dir}")
        make_zip(dist_dir, include_models=args.zip_with_models)
        return

    check_requirements()
    dist_dir = run_pyinstaller()
    print(f"[2/3] Copying extras into {dist_dir.name} ...")
    copy_extra_files(dist_dir, include_models=not args.skip_models,
                     copy_models=args.copy_models)
    if dist_dir.is_dir():
        total = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
    else:
        total = dist_dir.stat().st_size
    print(f"    bundle size: {total / 1e6:.1f} MB")
    if args.zip:
        print("[3/3] Creating ZIP ...")
        make_zip(dist_dir, include_models=args.zip_with_models)
    else:
        print("[3/3] Skipping ZIP (use --zip to create one)")
    print(f"\n[ok] Done. Bundle at: {dist_dir}")
    if not args.copy_models and not args.skip_models:
        print("    Models are linked for local testing; use --copy-models for a relocatable release.")
    print("    End users double-click WordAlign.exe — no Python needed.")


if __name__ == "__main__":
    main()
