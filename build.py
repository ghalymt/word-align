# build.py — one-command Windows portable build
"""Build the portable WordAlign.exe distribution.

Usage:
    python build.py            # builds dist/WordAlign/ (folder)
    python build.py --zip      # also creates dist/WordAlign-portable.zip

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
    """Run PyInstaller with build.spec. Returns dist directory."""
    spec = ROOT / "build.spec"
    print(f"[1/3] Running PyInstaller with {spec.name} ...")
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm",
                    "--clean", str(spec)], cwd=str(ROOT), check=True)
    dist_dir = ROOT / "dist" / "WordAlign"
    if not dist_dir.exists():
        raise FileNotFoundError(f"Build output missing: {dist_dir}")
    return dist_dir


def copy_extra_files(dist_dir: Path) -> None:
    """Copy README, LICENSE, and docs into the bundle."""
    for name in ("README.md", "LICENSE", "TODO.md", "HANDOFF.md"):
        src = ROOT / name
        if src.exists():
            shutil.copy2(src, dist_dir / name)
    docs_dir = dist_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    for doc in (ROOT / "docs").glob("*.md"):
        shutil.copy2(doc, docs_dir / doc.name)


def make_zip(dist_dir: Path) -> Path:
    """Zip the bundle for distribution."""
    zip_path = ROOT / "dist" / "WordAlign-portable.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in dist_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(dist_dir.parent))
    print(f"    -> {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portable WordAlign")
    parser.add_argument("--zip", action="store_true",
                        help="also create a distributable ZIP")
    args = parser.parse_args()

    check_requirements()
    dist_dir = run_pyinstaller()
    print(f"[2/3] Copying extras into {dist_dir.name} ...")
    copy_extra_files(dist_dir)
    total = sum(f.stat().st_size for f in dist_dir.rglob("*") if f.is_file())
    print(f"    bundle size: {total / 1e6:.1f} MB")
    if args.zip:
        print("[3/3] Creating ZIP ...")
        make_zip(dist_dir)
    else:
        print("[3/3] Skipping ZIP (use --zip to create one)")
    print(f"\n[ok] Done. Portable app at: {dist_dir}")
    print("    End users double-click WordAlign.exe — no Python needed.")


if __name__ == "__main__":
    main()
