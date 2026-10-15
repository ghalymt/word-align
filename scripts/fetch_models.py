"""Download the models used by the robust (bad-audio) pipeline.

Everything lands under one models root (default ``<project>/models``, or
``WORDALIGN_MODELS_DIR``, or ``--models``), so the app stays portable and
nothing is read from ``~/.cache``:

    vad/silero_vad.onnx                       Silero VAD v6.2.1 (MIT)
    sed/PretrainedSED/...                     CP-JKU PretrainedSED code (MIT)
    sed/PretrainedSED/resources/BEATs_strong_1.pt
    enhance/checkpoints/MossFormerGAN_SE_16K/ ClearerVoice MossFormerGAN
    huggingface/hub/models--<org>--<name>/    Hugging Face cache layout:
        CohereLabs/cohere-transcribe-03-2026  (gated: accept on the model page)
        moondream/parakeet-ultra              (CC-BY-4.0)
        pyannote/speaker-diarization-community-1 (gated, CC-BY-4.0)
        BUT-FIT/SE-DiCoW                      (Apache-2.0)

The RoFormer separation checkpoints are fetched by audio-separator itself
(``scripts/install_robust.ps1`` does that inside the ``separate`` venv).

Run it with any Python that has ``huggingface_hub`` (the WhisperX or Qwen
backend venv works). Gated models need a Hugging Face token: ``HF_TOKEN``,
``--token``, or a prior ``huggingface-cli login``.

    python scripts/fetch_models.py                       # everything
    python scripts/fetch_models.py --only vad,sed        # a subset
    python scripts/fetch_models.py --list                # show what exists
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SILERO_VERSION = "v6.2.1"
SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_URLS = [
    f"https://github.com/snakers4/silero-vad/raw/{SILERO_VERSION}/src/silero_vad/data/silero_vad.onnx",
]
SILERO_WHEEL = ("https://files.pythonhosted.org/packages/py3/s/silero-vad/"
                "silero_vad-6.2.1-py3-none-any.whl")

PRETRAINED_SED_COMMIT = "1aa47e482f7e89904cba2338999345025d8b4e36"
PRETRAINED_SED_ZIP = ("https://codeload.github.com/fschmid56/PretrainedSED/zip/"
                      + PRETRAINED_SED_COMMIT)
BEATS_STRONG_URL = ("https://github.com/fschmid56/PretrainedSED/releases/"
                    "download/v0.0.1/BEATs_strong_1.pt")

HF_MODELS = {
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
    "parakeet": "moondream/parakeet-ultra",
    "diarize": "pyannote/speaker-diarization-community-1",
    "sedicow": "BUT-FIT/SE-DiCoW",
}
COMPONENTS = ("vad", "sed", "denoise", "cohere", "parakeet", "diarize", "sedicow")


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def models_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    if os.environ.get("WORDALIGN_MODELS_DIR"):
        return Path(os.environ["WORDALIGN_MODELS_DIR"])
    return ROOT / "models"


def _download(url: str, dest: Path, expected_sha256: str | None = None) -> None:
    """Stream *url* to *dest* atomically (a .part file renamed at the end)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "wordalign-fetch"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, time.time()
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            digest.update(block)
            done += len(block)
            if time.time() - last > 15 and total:
                log(f"  {dest.name}: {done / total:.0%} of {total / 1e6:.0f} MB")
                last = time.time()
    if expected_sha256 and digest.hexdigest() != expected_sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}")
    os.replace(part, dest)


def fetch_vad(root: Path) -> None:
    dest = root / "vad" / "silero_vad.onnx"
    if dest.is_file() and _sha256(dest) == SILERO_SHA256:
        log(f"vad: present ({dest})")
        return
    for url in SILERO_URLS:
        try:
            _download(url, dest, SILERO_SHA256)
            log(f"vad: downloaded {dest}")
            return
        except Exception as exc:  # fall back to the PyPI wheel
            log(f"vad: {url} failed ({exc}); trying the PyPI wheel")
    with tempfile.TemporaryDirectory() as tmp:
        wheel = Path(tmp) / "silero.whl"
        _download(SILERO_WHEEL, wheel)
        with zipfile.ZipFile(wheel) as zf:
            data = zf.read("silero_vad/data/silero_vad.onnx")
    if hashlib.sha256(data).hexdigest() != SILERO_SHA256:
        raise RuntimeError("Silero VAD from the wheel failed its checksum")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    log(f"vad: extracted {dest} from the PyPI wheel")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_sed(root: Path) -> None:
    code_dir = root / "sed" / "PretrainedSED"
    if not (code_dir / "models" / "beats").is_dir():
        log("sed: downloading PretrainedSED code "
            f"@{PRETRAINED_SED_COMMIT[:8]}")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "sed.zip"
            _download(PRETRAINED_SED_ZIP, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
            extracted = next(p for p in Path(tmp).iterdir()
                             if p.is_dir() and p.name.startswith("PretrainedSED"))
            if code_dir.exists():
                shutil.rmtree(code_dir)
            code_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(extracted), str(code_dir))
        (code_dir / "COMMIT").write_text(PRETRAINED_SED_COMMIT + "\n",
                                         encoding="utf-8")
    ckpt = code_dir / "resources" / "BEATs_strong_1.pt"
    if ckpt.is_file() and ckpt.stat().st_size > 1_000_000:
        log(f"sed: checkpoint present ({ckpt})")
        return
    log("sed: downloading BEATs_strong_1.pt")
    _download(BEATS_STRONG_URL, ckpt)
    log(f"sed: downloaded {ckpt}")


def _hf_token(cli_token: str | None):
    return cli_token or os.environ.get("HF_TOKEN") or None


def _snapshot(repo_id: str, token, **kwargs) -> str:
    from huggingface_hub import snapshot_download
    for attempt in range(1, 4):
        try:
            return snapshot_download(repo_id, token=token, **kwargs)
        except Exception as exc:
            name = type(exc).__name__
            if "Gated" in name or "401" in str(exc) or "403" in str(exc):
                raise RuntimeError(
                    f"{repo_id} is gated: accept its terms on "
                    f"https://huggingface.co/{repo_id} and log in "
                    "(huggingface-cli login) or set HF_TOKEN") from exc
            if attempt == 3:
                raise
            log(f"  {repo_id}: {name} ({exc}); retrying in {10 * attempt}s")
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def fetch_denoise(root: Path, token) -> None:
    dest = root / "enhance" / "checkpoints" / "MossFormerGAN_SE_16K"
    if (dest / "last_best_checkpoint").is_file():
        log(f"denoise: present ({dest})")
        return
    log("denoise: downloading alibabasglab/MossFormerGAN_SE_16K")
    _snapshot("alibabasglab/MossFormerGAN_SE_16K", token, local_dir=str(dest))
    log(f"denoise: downloaded {dest}")


def fetch_hf(root: Path, key: str, token) -> None:
    repo_id = HF_MODELS[key]
    cache = root / "huggingface" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    log(f"{key}: downloading {repo_id} into {cache}")
    path = _snapshot(repo_id, token, cache_dir=str(cache))
    log(f"{key}: ready at {path}")


def list_models(root: Path) -> None:
    checks = {
        "vad": root / "vad" / "silero_vad.onnx",
        "sed": root / "sed" / "PretrainedSED" / "resources" / "BEATs_strong_1.pt",
        "denoise": root / "enhance" / "checkpoints" / "MossFormerGAN_SE_16K"
                   / "last_best_checkpoint",
    }
    for key, repo in HF_MODELS.items():
        checks[key] = (root / "huggingface" / "hub"
                       / ("models--" + repo.replace("/", "--")) / "snapshots")
    for key in COMPONENTS:
        path = checks[key]
        print(f"  {key:9s} {'yes' if path.exists() else 'no ':3s} {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", help="models root (default: WORDALIGN_MODELS_DIR "
                                     "or <project>/models)")
    ap.add_argument("--only", help="comma-separated subset of: "
                                   + ",".join(COMPONENTS))
    ap.add_argument("--token", help="Hugging Face token for gated models")
    ap.add_argument("--list", action="store_true", help="show what is present")
    args = ap.parse_args(argv)
    root = models_root(args.models)
    if args.list:
        list_models(root)
        return 0
    wanted = [c.strip() for c in (args.only or ",".join(COMPONENTS)).split(",")
              if c.strip()]
    unknown = sorted(set(wanted) - set(COMPONENTS))
    if unknown:
        ap.error(f"unknown component(s): {', '.join(unknown)}")
    token = _hf_token(args.token)
    failed = []
    for key in wanted:
        try:
            if key == "vad":
                fetch_vad(root)
            elif key == "sed":
                fetch_sed(root)
            elif key == "denoise":
                fetch_denoise(root, token)
            else:
                fetch_hf(root, key, token)
        except Exception as exc:
            failed.append(key)
            log(f"{key}: FAILED: {exc}")
    log("done" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
