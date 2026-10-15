"""Download the models used by word-align's bad-audio stages.

Standalone on purpose: it imports only the standard library (plus
``huggingface_hub`` for Hugging Face repos), so it runs under any of the
backend venvs before the ``wordalign`` package is importable there.

    python scripts/fetch_models.py --models-dir E:/Wordalign/models vad sed cohere

Every model lands under ``--models-dir`` (the same folder word-align reads
through ``WORDALIGN_MODELS_DIR`` or ``<project>/models``):

    vad/silero_vad.onnx                     Silero VAD 6.2.1 (MIT)
    sed/PretrainedSED/...                   PretrainedSED code + BEATs strong checkpoint (MIT)
    huggingface/hub/models--<org>--<name>   Hugging Face snapshots (Cohere, parakeet-ultra,
                                            pyannote Community-1, SE-DiCoW)
    denoise/checkpoints/MossFormerGAN_SE_16K  ClearerVoice MossFormerGAN (Apache-2.0)

Separation (RoFormer) checkpoints are fetched by audio-separator itself
(see scripts/install_robust.ps1), because it knows its own mirrors.

Gated Hugging Face repos (Cohere Transcribe, pyannote Community-1) need a
token: log in once with ``huggingface-cli login`` or set ``HF_TOKEN``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SILERO_VERSION = "6.2.1"
SILERO_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_URL = ("https://raw.githubusercontent.com/snakers4/silero-vad/"
              f"v{SILERO_VERSION}/src/silero_vad/data/silero_vad.onnx")

PRETRAINED_SED_COMMIT = "1aa47e482f7e89904cba2338999345025d8b4e36"
PRETRAINED_SED_ZIP = ("https://codeload.github.com/fschmid56/PretrainedSED/zip/"
                      + PRETRAINED_SED_COMMIT)
PRETRAINED_SED_CKPT = ("https://github.com/fschmid56/PretrainedSED/releases/"
                       "download/v0.0.1/BEATs_strong_1.pt")

HF_REPOS = {
    "cohere": "CohereLabs/cohere-transcribe-03-2026",
    "parakeet": "moondream/parakeet-ultra",
    "diarize": "pyannote/speaker-diarization-community-1",
    "sedicow": "BUT-FIT/SE-DiCoW",
}
DENOISE_REPO = "alibabasglab/MossFormerGAN_SE_16K"

COMPONENTS = ("vad", "sed", "cohere", "parakeet", "diarize", "sedicow", "denoise")


def log(msg: str) -> None:
    print(f"[fetch] {msg}", flush=True)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "word-align"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    os.replace(tmp, dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_vad(models: Path) -> None:
    dest = models / "vad" / "silero_vad.onnx"
    if dest.exists() and _sha256(dest) == SILERO_SHA256:
        log(f"vad: already present ({dest})")
        return
    try:
        _download(SILERO_URL, dest)
        if _sha256(dest) == SILERO_SHA256:
            log(f"vad: ok ({dest})")
            return
        log("vad: checksum mismatch from GitHub; trying the PyPI wheel")
    except Exception as exc:  # network / 404
        log(f"vad: GitHub download failed ({exc}); trying the PyPI wheel")
    meta_url = f"https://pypi.org/pypi/silero-vad/{SILERO_VERSION}/json"
    with urllib.request.urlopen(meta_url, timeout=60) as resp:
        meta = json.load(resp)
    wheel = next(u for u in meta["urls"] if u["filename"].endswith(".whl"))
    with urllib.request.urlopen(wheel["url"], timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        payload = zf.read("silero_vad/data/silero_vad.onnx")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    if _sha256(dest) != SILERO_SHA256:
        raise RuntimeError("Silero VAD checksum mismatch; refusing to use it")
    log(f"vad: ok from PyPI ({dest})")


def fetch_sed(models: Path) -> None:
    root = models / "sed" / "PretrainedSED"
    marker = root / ".commit"
    if not (marker.exists() and marker.read_text().strip() == PRETRAINED_SED_COMMIT):
        log("sed: fetching PretrainedSED code")
        with tempfile.TemporaryDirectory() as tmp:
            zpath = _download(PRETRAINED_SED_ZIP, Path(tmp) / "sed.zip")
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(tmp)
            src = next(p for p in Path(tmp).iterdir()
                       if p.is_dir() and p.name.startswith("PretrainedSED"))
            if root.exists():
                # Keep the superseded checkout instead of deleting it.
                old = root.with_name(f"PretrainedSED.previous-{_stamp()}")
                shutil.move(str(root), str(old))
                log(f"sed: kept the previous checkout as {old.name}")
            root.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(root))
        marker.write_text(PRETRAINED_SED_COMMIT)
    ckpt = root / "resources" / "BEATs_strong_1.pt"
    if not ckpt.exists():
        _download(PRETRAINED_SED_CKPT, ckpt)
    log(f"sed: ok ({root})")


def _hf_snapshot(repo: str, hub: Path, **kwargs) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the venv
        raise RuntimeError("huggingface_hub is not installed in this Python; "
                           "run this script with a backend venv") from exc
    log(f"hf: {repo} -> {hub if 'local_dir' not in kwargs else kwargs['local_dir']}")
    if "local_dir" in kwargs:
        return snapshot_download(repo_id=repo, **kwargs)
    return snapshot_download(repo_id=repo, cache_dir=str(hub), **kwargs)


def fetch_hf(component: str, models: Path) -> None:
    hub = models / "huggingface" / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    path = _hf_snapshot(HF_REPOS[component], hub)
    log(f"{component}: ok ({path})")


def fetch_denoise(models: Path) -> None:
    target = models / "denoise" / "checkpoints" / "MossFormerGAN_SE_16K"
    target.mkdir(parents=True, exist_ok=True)
    _hf_snapshot(DENOISE_REPO, models, local_dir=str(target))
    log(f"denoise: ok ({target})")


def _stamp() -> str:
    import time
    return time.strftime("%Y%m%d-%H%M%S")


def component_target(component: str, models: Path) -> Path:
    """Where a component lands under the models folder."""
    if component == "vad":
        return models / "vad" / "silero_vad.onnx"
    if component == "sed":
        return models / "sed" / "PretrainedSED" / "resources" / "BEATs_strong_1.pt"
    if component == "denoise":
        return models / "denoise" / "checkpoints" / "MossFormerGAN_SE_16K"
    org, name = HF_REPOS[component].split("/")
    return models / "huggingface" / "hub" / f"models--{org}--{name}"


def is_present(component: str, models: Path) -> bool:
    target = component_target(component, models)
    if component == "vad":
        return target.exists() and _sha256(target) == SILERO_SHA256
    if target.is_dir():
        if component == "denoise":
            return any(target.iterdir())
        snaps = target / "snapshots"
        return snaps.is_dir() and any(snaps.iterdir())
    return target.exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("components", nargs="*", default=list(COMPONENTS),
                    help=f"any of: {', '.join(COMPONENTS)} (default: all)")
    ap.add_argument("--list", action="store_true",
                    help="show each component's folder and whether it is present; "
                         "downloads nothing")
    args = ap.parse_args(argv)
    models = Path(args.models_dir).expanduser().resolve()
    if args.list:
        for comp in args.components:
            if comp not in COMPONENTS:
                ap.error(f"unknown component {comp!r}")
            state = "present" if is_present(comp, models) else "missing"
            print(f"{comp:<9} {state:<8} {component_target(comp, models)}")
        return 0
    failed = []
    for comp in args.components:
        if comp not in COMPONENTS:
            ap.error(f"unknown component {comp!r}")
        try:
            if comp == "vad":
                fetch_vad(models)
            elif comp == "sed":
                fetch_sed(models)
            elif comp == "denoise":
                fetch_denoise(models)
            else:
                fetch_hf(comp, models)
        except Exception as exc:
            log(f"{comp}: FAILED: {exc}")
            failed.append(comp)
    if failed:
        log(f"failed: {', '.join(failed)}")
        return 1
    log("all requested models are present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
