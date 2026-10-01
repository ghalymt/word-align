"""Download manager — resumable model downloads with validation."""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional


class DownloadManager:
    """Manages model downloads with progress reporting and atomic writes."""

    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)

    def download_huggingface(self, repo_id: str, revision: str = "main",
                             target_dir: Optional[str] = None,
                             on_progress: Optional[Callable[[float, str], None]] = None) -> str:
        """Download a model from HuggingFace Hub.

        Returns the path to the downloaded model directory.
        Raises ImportError if huggingface_hub is not installed.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloads. "
                "pip install huggingface_hub")

        target = Path(target_dir) if target_dir else (
            self.models_dir / repo_id.replace("/", "_"))
        target.mkdir(parents=True, exist_ok=True)

        # Download to temp first, then move atomically
        temp_dir = str(target / ".downloading")
        os.makedirs(temp_dir, exist_ok=True)

        def hf_progressCallback(copied, total):
            if on_progress and total > 0:
                on_progress(copied / total, f"{copied}/{total} bytes")

        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=temp_dir,
                local_dir_use_symlinks=False,
            )
            # Move temp contents to target
            for item in os.listdir(temp_dir):
                src = os.path.join(temp_dir, item)
                dst = os.path.join(str(target), item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)
            os.rmdir(temp_dir)
            if on_progress:
                on_progress(1.0, "Complete")
            return str(target)
        except Exception:
            # Cleanup temp on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def download_direct(self, url: str, target_path: str,
                        on_progress: Optional[Callable[[float, str], None]] = None,
                        expected_checksum: Optional[str] = None) -> str:
        """Download a file directly with progress and optional checksum."""
        import urllib.request

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 256  # 256KB
                with open(temp, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_progress and total > 0:
                            on_progress(downloaded / total,
                                        f"{downloaded}/{total} bytes")

            # Checksum validation
            if expected_checksum:
                actual = self._checksum(temp)
                if actual != expected_checksum:
                    os.remove(temp)
                    raise ValueError(
                        f"Checksum mismatch: expected {expected_checksum}, "
                        f"got {actual}")

            # Atomic move
            shutil.move(str(temp), str(target))
            if on_progress:
                on_progress(1.0, "Complete")
            return str(target)
        except Exception:
            if temp.exists():
                os.remove(temp)
            raise

    @staticmethod
    def _checksum(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    def check_disk_space(self, required_bytes: int) -> bool:
        """Check if there's enough disk space for a download."""
        usage = shutil.disk_usage(self.models_dir)
        return usage.free > required_bytes
