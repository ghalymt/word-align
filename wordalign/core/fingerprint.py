"""Audio fingerprinting for cache keys.

Generates a fast, stable hash of audio file identity so that cache entries
can be matched across runs without storing the full audio.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AudioFingerprint:
    """Stable identity for an audio file."""
    file_hash: str           # first 64KB hash (fast, not full-file)
    file_size: int           # bytes
    file_mtime: float        # modification time
    duration_hint: float     # seconds, from ffprobe if available

    @property
    def cache_key(self) -> str:
        """Short stable key for cache lookup."""
        return hashlib.sha256(
            f"{self.file_hash}:{self.file_size}:{int(self.file_mtime)}".encode()
        ).hexdigest()[:16]


def fingerprint_audio(audio_path: str | Path) -> AudioFingerprint:
    """Generate a fingerprint for an audio file.

    Reads the first 64KB + last 4KB for a fast but stable hash.
    Falls back to full-file hash for small files.
    """
    path = Path(audio_path)
    stat = path.stat()
    file_size = stat.st_size
    mtime = stat.st_mtime

    h = hashlib.sha256()
    with open(path, "rb") as f:
        if file_size <= 65536 + 4096:
            # Small file: hash everything
            h.update(f.read())
        else:
            # Large file: hash first 64KB + last 4KB
            h.update(f.read(65536))
            f.seek(-4096, os.SEEK_END)
            h.update(f.read(4096))

    file_hash = h.hexdigest()

    # Try to get duration via ffprobe
    duration_hint = 0.0
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            duration_hint = float(result.stdout.strip())
    except Exception:
        pass

    return AudioFingerprint(
        file_hash=file_hash,
        file_size=file_size,
        file_mtime=mtime,
        duration_hint=duration_hint,
    )


def engine_cache_key(fingerprint: AudioFingerprint,
                     plugin_id: str,
                     plugin_version: str,
                     model_id: Optional[str] = None,
                     model_revision: Optional[str] = None,
                     language: Optional[str] = None,
                     engine_settings: Optional[dict] = None) -> str:
    """Build a deterministic cache key for a single engine's output.

    Components: audio fingerprint, plugin identity, model identity,
    language, and a hash of engine-specific settings.
    """
    parts = [
        fingerprint.cache_key,
        plugin_id,
        plugin_version,
        model_id or "none",
        model_revision or "none",
        language or "auto",
    ]
    if engine_settings:
        # Sort keys for deterministic hashing
        settings_str = str(sorted(engine_settings.items()))
        settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()[:8]
        parts.append(settings_hash)
    return "_".join(str(p) for p in parts)
