"""Stage-level cache for pipeline outputs.

Stores JSON-serializable outputs keyed by audio fingerprint + engine identity.
Changing subtitle formatting (CPL, lines) reuses all ASR outputs and only
re-runs segmentation — the main motivation for this cache.

Usage:
    cache = StageCache(cache_dir="~/.wordalign/cache")
    cache.put(key, {"words": [...], "events": [...]})
    data = cache.get(key)  # None if miss
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .fingerprint import AudioFingerprint, engine_cache_key, fingerprint_audio


class StageCache:
    """Disk-based cache for pipeline stage outputs."""

    def __init__(self, cache_dir: Optional[str] = None,
                 max_age_days: int = 30,
                 max_size_mb: int = 2048):
        self.cache_dir = Path(cache_dir or self._default_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max_age_days * 86400
        self.max_size_bytes = max_size_mb * 1024 * 1024

    @staticmethod
    def _default_cache_dir() -> str:
        return os.path.join(os.path.expanduser("~"), ".wordalign", "cache")

    def _key_path(self, key: str) -> Path:
        # Use first 2 chars as subdirectory to avoid huge flat dirs
        sub = self.cache_dir / key[:2]
        sub.mkdir(exist_ok=True)
        return sub / f"{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Retrieve cached output, or None if missing/expired."""
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            stat = path.stat()
            if time.time() - stat.st_mtime > self.max_age_seconds:
                return None  # expired
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, data: dict[str, Any]) -> None:
        """Store output in cache. Must be JSON-serializable."""
        path = self._key_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except OSError:
            pass  # cache write failures are non-fatal

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def invalidate(self, key: str) -> None:
        path = self._key_path(key)
        if path.exists():
            path.unlink()

    def clear(self) -> int:
        """Remove all cache entries. Returns count of deleted files."""
        count = 0
        for sub in self.cache_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.glob("*.json"):
                f.unlink()
                count += 1
        return count

    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of deleted files."""
        count = 0
        now = time.time()
        for sub in self.cache_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.glob("*.json"):
                try:
                    stat = f.stat()
                    if now - stat.st_mtime > self.max_age_seconds:
                        f.unlink()
                        count += 1
                except OSError:
                    pass
        return count

    def total_size_bytes(self) -> int:
        total = 0
        for sub in self.cache_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.glob("*.json"):
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
        return total

    # ---- High-level convenience methods ----

    def cached_engine_output(self,
                             audio_path: str,
                             plugin_id: str,
                             plugin_version: str,
                             model_id: Optional[str] = None,
                             language: Optional[str] = None,
                             engine_settings: Optional[dict] = None) -> Optional[dict]:
        """Look up a cached engine output by audio + engine identity."""
        fp = fingerprint_audio(audio_path)
        key = engine_cache_key(fp, plugin_id, plugin_version,
                               model_id, None, language, engine_settings)
        return self.get(key)

    def store_engine_output(self,
                            audio_path: str,
                            plugin_id: str,
                            plugin_version: str,
                            output: dict[str, Any],
                            model_id: Optional[str] = None,
                            language: Optional[str] = None,
                            engine_settings: Optional[dict] = None) -> None:
        """Store an engine output for later reuse."""
        fp = fingerprint_audio(audio_path)
        key = engine_cache_key(fp, plugin_id, plugin_version,
                               model_id, None, language, engine_settings)
        self.put(key, output)
