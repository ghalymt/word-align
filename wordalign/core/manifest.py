"""Job manifest — output provenance for pipeline runs.

Writes a JSON manifest alongside the output files recording:
- WordAlign version
- Input file fingerprint
- Pipeline profile used
- Engines and models used
- Runtime versions (Python, CUDA, FFmpeg)
- Stage durations
- Warnings
- Timestamp source statistics
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .. import __version__


def generate_manifest(audio_path: str,
                      profile_name: str = "default",
                      engines_used: Optional[list[str]] = None,
                      models_used: Optional[list[str]] = None,
                      stage_durations: Optional[dict[str, float]] = None,
                      warnings: Optional[list[str]] = None,
                      word_stats: Optional[dict[str, Any]] = None,
                      output_files: Optional[list[str]] = None) -> dict:
    """Build a job manifest dict."""
    from .fingerprint import fingerprint_audio

    fp = fingerprint_audio(audio_path)

    manifest = {
        "manifest_version": 1,
        "wordalign_version": __version__,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input": {
            "audio_path": os.path.abspath(audio_path),
            "audio_fingerprint": fp.cache_key,
            "audio_size_bytes": fp.file_size,
            "audio_duration_seconds": fp.duration_hint,
        },
        "profile": profile_name,
        "engines": engines_used or [],
        "models": models_used or [],
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "stages": stage_durations or {},
        "warnings": warnings or [],
        "word_stats": word_stats or {},
        "output_files": output_files or [],
    }

    # Add GPU info if available
    try:
        from ..runtimes.detector import HardwareDetector
        detector = HardwareDetector()
        hw = detector.detect()
        manifest["hardware"] = {
            "os": hw.os,
            "cpu_count": hw.cpu_count,
            "ram_gb": hw.ram_gb,
            "gpu": hw.gpu_name,
            "gpu_vram_gb": hw.gpu_vram_gb,
            "cuda_available": hw.cuda_available,
            "ffmpeg_available": hw.ffmpeg_available,
        }
    except Exception:
        manifest["hardware"] = {}

    return manifest


def save_manifest(manifest: dict, output_path: str) -> str:
    """Write the manifest to a JSON file. Returns the path."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return output_path


def write_job_manifest(audio_path: str,
                       output_base: str,
                       **kwargs) -> str:
    """Generate and write a job manifest. Returns the manifest path."""
    manifest = generate_manifest(audio_path, **kwargs)
    manifest_path = f"{output_base}_job_manifest.json"
    return save_manifest(manifest, manifest_path)
