"""Model validator — checks directory structure for different model types."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .catalog import ModelCatalog


class ModelValidator:
    """Validates model directories for different engine types."""

    # Expected file/dir patterns per plugin
    EXPECTED_PATTERNS = {
        "whisperx": ["config.json", "pytorch_model.bin", "tokenizer.json"],
        "qwen_asr": ["config.json"],
        "vosk": ["am/final.mdl", "conf"],
        "mfa": ["english_us_arpa.zip"],
    }

    def __init__(self, catalog: Optional[ModelCatalog] = None):
        self.catalog = catalog

    def validate(self, model_path: str | Path,
                 plugin_id: str) -> dict:
        """Check if a directory contains a valid model for the given plugin."""
        path = Path(model_path)
        if not path.is_dir():
            return {"valid": False, "reason": f"Not a directory: {path}",
                    "expected": [], "found": []}

        patterns = self.EXPECTED_PATTERNS.get(plugin_id, [])
        found = []
        missing = []
        for pattern in patterns:
            # Check direct file
            if (path / pattern).exists():
                found.append(pattern)
            else:
                # Check if it's a glob pattern (e.g. for subdirs)
                matches = list(path.rglob(pattern))
                if matches:
                    found.append(pattern)
                else:
                    missing.append(pattern)

        if missing:
            return {"valid": False,
                    "reason": f"Missing: {', '.join(missing)}",
                    "expected": patterns, "found": found, "missing": missing}
        return {"valid": True, "reason": "OK",
                "expected": patterns, "found": found, "missing": []}
