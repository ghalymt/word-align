"""Model catalog — parses and serves model metadata from model_catalog.json."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

CATALOG_SCHEMA_VERSION = 1


@dataclass
class ModelEntry:
    """One model in the catalog."""
    id: str
    display_name: str
    plugin_id: str
    source_type: str            # "huggingface" | "direct_url" | "manual" | "existing_path"
    source_repo: str = ""       # HF repo or URL
    source_revision: str = "main"
    size_bytes: int = 0
    languages: list[str] = field(default_factory=lambda: ["*"])
    capabilities: list[str] = field(default_factory=list)
    minimum_ram_gb: float = 0
    recommended_ram_gb: float = 0
    minimum_vram_gb: float = 0
    recommended_vram_gb: float = 0
    quantization: Optional[str] = None
    license: str = ""
    description: str = ""
    recommended: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        source = d.get("source", {})
        return cls(
            id=d["id"],
            display_name=d.get("display_name", d["id"]),
            plugin_id=d["plugin_id"],
            source_type=source.get("type", "manual"),
            source_repo=source.get("repository", source.get("url", "")),
            source_revision=source.get("revision", "main"),
            size_bytes=d.get("size_bytes", 0),
            languages=d.get("languages", ["*"]),
            capabilities=d.get("capabilities", []),
            minimum_ram_gb=d.get("minimum_ram_gb", 0),
            recommended_ram_gb=d.get("recommended_ram_gb", 0),
            minimum_vram_gb=d.get("minimum_vram_gb", 0),
            recommended_vram_gb=d.get("recommended_vram_gb", 0),
            quantization=d.get("quantization"),
            license=d.get("license", ""),
            description=d.get("description", ""),
            recommended=d.get("recommended", False),
        )


class ModelCatalog:
    """Loads and queries the model catalog."""

    def __init__(self, catalog_path: Optional[str] = None):
        self._entries: dict[str, ModelEntry] = {}
        if catalog_path:
            self.load(catalog_path)

    def load(self, path: str | Path) -> None:
        """Load a catalog JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry_dict in data.get("models", []):
            entry = ModelEntry.from_dict(entry_dict)
            self._entries[entry.id] = entry

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self._entries.get(model_id)

    def list_all(self) -> list[ModelEntry]:
        return list(self._entries.values())

    def list_for_plugin(self, plugin_id: str) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.plugin_id == plugin_id]

    def list_recommended(self) -> list[ModelEntry]:
        return [e for e in self._entries.values() if e.recommended]

    def list_by_language(self, language: str) -> list[ModelEntry]:
        return [e for e in self._entries.values()
                if "*" in e.languages or language in e.languages]
