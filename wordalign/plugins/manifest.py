"""Plugin manifest dataclass and JSON schema validation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .capabilities import Capability

MANIFEST_SCHEMA_VERSION = 1


@dataclass
class PluginManifest:
    """Metadata for a plugin, loaded from JSON or constructed in code."""
    schema_version: int = MANIFEST_SCHEMA_VERSION
    id: str = ""
    name: str = ""
    version: str = "1.0"
    capabilities: list[str] = field(default_factory=list)
    runtime: str = "in_process"         # "in_process" | "subprocess" | "external"
    supported_languages: list[str] = field(default_factory=lambda: ["*"])
    models: list[str] = field(default_factory=list)
    hardware: dict = field(default_factory=lambda: {"cpu": True, "cuda": False})
    description: str = ""
    help_text: str = ""
    engine_compatibility: str = "*"     # semver range

    @classmethod
    def from_dict(cls, d: dict) -> "PluginManifest":
        return cls(
            schema_version=d.get("schema_version", MANIFEST_SCHEMA_VERSION),
            id=d["id"],
            name=d.get("name", d["id"]),
            version=d.get("version", "1.0"),
            capabilities=d.get("capabilities", []),
            runtime=d.get("runtime", "in_process"),
            supported_languages=d.get("supported_languages", ["*"]),
            models=d.get("models", []),
            hardware=d.get("hardware", {"cpu": True, "cuda": False}),
            description=d.get("description", ""),
            help_text=d.get("help_text", ""),
            engine_compatibility=d.get("engine_compatibility", "*"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PluginManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def capability_flags(self) -> Capability:
        """Convert string capability list to Capability flags."""
        flags = Capability(0)
        cap_map = {c.name.lower(): c for c in Capability}
        for cap_str in self.capabilities:
            cap = cap_map.get(cap_str.lower())
            if cap:
                flags |= cap
        return flags

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors = []
        if not self.id:
            errors.append("Manifest missing 'id'")
        if self.schema_version > MANIFEST_SCHEMA_VERSION:
            errors.append(f"Schema version {self.schema_version} > "
                          f"supported {MANIFEST_SCHEMA_VERSION}")
        if self.runtime not in ("in_process", "subprocess", "external"):
            errors.append(f"Invalid runtime: {self.runtime}")
        valid_caps = {c.name.lower() for c in Capability}
        for cap in self.capabilities:
            if cap.lower() not in valid_caps:
                errors.append(f"Unknown capability: {cap}")
        return errors
