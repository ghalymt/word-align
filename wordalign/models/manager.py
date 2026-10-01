"""Model manager — install, validate, delete, list, change location."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .catalog import ModelCatalog, ModelEntry


@dataclass
class RegisteredModel:
    """A model that has been registered (installed or linked)."""
    model_id: str
    install_path: str
    managed: bool = True         # True = word-align manages, False = external
    source_type: str = "manual"
    source_revision: str = ""
    installed_at: str = ""
    validated: bool = False
    checksum: Optional[str] = None


class ModelManager:
    """Manages model installation, validation, and lifecycle."""

    def __init__(self, models_dir: str, catalog: Optional[ModelCatalog] = None,
                 registry_path: Optional[str] = None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog or ModelCatalog()
        self.registry_path = Path(registry_path or
                                  (self.models_dir / "model_registry.json"))
        self._registry: dict[str, RegisteredModel] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for model_id, info in data.items():
                self._registry[model_id] = RegisteredModel(**info)

    def _save_registry(self) -> None:
        data = {k: asdict(v) for k, v in self._registry.items()}
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_registered(self) -> list[RegisteredModel]:
        return list(self._registry.values())

    def is_installed(self, model_id: str) -> bool:
        return model_id in self._registry and self._registry[model_id].validated

    def get_install_path(self, model_id: str) -> Optional[str]:
        reg = self._registry.get(model_id)
        return reg.install_path if reg else None

    def register_external(self, model_id: str, path: str) -> RegisteredModel:
        """Register an existing model directory without copying it."""
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory not found: {path}")
        reg = RegisteredModel(
            model_id=model_id,
            install_path=path,
            managed=False,
            source_type="existing_path",
            validated=True,
            installed_at=os.path.basename(path),
        )
        self._registry[model_id] = reg
        self._save_registry()
        return reg

    def delete(self, model_id: str) -> bool:
        """Delete a managed model. External models are unregistered only."""
        reg = self._registry.get(model_id)
        if not reg:
            return False
        if reg.managed and os.path.isdir(reg.install_path):
            shutil.rmtree(reg.install_path, ignore_errors=True)
        del self._registry[model_id]
        self._save_registry()
        return True

    def change_location(self, model_id: str, new_path: str) -> RegisteredModel:
        """Change the install path for a model (move or re-link)."""
        reg = self._registry.get(model_id)
        if not reg:
            raise KeyError(f"Model not registered: {model_id}")
        old_path = reg.install_path
        new_path = os.path.abspath(new_path)
        if reg.managed:
            if os.path.isdir(old_path):
                shutil.move(old_path, new_path)
        reg.install_path = new_path
        self._save_registry()
        return reg

    def get_model_path_for_plugin(self, plugin_id: str) -> Optional[str]:
        """Get the install path for the first registered model of a plugin."""
        for entry in self.catalog.list_for_plugin(plugin_id):
            reg = self._registry.get(entry.id)
            if reg and reg.validated:
                return reg.install_path
        return None

    def hardware_check(self, model_id: str, vram_gb: float,
                       ram_gb: float) -> dict:
        """Check if hardware meets model requirements."""
        entry = self.catalog.get(model_id)
        if not entry:
            return {"ok": False, "reason": "Unknown model"}
        issues = []
        if entry.minimum_vram_gb and vram_gb < entry.minimum_vram_gb:
            issues.append(f"VRAM: {vram_gb:.1f} GB < required {entry.minimum_vram_gb} GB")
        if entry.minimum_ram_gb and ram_gb < entry.minimum_ram_gb:
            issues.append(f"RAM: {ram_gb:.1f} GB < required {entry.minimum_ram_gb} GB")
        return {"ok": len(issues) == 0, "issues": issues,
                "model": entry.display_name}
