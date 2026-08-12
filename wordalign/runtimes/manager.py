"""Runtime manager — tracks isolated runtime environments for engines."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class RuntimeStatus(str, Enum):
    READY = "ready"
    MISSING = "missing"
    BROKEN = "broken"
    INCOMPATIBLE = "incompatible"
    DOWNLOADING = "downloading"
    UPDATE_AVAILABLE = "update_available"


@dataclass
class RuntimeInfo:
    """Information about an installed runtime."""
    runtime_id: str
    name: str
    version: str
    status: str = "missing"
    install_path: str = ""
    engine_id: str = ""
    platform: str = ""
    python_version: str = ""
    packages: list[dict] = field(default_factory=list)
    checksum: str = ""
    installed_at: str = ""


class RuntimeManager:
    """Manages runtime environments for engines.

    Possible runtimes:
    - main: the main application's Python environment
    - isolated: a separate venv with different dependencies
    - bundled: shipped with the portable application
    - external: an external executable (e.g. MFA)
    """

    def __init__(self, runtimes_dir: str,
                 registry_path: Optional[str] = None):
        self.runtimes_dir = Path(runtimes_dir)
        self.runtimes_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = Path(registry_path or
                                  (self.runtimes_dir / "runtime_registry.json"))
        self._runtimes: dict[str, RuntimeInfo] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rid, info in data.items():
                self._runtimes[rid] = RuntimeInfo(**info)

    def _save_registry(self) -> None:
        data = {k: asdict(v) for k, v in self._runtimes.items()}
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_runtimes(self) -> list[RuntimeInfo]:
        return list(self._runtimes.values())

    def get(self, runtime_id: str) -> Optional[RuntimeInfo]:
        return self._runtimes.get(runtime_id)

    def get_for_engine(self, engine_id: str) -> Optional[RuntimeInfo]:
        for rt in self._runtimes.values():
            if rt.engine_id == engine_id and rt.status == RuntimeStatus.READY:
                return rt
        return None

    def register(self, runtime: RuntimeInfo) -> None:
        self._runtimes[runtime.runtime_id] = runtime
        self._save_registry()

    def unregister(self, runtime_id: str) -> None:
        self._runtimes.pop(runtime_id, None)
        self._save_registry()

    def check_status(self, runtime_id: str) -> RuntimeStatus:
        """Verify a runtime's installed state."""
        rt = self._runtimes.get(runtime_id)
        if not rt:
            return RuntimeStatus.MISSING
        if not rt.install_path or not os.path.isdir(rt.install_path):
            return RuntimeStatus.MISSING
        # Check for python.exe
        if os.name == "nt":
            exe = os.path.join(rt.install_path, "Scripts", "python.exe")
        else:
            exe = os.path.join(rt.install_path, "bin", "python")
        if not os.path.isfile(exe):
            return RuntimeStatus.BROKEN
        rt.status = RuntimeStatus.READY
        self._save_registry()
        return RuntimeStatus.READY
