"""Plugin registry: discovery, lookup, and capability-based dispatch."""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import EngineDescriptor, EnginePlugin
from .capabilities import Capability
from .manifest import PluginManifest


class PluginRegistry:
    """Central registry for all engine plugins.

    Plugins are registered with a manifest. The pipeline queries by
    capability (e.g., "give me all TRANSCRIBE engines") rather than by name.
    """

    def __init__(self):
        self._plugins: Dict[str, EnginePlugin] = {}
        self._manifests: Dict[str, PluginManifest] = {}

    def register(self, plugin: EnginePlugin,
                 manifest: Optional[PluginManifest] = None) -> None:
        """Register a plugin instance with its manifest."""
        desc = plugin.descriptor()
        if manifest is None:
            # Build manifest from descriptor
            desc = plugin.descriptor()
            cap_labels = []
            # EngineDescriptor stores Capability flags in 'capabilities' field
            caps = desc.capabilities
            if hasattr(caps, 'labels'):
                cap_labels = caps.labels()
            elif isinstance(caps, (list, tuple)):
                cap_labels = list(caps)
            manifest = PluginManifest(
                id=desc.engine_id,
                name=desc.display_name,
                version=desc.version,
                capabilities=cap_labels,
                runtime=desc.runtime_type,
                supported_languages=desc.supported_languages,
                models=desc.models,
                hardware=desc.hardware,
                description=desc.help_text,
            )
        errors = manifest.validate()
        if errors:
            raise ValueError(f"Invalid manifest for '{manifest.id}': {'; '.join(errors)}")
        self._plugins[manifest.id] = plugin
        self._manifests[manifest.id] = manifest

    def unregister(self, plugin_id: str) -> None:
        """Remove a plugin from the registry."""
        self._plugins.pop(plugin_id, None)
        self._manifests.pop(plugin_id, None)

    def get(self, plugin_id: str) -> Optional[EnginePlugin]:
        """Retrieve a plugin by ID."""
        return self._plugins.get(plugin_id)

    def get_manifest(self, plugin_id: str) -> Optional[PluginManifest]:
        """Retrieve a plugin's manifest by ID."""
        return self._manifests.get(plugin_id)

    def list_plugins(self) -> List[str]:
        """Return all registered plugin IDs."""
        return list(self._plugins.keys())

    def list_manifests(self) -> List[PluginManifest]:
        """Return all registered manifests."""
        return list(self._manifests.values())

    def query(self, capability: Capability) -> List[str]:
        """Return plugin IDs that have the given capability."""
        result = []
        for pid, manifest in self._manifests.items():
            if manifest.capability_flags & capability:
                result.append(pid)
        return result

    def query_plugins(self, capability: Capability) -> List[EnginePlugin]:
        """Return plugin instances that have the given capability."""
        return [self._plugins[pid] for pid in self.query(capability)]

    def health_check_all(self) -> Dict[str, dict]:
        """Run health checks on all registered plugins."""
        results = {}
        for pid, plugin in self._plugins.items():
            status = plugin.health_check()
            results[pid] = {
                "ready": status.ready,
                "runtime_status": status.runtime_status,
                "missing": status.missing_components,
                "message": status.message,
                "capabilities": self._manifests[pid].capability_flags.labels(),
            }
        return results

    def __len__(self) -> int:
        return len(self._plugins)

    def __contains__(self, plugin_id: str) -> bool:
        return plugin_id in self._plugins
