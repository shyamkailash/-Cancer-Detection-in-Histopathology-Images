"""
Dataset Registry for tracking and configuring multi-source histopathology datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "metadata" / "dataset_registry.json"


class DatasetRegistry:
    """
    Manages metadata, download strategies, and specifications for supported dataset sources.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._sources: Dict[str, Dict[str, Any]] = {}
        self._version = "1.0.0"
        self.load()

    def load(self) -> None:
        """Load registry from JSON file if it exists."""
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._version = data.get("version", "1.0.0")
                self._sources = data.get("sources", {})
        else:
            self._sources = {}

    def save(self, output_path: Optional[Path] = None) -> None:
        """Save registry to JSON file."""
        target = Path(output_path) if output_path else self.registry_path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self._version,
            "sources": self._sources,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_source(self, source_id: str, metadata: Dict[str, Any]) -> None:
        """Register or update a dataset source metadata specification."""
        required_fields = ["source_id", "dataset_name", "task", "problem_type"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing required metadata field: '{field}'")

        self._sources[source_id] = metadata

    def get_source(self, source_id: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset source."""
        if source_id not in self._sources:
            raise KeyError(f"Dataset source '{source_id}' is not registered. Available sources: {self.list_sources()}")
        return self._sources[source_id]

    def has_source(self, source_id: str) -> bool:
        """Check if source is registered."""
        return source_id in self._sources

    def list_sources(self) -> List[str]:
        """List all registered source IDs."""
        return list(self._sources.keys())

    def update_source_status(self, source_id: str, download_status: Optional[str] = None, validation_status: Optional[str] = None) -> None:
        """Update runtime status for a source."""
        source = self.get_source(source_id)
        if download_status is not None:
            source["download_status"] = download_status
        if validation_status is not None:
            source["validation_status"] = validation_status
        self._sources[source_id] = source
