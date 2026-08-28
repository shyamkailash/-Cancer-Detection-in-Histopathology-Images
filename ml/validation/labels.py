"""
Label validation and standardization across multi-source histopathology datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


DEFAULT_MAPPING_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "metadata" / "class_mapping.json"


class LabelValidator:
    """
    Validates source-specific labels and maps them to canonical integer and string representations.
    """

    def __init__(self, mapping_path: Optional[Path] = None):
        self.mapping_path = Path(mapping_path) if mapping_path else DEFAULT_MAPPING_PATH
        self.mappings: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load class mapping configurations from JSON."""
        if self.mapping_path.exists():
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.mappings = data.get("mappings", {})
        else:
            self.mappings = {}

    def get_source_mapping(self, source_id: str) -> Dict[str, Any]:
        """Get mapping config for a specific source."""
        if source_id not in self.mappings:
            raise KeyError(f"No class mapping found for source '{source_id}'. Registered sources: {list(self.mappings.keys())}")
        return self.mappings[source_id]

    def validate_and_map(self, source_id: str, raw_label: Any) -> Tuple[bool, Optional[int], Optional[str], Optional[str]]:
        """
        Validate raw label for a dataset source and return standardized label representation.
        Returns: (is_valid, standardized_label_int, canonical_label_name, error_message)
        """
        if source_id not in self.mappings:
            return False, None, None, f"Source '{source_id}' does not have a registered label mapping."

        source_cfg = self.mappings[source_id]
        id_map = source_cfg.get("to_standardized_id", {})
        name_map = source_cfg.get("to_label_name", {})

        # Normalize key for lookup
        key = str(raw_label).strip()
        key_lower = key.lower()

        std_id = None
        if key in id_map:
            std_id = id_map[key]
        elif key_lower in id_map:
            std_id = id_map[key_lower]

        if std_id is None:
            return False, None, None, f"Unrecognized label '{raw_label}' for source '{source_id}'. Expected one of: {list(id_map.keys())}"

        label_name = name_map.get(str(std_id), str(std_id))
        return True, int(std_id), label_name, None
