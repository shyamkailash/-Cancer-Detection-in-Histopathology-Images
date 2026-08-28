"""
Base data structures and metadata schemas for histopathology datasets.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class SampleRecord:
    """
    Standardized schema for a single histopathology image sample.
    """
    sample_id: str
    source_id: str
    dataset_name: str
    image_path: str
    original_label: Any
    standardized_label: Optional[int] = None
    label_name: Optional[str] = None
    task: str = "binary_classification"
    cancer_type: str = "unknown"
    magnification: Optional[str] = None
    patient_id: Optional[str] = None
    slide_id: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_format: Optional[str] = None
    file_size_bytes: Optional[int] = None
    file_hash: Optional[str] = None
    split: str = "unassigned"  # 'train', 'val', 'test', 'unassigned'
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample record to dictionary."""
        data = asdict(self)
        # Ensure path is string
        data["image_path"] = str(data["image_path"])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleRecord":
        """Create sample record from dictionary."""
        clean_data = dict(data)
        if "validation_errors" not in clean_data or clean_data["validation_errors"] is None:
            clean_data["validation_errors"] = []
        if "metadata" not in clean_data or clean_data["metadata"] is None:
            clean_data["metadata"] = {}
        return cls(**clean_data)

    def get_absolute_path(self, base_dir: Optional[Path] = None) -> Path:
        """Resolve path to absolute path using optional base directory."""
        path = Path(self.image_path)
        if path.is_absolute() or base_dir is None:
            return path
        return (base_dir / path).resolve()

