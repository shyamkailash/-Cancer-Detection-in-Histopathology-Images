"""
PatchCamelyon (PCam) Dataset Source Adapter.
Binary classification of lymph node metastasis in histopathology image patches.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import DatasetSource
from ..data.base import SampleRecord


class PCamSource(DatasetSource):
    """
    Adapter for the PatchCamelyon (PCam) histopathology dataset.
    PCam consists of 96x96px histopathology image patches extracted from the Camelyon16 dataset.
    Target: 0 = normal (no tumor), 1 = metastasis (tumor present in central 32x32px region).
    """

    @property
    def source_id(self) -> str:
        return "pcam"

    @property
    def dataset_name(self) -> str:
        return "PatchCamelyon"

    @property
    def task(self) -> str:
        return "lymph_node_metastasis_detection"

    @property
    def cancer_type(self) -> str:
        return "breast_cancer_lymph_node_metastasis"

    def is_available(self, data_dir: Optional[Path] = None) -> bool:
        """Check if PCam dataset directory contains valid image files."""
        dir_path = Path(data_dir) if data_dir else Path("data/raw/pcam")
        if not dir_path.exists() or not dir_path.is_dir():
            return False

        # Look for at least one supported image file
        for ext in self.image_validator.supported_extensions:
            if any(dir_path.rglob(f"*{ext}")):
                return True
        return False

    def download(self, dest_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Provide acquisition metadata and instructions for acquiring PCam.
        """
        target = Path(dest_dir)
        target.mkdir(parents=True, exist_ok=True)

        instructions = [
            "1. Official PCam repository: https://github.com/basveeling/pcam",
            "2. Kaggle Histopathologic Cancer Detection: 'kaggle competitions download -c histopathologic-cancer-detection'",
            "3. Extract image patches into subdirectories: data/raw/pcam/0/ (normal) and data/raw/pcam/1/ (metastasis)",
            "4. Or place images in train/0 and train/1 folders.",
        ]

        return {
            "source_id": self.source_id,
            "destination": str(target.resolve()),
            "status": "manual_or_cli_acquisition_required",
            "instructions": instructions,
            "official_url": "https://github.com/basveeling/pcam",
            "license": "CC0: Public Domain",
        }

    def discover_samples(self, data_dir: Path) -> List[SampleRecord]:
        """
        Discover PCam samples from directory structure.
        Supports folder-based categorization (e.g. .../0/img.png, .../1/img.png, .../normal/img.png, .../tumor/img.png).
        """
        root = Path(data_dir)
        samples: List[SampleRecord] = []
        if not root.exists():
            return samples

        # Traverse directory for image files
        for dirpath, _, filenames in os.walk(root):
            dir_path = Path(dirpath)
            folder_name = dir_path.name.lower()

            # Infer label from folder name if possible
            raw_label = None
            if folder_name in ("0", "normal", "negative", "non-tumor", "nontumor"):
                raw_label = 0
            elif folder_name in ("1", "tumor", "positive", "metastasis", "cancer"):
                raw_label = 1

            for fname in filenames:
                file_ext = Path(fname).suffix.lower()
                if file_ext in self.image_validator.supported_extensions:
                    filepath = dir_path / fname

                    # If label not inferred from folder, check filename or parent folder
                    sample_label = raw_label
                    if sample_label is None:
                        # Check filename prefix/suffix (e.g., '0_patch123.png' or 'tumor_123.png')
                        stem = filepath.stem.lower()
                        if stem.startswith("0_") or stem.endswith("_0") or "normal" in stem:
                            sample_label = 0
                        elif stem.startswith("1_") or stem.endswith("_1") or "tumor" in stem:
                            sample_label = 1
                        else:
                            sample_label = "unknown"

                    sample = self.build_sample_record(
                        filepath=filepath,
                        raw_label=sample_label,
                        magnification="10x",
                        extra_metadata={"native_patch_size": [96, 96]},
                    )
                    samples.append(sample)

        return samples

