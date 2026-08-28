"""
BreaKHis (Breast Cancer Histopathological Image Classification) Dataset Source Adapter.
Binary and subtype classification of breast tumors with patient-level grouping.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import DatasetSource
from ..data.base import SampleRecord


# Regex pattern for standard BreaKHis file naming convention:
# SOB_<BiopsyType>_<TumorType>-<PatientID>-<Magnification>-<SeqNum>.png
# Examples:
#   SOB_B_A-14-22549AB-100-001.png
#   SOB_M_DC-14-2773-40-003.png
BREAKHIS_PATTERN = re.compile(
    r"^SOB_(?P<biopsy>[BM])_(?P<subtype>[A-Z]+)-(?P<patient>[\w-]+)-(?P<mag>\d+)-(?P<seq>\d+)$",
    re.IGNORECASE
)


class BreakHisSource(DatasetSource):
    """
    Adapter for BreaKHis breast cancer histopathology dataset.
    Extracts slide-level and patient-level metadata from BreaKHis filenames to ensure leakage-safe splitting.
    """

    @property
    def source_id(self) -> str:
        return "breakhis"

    @property
    def dataset_name(self) -> str:
        return "Breast Cancer Histopathological Image Classification (BreaKHis)"

    @property
    def task(self) -> str:
        return "breast_cancer_malignancy_detection"

    @property
    def cancer_type(self) -> str:
        return "breast_cancer"

    def is_available(self, data_dir: Optional[Path] = None) -> bool:
        """Check if BreaKHis dataset directory contains valid image files."""
        dir_path = Path(data_dir) if data_dir else Path("data/raw/breakhis")
        if not dir_path.exists() or not dir_path.is_dir():
            return False

        for ext in self.image_validator.supported_extensions:
            if any(dir_path.rglob(f"*{ext}")):
                return True
        return False

    def download(self, dest_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Provide acquisition metadata and instructions for acquiring BreaKHis.
        """
        target = Path(dest_dir)
        target.mkdir(parents=True, exist_ok=True)

        instructions = [
            "1. Official BreaKHis URL: https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
            "2. Kaggle dataset: 'kaggle datasets download -d amirassov/breakhis'",
            "3. Extract into: data/raw/breakhis/",
            "4. File names will follow the pattern SOB_<B/M>_<Subtype>-<PatientID>-<Mag>-<Seq>.png",
        ]

        return {
            "source_id": self.source_id,
            "destination": str(target.resolve()),
            "status": "manual_or_cli_acquisition_required",
            "instructions": instructions,
            "official_url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
            "license": "Academic Free License (Non-commercial research use)",
        }

    def parse_filename(self, stem: str) -> Dict[str, Optional[str]]:
        """Parse structured metadata from BreaKHis filename stem."""
        match = BREAKHIS_PATTERN.match(stem)
        if match:
            group = match.groupdict()
            biopsy = "benign" if group["biopsy"].upper() == "B" else "malignant"
            subtype = group["subtype"]
            patient_id = group["patient"]
            magnification = f"{group['mag']}X"
            slide_id = f"SOB_{group['biopsy'].upper()}_{subtype}-{patient_id}"
            return {
                "raw_label": biopsy,
                "subtype": subtype,
                "patient_id": patient_id,
                "slide_id": slide_id,
                "magnification": magnification,
            }

        # Fallback heuristic if filenames don't follow exact SOB pattern
        lower_stem = stem.lower()
        if "benign" in lower_stem:
            raw_label = "benign"
        elif "malignant" in lower_stem:
            raw_label = "malignant"
        else:
            raw_label = "unknown"

        return {
            "raw_label": raw_label,
            "subtype": None,
            "patient_id": None,
            "slide_id": None,
            "magnification": None,
        }

    def discover_samples(self, data_dir: Path) -> List[SampleRecord]:
        """
        Discover BreaKHis samples from directory and extract patient/slide IDs.
        """
        root = Path(data_dir)
        samples: List[SampleRecord] = []
        if not root.exists():
            return samples

        for dirpath, _, filenames in os.walk(root):
            dir_path = Path(dirpath)
            for fname in filenames:
                file_ext = Path(fname).suffix.lower()
                if file_ext in self.image_validator.supported_extensions:
                    filepath = dir_path / fname
                    meta = self.parse_filename(filepath.stem)

                    sample = self.build_sample_record(
                        filepath=filepath,
                        raw_label=meta["raw_label"],
                        patient_id=meta["patient_id"],
                        slide_id=meta["slide_id"],
                        magnification=meta["magnification"],
                        extra_metadata={"subtype": meta["subtype"]},
                    )
                    samples.append(sample)

        return samples
