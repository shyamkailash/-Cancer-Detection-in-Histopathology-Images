"""
Abstract Base Class for Dataset Source Adapters in the Histopathology Pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ..data.base import SampleRecord
from ..validation.images import ImageValidator
from ..validation.labels import LabelValidator


class DatasetSource(ABC):
    """
    Abstract interface for acquiring, discovering, validating, and extracting metadata
    from a specific external histopathology dataset source.
    """

    def __init__(
        self,
        image_validator: Optional[ImageValidator] = None,
        label_validator: Optional[LabelValidator] = None,
    ):
        self.image_validator = image_validator or ImageValidator()
        self.label_validator = label_validator or LabelValidator()

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier for this dataset source (e.g. 'pcam', 'breakhis')."""
        pass

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Human readable name of the dataset."""
        pass

    @property
    @abstractmethod
    def task(self) -> str:
        """Canonical clinical task definition."""
        pass

    @property
    @abstractmethod
    def cancer_type(self) -> str:
        """Type of cancer represented in the dataset."""
        pass

    @abstractmethod
    def is_available(self, data_dir: Optional[Path] = None) -> bool:
        """Check if dataset files are available locally."""
        pass

    @abstractmethod
    def download(self, dest_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Download or provide step-by-step acquisition instructions for the dataset.
        Returns status dictionary with instructions or result details.
        """
        pass

    @abstractmethod
    def discover_samples(self, data_dir: Path) -> List[SampleRecord]:
        """
        Scan directory, discover all image files, extract source-specific metadata (patient, slide, magnification),
        validate images and labels, and return a list of standardized SampleRecord objects.
        """
        pass

    def build_sample_record(
        self,
        filepath: Path,
        raw_label: Any,
        patient_id: Optional[str] = None,
        slide_id: Optional[str] = None,
        magnification: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SampleRecord:
        """
        Helper method to validate image file and label, extract metadata, and produce a SampleRecord.
        """
        is_img_valid, img_errors, img_meta = self.image_validator.validate_image(filepath)
        is_lbl_valid, std_label, label_name, lbl_error = self.label_validator.validate_and_map(
            self.source_id, raw_label
        )

        all_errors = list(img_errors)
        if not is_lbl_valid and lbl_error:
            all_errors.append(lbl_error)

        file_hash = img_meta.get("file_hash")
        # Generate deterministic sample_id: {source_id}_{file_hash[:12]} or fallback to file stem
        hash_suffix = file_hash[:12] if file_hash else filepath.stem
        sample_id = f"{self.source_id}_{hash_suffix}"

        return SampleRecord(
            sample_id=sample_id,
            source_id=self.source_id,
            dataset_name=self.dataset_name,
            image_path=str(filepath.resolve()),
            original_label=raw_label,
            standardized_label=std_label,
            label_name=label_name,
            task=self.task,
            cancer_type=self.cancer_type,
            magnification=magnification,
            patient_id=patient_id,
            slide_id=slide_id,
            image_width=img_meta.get("image_width"),
            image_height=img_meta.get("image_height"),
            image_format=img_meta.get("image_format"),
            file_size_bytes=img_meta.get("file_size_bytes"),
            file_hash=file_hash,
            is_valid=(is_img_valid and is_lbl_valid),
            validation_errors=all_errors,
            metadata=extra_metadata or {},
        )

