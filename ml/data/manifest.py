"""
Dataset Manifest for managing sample records, serialization (CSV/JSONL), and summary statistics.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from .base import SampleRecord


class DatasetManifest:
    """
    Collection of SampleRecords with serialization, indexing, and statistical querying capabilities.
    """

    def __init__(self, samples: Optional[List[SampleRecord]] = None, source_id: Optional[str] = None):
        self.samples: List[SampleRecord] = samples or []
        self.source_id = source_id
        self._index_by_id: Dict[str, SampleRecord] = {}
        self._index_by_hash: Dict[str, List[SampleRecord]] = {}
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild internal lookup indices."""
        self._index_by_id.clear()
        self._index_by_hash.clear()
        for sample in self.samples:
            self._index_by_id[sample.sample_id] = sample
            if sample.file_hash:
                if sample.file_hash not in self._index_by_hash:
                    self._index_by_hash[sample.file_hash] = []
                self._index_by_hash[sample.file_hash].append(sample)

    def add_sample(self, sample: SampleRecord) -> None:
        """Add a sample record to manifest."""
        self.samples.append(sample)
        self._index_by_id[sample.sample_id] = sample
        if sample.file_hash:
            if sample.file_hash not in self._index_by_hash:
                self._index_by_hash[sample.file_hash] = []
            self._index_by_hash[sample.file_hash].append(sample)

    def get_by_id(self, sample_id: str) -> Optional[SampleRecord]:
        """Lookup sample by sample_id."""
        return self._index_by_id.get(sample_id)

    def get_by_hash(self, file_hash: str) -> List[SampleRecord]:
        """Lookup samples by file_hash."""
        return self._index_by_hash.get(file_hash, [])

    def filter(
        self,
        is_valid: Optional[bool] = None,
        split: Optional[str] = None,
        source_id: Optional[str] = None,
        standardized_label: Optional[int] = None,
    ) -> "DatasetManifest":
        """Filter samples based on criteria and return a new DatasetManifest."""
        filtered = self.samples
        if is_valid is not None:
            filtered = [s for s in filtered if s.is_valid == is_valid]
        if split is not None:
            filtered = [s for s in filtered if s.split == split]
        if source_id is not None:
            filtered = [s for s in filtered if s.source_id == source_id]
        if standardized_label is not None:
            filtered = [s for s in filtered if s.standardized_label == standardized_label]

        return DatasetManifest(samples=filtered, source_id=self.source_id)

    def get_summary(self) -> Dict[str, Any]:
        """Compute statistical summary of samples in the manifest."""
        total = len(self.samples)
        valid_count = sum(1 for s in self.samples if s.is_valid)
        invalid_count = total - valid_count

        label_counts: Dict[str, int] = {}
        split_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        patient_ids = set()
        slide_ids = set()
        unique_hashes = set()
        formats: Dict[str, int] = {}

        for s in self.samples:
            # Labels
            lbl = str(s.label_name if s.label_name is not None else s.standardized_label)
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

            # Splits
            split_counts[s.split] = split_counts.get(s.split, 0) + 1

            # Sources
            source_counts[s.source_id] = source_counts.get(s.source_id, 0) + 1

            # Patients / Slides / Hashes
            if s.patient_id:
                patient_ids.add(s.patient_id)
            if s.slide_id:
                slide_ids.add(s.slide_id)
            if s.file_hash:
                unique_hashes.add(s.file_hash)
            if s.image_format:
                formats[s.image_format] = formats.get(s.image_format, 0) + 1

        exact_duplicate_count = total - len(unique_hashes) if unique_hashes else 0

        return {
            "total_samples": total,
            "valid_samples": valid_count,
            "invalid_samples": invalid_count,
            "unique_file_hashes": len(unique_hashes),
            "exact_duplicates": exact_duplicate_count,
            "unique_patients": len(patient_ids),
            "unique_slides": len(slide_ids),
            "label_distribution": label_counts,
            "split_distribution": split_counts,
            "source_distribution": source_counts,
            "format_distribution": formats,
        }

    def save_jsonl(self, filepath: Union[str, Path]) -> None:
        """Save manifest to JSONL format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(s.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, filepath: Union[str, Path]) -> "DatasetManifest":
        """Load manifest from JSONL format."""
        path = Path(filepath)
        samples: List[SampleRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(SampleRecord.from_dict(json.loads(line)))
        return cls(samples=samples)

    def save_csv(self, filepath: Union[str, Path]) -> None:
        """Save manifest to CSV format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            with open(path, "w", encoding="utf-8", newline="") as f:
                pass
            return

        fieldnames = list(self.samples[0].to_dict().keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.samples:
                row = s.to_dict()
                # Serialize list/dict fields as JSON strings for CSV compatibility
                row["validation_errors"] = json.dumps(row.get("validation_errors", []))
                row["metadata"] = json.dumps(row.get("metadata", {}))
                writer.writerow(row)

    @classmethod
    def load_csv(cls, filepath: Union[str, Path]) -> "DatasetManifest":
        """Load manifest from CSV format."""
        path = Path(filepath)
        samples: List[SampleRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Deserialize complex fields
                if "validation_errors" in row and row["validation_errors"]:
                    try:
                        row["validation_errors"] = json.loads(row["validation_errors"])
                    except Exception:
                        row["validation_errors"] = []
                if "metadata" in row and row["metadata"]:
                    try:
                        row["metadata"] = json.loads(row["metadata"])
                    except Exception:
                        row["metadata"] = {}

                # Cast numeric and boolean fields
                if "is_valid" in row:
                    row["is_valid"] = str(row["is_valid"]).lower() in ("true", "1", "yes")
                if "standardized_label" in row and row["standardized_label"] != "":
                    try:
                        row["standardized_label"] = int(row["standardized_label"])
                    except (ValueError, TypeError):
                        pass
                if "image_width" in row and row["image_width"]:
                    try:
                        row["image_width"] = int(row["image_width"])
                    except (ValueError, TypeError):
                        pass
                if "image_height" in row and row["image_height"]:
                    try:
                        row["image_height"] = int(row["image_height"])
                    except (ValueError, TypeError):
                        pass
                if "file_size_bytes" in row and row["file_size_bytes"]:
                    try:
                        row["file_size_bytes"] = int(row["file_size_bytes"])
                    except (ValueError, TypeError):
                        pass

                samples.append(SampleRecord.from_dict(row))
        return cls(samples=samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

