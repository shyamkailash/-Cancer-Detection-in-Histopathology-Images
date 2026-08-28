"""
Data Leakage-Safe Dataset Splitting for Histopathology Images.
Supports patient-level, slide-level, and stratified image-level splitting.
"""

import random
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from .base import SampleRecord
from .manifest import DatasetManifest


class DatasetSplitter:
    """
    Splits dataset manifests into train, validation, and test subsets reproducibly
    while preventing data leakage through patient/slide grouping and duplicate hash isolation.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0. Got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio}")
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(
        self,
        manifest: DatasetManifest,
        strategy: str = "auto",  # 'auto', 'patient', 'slide', 'stratified'
        only_valid: bool = True,
    ) -> DatasetManifest:
        """
        Split samples in manifest into train/val/test splits.
        Returns a new DatasetManifest with split values assigned.
        """
        samples = [s for s in manifest.samples if s.is_valid] if only_valid else list(manifest.samples)
        if not samples:
            return DatasetManifest(samples=[], source_id=manifest.source_id)

        rng = random.Random(self.seed)

        # Detect grouping metadata availability
        has_patients = any(bool(s.patient_id) for s in samples)
        has_slides = any(bool(s.slide_id) for s in samples)

        chosen_strategy = strategy
        if strategy == "auto":
            if has_patients:
                chosen_strategy = "patient"
            elif has_slides:
                chosen_strategy = "slide"
            else:
                chosen_strategy = "stratified"

        # Apply splitting strategy
        if chosen_strategy == "patient" and has_patients:
            split_samples = self._split_by_group(samples, group_key="patient_id", rng=rng)
        elif chosen_strategy == "slide" and has_slides:
            split_samples = self._split_by_group(samples, group_key="slide_id", rng=rng)
        else:
            split_samples = self._split_stratified(samples, rng=rng)

        # Data Leakage Protection: Resolve exact duplicate file hashes across different splits
        split_samples = self._resolve_cross_split_duplicates(split_samples)

        # Record metadata on samples
        for s in split_samples:
            s.metadata["split_strategy"] = chosen_strategy
            s.metadata["split_seed"] = self.seed

        # Re-attach invalid samples as unassigned if only_valid was requested
        if only_valid:
            invalid_samples = [s for s in manifest.samples if not s.is_valid]
            for inv in invalid_samples:
                inv.split = "unassigned"
            all_samples = split_samples + invalid_samples
        else:
            all_samples = split_samples

        return DatasetManifest(samples=all_samples, source_id=manifest.source_id)

    def _split_by_group(
        self,
        samples: List[SampleRecord],
        group_key: str,
        rng: random.Random,
    ) -> List[SampleRecord]:
        """Group samples by group_key (e.g. patient_id) and assign entire groups to splits."""
        groups: Dict[str, List[SampleRecord]] = defaultdict(list)
        ungrouped: List[SampleRecord] = []

        for s in samples:
            val = getattr(s, group_key, None)
            if val:
                groups[val].append(s)
            else:
                ungrouped.append(s)

        group_keys = list(groups.keys())
        rng.shuffle(group_keys)

        n_groups = len(group_keys)
        n_train = max(1, int(round(n_groups * self.train_ratio))) if n_groups >= 3 else n_groups
        n_val = max(1, int(round(n_groups * self.val_ratio))) if n_groups >= 3 else 0
        if n_train + n_val >= n_groups and n_groups >= 3:
            n_train = max(1, n_groups - 2)
            n_val = 1

        train_groups = set(group_keys[:n_train])
        val_groups = set(group_keys[n_train:n_train + n_val])
        test_groups = set(group_keys[n_train + n_val:])

        # Assign splits to grouped samples
        result: List[SampleRecord] = []
        for g_id, g_samples in groups.items():
            if g_id in train_groups:
                split_name = "train"
            elif g_id in val_groups:
                split_name = "val"
            else:
                split_name = "test"

            for s in g_samples:
                s.split = split_name
                result.append(s)

        # Handle ungrouped samples with stratified splitting
        if ungrouped:
            ungrouped_split = self._split_stratified(ungrouped, rng=rng)
            result.extend(ungrouped_split)

        return result

    def _split_stratified(
        self,
        samples: List[SampleRecord],
        rng: random.Random,
    ) -> List[SampleRecord]:
        """Stratified splitting by standardized_label."""
        by_label: Dict[Any, List[SampleRecord]] = defaultdict(list)
        for s in samples:
            by_label[s.standardized_label].append(s)

        result: List[SampleRecord] = []
        for label, label_samples in by_label.items():
            shuffled = list(label_samples)
            rng.shuffle(shuffled)

            n = len(shuffled)
            if n == 1:
                shuffled[0].split = "train"
                result.extend(shuffled)
                continue

            n_train = max(1, int(round(n * self.train_ratio)))
            n_val = max(1, int(round(n * self.val_ratio))) if n >= 3 else 0
            if n_train + n_val >= n and n >= 3:
                n_train = max(1, n - 2)
                n_val = 1

            for i, s in enumerate(shuffled):
                if i < n_train:
                    s.split = "train"
                elif i < n_train + n_val:
                    s.split = "val"
                else:
                    s.split = "test"
                result.append(s)

        return result

    def _resolve_cross_split_duplicates(self, samples: List[SampleRecord]) -> List[SampleRecord]:
        """
        Ensure that exact duplicate images (identical file_hash) do not cross split boundaries.
        If a hash exists in multiple splits, collapse all instances into the split of the first occurrence.
        """
        hash_to_split: Dict[str, str] = {}
        for s in samples:
            if s.file_hash:
                if s.file_hash not in hash_to_split:
                    hash_to_split[s.file_hash] = s.split
                else:
                    # Enforce consistent split assignment for identical hashes
                    s.split = hash_to_split[s.file_hash]
        return samples

    def verify_no_leakage(self, manifest: DatasetManifest) -> Tuple[bool, List[str]]:
        """
        Verify that there is no data leakage between splits.
        Checks:
          1. Exact duplicate file_hashes across splits
          2. Patient IDs appearing across multiple splits
          3. Slide IDs appearing across multiple splits
        """
        issues: List[str] = []

        split_hashes: Dict[str, Set[str]] = defaultdict(set)
        split_patients: Dict[str, Set[str]] = defaultdict(set)
        split_slides: Dict[str, Set[str]] = defaultdict(set)

        for s in manifest.samples:
            if not s.is_valid or s.split == "unassigned":
                continue
            if s.file_hash:
                split_hashes[s.split].add(s.file_hash)
            if s.patient_id:
                split_patients[s.split].add(s.patient_id)
            if s.slide_id:
                split_slides[s.split].add(s.slide_id)

        splits = list(split_hashes.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                # Duplicate hash leakage
                dup_hashes = split_hashes[s1].intersection(split_hashes[s2])
                if dup_hashes:
                    issues.append(f"Data Leakage: {len(dup_hashes)} identical file hashes found across '{s1}' and '{s2}' splits.")

                # Patient leakage
                dup_patients = split_patients[s1].intersection(split_patients[s2])
                if dup_patients:
                    issues.append(f"Patient Leakage: {len(dup_patients)} patient IDs ({list(dup_patients)[:3]}...) found across '{s1}' and '{s2}' splits.")

                # Slide leakage
                dup_slides = split_slides[s1].intersection(split_slides[s2])
                if dup_slides:
                    issues.append(f"Slide Leakage: {len(dup_slides)} slide IDs ({list(dup_slides)[:3]}...) found across '{s1}' and '{s2}' splits.")

        return len(issues) == 0, issues
