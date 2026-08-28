"""
Duplicate image detection using cryptographic hashing for histopathology datasets.
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
from ..data.base import SampleRecord
from ..data.manifest import DatasetManifest


class DuplicateDetector:
    """
    Identifies exact duplicate images using SHA-256 hashes and detects cross-source/cross-split duplicates.
    """

    def find_exact_duplicates(self, samples: List[SampleRecord]) -> Dict[str, List[SampleRecord]]:
        """
        Find exact duplicate images based on file_hash.
        Returns a dictionary mapping file_hash to list of duplicate SampleRecords (for hashes with >1 samples).
        """
        hash_map: Dict[str, List[SampleRecord]] = defaultdict(list)
        for s in samples:
            if s.file_hash:
                hash_map[s.file_hash].append(s)

        # Filter to only clusters with >1 occurrence
        return {h: recs for h, recs in hash_map.items() if len(recs) > 1}

    def find_cross_source_duplicates(self, samples: List[SampleRecord]) -> Dict[str, List[SampleRecord]]:
        """Find duplicate images that appear in multiple dataset sources."""
        duplicates = self.find_exact_duplicates(samples)
        cross_source: Dict[str, List[SampleRecord]] = {}

        for h, recs in duplicates.items():
            sources = {r.source_id for r in recs}
            if len(sources) > 1:
                cross_source[h] = recs

        return cross_source

    def find_cross_split_duplicates(self, manifest: DatasetManifest) -> Dict[str, List[SampleRecord]]:
        """Find duplicate images that span across different data splits (train/val/test)."""
        duplicates = self.find_exact_duplicates(manifest.samples)
        cross_split: Dict[str, List[SampleRecord]] = {}

        for h, recs in duplicates.items():
            splits = {r.split for r in recs if r.split != "unassigned"}
            if len(splits) > 1:
                cross_split[h] = recs

        return cross_split

    def generate_duplicate_report(self, samples: List[SampleRecord]) -> Dict[str, Any]:
        """Generate a complete duplicate detection report."""
        exact_dups = self.find_exact_duplicates(samples)
        cross_source = self.find_cross_source_duplicates(samples)

        total_samples = len(samples)
        total_duplicate_instances = sum(len(recs) for recs in exact_dups.values())
        unique_duplicate_hashes = len(exact_dups)
        redundant_samples_count = total_duplicate_instances - unique_duplicate_hashes

        clusters = []
        for h, recs in exact_dups.items():
            clusters.append({
                "file_hash": h,
                "count": len(recs),
                "sample_ids": [r.sample_id for r in recs],
                "sources": list({r.source_id for r in recs}),
                "paths": [r.image_path for r in recs],
                "splits": list({r.split for r in recs}),
            })

        return {
            "total_samples": total_samples,
            "unique_duplicate_hashes": unique_duplicate_hashes,
            "total_duplicate_instances": total_duplicate_instances,
            "redundant_samples_count": redundant_samples_count,
            "cross_source_duplicate_hashes": len(cross_source),
            "duplicate_clusters": clusters,
        }

