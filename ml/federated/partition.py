"""
Reproducible Multi-Site Dataset Partitioning for Federated Learning (IID and Dirichlet Non-IID).
Simulates distributed healthcare site / hospital silos without raw data sharing.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from ml.data.base import SampleRecord
from ml.data.manifest import DatasetManifest


DEFAULT_SITE_NAMES = [
    "Hospital A (Simulated Site)",
    "Hospital B (Simulated Site)",
    "Hospital C (Simulated Site)",
    "Hospital D (Simulated Site)",
    "Hospital E (Simulated Site)",
    "Hospital F (Simulated Site)",
    "Hospital G (Simulated Site)",
    "Hospital H (Simulated Site)",
]


@dataclass
class SitePartition:
    """Metadata and sample assignments for a single simulated healthcare site."""
    client_id: str
    site_name: str
    sample_ids: List[str]
    sample_count: int
    normal_count: int
    metastasis_count: int
    normal_percentage: float
    metastasis_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SitePartition":
        return cls(**data)


class FederatedPartitioner:
    """
    Partitions a centralized dataset manifest across simulated healthcare client sites.
    Supports:
    - 'iid': Uniform random distribution across clients
    - 'noniid': Dirichlet-based heterogeneous label distribution (Non-IID)
    """

    def __init__(self, num_clients: int = 5, seed: int = 42):
        if num_clients < 2:
            raise ValueError(f"num_clients must be at least 2. Got {num_clients}")
        self.num_clients = num_clients
        self.seed = seed

    def partition_manifest(
        self,
        manifest: DatasetManifest,
        partition_type: str = "noniid",
        alpha: float = 0.5,
        target_split: str = "train",
    ) -> Dict[str, SitePartition]:
        """
        Partition samples of target_split (default: 'train') among clients.
        Validation and test splits are strictly preserved untouched for global evaluation.
        """
        train_samples = [
            s for s in manifest.samples
            if s.split == target_split and s.is_valid
        ]

        if not train_samples:
            raise ValueError(f"No valid samples found with split='{target_split}'.")

        rng = np.random.default_rng(self.seed)

        # Map sample_id to SampleRecord for fast lookups
        sample_map = {s.sample_id: s for s in train_samples}

        if partition_type == "iid":
            client_sample_ids = self._partition_iid(train_samples, rng)
        elif partition_type == "noniid":
            client_sample_ids = self._partition_dirichlet(train_samples, alpha, rng)
        else:
            raise ValueError(f"Unknown partition_type '{partition_type}'. Supported: 'iid', 'noniid'.")

        # Build SitePartition metadata objects
        partitions: Dict[str, SitePartition] = {}
        for idx in range(self.num_clients):
            client_id = f"site_{idx + 1}"
            site_name = DEFAULT_SITE_NAMES[idx] if idx < len(DEFAULT_SITE_NAMES) else f"Hospital {idx + 1} (Simulated)"
            s_ids = client_sample_ids[idx]

            normal_c = sum(1 for s_id in s_ids if sample_map[s_id].standardized_label == 0)
            meta_c = sum(1 for s_id in s_ids if sample_map[s_id].standardized_label == 1)
            total_c = len(s_ids)

            norm_pct = (normal_c / total_c * 100.0) if total_c > 0 else 0.0
            meta_pct = (meta_c / total_c * 100.0) if total_c > 0 else 0.0

            partitions[client_id] = SitePartition(
                client_id=client_id,
                site_name=site_name,
                sample_ids=s_ids,
                sample_count=total_c,
                normal_count=normal_c,
                metastasis_count=meta_c,
                normal_percentage=round(norm_pct, 2),
                metastasis_percentage=round(meta_pct, 2),
            )

        return partitions

    def _partition_iid(self, samples: List[SampleRecord], rng: np.random.Generator) -> List[List[str]]:
        """Evenly and randomly distribute samples across clients."""
        indices = np.arange(len(samples))
        rng.shuffle(indices)

        splits = np.array_split(indices, self.num_clients)
        return [[samples[i].sample_id for i in client_indices] for client_indices in splits]

    def _partition_dirichlet(
        self,
        samples: List[SampleRecord],
        alpha: float,
        rng: np.random.Generator,
    ) -> List[List[str]]:
        """
        Partition samples using Dirichlet distribution over class labels.
        Each client receives a skewed non-IID proportion of normal vs metastasis samples.
        """
        # Separate sample indices by class
        class_indices: Dict[int, List[int]] = {0: [], 1: []}
        for i, s in enumerate(samples):
            lbl = int(s.standardized_label) if s.standardized_label is not None else 0
            if lbl not in class_indices:
                class_indices[lbl] = []
            class_indices[lbl].append(i)

        client_assigned_indices: List[List[int]] = [[] for _ in range(self.num_clients)]

        for lbl, indices in class_indices.items():
            shuffled_idx = np.array(indices)
            rng.shuffle(shuffled_idx)

            # Sample class distribution proportions from Dirichlet(alpha * 1_K)
            proportions = rng.dirichlet(np.repeat(alpha, self.num_clients))
            # Normalize to match actual sample count
            proportions = proportions / proportions.sum()

            # Compute split points
            counts = (proportions * len(shuffled_idx)).astype(int)
            # Adjust rounding difference
            diff = len(shuffled_idx) - counts.sum()
            for k in range(diff):
                counts[k % self.num_clients] += 1

            # Slice and assign
            start_idx = 0
            for k in range(self.num_clients):
                end_idx = start_idx + counts[k]
                client_assigned_indices[k].extend(shuffled_idx[start_idx:end_idx].tolist())
                start_idx = end_idx

        # Shuffle each client's assigned samples
        for k in range(self.num_clients):
            rng.shuffle(client_assigned_indices[k])

        return [[samples[i].sample_id for i in client_indices] for client_indices in client_assigned_indices]

    @staticmethod
    def save_partitions(
        partitions: Dict[str, SitePartition],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save site partition metadata to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "disclaimer": "SIMULATED HEALTHCARE SITES - For academic/research benchmarking only. Not real patient or hospital data.",
            "num_clients": len(partitions),
            "metadata": metadata or {},
            "sites": {cid: p.to_dict() for cid, p in partitions.items()},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load_partitions(input_path: Path) -> Dict[str, SitePartition]:
        """Load site partitions from a JSON file."""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            sites_data = data.get("sites", {})
            return {cid: SitePartition.from_dict(p_dict) for cid, p_dict in sites_data.items()}

    @staticmethod
    def get_client_manifest(full_manifest: DatasetManifest, partition: SitePartition) -> DatasetManifest:
        """Create a client-specific sub-manifest from assigned sample IDs."""
        sample_id_set = set(partition.sample_ids)
        client_samples = [s for s in full_manifest.samples if s.sample_id in sample_id_set]
        return DatasetManifest(samples=client_samples, source_id=full_manifest.source_id)

