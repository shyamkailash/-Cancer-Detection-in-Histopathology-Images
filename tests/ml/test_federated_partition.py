"""
Tests for Federated Multi-Site Dataset Partitioning.
"""

from ml.data.base import SampleRecord
from ml.data.manifest import DatasetManifest
from ml.federated.partition import FederatedPartitioner


def make_dummy_manifest(n_samples: int = 100) -> DatasetManifest:
    samples = []
    for i in range(n_samples):
        # 70 train, 15 val, 15 test
        if i < 70:
            split = "train"
        elif i < 85:
            split = "val"
        else:
            split = "test"

        label = 0 if i % 2 == 0 else 1
        samples.append(SampleRecord(
            sample_id=f"sample_{i}",
            source_id="pcam",
            dataset_name="PatchCamelyon",
            image_path=f"/dummy/{i}.png",
            original_label=label,
            standardized_label=label,
            split=split,
            file_hash=f"hash_{i}",
            is_valid=True,
        ))
    return DatasetManifest(samples=samples)


def test_federated_partitioner_iid():
    """Test uniform IID partitioning across 5 simulated sites."""
    manifest = make_dummy_manifest(100)
    partitioner = FederatedPartitioner(num_clients=5, seed=42)

    partitions = partitioner.partition_manifest(manifest, partition_type="iid", target_split="train")

    assert len(partitions) == 5
    all_assigned_ids = set()
    total_assigned = 0

    for cid, part in partitions.items():
        assert part.sample_count == 14  # 70 train samples / 5 = 14 per client
        assert len(part.sample_ids) == 14
        all_assigned_ids.update(part.sample_ids)
        total_assigned += part.sample_count

    # Zero overlap check
    assert len(all_assigned_ids) == 70
    assert total_assigned == 70


def test_federated_partitioner_noniid_dirichlet():
    """Test Non-IID Dirichlet partitioning creates class distribution variation without overlap."""
    manifest = make_dummy_manifest(100)
    partitioner = FederatedPartitioner(num_clients=4, seed=42)

    partitions = partitioner.partition_manifest(manifest, partition_type="noniid", alpha=0.5, target_split="train")

    assert len(partitions) == 4
    all_assigned_ids = set()
    total_samples = 0

    for cid, part in partitions.items():
        assert part.sample_count > 0
        assert part.normal_count + part.metastasis_count == part.sample_count
        all_assigned_ids.update(part.sample_ids)
        total_samples += part.sample_count

    # Zero overlap
    assert len(all_assigned_ids) == 70
    assert total_samples == 70


def test_partition_metadata_save_and_load(tmp_path):
    """Test serializing partition metadata to JSON and reloading."""
    manifest = make_dummy_manifest(50)
    partitioner = FederatedPartitioner(num_clients=3, seed=42)
    partitions = partitioner.partition_manifest(manifest, partition_type="iid", target_split="train")

    json_path = tmp_path / "site_partitions.json"
    partitioner.save_partitions(partitions, json_path)
    assert json_path.exists()

    loaded = partitioner.load_partitions(json_path)
    assert len(loaded) == 3
    assert loaded["site_1"].sample_count == partitions["site_1"].sample_count
    assert loaded["site_1"].site_name == partitions["site_1"].site_name

