"""
Tests for Data Leakage-Safe Dataset Splitting.
"""

from ml.data.base import SampleRecord
from ml.data.manifest import DatasetManifest
from ml.data.split import DatasetSplitter


def test_stratified_split_proportions_and_reproducibility():
    """Test stratified splitting generates valid proportions and is deterministic."""
    samples = []
    for i in range(100):
        label = 0 if i < 60 else 1
        samples.append(SampleRecord(
            sample_id=f"sample_{i}",
            source_id="pcam",
            dataset_name="PCam",
            image_path=f"/path/{i}.png",
            original_label=label,
            standardized_label=label,
            file_hash=f"hash_{i}",
        ))

    manifest = DatasetManifest(samples=samples)
    splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

    split_manifest_1 = splitter.split(manifest, strategy="stratified")
    summary_1 = split_manifest_1.get_summary()

    assert summary_1["split_distribution"]["train"] >= 65
    assert summary_1["split_distribution"]["val"] >= 10
    assert summary_1["split_distribution"]["test"] >= 10

    # Test exact reproducibility with same seed
    split_manifest_2 = splitter.split(manifest, strategy="stratified")
    splits_1 = [s.split for s in split_manifest_1.samples]
    splits_2 = [s.split for s in split_manifest_2.samples]
    assert splits_1 == splits_2

    # Verify no leakage
    is_clean, issues = splitter.verify_no_leakage(split_manifest_1)
    assert is_clean is True
    assert len(issues) == 0


def test_patient_level_grouped_splitting():
    """Test patient grouping ensures all samples from a patient stay within one split."""
    samples = []
    patients = [f"patient_{p}" for p in range(10)]
    for i in range(50):
        pat = patients[i % 10]
        samples.append(SampleRecord(
            sample_id=f"sample_{i}",
            source_id="breakhis",
            dataset_name="BreaKHis",
            image_path=f"/path/{i}.png",
            original_label="benign",
            standardized_label=0,
            patient_id=pat,
            file_hash=f"hash_{i}",
        ))

    manifest = DatasetManifest(samples=samples)
    splitter = DatasetSplitter(seed=42)
    split_manifest = splitter.split(manifest, strategy="patient")

    # Verify patient grouping: a patient must belong to only ONE split
    patient_splits = {}
    for s in split_manifest.samples:
        if s.patient_id not in patient_splits:
            patient_splits[s.patient_id] = s.split
        else:
            assert patient_splits[s.patient_id] == s.split, f"Patient {s.patient_id} leaked across splits!"

    is_clean, issues = splitter.verify_no_leakage(split_manifest)
    assert is_clean is True


def test_duplicate_hash_isolation_across_splits():
    """Test duplicate images with identical hashes are forced into the same split."""
    s1 = SampleRecord(sample_id="s1", source_id="pcam", dataset_name="PCam", image_path="/1.png", original_label=0, standardized_label=0, file_hash="DUP_HASH")
    s2 = SampleRecord(sample_id="s2", source_id="pcam", dataset_name="PCam", image_path="/2.png", original_label=0, standardized_label=0, file_hash="DUP_HASH")
    s3 = SampleRecord(sample_id="s3", source_id="pcam", dataset_name="PCam", image_path="/3.png", original_label=1, standardized_label=1, file_hash="UNIQUE_HASH")

    manifest = DatasetManifest(samples=[s1, s2, s3])
    splitter = DatasetSplitter(seed=42)
    split_manifest = splitter.split(manifest)

    # DUP_HASH instances must have identical split
    dups = [s for s in split_manifest.samples if s.file_hash == "DUP_HASH"]
    assert dups[0].split == dups[1].split

    is_clean, _ = splitter.verify_no_leakage(split_manifest)
    assert is_clean is True

