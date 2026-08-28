"""
Tests for DatasetManifest serialization, filtering, and summary statistics.
"""

from ml.data.base import SampleRecord
from ml.data.manifest import DatasetManifest


def test_manifest_add_lookup_and_filter():
    """Test adding samples, indexing by ID/hash, and filtering."""
    s1 = SampleRecord(
        sample_id="s1",
        source_id="pcam",
        dataset_name="PCam",
        image_path="/1.png",
        original_label=0,
        standardized_label=0,
        file_hash="hash1",
        split="train",
        is_valid=True,
    )
    s2 = SampleRecord(
        sample_id="s2",
        source_id="pcam",
        dataset_name="PCam",
        image_path="/2.png",
        original_label=1,
        standardized_label=1,
        file_hash="hash2",
        split="val",
        is_valid=True,
    )
    s3 = SampleRecord(
        sample_id="s3",
        source_id="breakhis",
        dataset_name="BreaKHis",
        image_path="/3.png",
        original_label="benign",
        standardized_label=0,
        file_hash="hash3",
        split="unassigned",
        is_valid=False,
    )

    manifest = DatasetManifest(samples=[s1, s2, s3])
    assert len(manifest) == 3
    assert manifest.get_by_id("s1") == s1
    assert manifest.get_by_hash("hash2") == [s2]

    # Filter by validity
    valid_manifest = manifest.filter(is_valid=True)
    assert len(valid_manifest) == 2

    # Filter by source
    breakhis_manifest = manifest.filter(source_id="breakhis")
    assert len(breakhis_manifest) == 1


def test_manifest_csv_and_jsonl_roundtrip(tmp_path):
    """Test serializing manifest to CSV/JSONL and reloading."""
    s1 = SampleRecord(
        sample_id="s1",
        source_id="pcam",
        dataset_name="PCam",
        image_path="/1.png",
        original_label=0,
        standardized_label=0,
        label_name="normal",
        image_width=96,
        image_height=96,
        image_format="PNG",
        file_size_bytes=1000,
        file_hash="hash1",
        split="train",
        is_valid=True,
        validation_errors=["minor warning"],
        metadata={"custom_attr": "value1"},
    )
    manifest = DatasetManifest(samples=[s1])

    # JSONL roundtrip
    jsonl_path = tmp_path / "manifest.jsonl"
    manifest.save_jsonl(jsonl_path)
    loaded_jsonl = DatasetManifest.load_jsonl(jsonl_path)
    assert len(loaded_jsonl) == 1
    assert loaded_jsonl.samples[0].sample_id == "s1"
    assert loaded_jsonl.samples[0].is_valid is True

    # CSV roundtrip
    csv_path = tmp_path / "manifest.csv"
    manifest.save_csv(csv_path)
    loaded_csv = DatasetManifest.load_csv(csv_path)
    assert len(loaded_csv) == 1
    assert loaded_csv.samples[0].sample_id == "s1"
    assert loaded_csv.samples[0].image_width == 96
    assert loaded_csv.samples[0].is_valid is True
