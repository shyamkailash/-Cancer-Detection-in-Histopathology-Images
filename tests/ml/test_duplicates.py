"""
Tests for Duplicate Image Detection.
"""

from ml.data.base import SampleRecord
from ml.validation.duplicates import DuplicateDetector


def test_duplicate_detector_exact_duplicates():
    """Test detecting exact duplicate hash clusters."""
    detector = DuplicateDetector()

    s1 = SampleRecord(sample_id="s1", source_id="pcam", dataset_name="PCam", image_path="/p1.png", original_label=0, file_hash="hash_AAA")
    s2 = SampleRecord(sample_id="s2", source_id="pcam", dataset_name="PCam", image_path="/p2.png", original_label=0, file_hash="hash_AAA")
    s3 = SampleRecord(sample_id="s3", source_id="pcam", dataset_name="PCam", image_path="/p3.png", original_label=1, file_hash="hash_BBB")

    samples = [s1, s2, s3]
    dups = detector.find_exact_duplicates(samples)

    assert "hash_AAA" in dups
    assert len(dups["hash_AAA"]) == 2
    assert "hash_BBB" not in dups

    report = detector.generate_duplicate_report(samples)
    assert report["unique_duplicate_hashes"] == 1
    assert report["total_duplicate_instances"] == 2
    assert report["redundant_samples_count"] == 1


def test_duplicate_detector_cross_source_duplicates():
    """Test identifying images duplicated across different datasets."""
    detector = DuplicateDetector()

    s1 = SampleRecord(sample_id="pcam_1", source_id="pcam", dataset_name="PCam", image_path="/p1.png", original_label=0, file_hash="shared_hash")
    s2 = SampleRecord(sample_id="ext_1", source_id="external", dataset_name="Ext", image_path="/e1.png", original_label=0, file_hash="shared_hash")

    cross_dups = detector.find_cross_source_duplicates([s1, s2])
    assert "shared_hash" in cross_dups
    assert len(cross_dups["shared_hash"]) == 2
