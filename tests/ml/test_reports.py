"""
Tests for Dataset Report Generator.
"""

from ml.data.base import SampleRecord
from ml.data.manifest import DatasetManifest
from ml.reports.dataset_report import DatasetReportGenerator


def test_dataset_report_generator_json_and_markdown(tmp_path):
    """Test generating and saving JSON and Markdown quality reports."""
    s1 = SampleRecord(
        sample_id="pcam_001",
        source_id="pcam",
        dataset_name="PatchCamelyon",
        image_path="/path/001.png",
        original_label=0,
        standardized_label=0,
        label_name="normal",
        image_width=96,
        image_height=96,
        image_format="PNG",
        file_size_bytes=1024,
        file_hash="hash_1",
        split="train",
        is_valid=True,
    )
    s2 = SampleRecord(
        sample_id="pcam_002",
        source_id="pcam",
        dataset_name="PatchCamelyon",
        image_path="/path/002.png",
        original_label=1,
        standardized_label=1,
        label_name="metastasis",
        image_width=96,
        image_height=96,
        image_format="PNG",
        file_size_bytes=1024,
        file_hash="hash_2",
        split="val",
        is_valid=True,
    )

    manifest = DatasetManifest(samples=[s1, s2], source_id="pcam")
    gen = DatasetReportGenerator()
    report = gen.generate_report(manifest, dataset_name="PatchCamelyon", source_id="pcam")

    assert report["source_id"] == "pcam"
    assert report["summary"]["total_samples"] == 2
    assert report["data_leakage_audit"]["is_leakage_free"] is True

    # Test Markdown generation
    md_text = gen.generate_markdown(report)
    assert "# Dataset Quality Report: PatchCamelyon" in md_text
    assert "normal" in md_text
    assert "metastasis" in md_text

    # Test saving
    json_file = tmp_path / "report.json"
    md_file = tmp_path / "report.md"
    gen.save_json(report, json_file)
    gen.save_markdown(report, md_file)

    assert json_file.exists()
    assert md_file.exists()
