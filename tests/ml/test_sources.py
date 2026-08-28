"""
Tests for PCam and BreaKHis Dataset Sources.
"""

from pathlib import Path
from PIL import Image
import numpy as np
from ml.sources.pcam import PCamSource
from ml.sources.breakhis import BreakHisSource


def create_synthetic_image(path: Path, size=(96, 96), color=(200, 100, 150)):
    """Helper to create a small test image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)


def test_pcam_source_discovery_folder_structure(tmp_path):
    """Test PCamSource discovery with standard subfolder structure (0/ and 1/)."""
    pcam_dir = tmp_path / "pcam"
    create_synthetic_image(pcam_dir / "0" / "patch_001.png", size=(96, 96), color=(220, 180, 200))
    create_synthetic_image(pcam_dir / "0" / "patch_002.png", size=(96, 96), color=(220, 180, 201))
    create_synthetic_image(pcam_dir / "1" / "patch_101.png", size=(96, 96), color=(180, 50, 100))

    adapter = PCamSource()
    assert adapter.is_available(pcam_dir) is True

    samples = adapter.discover_samples(pcam_dir)
    assert len(samples) == 3

    labels = {s.standardized_label for s in samples}
    assert labels == {0, 1}

    for s in samples:
        assert s.source_id == "pcam"
        assert s.image_width == 96
        assert s.image_height == 96
        assert s.is_valid is True
        assert s.file_hash is not None


def test_breakhis_source_filename_parsing():
    """Test BreakHisSource structured filename parsing."""
    adapter = BreakHisSource()

    stem_benign = "SOB_B_A-14-22549AB-100-001"
    meta_b = adapter.parse_filename(stem_benign)
    assert meta_b["raw_label"] == "benign"
    assert meta_b["subtype"] == "A"
    assert meta_b["patient_id"] == "14-22549AB"
    assert meta_b["magnification"] == "100X"
    assert meta_b["slide_id"] == "SOB_B_A-14-22549AB"

    stem_malignant = "SOB_M_DC-14-2773-40-003"
    meta_m = adapter.parse_filename(stem_malignant)
    assert meta_m["raw_label"] == "malignant"
    assert meta_m["subtype"] == "DC"
    assert meta_m["patient_id"] == "14-2773"
    assert meta_m["magnification"] == "40X"


def test_breakhis_source_discovery(tmp_path):
    """Test BreakHisSource sample discovery and metadata population."""
    breakhis_dir = tmp_path / "breakhis"
    create_synthetic_image(breakhis_dir / "SOB_B_F-14-1234-100-001.png", size=(100, 80))
    create_synthetic_image(breakhis_dir / "SOB_M_DC-14-5678-200-002.png", size=(100, 80))

    adapter = BreakHisSource()
    assert adapter.is_available(breakhis_dir) is True

    samples = adapter.discover_samples(breakhis_dir)
    assert len(samples) == 2

    s0 = next(s for s in samples if s.standardized_label == 0)
    assert s0.label_name == "benign"
    assert s0.patient_id == "14-1234"
    assert s0.magnification == "100X"

    s1 = next(s for s in samples if s.standardized_label == 1)
    assert s1.label_name == "malignant"
    assert s1.patient_id == "14-5678"
    assert s1.magnification == "200X"
