"""
Tests for Image Validation and Label Standardization.
"""

from PIL import Image
import numpy as np
from ml.validation.images import ImageValidator
from ml.validation.labels import LabelValidator


def test_validate_valid_image(tmp_path):
    """Test validator on a healthy valid PNG image."""
    img_path = tmp_path / "healthy.png"
    arr = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    validator = ImageValidator()
    is_valid, errors, meta = validator.validate_image(img_path)

    assert is_valid is True
    assert len(errors) == 0
    assert meta["image_width"] == 96
    assert meta["image_height"] == 96
    assert meta["file_size_bytes"] > 0
    assert meta["file_hash"] is not None


def test_validate_zero_byte_image(tmp_path):
    """Test validator on empty 0-byte file."""
    empty_path = tmp_path / "empty.png"
    empty_path.touch()

    validator = ImageValidator()
    is_valid, errors, meta = validator.validate_image(empty_path)

    assert is_valid is False
    assert any("empty" in e.lower() for e in errors)


def test_validate_corrupted_image(tmp_path):
    """Test validator on corrupted pseudo-image with bad binary content."""
    corrupt_path = tmp_path / "corrupt.png"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_REAL_PNG_HEADER_GARBAGE_BYTES")

    validator = ImageValidator()
    is_valid, errors, meta = validator.validate_image(corrupt_path)

    assert is_valid is False
    assert len(errors) > 0


def test_validate_nonexistent_file(tmp_path):
    """Test validator on non-existent path."""
    missing_path = tmp_path / "does_not_exist.png"
    validator = ImageValidator()
    is_valid, errors = validator.validate_file(missing_path)

    assert is_valid is False
    assert "does not exist" in errors[0].lower()


def test_validate_unsupported_extension(tmp_path):
    """Test validator on unsupported file extension."""
    txt_path = tmp_path / "notes.txt"
    txt_path.write_text("histopathology metadata notes")

    validator = ImageValidator()
    is_valid, errors = validator.validate_file(txt_path)

    assert is_valid is False
    assert "unsupported" in errors[0].lower()


def test_label_validator_mappings():
    """Test LabelValidator for pcam and breakhis."""
    validator = LabelValidator()

    # PCam
    valid, std_id, name, err = validator.validate_and_map("pcam", 0)
    assert valid is True and std_id == 0 and name == "normal"

    valid, std_id, name, err = validator.validate_and_map("pcam", "tumor")
    assert valid is True and std_id == 1 and name == "metastasis"

    # BreakHis
    valid, std_id, name, err = validator.validate_and_map("breakhis", "adenosis")
    assert valid is True and std_id == 0 and name == "benign"

    valid, std_id, name, err = validator.validate_and_map("breakhis", "ductal_carcinoma")
    assert valid is True and std_id == 1 and name == "malignant"

    # Unknown label
    valid, std_id, name, err = validator.validate_and_map("pcam", "invalid_category")
    assert valid is False
    assert err is not None
