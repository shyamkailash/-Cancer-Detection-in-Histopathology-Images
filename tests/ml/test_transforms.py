"""
Tests for Preprocessing Transforms and Quality Metrics.
"""

from PIL import Image
import numpy as np
from ml.preprocessing.transforms import (
    ResizeTransform,
    ColorModeTransform,
    ToTensorTransform,
    NormalizeTransform,
    RandomHorizontalFlip,
    RandomRotation90,
    get_train_transforms,
    get_eval_transforms,
)
from ml.preprocessing.quality import ImageQualityChecker


def test_resize_and_color_mode_transforms():
    """Test image resizing and RGB conversion."""
    img = Image.new("L", (200, 150), color=128)
    color_t = ColorModeTransform("RGB")
    rgb_img = color_t(img)
    assert rgb_img.mode == "RGB"

    resize_t = ResizeTransform(size=(96, 96))
    resized_img = resize_t(rgb_img)
    assert resized_img.size == (96, 96)


def test_to_tensor_and_normalize_transforms():
    """Test converting PIL image to normalized float32 (C, H, W) tensor."""
    img = Image.new("RGB", (96, 96), color=(255, 128, 64))
    to_tensor = ToTensorTransform()
    tensor = to_tensor(img)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 96, 96)
    assert tensor.dtype == np.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0

    norm_t = NormalizeTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    normed = norm_t(tensor)
    assert normed.shape == (3, 96, 96)


def test_train_and_eval_transform_pipelines():
    """Test full compose pipelines."""
    train_pipe = get_train_transforms(target_size=(96, 96))
    eval_pipe = get_eval_transforms(target_size=(96, 96))

    img = Image.new("RGB", (120, 120), color=(180, 120, 150))
    train_out = train_pipe(img)
    eval_out = eval_pipe(img)

    assert train_out.shape == (3, 96, 96)
    assert eval_out.shape == (3, 96, 96)


def test_image_quality_checker_whitespace_and_blur():
    """Test ImageQualityChecker metrics on synthetic white vs patterned images."""
    checker = ImageQualityChecker()

    # Mostly white image (empty slide)
    white_img = Image.new("RGB", (100, 100), color=(250, 250, 250))
    ws_white = checker.calculate_whitespace_ratio(white_img, threshold=220)
    assert ws_white == 1.0

    # Cellular / textured image
    arr = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
    tissue_img = Image.fromarray(arr)
    ws_tissue = checker.calculate_whitespace_ratio(tissue_img, threshold=220)
    assert ws_tissue < 0.1

    blur_tissue = checker.calculate_blurriness(tissue_img)
    assert blur_tissue > 0.0

    quality_assessment = checker.assess_quality(tissue_img)
    assert "whitespace_ratio" in quality_assessment
    assert "blurriness_laplacian_var" in quality_assessment
    assert "contrast_std" in quality_assessment

