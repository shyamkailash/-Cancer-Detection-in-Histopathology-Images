"""
Tests for ML Inference Pipeline and Grad-CAM Visual Explainability.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import torch
from ml.models.resnet import create_resnet18
from ml.inference.model_manager import ModelManager, MODEL_REGISTRY_CONFIG
from ml.inference.gradcam import GradCAM, generate_heatmap_overlay, pil_to_base64, base64_to_pil
from ml.inference.pipeline import InferencePipeline


def create_synthetic_patch(size=(96, 96), color=(200, 100, 150)) -> Image.Image:
    """Helper to create a small test patch."""
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def test_model_manager_list_and_canonicalize():
    """Test model registry and alias resolution."""
    manager = ModelManager(device=torch.device("cpu"))
    models = manager.list_models()

    assert len(models) == len(MODEL_REGISTRY_CONFIG)
    assert any(m["id"] == "centralized" for m in models)
    assert any(m["id"] == "fedavg" for m in models)
    assert any(m["id"] == "fedprox" for m in models)
    assert any(m["id"] == "fedbn" for m in models)
    assert any(m["id"] == "dp_fedavg" for m in models)

    assert manager.canonicalize_model_name("baseline") == "centralized"
    assert manager.canonicalize_model_name("federated") == "fedavg"
    assert manager.canonicalize_model_name("proximal") == "fedprox"
    assert manager.canonicalize_model_name("batchnorm") == "fedbn"
    assert manager.canonicalize_model_name("privacy_federated") == "dp_fedavg"


def test_centralized_default_resolves_to_baseline_checkpoint():
    manager = ModelManager(device=torch.device("cpu"))

    checkpoint = manager.resolve_checkpoint_path("centralized")

    assert checkpoint == Path(MODEL_REGISTRY_CONFIG["centralized"]["default_checkpoint"])


def test_custom_structured_checkpoint_loads(tmp_path):
    checkpoint_path = tmp_path / "candidate.pt"
    source_model = create_resnet18(num_classes=2, pretrained=False)
    torch.save({"epoch": 3, "model_state_dict": source_model.state_dict()}, checkpoint_path)
    manager = ModelManager(device=torch.device("cpu"))

    model = manager.get_model("centralized", checkpoint_path=str(checkpoint_path))

    assert model.training is False


def test_invalid_checkpoint_path_has_clear_error(tmp_path):
    manager = ModelManager(device=torch.device("cpu"))
    missing_path = tmp_path / "missing.pt"

    try:
        manager.get_model("centralized", checkpoint_path=str(missing_path))
    except FileNotFoundError as exc:
        assert "Checkpoint file not found" in str(exc)
    else:
        raise AssertionError("Expected a missing checkpoint error")


def test_incompatible_checkpoint_has_clear_error(tmp_path):
    checkpoint_path = tmp_path / "incompatible.pt"
    incompatible_model = create_resnet18(num_classes=3, pretrained=False)
    torch.save({"model_state_dict": incompatible_model.state_dict()}, checkpoint_path)
    manager = ModelManager(device=torch.device("cpu"))

    try:
        manager.get_model("centralized", checkpoint_path=str(checkpoint_path))
    except ValueError as exc:
        assert "incompatible with the expected ResNet-18 architecture" in str(exc)
    else:
        raise AssertionError("Expected an incompatible checkpoint error")


def test_model_manager_fallback_checkpoint_resolution(tmp_path, monkeypatch):
    """Test fallback checkpoint resolution when default checkpoint is absent."""
    fallback_file = tmp_path / "fallback_model.pt"
    dummy_model = create_resnet18(num_classes=2, pretrained=False)
    torch.save({"model_state_dict": dummy_model.state_dict()}, fallback_file)

    test_config = {
        "test_model": {
            "name": "test_model",
            "display_name": "Test Model",
            "description": "Test model for fallback verification",
            "paradigm": "Test",
            "default_checkpoint": str(tmp_path / "nonexistent_default.pt"),
            "fallback_checkpoint": str(fallback_file),
        }
    }
    monkeypatch.setattr("ml.inference.model_manager.MODEL_REGISTRY_CONFIG", test_config)

    manager = ModelManager(device=torch.device("cpu"))
    resolved = manager.resolve_checkpoint_path("test_model")
    assert resolved == Path(str(fallback_file))

    catalog = manager.list_models()
    assert len(catalog) == 1
    assert catalog[0]["checkpoint_exists"] is True
    assert catalog[0]["checkpoint_path"] == str(fallback_file)

    loaded = manager.get_model("test_model")
    assert loaded is not None
    assert manager.list_models()[0]["is_cached"] is True


def test_model_manager_missing_default_and_fallback_raises(tmp_path, monkeypatch):
    """Test error raised when neither default nor fallback checkpoint exists."""
    test_config = {
        "missing_model": {
            "name": "missing_model",
            "display_name": "Missing Model",
            "description": "Test missing",
            "paradigm": "Test",
            "default_checkpoint": str(tmp_path / "missing_default.pt"),
            "fallback_checkpoint": str(tmp_path / "missing_fallback.pt"),
        }
    }
    monkeypatch.setattr("ml.inference.model_manager.MODEL_REGISTRY_CONFIG", test_config)

    manager = ModelManager(device=torch.device("cpu"))
    catalog = manager.list_models()
    assert catalog[0]["checkpoint_exists"] is False
    assert catalog[0]["checkpoint_path"] == ""

    try:
        manager.resolve_checkpoint_path("missing_model")
    except FileNotFoundError as exc:
        assert "No valid checkpoint found for model 'missing_model'" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")

    try:
        manager.get_model("missing_model")
    except FileNotFoundError as exc:
        assert "No valid checkpoint found for model 'missing_model'" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_model_manager_corrupted_checkpoint_raises(tmp_path):
    """Test corrupted checkpoint file raises descriptive ValueError."""
    corrupt_file = tmp_path / "corrupt.pt"
    corrupt_file.write_bytes(b"This is not a valid torch checkpoint file!!!")

    manager = ModelManager(device=torch.device("cpu"))
    try:
        manager.get_model("centralized", checkpoint_path=str(corrupt_file))
    except ValueError as exc:
        assert "corrupted or unreadable" in str(exc)
    else:
        raise AssertionError("Expected ValueError for corrupted checkpoint")


def test_model_manager_raw_state_dict_and_module_prefix(tmp_path):
    """Test loading raw state dict and DataParallel module. prefix stripping."""
    raw_ckpt = tmp_path / "dataparallel.pt"
    source_model = create_resnet18(num_classes=2, pretrained=False)
    # Add 'module.' prefix to all keys
    dp_state_dict = {f"module.{k}": v for k, v in source_model.state_dict().items()}
    torch.save(dp_state_dict, raw_ckpt)

    manager = ModelManager(device=torch.device("cpu"))
    model = manager.get_model("centralized", checkpoint_path=str(raw_ckpt))
    assert model is not None
    assert model.training is False


def test_gradcam_generation():
    """Test Grad-CAM activation map extraction on ResNet-18."""
    model = create_resnet18(num_classes=2, pretrained=False)
    gradcam = GradCAM(model)

    dummy_input = torch.randn(1, 3, 96, 96)
    cam_map, pred_class, prob = gradcam.generate_cam(dummy_input)

    assert isinstance(cam_map, np.ndarray)
    assert cam_map.ndim == 2
    assert 0.0 <= np.min(cam_map) and np.max(cam_map) <= 1.0
    assert pred_class in (0, 1)
    assert 0.0 <= prob <= 1.0

    gradcam.remove_hooks()


def test_heatmap_overlay_and_base64_converters():
    """Test blending heatmap onto RGB image and roundtripping base64."""
    img = create_synthetic_patch((96, 96), (180, 120, 150))
    cam = np.random.uniform(0.0, 1.0, (7, 7)).astype(np.float32)

    heat_pil, overlay_pil = generate_heatmap_overlay(img, cam, alpha=0.5)

    assert heat_pil.size == (96, 96)
    assert overlay_pil.size == (96, 96)

    # Base64 test
    b64_str = pil_to_base64(overlay_pil, "PNG")
    assert b64_str.startswith("data:image/png;base64,")

    reloaded_pil = base64_to_pil(b64_str)
    assert reloaded_pil.size == (96, 96)


def test_inference_pipeline_predict():
    """Test complete inference pipeline end-to-end."""
    pipeline = InferencePipeline()
    patch = create_synthetic_patch((96, 96))

    result = pipeline.predict(patch, model_name="centralized", include_gradcam=True)

    assert result["status"] == "success"
    assert "prediction" in result
    assert result["prediction"]["class_id"] in (0, 1)
    assert "probabilities" in result["prediction"]
    assert "explainability" in result
    assert "heatmap_base64" in result["explainability"]
    assert "overlay_base64" in result["explainability"]
    assert result["inference_time_ms"] > 0


def test_gradcam_context_manager_and_hook_cleanup():
    """Test GradCAM context manager cleans up hooks properly."""
    model = create_resnet18(num_classes=2, pretrained=False)
    dummy_input = torch.randn(1, 3, 96, 96)

    with GradCAM(model) as gradcam:
        assert len(gradcam._handlers) == 2
        cam_map, _, _ = gradcam.generate_cam(dummy_input)
        assert cam_map is not None

    assert len(gradcam._handlers) == 0


def test_gradcam_invalid_target_class_raises():
    """Test GradCAM raises ValueError when target class is out of bounds."""
    model = create_resnet18(num_classes=2, pretrained=False)
    dummy_input = torch.randn(1, 3, 96, 96)

    with GradCAM(model) as gradcam:
        try:
            gradcam.generate_cam(dummy_input, target_class=5)
        except ValueError as exc:
            assert "out of bounds" in str(exc)
        else:
            raise AssertionError("Expected ValueError for out of bounds target class")


def test_pipeline_invalid_image_inputs_raise():
    """Test pipeline raises ValueError on invalid or empty image inputs."""
    pipeline = InferencePipeline()

    try:
        pipeline.predict(None)
    except ValueError as exc:
        assert "Empty image input" in str(exc)
    else:
        raise AssertionError("Expected ValueError for None image")

    try:
        pipeline.predict(b"")
    except ValueError as exc:
        assert "Empty image bytes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty bytes")

    try:
        pipeline.predict(b"not an image binary content")
    except ValueError as exc:
        assert "Invalid or unreadable image bytes" in str(exc)
    else:
        raise AssertionError("Expected ValueError for corrupted bytes")


def test_pipeline_invalid_target_class_raises():
    """Test pipeline raises ValueError when target_class is not in (0, 1)."""
    pipeline = InferencePipeline()
    patch = create_synthetic_patch((96, 96))

    try:
        pipeline.predict(patch, target_class=2)
    except ValueError as exc:
        assert "Invalid target_class" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid target_class")


def test_pipeline_gradcam_failure_graceful_degradation(monkeypatch):
    """Test pipeline returns valid prediction even if Grad-CAM raises an exception."""
    pipeline = InferencePipeline()
    patch = create_synthetic_patch((96, 96))

    def mock_generate_cam(*args, **kwargs):
        raise RuntimeError("Synthetic Grad-CAM failure")

    monkeypatch.setattr(GradCAM, "generate_cam", mock_generate_cam)

    result = pipeline.predict(patch, model_name="centralized", include_gradcam=True)

    assert result["status"] == "success"
    assert "prediction" in result
    assert result["prediction"]["class_id"] in (0, 1)
    assert result["explainability"] is not None
    assert "error" in result["explainability"]
    assert "Grad-CAM generation failed" in result["explainability"]["error"]

