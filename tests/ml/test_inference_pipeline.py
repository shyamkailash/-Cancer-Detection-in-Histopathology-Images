"""
Tests for ML Inference Pipeline and Grad-CAM Visual Explainability.
"""

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
    assert any(m["id"] == "dp_fedavg" for m in models)

    assert manager.canonicalize_model_name("baseline") == "centralized"
    assert manager.canonicalize_model_name("federated") == "fedavg"
    assert manager.canonicalize_model_name("privacy_federated") == "dp_fedavg"


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
