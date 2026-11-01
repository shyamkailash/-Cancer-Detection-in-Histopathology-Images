"""
API integration tests for predictions and explainability endpoints.
"""

import io
from PIL import Image
import numpy as np
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient
from rest_framework import status

from ml.inference.gradcam import pil_to_base64


def create_test_image_file(name: str = "test_patch.png", size=(96, 96)) -> SimpleUploadedFile:
    """Create in-memory PNG image file for multipart upload."""
    arr = np.random.randint(50, 200, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return SimpleUploadedFile(name, buffer.getvalue(), content_type="image/png")


@pytest.fixture
def api_client():
    return APIClient()


def test_models_list_endpoint(api_client):
    """Test GET /api/models/ returns list of available models."""
    response = api_client.get("/api/models/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "success"
    assert len(data["models"]) == 6
    model_ids = [m["id"] for m in data["models"]]
    assert "centralized" in model_ids
    assert "centralized_finetuned" in model_ids
    assert "fedavg" in model_ids
    assert "fedprox" in model_ids
    assert "fedbn" in model_ids
    assert "dp_fedavg" in model_ids


def test_predict_endpoint_get_documentation(api_client):
    """Test GET /api/predict/ returns 200 OK with usage instructions and schema."""
    response = api_client.get("/api/predict/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "info"
    assert data["method"] == "POST"
    assert "expected_payload" in data
    assert "image" in data["expected_payload"]["fields"]


def test_predict_endpoint_multipart_upload(api_client):
    """Test POST /api/predict/ with multipart image upload."""
    img_file = create_test_image_file()
    payload = {
        "image": img_file,
        "model_name": "centralized",
        "include_gradcam": "true",
    }

    response = api_client.post("/api/predict/", data=payload, format="multipart")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["status"] == "success"
    assert "prediction" in data
    assert data["prediction"]["class_id"] in (0, 1)
    assert data["prediction"]["class_name"] in ("normal", "metastasis")
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0
    assert "probabilities" in data["prediction"]
    assert "normal" in data["prediction"]["probabilities"]
    assert "metastasis" in data["prediction"]["probabilities"]

    # Verify Grad-CAM explainability visuals
    assert "explainability" in data
    assert data["explainability"]["method"] == "Grad-CAM"
    assert data["explainability"]["heatmap_base64"].startswith("data:image/png;base64,")
    assert data["explainability"]["overlay_base64"].startswith("data:image/png;base64,")
    assert data["explainability"]["original_base64"].startswith("data:image/png;base64,")


def test_predict_endpoint_base64_payload(api_client):
    """Test POST /api/predict/ with base64 data URI payload."""
    img = Image.new("RGB", (96, 96), color=(200, 100, 120))
    b64_uri = pil_to_base64(img, "PNG")

    payload = {
        "image_base64": b64_uri,
        "model_name": "fedavg",
        "include_gradcam": False,
    }

    response = api_client.post("/api/predict/", data=payload, format="json")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["status"] == "success"
    assert data["model"]["id"] == "fedavg"
    assert data["explainability"] is None


def test_predict_endpoint_all_models(api_client):
    """Test POST /api/predict/ with centralized_finetuned, fedprox, fedbn, and dp_fedavg."""
    for model_name in ["centralized_finetuned", "fedprox", "fedbn", "dp_fedavg"]:
        img_file = create_test_image_file(f"patch_{model_name}.png")
        resp = api_client.post(
            "/api/predict/",
            data={"image": img_file, "model_name": model_name},
            format="multipart",
        )
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["model"]["id"] == model_name
        assert data["status"] == "success"


def test_predict_endpoint_missing_image(api_client):
    """Test POST /api/predict/ without image returns 400 Bad Request."""
    response = api_client.post("/api/predict/", data={"model_name": "centralized"}, format="json")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["status"] == "error"
    assert "No image provided" in data["error"]


def test_predict_endpoint_corrupted_image(api_client):
    """Test POST /api/predict/ with corrupted bytes returns 400 Bad Request."""
    corrupted_file = SimpleUploadedFile("corrupt.png", b"not an image", content_type="image/png")
    response = api_client.post("/api/predict/", data={"image": corrupted_file}, format="multipart")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["status"] == "error"
    assert "Invalid or unreadable image bytes" in data["error"]


def test_predict_endpoint_invalid_model(api_client):
    """Test POST /api/predict/ with non-existent model name returns 400 Bad Request."""
    img_file = create_test_image_file()
    payload = {
        "image": img_file,
        "model_name": "invalid_model_xyz",
    }
    response = api_client.post("/api/predict/", data=payload, format="multipart")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "Unknown model" in data["error"]


def test_demo_view_endpoint(api_client):
    """Test GET /api/demo/ returns 200 OK and HTML demonstration UI."""
    response = api_client.get("/api/demo/")
    assert response.status_code == status.HTTP_200_OK
    assert "text/html" in response["Content-Type"]
    assert b"Privacy-Preserving Federated Cancer Detection" in response.content
    assert b"Centralized ResNet-18 Fine-Tuned (95.35% Acc)" in response.content
    assert b"Privacy-Preserving DP-FedAvg (94.33% Acc)" in response.content
    assert b"errorBox" in response.content


def test_predict_response_schema_and_probability_constraints(api_client):
    """Validate full response contract, probability bounds, and metadata consistency."""
    img_file = create_test_image_file("schema_test.png", size=(96, 96))
    response = api_client.post(
        "/api/predict/",
        data={"image": img_file, "model_name": "centralized_finetuned", "include_gradcam": True},
        format="multipart",
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Schema checks
    assert data["status"] == "success"
    assert "prediction" in data
    assert "model" in data
    assert "image_metadata" in data
    assert "explainability" in data
    assert "inference_time_ms" in data
    assert "disclaimer" in data

    # Prediction checks
    pred = data["prediction"]
    assert pred["class_id"] in (0, 1)
    assert pred["class_name"] in ("normal", "metastasis")
    assert isinstance(pred["description"], str) and len(pred["description"]) > 0
    assert 0.0 <= pred["confidence"] <= 1.0

    probs = pred["probabilities"]
    assert 0.0 <= probs["normal"] <= 1.0
    assert 0.0 <= probs["metastasis"] <= 1.0
    assert abs((probs["normal"] + probs["metastasis"]) - 1.0) < 1e-3
    assert pred["confidence"] == max(probs["normal"], probs["metastasis"])

    # Model metadata checks
    model_meta = data["model"]
    assert model_meta["id"] == "centralized_finetuned"
    assert "Centralized ResNet-18 Fine-Tuned" in model_meta["display_name"]
    assert model_meta["paradigm"] == "Centralized Transfer Learning"
    assert "device" in model_meta

    # Image metadata checks
    img_meta = data["image_metadata"]
    assert img_meta["original_width"] == 96
    assert img_meta["original_height"] == 96
    assert img_meta["input_channels"] == 3
    assert img_meta["processed_size"] == [96, 96]

    # Explainability checks
    exp = data["explainability"]
    assert exp["method"] == "Grad-CAM"
    assert exp["target_layer"] == "layer4"
    assert exp["explained_class_id"] in (0, 1)
    assert exp["heatmap_base64"].startswith("data:image/png;base64,")
    assert exp["overlay_base64"].startswith("data:image/png;base64,")
    assert exp["original_base64"].startswith("data:image/png;base64,")
    assert data["inference_time_ms"] > 0


def test_predict_sequential_error_recovery(api_client):
    """Verify that failed requests do not poison model cache or break subsequent valid requests."""
    # 1. Failed request (bad image)
    corrupted = SimpleUploadedFile("broken.png", b"invalid_bytes", content_type="image/png")
    resp_fail1 = api_client.post("/api/predict/", data={"image": corrupted}, format="multipart")
    assert resp_fail1.status_code == status.HTTP_400_BAD_REQUEST

    # 2. Failed request (bad model name)
    valid_file = create_test_image_file("valid1.png")
    resp_fail2 = api_client.post("/api/predict/", data={"image": valid_file, "model_name": "non_existent"}, format="multipart")
    assert resp_fail2.status_code == status.HTTP_400_BAD_REQUEST

    # 3. Subsequent valid request should succeed normally
    valid_file2 = create_test_image_file("valid2.png")
    resp_ok = api_client.post("/api/predict/", data={"image": valid_file2, "model_name": "fedavg"}, format="multipart")
    assert resp_ok.status_code == status.HTTP_200_OK
    assert resp_ok.json()["status"] == "success"
    assert resp_ok.json()["model"]["id"] == "fedavg"


def test_predict_explicit_target_class_and_invalid_target_class(api_client):
    """Test explicit target class selection and invalid target class rejection."""
    img_file = create_test_image_file("target_class.png")
    resp_class0 = api_client.post(
        "/api/predict/",
        data={"image": img_file, "target_class": "0", "include_gradcam": "true"},
        format="multipart",
    )
    assert resp_class0.status_code == status.HTTP_200_OK
    assert resp_class0.json()["explainability"]["explained_class_id"] == 0

    img_file2 = create_test_image_file("target_class2.png")
    resp_invalid = api_client.post(
        "/api/predict/",
        data={"image": img_file2, "target_class": "99"},
        format="multipart",
    )
    assert resp_invalid.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid target_class" in resp_invalid.json()["error"]
