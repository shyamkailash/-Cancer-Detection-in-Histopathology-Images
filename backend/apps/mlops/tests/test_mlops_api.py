"""
API integration tests for MLOps REST endpoints.
"""

import pytest
from rest_framework.test import APIClient
from rest_framework import status


@pytest.fixture
def api_client():
    return APIClient()


def test_mlops_status_endpoint(api_client):
    """Test GET /api/mlops/status/ returns subsystem diagnostics."""
    response = api_client.get("/api/mlops/status/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "tracking" in data["components"]
    assert "drift_monitoring" in data["components"]
    assert "registry" in data["components"]
    assert "distributed_coordination" in data["components"]


def test_mlops_models_endpoint(api_client):
    """Test GET /api/mlops/models/ returns registered models with lifecycle states."""
    response = api_client.get("/api/mlops/models/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] >= 6
    model_ids = [m["model_id"] for m in data["models"]]
    assert "centralized" in model_ids
    assert "fedavg" in model_ids


def test_mlops_drift_endpoints(api_client):
    """Test GET /api/mlops/drift/ and POST /api/mlops/drift/check/."""
    # GET
    res_get = api_client.get("/api/mlops/drift/")
    assert res_get.status_code == status.HTTP_200_OK
    data_get = res_get.json()
    assert "current_config" in data_get

    # POST check
    payload = {
        "baseline": [0.1, 0.9],
        "current": [0.8, 0.2],
        "method": "psi",
    }
    res_post = api_client.post("/api/mlops/drift/check/", payload, format="json")
    assert res_post.status_code == status.HTTP_200_OK
    data_post = res_post.json()
    assert data_post["status"] == "SUCCESS"
    assert data_post["details"]["drift_detected"] is True


def test_mlops_retraining_endpoints(api_client):
    """Test GET /api/mlops/retraining/ and POST /api/mlops/retraining/evaluate/."""
    payload = {
        "drift_detected": True,
        "strategy": "fedprox",
        "sample_count": 100,
    }
    res_eval = api_client.post("/api/mlops/retraining/evaluate/", payload, format="json")
    assert res_eval.status_code == status.HTTP_200_OK
    data_eval = res_eval.json()
    assert data_eval["decision"] == "RETRAINING_PLANNED"

    res_list = api_client.get("/api/mlops/retraining/")
    assert res_list.status_code == status.HTTP_200_OK
    assert res_list.json()["count"] >= 1


def test_mlops_orchestrate_endpoint(api_client):
    """Test POST /api/mlops/orchestrate/ runs full 14-stage lifecycle safely."""
    payload = {
        "strategy": "fedavg",
        "dry_run": True,
        "simulate_drift": False,
    }
    response = api_client.post("/api/mlops/orchestrate/", payload, format="json")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "DRY_RUN_COMPLETED"
    assert len(data["stages"]) == 14


def test_mlops_registry_endpoints(api_client):
    """Test POST /api/mlops/registry/validate/ and POST /api/mlops/registry/approve/."""
    # Validate centralized baseline
    val_payload = {
        "model_id": "centralized",
        "quality_gate": {"min_accuracy": 0.90, "min_roc_auc": 0.90},
    }
    res_val = api_client.post("/api/mlops/registry/validate/", val_payload, format="json")
    assert res_val.status_code == status.HTTP_200_OK
    assert res_val.json()["passed"] is True

    # Approve
    app_payload = {
        "model_id": "centralized",
        "approver_id": "chief_medical_officer",
    }
    res_app = api_client.post("/api/mlops/registry/approve/", app_payload, format="json")
    assert res_app.status_code == status.HTTP_200_OK
    assert res_app.json()["approved"] is True
    assert res_app.json()["model"]["approver"] == "chief_medical_officer"
