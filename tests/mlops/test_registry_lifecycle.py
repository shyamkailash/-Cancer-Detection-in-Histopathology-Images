"""
Unit tests for Model Registry Lifecycle and Quality Gating.
"""

import pytest
from mlops.registry_lifecycle import ModelRegistryLifecycle, ModelLifecycleState
from mlops.contracts import QualityGateConfig


def test_registry_lifecycle_seed_models():
    lifecycle = ModelRegistryLifecycle()
    models = lifecycle.list_models()
    assert len(models) >= 6
    approved_models = lifecycle.list_models(state_filter=ModelLifecycleState.APPROVED)
    assert len(approved_models) >= 6


def test_registry_lifecycle_candidate_validation_and_approval():
    lifecycle = ModelRegistryLifecycle()
    candidate_id = "test_candidate_resnet"

    lifecycle.register_candidate(
        model_id=candidate_id,
        checkpoint_path="artifacts/federated/fedavg/best_global_model.pt",
        metrics={"accuracy": 0.952, "roc_auc": 0.982, "sensitivity": 0.941},
        display_name="Test Candidate Model",
    )

    model = lifecycle.get_model(candidate_id)
    assert model["state"] == ModelLifecycleState.CANDIDATE.value

    passed, note, record = lifecycle.validate_candidate(
        model_id=candidate_id,
        quality_gate=QualityGateConfig(min_accuracy=0.90, min_roc_auc=0.90),
    )
    assert passed is True
    assert record["state"] == ModelLifecycleState.VALIDATED.value

    approved = lifecycle.approve_model(model_id=candidate_id, approver_id="lead_pathologist")
    assert approved["state"] == ModelLifecycleState.APPROVED.value
    assert approved["approver"] == "lead_pathologist"

    archived = lifecycle.archive_model(model_id=candidate_id, reason="Superseded by newer iteration")
    assert archived["state"] == ModelLifecycleState.ARCHIVED.value
