"""
Unit tests for MLflow Experiment Tracking Abstraction and Fallback Mechanism.
"""

import pytest
from mlops.tracking import MLflowTracker, BaseExperimentTracker


def test_tracker_inheritance():
    tracker = MLflowTracker(enabled=False)
    assert isinstance(tracker, BaseExperimentTracker)
    assert tracker.is_available() is False


def test_tracker_fallback_logging():
    tracker = MLflowTracker(enabled=False)
    run_id = tracker.start_run(run_name="unit_test_run", tags={"env": "test"})
    assert run_id is not None
    assert tracker.active_run_id == run_id

    tracker.log_params({"learning_rate": 0.001, "batch_size": 64})
    tracker.log_metrics({"accuracy": 0.952, "loss": 0.12}, step=1)
    tracker.set_tags({"framework": "pytorch"})

    # Check fallback storage record
    run_data = tracker._fallback_runs.get(run_id)
    assert run_data is not None
    assert run_data["params"]["learning_rate"] == 0.001
    assert run_data["metrics"]["accuracy"] == 0.952
    assert run_data["tags"]["framework"] == "pytorch"

    reg_result = tracker.register_model(model_uri="models:/candidate", name="CandidateResNet")
    assert reg_result["status"] in ("REGISTERED_LOCAL", "FALLBACK_REGISTERED")

    tracker.end_run(status="FINISHED")
    assert tracker.active_run_id is None
