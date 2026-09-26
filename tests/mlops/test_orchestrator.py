"""
Unit and integration tests for MLOpsOrchestrator 14-stage workflow.
"""

import pytest
from agents.orchestrator import MLOpsOrchestrator
from mlops.contracts import OrchestrationStatus


def test_orchestrator_14_stage_dry_run_execution():
    orchestrator = MLOpsOrchestrator()
    result = orchestrator.run_full_lifecycle(
        strategy="fedavg",
        dry_run=True,
        simulate_drift=False,
    )

    assert result["status"] == OrchestrationStatus.DRY_RUN_COMPLETED.value
    stages = result["stages"]
    assert len(stages) == 14

    stage_names = [s["stage_name"] for s in stages]
    assert "Config Initialization" in stage_names[0]
    assert "Tracking Initialization" in stage_names[1]
    assert "Distributed Node Health Check" in stage_names[2]
    assert "Client Selection" in stage_names[3]
    assert "Privacy Accounting" in stage_names[4]
    assert "Data Manifest Verification" in stage_names[5]
    assert "Model Registry Assessment" in stage_names[6]
    assert "Drift Monitoring" in stage_names[7]
    assert "Retraining Planning" in stage_names[8]
    assert "Training Dispatch" in stage_names[9]
    assert "Candidate Evaluation" in stage_names[10]
    assert "Quality Gate Verification" in stage_names[11]
    assert "Registry Lifecycle Transition" in stage_names[12]
    assert "Run Summarization & Audit Logging" in stage_names[13]

    for stage in stages:
        assert stage["status"] in ("SUCCESS", "WARNING")


def test_orchestrator_simulated_drift_and_retraining_trigger():
    orchestrator = MLOpsOrchestrator()
    result = orchestrator.run_full_lifecycle(
        strategy="fedprox",
        dry_run=True,
        simulate_drift=True,
    )
    assert result["summary"]["drift_detected"] is True
    stage_9 = result["stages"][8]
    assert stage_9["details"]["decision"] == "RETRAINING_PLANNED"
