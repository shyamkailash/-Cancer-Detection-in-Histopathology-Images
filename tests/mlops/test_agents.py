"""
Unit tests for Autonomous Bounded Agents.
"""

import pytest
import numpy as np
from agents.client_selection import ClientSelectionAgent
from agents.drift import DriftAgent
from agents.evaluation import EvaluationAgent
from agents.privacy import PrivacyAgent
from agents.registry import RegistryAgent
from agents.retraining import RetrainingAgent
from agents.training import TrainingAgent
from mlops.contracts import QualityGateConfig


def test_client_selection_agent():
    agent = ClientSelectionAgent()
    res = agent.run({"strategy": "fedavg", "participation_rate": 1.0})
    assert res.status == "SUCCESS"
    assert res.decision == "CLIENTS_SELECTED"
    assert len(res.details["selected_client_ids"]) == 5


def test_drift_agent():
    agent = DriftAgent()
    res_clean = agent.run({"baseline": [0.5, 0.5], "current": [0.5, 0.5]})
    assert res_clean.status == "SUCCESS"
    assert res_clean.decision == "NO_DRIFT"

    res_shift = agent.run({"baseline": [0.1, 0.9], "current": [0.9, 0.1]})
    assert res_shift.status == "SUCCESS"
    assert res_shift.decision == "DRIFT_DETECTED"


def test_evaluation_agent():
    agent = EvaluationAgent()
    res_pass = agent.run({
        "metrics": {"accuracy": 0.953, "roc_auc": 0.985, "sensitivity": 0.940},
        "quality_gate": QualityGateConfig(min_accuracy=0.90),
    })
    assert res_pass.status == "SUCCESS"
    assert res_pass.decision == "QUALITY_GATE_PASSED"

    res_fail = agent.run({
        "metrics": {"accuracy": 0.850},
        "quality_gate": QualityGateConfig(min_accuracy=0.90),
    })
    assert res_fail.status == "WARNING"
    assert res_fail.decision == "QUALITY_GATE_FAILED"


def test_privacy_agent():
    agent = PrivacyAgent()
    res_dp = agent.run({
        "privacy_enabled": True,
        "strategy": "dp_fedavg",
        "steps": 15,
        "sample_rate": 64.0 / 30804.0,
        "noise_multiplier": 1.0,
    })
    assert res_dp.status == "SUCCESS"
    assert res_dp.decision == "PRIVACY_BUDGET_SATISFIED"
    assert res_dp.details["epsilon"] > 0.0


def test_retraining_agent():
    agent = RetrainingAgent()
    res_idle = agent.run({"drift_detected": False})
    assert res_idle.decision == "NO_ACTION"

    res_trigger = agent.run({"drift_detected": True, "sample_count": 100})
    assert res_trigger.decision == "RETRAINING_PLANNED"
    assert "plan" in res_trigger.details


def test_training_agent_dry_run_safety():
    agent = TrainingAgent()
    res_dry = agent.run({"strategy": "fedavg", "dry_run": True})
    assert res_dry.status == "SUCCESS"
    assert res_dry.decision == "DRY_RUN_DISPATCHED"
    assert res_dry.details["task_spec"]["dry_run"] is True
