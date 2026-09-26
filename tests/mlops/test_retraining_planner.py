"""
Unit tests for Safe Retraining Planner.
"""

import pytest
from mlops.contracts import RetrainingAction, RetrainingResult, QualityGateConfig
from mlops.retraining_planner import RetrainingPlanner, RetrainingPlan


def test_retraining_planner_creates_valid_plan():
    planner = RetrainingPlanner()
    decision = RetrainingResult(
        action=RetrainingAction.TRIGGER_RETRAINING,
        reason="Distribution drift detected (PSI > 0.25)",
        selected_strategy="fedavg",
        metadata={"drift_detected": True},
    )

    plan = planner.create_plan(
        retraining_decision=decision,
        current_model="centralized",
        strategy_override="fedprox",
        rounds=5,
        local_epochs=1,
        batch_size=64,
    )

    assert isinstance(plan, RetrainingPlan)
    assert plan.trigger_reason == "Distribution drift detected (PSI > 0.25)"
    assert plan.recommended_strategy == "fedprox"
    assert plan.training_parameters["strategy"] == "fedprox"
    assert plan.training_parameters["rounds"] == 5
    assert plan.execution_mode == "PLANNED_ONLY"
    assert plan.approval_required is True
    assert plan.safety_checks["safe_dry_run_default"] is True
