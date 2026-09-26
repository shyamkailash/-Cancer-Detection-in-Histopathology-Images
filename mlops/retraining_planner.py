"""
Safe Retraining Workflow Planner for Bounded MLOps Execution.
Translates drift alerts and performance degradation signals into structured, reviewable retraining plans.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time
import uuid

from mlops.contracts import RetrainingAction, RetrainingResult, QualityGateConfig


@dataclass
class RetrainingPlan:
    """Structured, safe retraining plan requiring review before compute execution."""
    plan_id: str
    created_at: float
    trigger_reason: str
    trigger_metric: str
    current_model: str
    target_model_name: str
    recommended_strategy: str
    training_parameters: Dict[str, Any]
    evaluation_requirements: Dict[str, Any]
    execution_mode: str  # "PLANNED_ONLY", "DRY_RUN", "EXECUTED"
    approval_required: bool = True
    approved_by: Optional[str] = None
    safety_checks: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RetrainingPlanner:
    """
    Generates structured, verifiable retraining plans from drift/audit signals.
    Enforces that training plans are created safely without triggering unreviewed GPU/CPU compute.
    """

    def __init__(self):
        pass

    def create_plan(
        self,
        retraining_decision: RetrainingResult,
        current_model: str = "centralized",
        strategy_override: Optional[str] = None,
        rounds: int = 5,
        local_epochs: int = 1,
        batch_size: int = 64,
        client_ids: Optional[List[str]] = None,
        privacy_enabled: bool = False,
        quality_gate: Optional[QualityGateConfig] = None,
    ) -> RetrainingPlan:
        """
        Formulate a structured RetrainingPlan from an upstream RetrainingDecision.
        """
        plan_id = f"plan_{str(uuid.uuid4())[:8]}"
        now = time.time()

        selected_strategy = strategy_override or retraining_decision.selected_strategy or "fedavg"
        clients = client_ids or ["site_0", "site_1", "site_2", "site_3", "site_4"]
        qg = quality_gate or QualityGateConfig(min_accuracy=0.90, min_roc_auc=0.90)

        # Build training parameter specifications
        training_params = {
            "strategy": selected_strategy,
            "rounds": rounds if "fed" in selected_strategy else 1,
            "epochs": local_epochs if "fed" in selected_strategy else rounds,
            "batch_size": batch_size,
            "clients": clients,
            "privacy_enabled": privacy_enabled or (selected_strategy == "dp_fedavg"),
            "max_grad_norm": 1.0 if (privacy_enabled or selected_strategy == "dp_fedavg") else None,
            "noise_multiplier": 1.0 if (privacy_enabled or selected_strategy == "dp_fedavg") else None,
        }

        eval_reqs = {
            "min_accuracy": qg.min_accuracy,
            "min_roc_auc": qg.min_roc_auc,
            "min_sensitivity": qg.min_sensitivity,
            "target_dataset": "PCam Global Test Set (33,003 samples)",
        }

        safety_checks = {
            "safe_dry_run_default": True,
            "no_automatic_production_overwrite": True,
            "quality_gate_enforced": True,
            "sample_buffer_sufficient": retraining_decision.action == RetrainingAction.TRIGGER_RETRAINING,
        }

        return RetrainingPlan(
            plan_id=plan_id,
            created_at=now,
            trigger_reason=retraining_decision.reason,
            trigger_metric=retraining_decision.metadata.get("drift_detected", "manual_or_audit"),
            current_model=current_model,
            target_model_name=f"{current_model}_retrained_{plan_id}",
            recommended_strategy=selected_strategy,
            training_parameters=training_params,
            evaluation_requirements=eval_reqs,
            execution_mode="PLANNED_ONLY",
            approval_required=True,
            safety_checks=safety_checks,
        )
