"""
Retraining Decision Agent for Bounded Continual Learning.
Translates drift telemetry and degradation alerts into safe, reviewable retraining plans.
"""

from typing import Dict, Any, Optional

from agents.base import BaseAgent
from mlops.contracts import AgentResult, RetrainingAction, RetrainingResult, QualityGateConfig
from mlops.retraining_planner import RetrainingPlanner, RetrainingPlan


class RetrainingAgent(BaseAgent):
    """
    Sub-agent responsible for determining retraining necessity and generating retraining specifications.
    """

    def __init__(self, planner: Optional[RetrainingPlanner] = None):
        super().__init__(name="RetrainingAgent", description="Evaluates retraining triggers and creates structured retraining plans")
        self.planner = planner or RetrainingPlanner()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        drift_detected = context.get("drift_detected", False)
        performance_drop = context.get("performance_drop", False)
        current_model = context.get("current_model", "centralized")
        strategy = context.get("strategy", "fedavg")
        sample_count = int(context.get("sample_count", 100))
        min_samples = int(context.get("min_samples", 50))

        if not drift_detected and not performance_drop:
            decision_result = RetrainingResult(
                action=RetrainingAction.NO_ACTION,
                reason="No statistical drift or performance degradation detected.",
                selected_strategy=strategy,
            )
            return AgentResult(
                agent_name=self.name,
                status="SUCCESS",
                decision="NO_ACTION",
                details={"retraining_needed": False, "reason": decision_result.reason},
            )

        if sample_count < min_samples:
            decision_result = RetrainingResult(
                action=RetrainingAction.MONITOR,
                reason=f"Drift/Degradation observed but sample count ({sample_count}) < min required ({min_samples}).",
                selected_strategy=strategy,
            )
            return AgentResult(
                agent_name=self.name,
                status="SUCCESS",
                decision="MONITOR",
                details={"retraining_needed": False, "reason": decision_result.reason},
            )

        decision_result = RetrainingResult(
            action=RetrainingAction.TRIGGER_RETRAINING,
            reason="Confirmed distribution drift / metric degradation with sufficient telemetry buffer.",
            selected_strategy=strategy,
            metadata={"drift_detected": drift_detected, "sample_count": sample_count},
        )

        plan = self.planner.create_plan(
            retraining_decision=decision_result,
            current_model=current_model,
            strategy_override=strategy,
            rounds=int(context.get("rounds", 3)),
            local_epochs=int(context.get("local_epochs", 1)),
            batch_size=int(context.get("batch_size", 64)),
            client_ids=context.get("client_ids"),
            privacy_enabled=context.get("privacy_enabled", False),
        )

        return AgentResult(
            agent_name=self.name,
            status="SUCCESS",
            decision="RETRAINING_PLANNED",
            details={
                "retraining_needed": True,
                "plan": plan.to_dict(),
            },
        )
