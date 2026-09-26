"""
Privacy Accounting Agent for Differential Privacy Management.
Computes formal Rényi Differential Privacy bounds and verifies privacy budgets.
"""

from typing import Dict, Any, Optional

from agents.base import BaseAgent
from mlops.contracts import AgentResult, PrivacyEvaluationResult
from mlops.privacy_accountant import PrivacyAccountant


class PrivacyAgent(BaseAgent):
    """
    Sub-agent responsible for auditing and validating formal differential privacy guarantees.
    """

    def __init__(self, accountant: Optional[PrivacyAccountant] = None):
        super().__init__(name="PrivacyAgent", description="Audits and calculates formal Differential Privacy epsilon bounds")
        self.accountant = accountant or PrivacyAccountant()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        enabled = context.get("privacy_enabled", False) or (context.get("strategy") == "dp_fedavg")
        if not enabled:
            return AgentResult(
                agent_name=self.name,
                status="SUCCESS",
                decision="PRIVACY_DISABLED",
                details={
                    "privacy_enabled": False,
                    "message": "Differential privacy was not enabled for this workflow.",
                },
            )

        steps = int(context.get("steps", 15))
        sample_rate = float(context.get("sample_rate", 64.0 / 30804.0))
        noise_multiplier = float(context.get("noise_multiplier", 1.0))
        delta = float(context.get("delta", 1e-5))
        clipping_norm = float(context.get("clipping_norm", 1.0))
        max_epsilon_budget = float(context.get("max_epsilon_budget", 10.0))

        spent = self.accountant.get_privacy_spent(
            steps=steps,
            sample_rate=sample_rate,
            noise_multiplier=noise_multiplier,
            delta=delta,
        )

        eps = spent.get("epsilon")
        budget_ok = eps is not None and (eps <= max_epsilon_budget)

        decision = "PRIVACY_BUDGET_SATISFIED" if budget_ok else "PRIVACY_BUDGET_EXCEEDED"
        status = "SUCCESS" if budget_ok else "WARNING"

        return AgentResult(
            agent_name=self.name,
            status=status,
            decision=decision,
            details={
                "privacy_enabled": True,
                "epsilon": eps,
                "delta": delta,
                "clipping_norm": clipping_norm,
                "noise_multiplier": noise_multiplier,
                "steps": steps,
                "sample_rate": sample_rate,
                "max_epsilon_budget": max_epsilon_budget,
                "guarantee_satisfied": budget_ok,
                "formal_diagnostics": spent,
            },
        )
