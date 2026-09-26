"""
Training Orchestration Agent for Federated and Centralized Training.
Validates training specs, enforces safety bounds, and manages bounded execution.
"""

from typing import Dict, Any, Optional

from agents.base import BaseAgent
from mlops.contracts import AgentResult
from mlops.distributed import DistributedCoordinator


class TrainingAgent(BaseAgent):
    """
    Sub-agent responsible for validating training configurations and coordinating bounded execution.
    """

    def __init__(self, coordinator: Optional[DistributedCoordinator] = None):
        super().__init__(name="TrainingAgent", description="Validates training configurations and safely executes training workflows")
        self.coordinator = coordinator or DistributedCoordinator()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        strategy = context.get("strategy", "fedavg")
        dry_run = context.get("dry_run", True)
        rounds = int(context.get("rounds", 3))
        local_epochs = int(context.get("local_epochs", 1))
        batch_size = int(context.get("batch_size", 64))
        clients = context.get("clients", ["site_0", "site_1", "site_2", "site_3", "site_4"])
        privacy_enabled = context.get("privacy_enabled", False) or (strategy == "dp_fedavg")

        if strategy not in ["centralized", "centralized_finetuned", "fedavg", "fedprox", "fedbn", "dp_fedavg"]:
            return AgentResult(
                agent_name=self.name,
                status="FAILED",
                decision="INVALID_STRATEGY",
                details={"error": f"Unknown training strategy: {strategy}"},
            )

        if rounds < 1 or batch_size < 1:
            return AgentResult(
                agent_name=self.name,
                status="FAILED",
                decision="INVALID_PARAMETERS",
                details={"error": "rounds and batch_size must be >= 1"},
            )

        task_spec = {
            "strategy": strategy,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "clients": clients,
            "privacy_enabled": privacy_enabled,
            "dry_run": dry_run,
        }

        if dry_run:
            dispatch_result = self.coordinator.dispatch_task(
                task_name=f"train_{strategy}",
                client_ids=clients,
                task_payload=task_spec,
                dry_run=True,
            )
            return AgentResult(
                agent_name=self.name,
                status="SUCCESS",
                decision="DRY_RUN_DISPATCHED",
                details={
                    "task_spec": task_spec,
                    "dispatch_summary": dispatch_result,
                    "message": "Safe dry-run training spec verified; no GPU compute consumed.",
                },
            )
        else:
            dispatch_result = self.coordinator.dispatch_task(
                task_name=f"train_{strategy}",
                client_ids=clients,
                task_payload=task_spec,
                dry_run=False,
            )
            return AgentResult(
                agent_name=self.name,
                status="SUCCESS",
                decision="TRAINING_DISPATCHED",
                details={
                    "task_spec": task_spec,
                    "dispatch_summary": dispatch_result,
                },
            )
