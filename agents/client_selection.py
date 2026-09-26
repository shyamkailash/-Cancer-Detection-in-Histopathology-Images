"""
Client Selection Agent for Federated Learning Orchestration.
Selects and balances simulated clinical hospital sites based on health, data distribution, and strategy.
"""

from typing import Dict, Any, List, Optional
import random

from agents.base import BaseAgent
from mlops.contracts import AgentResult
from mlops.distributed import DistributedCoordinator, NodeStatus


class ClientSelectionAgent(BaseAgent):
    """
    Sub-agent responsible for selecting federated client sites for training or evaluation rounds.
    """

    def __init__(self, coordinator: Optional[DistributedCoordinator] = None):
        super().__init__(name="ClientSelectionAgent", description="Selects and validates simulated hospital sites for FL rounds")
        self.coordinator = coordinator or DistributedCoordinator()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        strategy = context.get("strategy", "fedavg")
        participation_rate = float(context.get("participation_rate", 1.0))
        min_clients = int(context.get("min_clients", 1))
        requested_clients = context.get("requested_clients")

        self.coordinator.ping_nodes()
        available_nodes = [
            node for node in self.coordinator.nodes.values()
            if node.status == NodeStatus.HEALTHY
        ]

        if not available_nodes:
            return AgentResult(
                agent_name=self.name,
                status="FAILED",
                decision="NO_HEALTHY_CLIENTS",
                details={"available_nodes": 0, "message": "No healthy hospital sites available."},
            )

        if requested_clients:
            selected_nodes = [
                node for node in available_nodes if node.node_id in requested_clients
            ]
            if len(selected_nodes) < len(requested_clients):
                missing = set(requested_clients) - {n.node_id for n in selected_nodes}
                return AgentResult(
                    agent_name=self.name,
                    status="WARNING",
                    decision="PARTIAL_SELECTION",
                    details={
                        "selected_client_ids": [n.node_id for n in selected_nodes],
                        "missing_client_ids": list(missing),
                        "total_samples": sum(n.sample_count for n in selected_nodes),
                    },
                )
        else:
            k = max(min_clients, int(len(available_nodes) * participation_rate))
            selected_nodes = available_nodes[:k]

        selected_ids = [n.node_id for n in selected_nodes]
        total_samples = sum(n.sample_count for n in selected_nodes)

        return AgentResult(
            agent_name=self.name,
            status="SUCCESS",
            decision="CLIENTS_SELECTED",
            details={
                "strategy": strategy,
                "selected_client_ids": selected_ids,
                "client_count": len(selected_ids),
                "total_samples": total_samples,
                "client_metadata": {n.node_id: n.to_dict() for n in selected_nodes},
            },
        )
