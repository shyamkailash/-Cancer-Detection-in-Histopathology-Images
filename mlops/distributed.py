"""
Distributed Multi-Node Coordination Abstraction for Federated MLOps.
Coordinates simulated clinical hospital sites, monitors node health, and manages task distribution.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional
import time


class NodeStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"
    BUSY = "BUSY"


@dataclass
class FederatedNode:
    node_id: str
    display_name: str
    status: NodeStatus = NodeStatus.HEALTHY
    endpoint: str = "local://internal"
    sample_count: int = 6160
    dataset_name: str = "PCam Partition"
    hardware: str = "GPU/CUDA"
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "display_name": self.display_name,
            "status": self.status.value if isinstance(self.status, NodeStatus) else str(self.status),
            "endpoint": self.endpoint,
            "sample_count": self.sample_count,
            "dataset_name": self.dataset_name,
            "hardware": self.hardware,
            "last_heartbeat": self.last_heartbeat,
            "metadata": self.metadata,
        }


class DistributedCoordinator:
    """
    Coordinates distributed simulated hospital nodes in a federated network.
    """

    def __init__(self, heartbeat_timeout: float = 300.0):
        self.heartbeat_timeout = heartbeat_timeout
        self.nodes: Dict[str, FederatedNode] = {}
        self._init_default_simulated_sites()

    def _init_default_simulated_sites(self) -> None:
        default_sites = [
            ("site_0", "Simulated Hospital Alpha (Pathology)", 6161),
            ("site_1", "Simulated Hospital Beta (Oncology)", 6161),
            ("site_2", "Simulated Hospital Gamma (Diagnostics)", 6161),
            ("site_3", "Simulated Hospital Delta (Histology)", 6161),
            ("site_4", "Simulated Hospital Epsilon (Research)", 6160),
        ]
        now = time.time()
        for node_id, name, samples in default_sites:
            self.nodes[node_id] = FederatedNode(
                node_id=node_id,
                display_name=name,
                status=NodeStatus.HEALTHY,
                endpoint=f"ipc://hospital_{node_id}.sock",
                sample_count=samples,
                last_heartbeat=now,
            )

    def register_node(self, node: FederatedNode) -> None:
        self.nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[FederatedNode]:
        return self.nodes.get(node_id)

    def list_nodes(self) -> List[Dict[str, Any]]:
        return [node.to_dict() for node in self.nodes.values()]

    def ping_nodes(self) -> Dict[str, str]:
        now = time.time()
        statuses = {}
        for node_id, node in self.nodes.items():
            if now - node.last_heartbeat > self.heartbeat_timeout:
                node.status = NodeStatus.OFFLINE
            else:
                node.status = NodeStatus.HEALTHY
            statuses[node_id] = node.status.value
        return statuses

    def get_available_clients(self) -> List[str]:
        self.ping_nodes()
        return [nid for nid, node in self.nodes.items() if node.status == NodeStatus.HEALTHY]

    def dispatch_task(
        self,
        task_name: str,
        client_ids: Optional[List[str]] = None,
        task_payload: Optional[Dict[str, Any]] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        target_clients = client_ids or self.get_available_clients()
        now = time.time()
        dispatches = {}

        for cid in target_clients:
            node = self.nodes.get(cid)
            if not node:
                dispatches[cid] = {"status": "FAILED", "error": f"Client {cid} not found"}
                continue

            if dry_run:
                dispatches[cid] = {
                    "status": "SUCCESS",
                    "mode": "DRY_RUN",
                    "task": task_name,
                    "sample_count": node.sample_count,
                    "timestamp": now,
                    "message": f"Simulated dry-run dispatch to {node.display_name}",
                }
            else:
                dispatches[cid] = {
                    "status": "DISPATCHED",
                    "mode": "ACTIVE",
                    "task": task_name,
                    "sample_count": node.sample_count,
                    "timestamp": now,
                }

        return {
            "task_name": task_name,
            "dry_run": dry_run,
            "targeted_nodes": len(target_clients),
            "dispatches": dispatches,
            "timestamp": now,
        }
