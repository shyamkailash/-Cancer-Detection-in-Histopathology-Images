"""
Unit tests for Distributed Multi-Node Coordinator.
"""

import pytest
from mlops.distributed import DistributedCoordinator, FederatedNode, NodeStatus


def test_distributed_coordinator_initialization():
    coord = DistributedCoordinator()
    assert len(coord.nodes) == 5
    healthy = coord.get_available_clients()
    assert len(healthy) == 5


def test_distributed_coordinator_dispatch_dry_run():
    coord = DistributedCoordinator()
    dispatch_res = coord.dispatch_task(
        task_name="federated_round_1",
        client_ids=["site_0", "site_1"],
        dry_run=True,
    )
    assert dispatch_res["targeted_nodes"] == 2
    assert dispatch_res["dry_run"] is True
    assert dispatch_res["dispatches"]["site_0"]["status"] == "SUCCESS"
    assert dispatch_res["dispatches"]["site_0"]["mode"] == "DRY_RUN"
