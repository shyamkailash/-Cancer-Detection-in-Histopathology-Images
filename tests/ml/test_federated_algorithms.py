"""
Integration tests for multi-algorithm federated training (FedProx, FedBN, FedAvg, DP-FedAvg).
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from ml.models.resnet import create_resnet18
from ml.federated.client import FederatedClient
from ml.federated.fedprox import FedProxClient
from ml.federated.fedbn import FedBNClient
from ml.federated.server import FederatedServer
from ml.federated.privacy import PrivacyConfig
from ml.federated.utils import set_seed, estimate_model_size_mb, estimate_communication_volume


def create_synthetic_data(n_samples: int = 16):
    x = torch.randn(n_samples, 3, 96, 96)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(x, y)


def test_fedprox_server_integration_smoke(tmp_path):
    """Test 2-round federated training with FedProx clients."""
    device = torch.device("cpu")
    clients = [
        FedProxClient(f"prox_s{i}", f"Hospital Prox {i}", create_synthetic_data(8), device, batch_size=4, mu=0.01)
        for i in range(2)
    ]

    val_loader = DataLoader(create_synthetic_data(6), batch_size=3)
    test_loader = DataLoader(create_synthetic_data(6), batch_size=3)

    server = FederatedServer(
        clients=clients,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        algorithm="fedprox",
        checkpoint_dir=str(tmp_path / "fedprox_ckpts"),
    )

    res = server.fit(rounds=2)
    assert res["algorithm"] == "fedprox"
    assert res["rounds"] == 2
    assert len(res["history"]) == 2
    assert (tmp_path / "fedprox_ckpts" / "best_global_model.pt").exists()


def test_fedbn_server_integration_smoke(tmp_path):
    """Test 2-round federated training with FedBN clients."""
    device = torch.device("cpu")
    clients = [
        FedBNClient(f"bn_s{i}", f"Hospital BN {i}", create_synthetic_data(8), device, batch_size=4)
        for i in range(2)
    ]

    val_loader = DataLoader(create_synthetic_data(6), batch_size=3)
    test_loader = DataLoader(create_synthetic_data(6), batch_size=3)

    server = FederatedServer(
        clients=clients,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        algorithm="fedbn",
        checkpoint_dir=str(tmp_path / "fedbn_ckpts"),
    )

    res = server.fit(rounds=2)
    assert res["algorithm"] == "fedbn"
    assert res["rounds"] == 2
    assert len(res["history"]) == 2
    assert (tmp_path / "fedbn_ckpts" / "best_global_model.pt").exists()


def test_communication_metrics_estimation():
    """Test model size and communication volume calculations."""
    model = create_resnet18(num_classes=2, pretrained=False)
    size_mb = estimate_model_size_mb(model)

    assert 40.0 <= size_mb <= 50.0  # ResNet-18 is ~42.7MB in float32

    comm = estimate_communication_volume(num_rounds=5, num_participating_clients_per_round=5, model_size_mb=size_mb)
    assert comm["rounds"] == 5
    assert comm["clients_per_round"] == 5
    assert comm["total_communication_mb"] > 0
    assert comm["total_communication_gb"] > 0

