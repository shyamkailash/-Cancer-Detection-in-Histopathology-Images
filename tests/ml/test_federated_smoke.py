"""
End-to-End Federated Learning Smoke Tests (Server, Aggregation, and Checkpointing).
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
from ml.models.resnet import create_resnet18
from ml.federated.client import FederatedClient
from ml.federated.server import FederatedServer
from ml.federated.privacy import PrivacyConfig


def create_synthetic_data(n_samples: int = 24):
    x = torch.randn(n_samples, 3, 96, 96)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(x, y)


def test_federated_server_end_to_end_smoke(tmp_path):
    """Smoke test: 3 clients, 2 rounds of federated training, global validation and checkpointing."""
    device = torch.device("cpu")

    d1 = create_synthetic_data(16)
    d2 = create_synthetic_data(16)
    d3 = create_synthetic_data(16)

    c1 = FederatedClient("s1", "Hospital A", d1, device, batch_size=8)
    c2 = FederatedClient("s2", "Hospital B", d2, device, batch_size=8)
    c3 = FederatedClient("s3", "Hospital C", d3, device, batch_size=8)

    val_loader = DataLoader(create_synthetic_data(12), batch_size=6)
    test_loader = DataLoader(create_synthetic_data(12), batch_size=6)

    init_model = create_resnet18(num_classes=2, pretrained=False)
    ckpt_dir = tmp_path / "checkpoints"

    server = FederatedServer(
        clients=[c1, c2, c3],
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        initial_model=init_model,
        client_fraction=1.0,
        checkpoint_dir=str(ckpt_dir),
    )

    results = server.fit(rounds=2)

    assert results["rounds"] == 2
    assert len(results["history"]) == 2
    assert "test_metrics" in results
    assert results["test_metrics"]["total_samples"] == 12

    # Verify checkpoints created
    assert (ckpt_dir / "best_global_model.pt").exists()
    assert (ckpt_dir / "final_global_model.pt").exists()


def test_federated_server_with_dp_and_partial_participation(tmp_path):
    """Test federated server with DP enabled and 67% client participation."""
    device = torch.device("cpu")
    dp_cfg = PrivacyConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=0.05)

    clients = [
        FederatedClient(f"s{i}", f"Hospital {i}", create_synthetic_data(12), device, batch_size=6, privacy_config=dp_cfg)
        for i in range(3)
    ]

    val_loader = DataLoader(create_synthetic_data(8), batch_size=4)
    test_loader = DataLoader(create_synthetic_data(8), batch_size=4)

    server = FederatedServer(
        clients=clients,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        initial_model=create_resnet18(num_classes=2, pretrained=False),
        client_fraction=0.67,  # Select 2 out of 3 clients
        checkpoint_dir=str(tmp_path / "dp_checkpoints"),
    )

    results = server.fit(rounds=1)

    assert results["rounds"] == 1
    assert len(results["history"][0]["participating_clients"]) == 2

