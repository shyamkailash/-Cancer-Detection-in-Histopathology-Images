"""
Tests for Federated Client Local Training.
"""

import torch
from torch.utils.data import TensorDataset
from ml.models.resnet import create_resnet18
from ml.federated.client import FederatedClient
from ml.federated.privacy import PrivacyConfig


def create_synthetic_client_dataset(n_samples: int = 16):
    """Create synthetic (3, 96, 96) image dataset for testing client."""
    x = torch.randn(n_samples, 3, 96, 96)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(x, y)


def test_federated_client_local_training():
    """Test that a client can load global weights, train locally, and return clean updates."""
    device = torch.device("cpu")
    dataset = create_synthetic_client_dataset(16)

    client = FederatedClient(
        client_id="site_test_1",
        site_name="Test Hospital A",
        dataset=dataset,
        device=device,
        batch_size=8,
        local_epochs=1,
        learning_rate=1e-3,
    )

    global_model = create_resnet18(num_classes=2, pretrained=False)
    initial_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

    res = client.train(initial_weights)

    assert res["client_id"] == "site_test_1"
    assert res["num_samples"] == 16
    assert "state_dict" in res
    assert res["train_loss"] >= 0.0
    assert 0.0 <= res["train_accuracy"] <= 1.0

    # Verify weights actually updated
    updated_fc_weight = res["state_dict"]["fc.weight"]
    assert not torch.allclose(updated_fc_weight, initial_weights["fc.weight"])


def test_federated_client_privacy_mode():
    """Test client local training with DP gradient perturbation enabled."""
    device = torch.device("cpu")
    dataset = create_synthetic_client_dataset(16)

    privacy_cfg = PrivacyConfig(enabled=True, max_grad_norm=1.0, noise_multiplier=0.1)

    client = FederatedClient(
        client_id="site_test_dp",
        site_name="Test Hospital DP",
        dataset=dataset,
        device=device,
        batch_size=8,
        local_epochs=1,
        privacy_config=privacy_cfg,
    )

    global_model = create_resnet18(num_classes=2, pretrained=False)
    res = client.train(global_model.state_dict())

    assert res["privacy_enabled"] is True
    assert res["num_samples"] == 16

