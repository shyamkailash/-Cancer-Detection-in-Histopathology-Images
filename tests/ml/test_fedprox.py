"""
Unit and Integration Tests for FedProx (Proximal Regularization) Algorithm.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ml.models.resnet import create_resnet18
from ml.federated.fedprox import FedProxClient, compute_proximal_loss


def create_synthetic_data(n_samples: int = 16):
    x = torch.randn(n_samples, 3, 96, 96)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(x, y)


def test_compute_proximal_loss_zero_when_identical():
    """Test proximal loss is exactly zero when local parameters match global parameters."""
    model = create_resnet18(num_classes=2, pretrained=False)
    global_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    loss = compute_proximal_loss(model, global_params, mu=0.01)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_compute_proximal_loss_positive_when_diverged():
    """Test proximal loss is positive when parameters differ and scales with mu."""
    model = create_resnet18(num_classes=2, pretrained=False)
    global_params = {
        name: param.detach().clone() + 1.0  # Diverged by +1.0
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    loss_mu_01 = compute_proximal_loss(model, global_params, mu=0.01)
    loss_mu_02 = compute_proximal_loss(model, global_params, mu=0.02)

    assert loss_mu_01.item() > 0.0
    assert torch.isclose(loss_mu_02, loss_mu_01 * 2.0, rtol=1e-4)


def test_compute_proximal_loss_zero_when_mu_is_zero():
    """Test proximal loss is zero when mu is 0."""
    model = create_resnet18(num_classes=2, pretrained=False)
    global_params = {
        name: param.detach().clone() + 5.0
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    loss = compute_proximal_loss(model, global_params, mu=0.0)
    assert loss.item() == 0.0


def test_fedprox_client_local_training():
    """Test FedProx client local training and proximal metric tracking."""
    device = torch.device("cpu")
    dataset = create_synthetic_data(12)
    client = FedProxClient(
        client_id="test_site_1",
        site_name="Hospital Alpha",
        dataset=dataset,
        device=device,
        batch_size=4,
        local_epochs=1,
        mu=0.05,
    )

    init_model = create_resnet18(num_classes=2, pretrained=False)
    global_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}

    res = client.train(global_state)

    assert res["client_id"] == "test_site_1"
    assert res["algorithm"] == "fedprox"
    assert res["mu"] == 0.05
    assert "train_loss" in res
    assert "train_cls_loss" in res
    assert "train_prox_loss" in res
    assert res["num_samples"] == 12
    assert "state_dict" in res

