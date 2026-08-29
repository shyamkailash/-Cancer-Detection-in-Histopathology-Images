"""
Tests for Client-Side Differential Privacy and Gradient Perturbation.
"""

import torch
import torch.nn as nn
from ml.federated.privacy import DifferentialPrivacyEngine, PrivacyConfig


def test_privacy_engine_disabled():
    """Test that when privacy is disabled, gradients are unchanged."""
    model = nn.Linear(5, 2)
    x = torch.randn(4, 5)
    loss = model(x).sum()
    loss.backward()

    orig_grad = model.weight.grad.clone()

    config = PrivacyConfig(enabled=False)
    engine = DifferentialPrivacyEngine(config)
    diag = engine.process_gradients(model)

    assert torch.allclose(model.weight.grad, orig_grad)
    assert diag["noise_scale"] == 0.0


def test_privacy_engine_gradient_clipping_and_noise():
    """Test that gradient norm is bounded by max_grad_norm and perturbed by noise."""
    model = nn.Linear(10, 2)
    # Produce large gradients
    x = torch.randn(8, 10) * 100.0
    loss = model(x).sum()
    loss.backward()

    # Pre-clip norm
    total_norm = torch.sqrt(sum(p.grad.norm(2) ** 2 for p in model.parameters())).item()
    assert total_norm > 1.0

    max_norm = 1.0
    config = PrivacyConfig(enabled=True, max_grad_norm=max_norm, noise_multiplier=0.05)
    engine = DifferentialPrivacyEngine(config)

    diag = engine.process_gradients(model)

    assert diag["original_grad_norm"] > max_norm
    assert diag["noise_std"] == 0.05 * max_norm
    assert model.weight.grad is not None

