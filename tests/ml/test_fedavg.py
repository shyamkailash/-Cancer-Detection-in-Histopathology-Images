"""
Tests for Federated Averaging (FedAvg) aggregation.
"""

import pytest
import torch
from ml.models.resnet import create_resnet18
from ml.federated.aggregation import federated_averaging


def test_fedavg_exact_mathematical_weights():
    """
    Test exact weighted FedAvg aggregation formula:
    Client A: weights = 1.0, samples = 100
    Client B: weights = 3.0, samples = 300
    Expected aggregated weight = (100/400)*1.0 + (300/400)*3.0 = 2.5
    """
    state_a = {
        "weight": torch.tensor([1.0, 1.0]),
        "bias": torch.tensor([10.0]),
        "num_batches_tracked": torch.tensor(5, dtype=torch.int64),
    }
    state_b = {
        "weight": torch.tensor([3.0, 3.0]),
        "bias": torch.tensor([30.0]),
        "num_batches_tracked": torch.tensor(15, dtype=torch.int64),
    }

    client_updates = [
        (state_a, 100),
        (state_b, 300),
    ]

    aggregated = federated_averaging(client_updates)

    expected_weight = torch.tensor([2.5, 2.5])
    expected_bias = torch.tensor([25.0])

    assert torch.allclose(aggregated["weight"], expected_weight)
    assert torch.allclose(aggregated["bias"], expected_bias)
    assert aggregated["num_batches_tracked"] == 5  # Non-floating tracking tensor copied cleanly


def test_fedavg_empty_updates_raises_error():
    """Test that FedAvg on empty list raises ValueError."""
    with pytest.raises(ValueError):
        federated_averaging([])


def test_fedavg_resnet18_models():
    """Test aggregating full ResNet-18 state dicts across 3 clients."""
    model_a = create_resnet18(num_classes=2, pretrained=False)
    model_b = create_resnet18(num_classes=2, pretrained=False)
    model_c = create_resnet18(num_classes=2, pretrained=False)

    updates = [
        (model_a.state_dict(), 500),
        (model_b.state_dict(), 300),
        (model_c.state_dict(), 200),
    ]

    agg_state = federated_averaging(updates)

    assert isinstance(agg_state, dict)
    assert "fc.weight" in agg_state
    assert "conv1.weight" in agg_state
    assert agg_state["fc.weight"].shape == (2, 512)

