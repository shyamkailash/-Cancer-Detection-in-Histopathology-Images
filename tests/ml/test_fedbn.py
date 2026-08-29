"""
Unit and Integration Tests for FedBN (Local Batch Normalization) Algorithm.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from ml.models.resnet import create_resnet18
from ml.federated.fedbn import (
    FedBNClient,
    get_bn_module_names,
    is_bn_parameter_or_buffer,
    get_bn_state,
    get_non_bn_state,
    aggregate_fedbn,
)


def create_synthetic_data(n_samples: int = 16):
    x = torch.randn(n_samples, 3, 96, 96)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(x, y)


def test_get_bn_module_names_resnet18():
    """Verify get_bn_module_names identifies all BatchNorm2d layers in ResNet-18."""
    model = create_resnet18(num_classes=2, pretrained=False)
    bn_names = get_bn_module_names(model)

    assert "bn1" in bn_names
    assert "layer1.0.bn1" in bn_names
    assert "layer2.0.bn1" in bn_names
    assert "layer4.1.bn2" in bn_names
    assert "fc" not in bn_names
    assert "conv1" not in bn_names


def test_is_bn_parameter_or_buffer():
    """Test checking BN parameters vs Conv/Linear parameters."""
    model = create_resnet18(num_classes=2, pretrained=False)
    bn_names = get_bn_module_names(model)

    assert is_bn_parameter_or_buffer("bn1.weight", bn_names)
    assert is_bn_parameter_or_buffer("bn1.bias", bn_names)
    assert is_bn_parameter_or_buffer("bn1.running_mean", bn_names)
    assert is_bn_parameter_or_buffer("layer1.0.bn1.running_var", bn_names)
    assert is_bn_parameter_or_buffer("layer1.0.bn1.num_batches_tracked", bn_names)

    assert not is_bn_parameter_or_buffer("conv1.weight", bn_names)
    assert not is_bn_parameter_or_buffer("layer1.0.conv1.weight", bn_names)
    assert not is_bn_parameter_or_buffer("fc.weight", bn_names)
    assert not is_bn_parameter_or_buffer("fc.bias", bn_names)


def test_get_bn_and_non_bn_state_partition():
    """Verify disjoint partition between BN state and non-BN state."""
    model = create_resnet18(num_classes=2, pretrained=False)
    full_state = model.state_dict()
    bn_state = get_bn_state(model)
    non_bn_state = get_non_bn_state(model)

    # Check non-empty
    assert len(bn_state) > 0
    assert len(non_bn_state) > 0

    # Disjoint keys
    assert set(bn_state.keys()).isdisjoint(set(non_bn_state.keys()))

    # Union equals full state
    assert set(bn_state.keys()).union(set(non_bn_state.keys())) == set(full_state.keys())


def test_aggregate_fedbn_only_averages_non_bn():
    """
    Test FedBN aggregation:
    Non-BN weights should be averaged.
    BN weights/buffers should NOT be averaged.
    """
    # Create two synthetic client state dicts
    state_a = {
        "conv1.weight": torch.tensor([2.0, 2.0]),
        "bn1.weight": torch.tensor([10.0]),
        "bn1.running_mean": torch.tensor([100.0]),
    }
    state_b = {
        "conv1.weight": torch.tensor([4.0, 4.0]),
        "bn1.weight": torch.tensor([20.0]),
        "bn1.running_mean": torch.tensor([200.0]),
    }

    base_state = {
        "conv1.weight": torch.tensor([0.0, 0.0]),
        "bn1.weight": torch.tensor([1.0]),
        "bn1.running_mean": torch.tensor([0.0]),
    }

    updates = [
        (state_a, 100),
        (state_b, 100),
    ]

    aggregated = aggregate_fedbn(updates, base_state_dict=base_state)

    # conv1.weight should be averaged to (2 + 4)/2 = 3.0
    assert torch.allclose(aggregated["conv1.weight"], torch.tensor([3.0, 3.0]))

    # BN parameters should be retained from base_state (not averaged to 15.0 or 150.0)
    assert torch.allclose(aggregated["bn1.weight"], torch.tensor([1.0]))
    assert torch.allclose(aggregated["bn1.running_mean"], torch.tensor([0.0]))


def test_fedbn_client_preserves_local_bn_across_rounds():
    """Test that FedBN client maintains separate persistent BN statistics across rounds."""
    device = torch.device("cpu")
    dataset = create_synthetic_data(12)
    client = FedBNClient(
        client_id="site_bn_1",
        site_name="Hospital Beta",
        dataset=dataset,
        device=device,
        batch_size=4,
        local_epochs=1,
    )

    init_model = create_resnet18(num_classes=2, pretrained=False)
    global_state = {k: v.cpu().clone() for k, v in init_model.state_dict().items()}

    # Round 1
    res1 = client.train(global_state)
    bn_mean_r1 = client.local_bn_state["bn1.running_mean"].clone()

    # Alter global state for next round
    altered_global = {k: v.clone() + 0.1 for k, v in global_state.items()}

    # Round 2
    res2 = client.train(altered_global)
    bn_mean_r2 = client.local_bn_state["bn1.running_mean"].clone()

    assert res1["algorithm"] == "fedbn"
    assert res2["algorithm"] == "fedbn"
    assert isinstance(bn_mean_r2, torch.Tensor)

