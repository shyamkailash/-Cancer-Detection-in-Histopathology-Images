"""
Tests for unified federated CLI argument parsing and reproducibility utilities.
"""

import pytest
from ml.training.federated_trainer import parse_args
from ml.federated.utils import set_seed
import random
import numpy as np
import torch


def test_parse_args_defaults(monkeypatch):
    """Test default CLI arguments."""
    monkeypatch.setattr("sys.argv", ["federated_trainer.py"])
    args = parse_args()

    assert args.algorithm == "fedavg"
    assert args.rounds == 5
    assert args.num_clients == 5
    assert args.mu == 0.01
    assert args.partition == "noniid"
    assert args.alpha == 0.5


def test_parse_args_fedprox_and_fedbn_options(monkeypatch):
    """Test CLI parsing for fedprox and fedbn."""
    monkeypatch.setattr(
        "sys.argv",
        ["federated_trainer.py", "--algorithm", "fedprox", "--mu", "0.05", "--rounds", "3"]
    )
    args_prox = parse_args()
    assert args_prox.algorithm == "fedprox"
    assert args_prox.mu == 0.05
    assert args_prox.rounds == 3

    monkeypatch.setattr(
        "sys.argv",
        ["federated_trainer.py", "--algorithm", "fedbn", "--rounds", "4"]
    )
    args_bn = parse_args()
    assert args_bn.algorithm == "fedbn"
    assert args_bn.rounds == 4


def test_set_seed_reproducibility():
    """Verify set_seed ensures reproducible random sampling across libraries."""
    set_seed(1234)
    r1 = random.random()
    np1 = np.random.rand(5)
    t1 = torch.rand(5)

    set_seed(1234)
    r2 = random.random()
    np2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert r1 == r2
    assert np.allclose(np1, np2)
    assert torch.allclose(t1, t2)

