"""
Unit tests for Differential Privacy Accountant (RDP for Subsampled Gaussian).
"""

import pytest
import numpy as np
from mlops.privacy_accountant import PrivacyAccountant


def test_privacy_accountant_initialization():
    accountant = PrivacyAccountant()
    assert len(accountant.orders) > 0
    assert np.all(accountant.orders > 1.0)


def test_privacy_accountant_invalid_orders():
    with pytest.raises(ValueError):
        PrivacyAccountant(orders=[0.5, 1.0, 2.0])


def test_privacy_accountant_invalid_parameters():
    accountant = PrivacyAccountant()
    with pytest.raises(ValueError):
        accountant.compute_rdp_step(sample_rate=-0.1, noise_multiplier=1.0, alpha=2.0)
    with pytest.raises(ValueError):
        accountant.compute_rdp_step(sample_rate=0.1, noise_multiplier=-1.0, alpha=2.0)
    with pytest.raises(ValueError):
        accountant.compute_epsilon(steps=0, sample_rate=0.01, noise_multiplier=1.0)


def test_privacy_accountant_standard_gaussian_and_subsampling():
    accountant = PrivacyAccountant()
    rdp_full = accountant.compute_rdp_step(sample_rate=1.0, noise_multiplier=2.0, alpha=2.0)
    assert np.isclose(rdp_full, 0.25)

    rdp_sub = accountant.compute_rdp_step(sample_rate=0.1, noise_multiplier=2.0, alpha=2.0)
    assert rdp_sub < rdp_full


def test_privacy_accountant_epsilon_monotonicity_with_steps():
    accountant = PrivacyAccountant()
    q = 64.0 / 30804.0
    sigma = 1.0
    delta = 1e-5

    eps_10_steps = accountant.compute_epsilon(steps=10, sample_rate=q, noise_multiplier=sigma, delta=delta)
    eps_100_steps = accountant.compute_epsilon(steps=100, sample_rate=q, noise_multiplier=sigma, delta=delta)

    assert eps_10_steps > 0.0
    assert eps_100_steps > eps_10_steps


def test_privacy_accountant_get_privacy_spent():
    accountant = PrivacyAccountant()
    spent = accountant.get_privacy_spent(
        steps=15,
        sample_rate=64.0 / 30804.0,
        noise_multiplier=1.0,
        delta=1e-5,
    )
    assert spent["accountant"] == "Rényi Differential Privacy (RDP)"
    assert spent["epsilon"] is not None
    assert spent["epsilon"] > 0.0
    assert spent["delta"] == 1e-5
    assert spent["steps"] == 15
