"""
Unit tests for Statistical Drift Monitor (PSI and TVD).
"""

import pytest
import numpy as np
from mlops.drift_monitor import calculate_psi, calculate_tvd, DriftMonitor, DriftStatus


def test_psi_identical_distributions():
    d1 = np.random.normal(0.5, 0.1, 200)
    psi = calculate_psi(d1, d1)
    assert np.isclose(psi, 0.0, atol=1e-3)


def test_psi_divergent_distributions():
    d1 = np.random.normal(0.2, 0.05, 200)
    d2 = np.random.normal(0.8, 0.05, 200)
    psi = calculate_psi(d1, d2)
    assert psi > 0.5


def test_tvd_identical_and_divergent():
    d1 = np.array([0.5, 0.5])
    tvd_zero = calculate_tvd(d1, d1)
    assert np.isclose(tvd_zero, 0.0)

    d2 = np.array([0.9, 0.1])
    d3 = np.array([0.1, 0.9])
    tvd_divergent = calculate_tvd(d2, d3)
    assert np.isclose(tvd_divergent, 0.8)


def test_drift_monitor_threshold_levels():
    monitor = DriftMonitor(warning_threshold=0.10, critical_threshold=0.25, min_samples=50)

    b = np.linspace(0.1, 0.9, 100)
    res_no_drift = monitor.check_drift(baseline_distribution=b, current_distribution=b)
    assert res_no_drift["status"] == DriftStatus.NO_DRIFT.value
    assert res_no_drift["drift_detected"] is False

    res_insufficient = monitor.check_drift(baseline_distribution=[0.1, 0.2], current_distribution=[0.2, 0.3])
    assert res_insufficient["status"] == DriftStatus.INSUFFICIENT_DATA.value

    monitor.clear_buffer()
    monitor.add_samples(np.linspace(0.8, 0.99, 100))
    res_buffer = monitor.check_drift()
    assert res_buffer["drift_detected"] is True
