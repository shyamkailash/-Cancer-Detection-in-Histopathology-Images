"""
Statistical Drift Monitoring Engine for Histopathology Prediction Distributions.
Implements robust Population Stability Index (PSI) and Total Variation Distance (TVD) calculation
with epsilon smoothing, minimum sample validation, and multi-level alert thresholds.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np


class DriftStatus(str, Enum):
    NO_DRIFT = "NO_DRIFT"
    WARNING = "WARNING"
    DRIFT = "DRIFT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index (PSI) between baseline and current distributions.
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    if len(baseline) == 2 and len(current) == 2 and np.isclose(np.sum(baseline), 1.0) and np.isclose(np.sum(current), 1.0):
        p = np.clip(baseline, epsilon, 1.0)
        q = np.clip(current, epsilon, 1.0)
        p = p / np.sum(p)
        q = q / np.sum(q)
        return float(np.sum((q - p) * np.log(q / p)))

    min_val = min(float(np.min(baseline)), float(np.min(current)))
    max_val = max(float(np.max(baseline)), float(np.max(current)))

    if np.isclose(min_val, max_val):
        return 0.0

    bins = np.linspace(min_val, max_val + 1e-6, num_bins + 1)
    b_counts, _ = np.histogram(baseline, bins=bins)
    c_counts, _ = np.histogram(current, bins=bins)

    b_pct = (b_counts + epsilon) / (np.sum(b_counts) + epsilon * num_bins)
    c_pct = (c_counts + epsilon) / (np.sum(c_counts) + epsilon * num_bins)

    psi_val = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
    return float(np.clip(psi_val, 0.0, None))


def calculate_tvd(
    baseline: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Compute Total Variation Distance (TVD) between baseline and current distributions.
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    if len(baseline) == 2 and len(current) == 2 and np.isclose(np.sum(baseline), 1.0) and np.isclose(np.sum(current), 1.0):
        return float(0.5 * np.sum(np.abs(current - baseline)))

    min_val = min(float(np.min(baseline)), float(np.min(current)))
    max_val = max(float(np.max(baseline)), float(np.max(current)))

    if np.isclose(min_val, max_val):
        return 0.0

    bins = np.linspace(min_val, max_val + 1e-6, num_bins + 1)
    b_counts, _ = np.histogram(baseline, bins=bins)
    c_counts, _ = np.histogram(current, bins=bins)

    p = b_counts / max(1, np.sum(b_counts))
    q = c_counts / max(1, np.sum(c_counts))
    return float(0.5 * np.sum(np.abs(q - p)))


class DriftMonitor:
    """
    Monitors prediction score distributions for statistical drift.
    """

    def __init__(
        self,
        default_method: str = "psi",
        warning_threshold: float = 0.10,
        critical_threshold: float = 0.25,
        min_samples: int = 50,
        baseline_distribution: Optional[Union[List[float], np.ndarray]] = None,
    ):
        self.default_method = default_method
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples
        self.baseline_distribution = (
            np.asarray(baseline_distribution, dtype=float).ravel()
            if baseline_distribution is not None
            else np.linspace(0.05, 0.95, 100)
        )
        self._sample_buffer: List[float] = []

    def add_sample(self, score: float) -> None:
        self._sample_buffer.append(float(score))

    def add_samples(self, scores: Union[List[float], np.ndarray]) -> None:
        self._sample_buffer.extend([float(s) for s in np.asarray(scores).ravel()])

    def clear_buffer(self) -> None:
        self._sample_buffer.clear()

    def check_drift(
        self,
        baseline_distribution: Optional[Union[List[float], np.ndarray]] = None,
        current_distribution: Optional[Union[List[float], np.ndarray]] = None,
        method: Optional[str] = None,
        warning_threshold: Optional[float] = None,
        critical_threshold: Optional[float] = None,
        min_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        if baseline_distribution is None:
            b_arr = self.baseline_distribution
        else:
            b_arr = np.asarray(baseline_distribution, dtype=float).ravel()

        if current_distribution is None:
            c_arr = np.asarray(self._sample_buffer, dtype=float).ravel()
        else:
            c_arr = np.asarray(current_distribution, dtype=float).ravel()

        min_s = min_samples if min_samples is not None else self.min_samples
        w_thresh = warning_threshold if warning_threshold is not None else self.warning_threshold
        c_thresh = critical_threshold if critical_threshold is not None else self.critical_threshold
        m = (method or self.default_method).lower().strip()

        if len(b_arr) < min_s or len(c_arr) < min_s:
            is_categorical = (len(b_arr) == 2 and len(c_arr) == 2 and np.isclose(np.sum(b_arr), 1.0) and np.isclose(np.sum(c_arr), 1.0))
            if not is_categorical:
                return {
                    "status": DriftStatus.INSUFFICIENT_DATA.value,
                    "drift_detected": False,
                    "statistic_value": 0.0,
                    "metric_name": m,
                    "reason": f"Sample count below minimum threshold ({min(len(b_arr), len(c_arr))} < {min_s})",
                    "baseline_samples": int(len(b_arr)),
                    "current_samples": int(len(c_arr)),
                }

        if m == "psi":
            stat_val = calculate_psi(b_arr, c_arr)
            label = "Population Stability Index"
        elif m in ("tvd", "total_variation"):
            stat_val = calculate_tvd(b_arr, c_arr)
            label = "Total Variation Distance"
        else:
            raise ValueError(f"Unknown drift metric '{m}'. Supported: 'psi', 'tvd'.")

        if stat_val >= c_thresh:
            status = DriftStatus.DRIFT
            drift_detected = True
            msg = f"Critical statistical drift detected: {label} ({stat_val:.4f}) >= critical threshold ({c_thresh:.4f})."
        elif stat_val >= w_thresh:
            status = DriftStatus.WARNING
            drift_detected = True
            msg = f"Moderate statistical drift warning: {label} ({stat_val:.4f}) >= warning threshold ({w_thresh:.4f})."
        else:
            status = DriftStatus.NO_DRIFT
            drift_detected = False
            msg = f"No significant drift detected: {label} ({stat_val:.4f}) < threshold ({w_thresh:.4f})."

        return {
            "status": status.value,
            "drift_detected": drift_detected,
            "statistic_value": float(stat_val),
            "metric_name": m,
            "warning_threshold": float(w_thresh),
            "critical_threshold": float(c_thresh),
            "baseline_summary": {
                "mean": float(np.mean(b_arr)),
                "std": float(np.std(b_arr)),
                "count": int(len(b_arr)),
            },
            "current_summary": {
                "mean": float(np.mean(c_arr)),
                "std": float(np.std(c_arr)),
                "count": int(len(c_arr)),
            },
            "explanation": msg,
            "disclaimer": "Prediction distribution drift detector. Academic statistical monitoring, not a clinical diagnostic tool.",
        }
