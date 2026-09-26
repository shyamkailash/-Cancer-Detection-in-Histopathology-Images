# Statistical Drift Monitoring Engine

## Overview
In high-stakes histopathology computer vision, changes in staining protocols, scanner calibrations, or tissue preparation at different clinical sites cause statistical distribution shifts in model prediction confidence and latent features.

The Phase 6 Drift Monitoring Engine implements rigorous mathematical drift quantification using **Population Stability Index (PSI)** and **Total Variation Distance (TVD)** with epsilon smoothing and minimum sample size gating.

---

## 1. Mathematical Formulations

### 1.1 Population Stability Index (PSI)
Given expected baseline distribution $P = (p_1, p_2, \dots, p_k)$ and actual operational distribution $Q = (q_1, q_2, \dots, q_k)$ across $k$ bins:

$$\text{PSI} = \sum_{i=1}^{k} (q_i - p_i) \cdot \ln\left(\frac{q_i}{p_i}\right)$$

To prevent division by zero or undefined logarithms on empirical zero-count bins, Laplace epsilon smoothing is applied:
$$\tilde{p}_i = \frac{\text{count}(P_i) + \epsilon}{\sum_{j} \text{count}(P_j) + k\epsilon}, \quad \tilde{q}_i = \frac{\text{count}(Q_i) + \epsilon}{\sum_{j} \text{count}(Q_j) + k\epsilon}$$

### 1.2 Total Variation Distance (TVD)
$$\text{TVD}(P, Q) = \frac{1}{2} \sum_{i=1}^{k} |p_i - q_i|$$
Bounded strictly in $[0, 1]$, representing the maximum difference between probabilities assigned to any event.

---

## 2. Thresholds and Operational Alert Levels

| Drift Metric | Range | Alert Status | Action Triggered |
|---|---|---|---|
| **PSI** | $\text{PSI} < 0.10$ | `NO_DRIFT` | Normal operation. Log telemetry. |
| **PSI** | $0.10 \le \text{PSI} < 0.25$ | `WARNING` | Moderate shift. Increase sampling rate. |
| **PSI** | $\text{PSI} \ge 0.25$ | `DRIFT` (Critical) | Signal Retraining Planner to formulate update plan. |
| **Samples** | $N < 50$ | `INSUFFICIENT_DATA` | Retain buffer; defer statistical inference. |

---

## 3. Integration & Usage
```python
from mlops.drift_monitor import DriftMonitor, calculate_psi

monitor = DriftMonitor(warning_threshold=0.10, critical_threshold=0.25, min_samples=50)

# Accumulate streaming prediction scores
monitor.add_samples([0.89, 0.94, 0.72, 0.81, ...])

# Evaluate drift
report = monitor.check_drift()
print(report["status"], report["statistic_value"])
```
