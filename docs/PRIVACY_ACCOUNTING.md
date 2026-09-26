# Formal Differential Privacy Accounting for DP-FedAvg

## Overview
In multi-site federated histopathology image analysis, preserving patient privacy against gradient inversion and membership inference attacks is essential.

Phase 6 implements rigorous mathematical privacy accounting using **Rényi Differential Privacy (RDP)** for the **Subsampled Gaussian Mechanism** (Mironov 2017, 2019; Wang et al. 2019).

---

## 1. Formal Mathematical Accounting

### 1.1 Parameters
- $N$: Total local training samples per site ($30,804$ in the semester federated subset)
- $B$: Local mini-batch size ($64$)
- $q = B / N$: Subsampling ratio ($q \approx 0.002078$)
- $C$: $L_2$ gradient clipping norm ($C = 1.0$)
- $\sigma$: Noise multiplier ($\sigma = 1.0$)
- $T$: Total composition steps ($T = \text{rounds} \times \text{local\_steps}$)
- $\delta$: Privacy failure target ($\delta = 10^{-5}$)

### 1.2 RDP Computation
For Rényi order $\alpha \in (1, \infty)$, the RDP guarantee per step of the subsampled Gaussian mechanism is upper bounded by:

$$\varepsilon_{\text{RDP}}(\alpha) \le \frac{q^2 \cdot \alpha}{2 \sigma^2} + O\left(\frac{q^3}{\sigma^3}\right)$$

Under $T$ composition steps, the cumulative RDP is:
$$\varepsilon_{\text{total}}(\alpha) = T \cdot \varepsilon_{\text{RDP}}(\alpha)$$

### 1.3 Canonical Conversion to $(\varepsilon, \delta)$-DP
The tightest canonical $(\varepsilon, \delta)$-Differential Privacy guarantee is computed by minimizing over orders $\alpha \in [1.25, 128.0]$:

$$\varepsilon(\delta) = \min_{\alpha > 1} \left[ \varepsilon_{\text{total}}(\alpha) + \frac{\ln(1 / \delta)}{\alpha - 1} \right] $$

---

## 2. Theoretical Guarantee vs. Empirical Disclaimer

> **Important Academic Disclaimer:**
> Differential privacy guarantees are calculated with exact theoretical accounting for the subsampled Gaussian gradient perturbation. The 5 hospital sites are simulated local splits from the PCam dataset. Real clinical hospital deployment requires hardware root of trust, secure aggregation enclaves, and institutional review board governance.
