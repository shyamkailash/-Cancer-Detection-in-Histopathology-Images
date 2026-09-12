# Federated Learning Benchmark Comparison

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images
**Dataset:** PatchCamelyon (PCam) — 33,003 Global Held-Out Test Samples
**Evaluation Date:** Complete Evaluation

---

## 1. Global Test Set Performance Comparison

| Method | Accuracy | Sensitivity | Specificity | Precision | F1 | ROC-AUC | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | 94.13% | 89.68% | 97.16% | 95.56% | 92.52% | 0.9822 | 0.1615 |
| **Centralized Fine-Tuned** | 95.35% | 94.18% | 96.14% | 94.32% | 94.25% | 0.9882 | 0.1301 |
| **FedAvg** | 95.09% | 92.03% | 97.17% | 95.68% | 93.81% | 0.9872 | 0.1367 |
| **FedProx** | 95.05% | 91.91% | 97.19% | 95.70% | 93.77% | 0.9870 | 0.1375 |
| **FedBN** | 91.77% | 82.18% | 98.30% | 97.05% | 89.00% | 0.9796 | 0.2332 |
| **DP-FedAvg** | 94.33% | 90.44% | 96.98% | 95.32% | 92.82% | 0.9831 | 0.1561 |

---

## 2. Performance Differences Relative to Centralized Baseline

| Method | Accuracy Δ vs Baseline | Sensitivity Δ vs Baseline | F1 Δ vs Baseline | ROC-AUC Δ vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | +0.00% | -0.00% | +0.00% | +0.0000 |
| **Centralized Fine-Tuned** | +1.22% | +4.50% | +1.73% | +0.0060 |
| **FedAvg** | +0.96% | +2.35% | +1.29% | +0.0050 |
| **FedProx** | +0.92% | +2.23% | +1.25% | +0.0048 |
| **FedBN** | -2.36% | -7.50% | -3.52% | -0.0026 |
| **DP-FedAvg** | +0.20% | +0.76% | +0.30% | +0.0009 |

---

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
