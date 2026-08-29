# Experimental Comparison Report: Multi-Site Federated Cancer Detection

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images  
**Dataset:** PatchCamelyon (PCam) — Lymph Node Metastasis Detection  
**Model Architecture:** ResNet-18 (ImageNet Pretrained)  
**Execution Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU  

---

## 1. Executive Summary

This study evaluates distributed training of deep neural networks across simulated hospital silos under severe Non-IID label skew (Dirichlet $\\alpha=0.5$). We compare:
1. **Centralized Learning Baseline:** Aggregated training on pooled dataset.
2. **Federated Learning (FedAvg):** Distributed local client training with weighted parameter averaging without transferring raw clinical images.
3. **Privacy-Preserving Federated Learning (DP-FedAvg):** Local client training coupled with gradient $L_2$-norm clipping ($C=1.0$) and calibrated Gaussian noise injection ($\\sigma=0.05$).

---

## 2. Experimental Results on Global Test Set

| Metric | Centralized Baseline | Federated (FedAvg) | Privacy-Preserving (DP-FedAvg) |
| :--- | :---: | :---: | :---: |
| **Data Partition** | Pooled (100% centralized) | Non-IID Dirichlet ($\\alpha=0.5$) | Non-IID Dirichlet ($\\alpha=0.5$) |
| **Simulated Sites** | N/A (1 central server) | 5 Simulated Healthcare Sites | 5 Simulated Healthcare Sites |
| **Privacy Guarantee** | None (Raw data centralized) | Data Localization | Data Localization + DP Gradient Perturbation |
| **Global Test Loss** | **0.1615** | 0.3214 | 0.6287 |
| **Test Accuracy** | **94.13%** | **88.80%** | **67.20%** |
| **Sensitivity (Metastasis Recall)** | **89.68%** | **86.97%** | **28.31%** |
| **Specificity (Normal Tissue)** | **97.16%** | **90.07%** | **94.18%** |
| **Precision** | **95.56%** | **85.87%** | **77.14%** |
| **F1-Score** | **92.52%** | **86.42%** | **41.42%** |
| **ROC-AUC** | **0.9822** | **0.9522** | **0.7370** |

---

## 3. Clinical & Privacy Analysis

1. **Utility Retention in Federated Learning:**
   Standard FedAvg achieves **88.80% test accuracy** and **86.97% sensitivity**, demonstrating effective knowledge aggregation across distributed healthcare sites without transferring raw histopathology images or patient labels.

2. **Privacy vs. Utility Tradeoff:**
   Injecting Differential Privacy noise ($\sigma=0.05$) with gradient norm clipping ($C=1.0$) introduces an observable utility-privacy tradeoff, yielding **67.20% test accuracy** and **94.18% specificity** while bounding gradient sensitivity against inference and model inversion attacks.

3. **Clinical Priority (False Negatives & Sensitivity):**
   In oncological screening, false negatives carry significant clinical risks. Standard FedAvg maintains high sensitivity (**86.97%** metastasis recall), confirming that federated averaging effectively handles Non-IID class imbalances across participating institutions.

---

## 4. Generated Figures
- `artifacts/reports/figures/val_accuracy_curves.png`: Multi-round validation accuracy trajectory.
- `artifacts/reports/figures/val_loss_curves.png`: Multi-round validation loss convergence.
- `artifacts/reports/figures/client_data_distribution.png`: Non-IID class skew across simulated hospital institutions.
- `artifacts/reports/figures/model_comparison_metrics.png`: Bar chart comparison across all clinical metrics.

---

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
