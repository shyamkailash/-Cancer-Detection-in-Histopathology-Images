# Experimental Comparison Report: Multi-Site Federated Cancer Detection

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images  
**Dataset:** PatchCamelyon (PCam) — Lymph Node Metastasis Detection  
**Model Architecture:** ResNet-18 (ImageNet Pretrained)  
**Evaluated Paradigms:** Centralized, FedAvg, FedProx, FedBN, and DP-FedAvg

---

## 1. Executive Summary

This study evaluates distributed training of deep neural networks across simulated hospital silos under severe Non-IID label skew (Dirichlet $\alpha=0.5$). We benchmark:
1. **Centralized Learning Baseline:** Aggregated training on pooled dataset.
2. **Federated Learning (FedAvg):** Distributed client training with sample-weighted parameter averaging.
3. **Proximal Federated Learning (FedProx):** Local training regularized with proximal constraint ($\mu=0.01$) to bound client drift.
4. **Local Batch Normalization Federated Learning (FedBN):** Decentralized BatchNorm layers addressing site-specific stain and domain heterogeneity.
5. **Privacy-Preserving Federated Learning (DP-FedAvg):** Local client training with gradient $L_2$-norm clipping ($C=1.0$) and calibrated Gaussian noise injection ($\sigma=0.05$).

---

## 2. Experimental Results on Global Test Set (33,003 Samples)

| Model | Paradigm | Accuracy | Sensitivity (Metastasis Recall) | Specificity (Normal Tissue) | Precision | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Centralized Pooled Training | 94.13% | 89.68% | 97.16% | 95.56% | 92.52% | 0.9822 |
| **Federated (FedAvg)** | Federated Learning (FedAvg) | 88.80% | 86.97% | 90.07% | 85.87% | 86.42% | 0.9522 |
| **Federated (FedProx)** | Federated Learning (FedProx, mu=0.01) | 84.07% | 72.26% | 92.44% | 87.15% | 79.01% | 0.9138 |
| **Federated (FedBN)** | Federated Learning (Local BatchNorm) | 59.96% | 5.40% | 98.65% | 74.00% | 10.07% | 0.6636 |
| **Privacy-Preserving (DP-FedAvg)** | Federated Learning + DP (C=1.0, sigma=0.05) | 67.20% | 28.31% | 94.18% | 77.14% | 41.42% | 0.7370 |

---

## 3. Clinical & Privacy Analysis

1. **Utility Retention in Federated Learning:**
   Standard FedAvg demonstrates strong distributed convergence, maintaining high sensitivity for metastatic lesion detection without transferring raw patient histopathology images.

2. **Mitigating Non-IID Client Drift (FedProx & FedBN):**
   - **FedProx** restricts local divergence from the global objective using a proximal penalty ($\frac{\mu}{2} \|w - w^t\|^2$).
   - **FedBN** keeps batch normalization layers strictly local, adapting to domain shifts and staining variance across medical imaging equipment.

3. **Privacy vs. Utility Tradeoff in DP-FedAvg:**
   Perturbing gradient updates with calibrated Gaussian noise bounds individual patient contribution sensitivity, presenting an empirical tradeoff between differential privacy defense and clinical classification accuracy.

---

## 4. Generated Figures
- `artifacts/reports/figures/val_accuracy_curves.png`: Multi-round validation accuracy trajectory.
- `artifacts/reports/figures/val_loss_curves.png`: Multi-round validation loss convergence.
- `artifacts/reports/figures/client_data_distribution.png`: Non-IID class skew across simulated hospital institutions.
- `artifacts/reports/figures/model_comparison_metrics.png`: Bar chart comparison across clinical metrics.

---

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
