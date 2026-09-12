# Federated Learning Experiment Report: DP_FEDAVG

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images  
**Algorithm:** `DP_FEDAVG`  
**Training Subset:** `20.0%` (30,804 Training Samples, Deterministic Seed = `42`)  
**Global Held-Out Test Set:** `33,003` Samples | **Global Validation Set:** `33,004` Samples  
**Partition Strategy:** `noniid` (Dirichlet alpha = `0.5`)  
**Simulated Sites (Clients):** `5` (100% Client Participation)  
**Rounds:** `3` | **Local Epochs:** `1` | **Batch Size:** `64`  
**Hardware / Device:** `cuda` (NVIDIA GeForce RTX 3050 6GB Laptop GPU)  

---

## 1. Held-Out Global Test Performance (33,003 Samples)

| Metric | Result |
| :--- | :---: |
| **Accuracy** | **94.33%** |
| **Sensitivity (Metastasis Recall)** | **90.44%** |
| **Specificity (Normal Tissue)** | **96.98%** |
| **Precision** | **95.32%** |
| **F1-Score** | **92.82%** |
| **ROC-AUC** | **0.9831** |
| **Test Loss** | **0.1561** |

---

## 2. Training Dynamics & Round Progression

- **Selected Best Round:** Round `2` (Validation Accuracy: `94.44%`)
- **Total Training Duration:** `228.20s`

| Round | Duration (s) | Val Loss | Val Accuracy | Val Sensitivity | Val Specificity | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 67.0 | 0.1553 | 94.43% | 90.67% | 97.00% | 0.9832 |
| 2 | 67.7 | 0.1552 | 94.44% | 90.67% | 97.00% | 0.9832 |
| 3 | 67.2 | 0.1559 | 94.33% | 90.61% | 96.86% | 0.9831 |

---

## 3. Simulated Healthcare Site Distribution

| Site ID | Site Name | Samples | Normal (Class 0) | Metastasis (Class 1) | Metastasis Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `site_1` | Hospital A (Simulated Site) | 11,203 | 7,109 (63.46%) | 4,094 (36.54%) | 0.576 |
| `site_2` | Hospital B (Simulated Site) | 13,243 | 10,006 (75.56%) | 3,237 (24.44%) | 0.324 |
| `site_3` | Hospital C (Simulated Site) | 310 | 171 (55.16%) | 139 (44.84%) | 0.813 |
| `site_4` | Hospital D (Simulated Site) | 4,240 | 781 (18.42%) | 3,459 (81.58%) | 4.429 |
| `site_5` | Hospital E (Simulated Site) | 1,808 | 241 (13.33%) | 1,567 (86.67%) | 6.502 |

---

## 4. Privacy Configuration & Accounting

- **Differential Privacy Mechanism:** Gaussian DP on Local Gradients
- **Gradient Clipping Norm ($C$):** `1.0`
- **Gaussian Noise Multiplier (\sigma):** `1.0`
- **Target Delta (\delta):** `1e-05`
- **Formal Epsilon Accounting:** Formal epsilon accounting not implemented/verified (empirical gradient clipping and calibrated Gaussian perturbation applied)

---

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
