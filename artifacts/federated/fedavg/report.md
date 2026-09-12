# Federated Learning Experiment Report: FEDAVG

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images  
**Algorithm:** `FEDAVG`  
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
| **Accuracy** | **95.09%** |
| **Sensitivity (Metastasis Recall)** | **92.03%** |
| **Specificity (Normal Tissue)** | **97.17%** |
| **Precision** | **95.68%** |
| **F1-Score** | **93.81%** |
| **ROC-AUC** | **0.9872** |
| **Test Loss** | **0.1367** |

---

## 2. Training Dynamics & Round Progression

- **Selected Best Round:** Round `3` (Validation Accuracy: `95.29%`)
- **Total Training Duration:** `224.48s`

| Round | Duration (s) | Val Loss | Val Accuracy | Val Sensitivity | Val Specificity | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 70.0 | 0.1506 | 94.77% | 90.82% | 97.46% | 0.9847 |
| 2 | 63.3 | 0.1371 | 95.18% | 93.70% | 96.19% | 0.9868 |
| 3 | 64.6 | 0.1364 | 95.29% | 92.32% | 97.31% | 0.9871 |

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

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
