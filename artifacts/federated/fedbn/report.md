# Federated Learning Experiment Report: FEDBN

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images  
**Algorithm:** `FEDBN`  
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
| **Accuracy** | **91.77%** |
| **Sensitivity (Metastasis Recall)** | **82.18%** |
| **Specificity (Normal Tissue)** | **98.30%** |
| **Precision** | **97.05%** |
| **F1-Score** | **89.00%** |
| **ROC-AUC** | **0.9796** |
| **Test Loss** | **0.2332** |

---

## 2. Training Dynamics & Round Progression

- **Selected Best Round:** Round `1` (Validation Accuracy: `91.83%`)
- **Total Training Duration:** `216.89s`

| Round | Duration (s) | Val Loss | Val Accuracy | Val Sensitivity | Val Specificity | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 61.4 | 0.2355 | 91.83% | 82.04% | 98.50% | 0.9790 |
| 2 | 64.2 | 0.4296 | 85.48% | 65.25% | 99.26% | 0.9737 |
| 3 | 64.4 | 0.7016 | 81.32% | 54.44% | 99.61% | 0.9730 |

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
