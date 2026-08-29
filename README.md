# Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images

An end-to-end machine learning platform and distributed federated learning system for detecting metastatic cancer in histopathology image patches across multi-site healthcare institutions.

> **Clinical Disclaimer:** This software is an academic research and educational prototype. It is **not** certified, validated, or approved for clinical diagnosis, patient triage, or healthcare decision-making.

---

## 1. Project Overview & Motivation

In multi-institutional healthcare environments, clinical data governance policies (such as HIPAA and GDPR) prohibit centralizing patient histopathology slides into a single database. This creates data silos and severe label heterogeneity across hospital sites.

This project implements a complete **Privacy-Preserving Federated Learning (FL)** framework that enables distributed training of deep neural networks (ResNet-18) across simulated hospital nodes without centralizing raw clinical imagery.

---

## 2. System Architecture

```
                       PCam Dataset (220,025 samples)
                                      │
                                      ▼
                        Leakage-Safe Manifest (CSV / JSONL)
                                      │
                                      ▼
                   Simulated Multi-Site Healthcare Partitioning
                          (Dirichlet Non-IID: alpha=0.5)
                                      │
        ┌──────────────┬──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
     Site 1         Site 2         Site 3         Site 4         Site 5
  (Hospital A)   (Hospital B)   (Hospital C)   (Hospital D)   (Hospital E)
  100% Normal    88.4% Tumor    91.2% Normal   72.8% Normal   95.5% Tumor
        │              │              │              │              │
  Local ResNet   Local ResNet   Local ResNet   Local ResNet   Local ResNet
  (DP Clipping   (DP Clipping   (DP Clipping   (DP Clipping   (DP Clipping
   + Gaussian)    + Gaussian)    + Gaussian)    + Gaussian)    + Gaussian)
        │              │              │              │              │
        └──────────────┴──────────────┼──────────────┴──────────────┘
                                      │  (Only Model Weights Transmitted)
                                      ▼
                         Federated Aggregator (FedAvg)
                                      │
                                      ▼
                            Global ResNet-18 Model
                                      │
                                      ▼
                         Global Validation & Test Set
                       (Metastasis Sensitivity, ROC-AUC)
```

---

## 3. Mathematical Formulations

### A. Federated Averaging (FedAvg)
At each round $t$, the global server aggregates local client weight vectors $w_k^{(t)}$ weighted proportionally to the number of local training samples $n_k$:

$$w^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{N} w_k^{(t)}$$

where $N = \sum_{k=1}^K n_k$.

### B. Client-Side Differential Privacy (DP-SGD Prototype)
To protect model weights against gradient inversion and membership inference attacks, each client applies $L_2$-norm gradient clipping and Gaussian noise perturbation before optimizer updates:

1. **Gradient Clipping:**
   $$\tilde{g}_i = g_i \cdot \min\left(1.0, \frac{C}{\|g_i\|_2 + 10^{-6}}\right)$$
2. **Noise Perturbation:**
   $$\hat{g}_i = \tilde{g}_i + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$

where $C$ is `--max-grad-norm` (default: $1.0$) and $\sigma$ is `--noise-multiplier` (default: $0.05$).

---

## 4. Experimental Results on PCam Test Set (33,003 Samples)

| Metric | Centralized Baseline | Federated (FedAvg) | Privacy-Preserving (DP-FedAvg) |
| :--- | :---: | :---: | :---: |
| **Data Partition** | Pooled (Centralized) | Non-IID Dirichlet ($\alpha=0.5$) | Non-IID Dirichlet ($\alpha=0.5$) |
| **Simulated Sites** | 1 Central Server | 5 Simulated Sites | 5 Simulated Sites |
| **Privacy Guarantee** | None (Data Pooled) | Data Localization | Data Localization + DP Gradient Noise |
| **Global Test Loss** | **0.1615** | 0.3214 | 0.6287 |
| **Test Accuracy** | **94.13%** | **88.80%** | **67.20%** |
| **Sensitivity (Metastasis Recall)** | **89.68%** | **86.97%** | **28.31%** |
| **Specificity (Normal Tissue)** | **97.16%** | **90.07%** | **94.18%** |
| **Precision** | **95.56%** | **85.87%** | **77.14%** |
| **F1-Score** | **92.52%** | **86.42%** | **41.42%** |
| **ROC-AUC** | **0.9822** | **0.9522** | **0.7370** |

---

## 5. Privacy Threat Model & Analysis

1. **Local Data Isolation:** Raw histopathology image files and clinical labels are strictly bound to local nodes and are **never** transmitted over the network.
2. **Threats to Standard FL:** Standard FedAvg transmits model parameter updates, which may be vulnerable to deep leakage / gradient inversion attacks or membership inference under malicious aggregation.
3. **Differential Privacy Defense:** Adding calibrated Gaussian noise directly to clipped gradients bounds model sensitivity and limits reconstruction feasibility.
4. **Utility-Privacy Tradeoff:** As observed empirically, stronger DP noise ($\sigma$) reduces sensitivity to rare cancer features. Optimal clinical deployments should calibrate $\sigma$ against validation sensitivity thresholds.
5. **Future Hardening:** Future production versions should integrate **Secure Multiparty Computation (SMC)** or **Homomorphic Encryption (HE)** for cryptographically secure aggregation.

---

## 6. Execution & Reproducibility Commands

### A. Centralized Baseline Training & Evaluation
```bash
# Evaluate existing baseline checkpoint
python scripts/train_centralized.py --eval-only

# Or train centralized baseline from scratch
python scripts/train_centralized.py --epochs 1 --batch-size 32
```

### B. Generate Simulated Multi-Site Partitions (Dirichlet Non-IID)
```bash
python scripts/create_federated_partitions.py \
    --num-clients 5 \
    --partition noniid \
    --alpha 0.5 \
    --seed 42
```

### C. Run Baseline Federated Learning (FedAvg)
```bash
python -m ml.training.federated_trainer \
    --rounds 5 \
    --num-clients 5 \
    --local-epochs 1 \
    --batch-size 32 \
    --partition noniid \
    --alpha 0.5 \
    --privacy none \
    --seed 42
```

### D. Run Privacy-Preserving Federated Learning (DP-FedAvg)
```bash
python -m ml.training.federated_trainer \
    --rounds 5 \
    --num-clients 5 \
    --local-epochs 1 \
    --batch-size 32 \
    --partition noniid \
    --alpha 0.5 \
    --privacy dp \
    --max-grad-norm 1.0 \
    --noise-multiplier 0.05 \
    --seed 42
```

### E. Generate Plots & Reports
```bash
python scripts/generate_plots.py
python scripts/generate_comparison_report.py
```

### F. Run Complete Unit & Integration Test Suite (61 Tests)
```bash
pytest -v
```

---

## 7. Storage & Artifact Structure

```
artifacts/
├── checkpoints/
│   └── pcam_resnet18_best.pt        # Centralized baseline model
├── federated/
│   ├── best_global_model.pt         # Best global federated checkpoint
│   └── final_global_model.pt        # Final global federated checkpoint
└── reports/
    ├── centralized_metrics.json     # Centralized baseline test metrics
    ├── federated_metrics.json       # FedAvg multi-round metrics
    ├── privacy_federated_metrics.json # DP FedAvg metrics
    ├── comparison_report.json       # Comparative summary JSON
    ├── comparison_report.md         # Comparative summary Markdown
    └── figures/
        ├── val_accuracy_curves.png  # Accuracy progression across rounds
        ├── val_loss_curves.png      # Loss convergence across rounds
        ├── client_data_distribution.png # Non-IID class skew across sites
        └── model_comparison_metrics.png # Comparative bar chart
```
