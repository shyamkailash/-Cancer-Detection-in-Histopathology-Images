# Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images

An end-to-end machine learning platform, distributed federated learning framework, and REST API for detecting metastatic cancer in histopathology image patches across multi-site healthcare institutions.

> **Clinical Disclaimer:** This software is an academic research and educational prototype. It is **not** certified, validated, or approved for clinical diagnosis, patient triage, or healthcare decision-making.

---

## 1. Project Overview & Motivation

In multi-institutional healthcare environments, clinical data governance regulations (HIPAA, GDPR) prohibit centralizing patient histopathology slides into a single database. This creates data silos and severe label heterogeneity across hospital sites.

This project implements a complete **Privacy-Preserving Federated Learning (FL)** framework that enables distributed collaborative training of deep neural networks (ResNet-18) across simulated hospital nodes without transmitting raw patient histopathology images or clinical labels.

### Supported Federated Learning Paradigms:
1. **Centralized Baseline:** Aggregated training on pooled dataset (ImageNet Pretrained ResNet-18).
2. **Federated Averaging (FedAvg):** Standard distributed parameter averaging.
3. **Proximal Federated Learning (FedProx):** Constrains local client drift using proximal penalty ($\mu=0.01$).
4. **Local Batch Normalization Federated Learning (FedBN):** Keeps BatchNorm parameters and running statistics local to each healthcare site to adapt to multi-site staining and scanner variations.
5. **Privacy-Preserving Federated Learning (DP-FedAvg):** Local client training with $L_2$-norm gradient clipping and calibrated Gaussian noise injection.

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
  (FedProx/BN/   (FedProx/BN/   (FedProx/BN/   (FedProx/BN/   (FedProx/BN/
   DP-SGD)        DP-SGD)        DP-SGD)        DP-SGD)        DP-SGD)
        │              │              │              │              │
        └──────────────┴──────────────┼──────────────┴──────────────┘
                                      │  (Only Model Parameters Transmitted)
                                      ▼
                         Federated Aggregator (Server)
                          [FedAvg / FedProx / FedBN]
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

### B. Proximal Federated Regularization (FedProx)
To bound local divergence caused by Non-IID client skew, each client optimizes the proximal objective:

$$\min_w h_k(w; w^t) = \mathcal{L}_k(w) + \frac{\mu}{2} \|w - w^t\|_2^2$$

where $\mu \ge 0$ is the proximal regularization parameter (default: $\mu=0.01$).

### C. Local Batch Normalization (FedBN)
BatchNorm layers ($\gamma, \beta, \mu_{\text{running}}, \sigma^2_{\text{running}}$) remain strictly local to each participating healthcare node. During round aggregation, only convolutional and linear weights are synchronized:

$$w_{\text{conv, linear}}^{(t+1)} = \sum_{k=1}^K \frac{n_k}{N} w_{k, \text{conv, linear}}^{(t)}$$

### D. Client-Side Differential Privacy (DP-SGD Prototype)
To protect parameter updates against gradient inversion and membership inference attacks:

1. **Gradient Clipping:**
   $$\tilde{g}_i = g_i \cdot \min\left(1.0, \frac{C}{\|g_i\|_2 + 10^{-6}}\right)$$
2. **Calibrated Gaussian Perturbation:**
   $$\hat{g}_i = \tilde{g}_i + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$

where $C$ is `--max-grad-norm` ($1.0$) and $\sigma$ is `--noise-multiplier` ($1.0$) for the reported DP-FedAvg experiment.

**Privacy accounting limitation:** The implementation applies gradient
clipping and Gaussian noise, but formal epsilon accounting has not been
implemented or verified. Therefore, a formal differential-privacy
guarantee is not claimed.


---

## 4. Experimental Results on PCam Test Set (33,003 Samples)

| Model | Paradigm | Accuracy | Sensitivity (Metastasis Recall) | Specificity (Normal Tissue) | Precision | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Centralized Pooled Training | **94.13%** | **89.68%** | **97.16%** | **95.56%** | **92.52%** | **0.9822** |
| **Centralized Fine-Tuned** | 2-Stage Transfer Learning | **95.35%** | **94.18%** | **96.14%** | **94.32%** | **94.25%** | **0.9882** |
| **Federated (FedAvg)** | Federated Learning (FedAvg) | **95.09%** | **92.03%** | **97.17%** | **95.68%** | **93.81%** | **0.9872** |
| **Federated (FedProx)** | Federated Learning (FedProx, $\mu=0.01$) | **95.05%** | **91.91%** | **97.19%** | **95.70%** | **93.77%** | **0.9870** |
| **Federated (FedBN)** | Federated Learning (Local BatchNorm) | **91.77%** | **82.18%** | **98.30%** | **97.05%** | **89.00%** | **0.9796** |
| **Privacy-Preserving (DP-FedAvg)** | Federated Learning + DP ($C=1.0, \sigma=1.0$) | **94.33%** | **90.44%** | **96.98%** | **95.32%** | **92.82%** | **0.9831** |

---

## 5. Execution & Reproducibility Commands

### A. Centralized Baseline Evaluation
```bash
python scripts/train_centralized.py --eval-only
```

### B. Run Baseline Federated Learning (FedAvg)
```bash
python scripts/run_federated.py \
    --algorithm fedavg \
    --rounds 5 \
    --clients 5 \
    --local-epochs 1 \
    --batch-size 32 \
    --seed 42
```

### C. Run Proximal Federated Learning (FedProx)
```bash
python scripts/run_federated.py \
    --algorithm fedprox \
    --rounds 5 \
    --clients 5 \
    --local-epochs 1 \
    --batch-size 32 \
    --mu 0.01 \
    --seed 42
```

### D. Run Local Batch Normalization Federated Learning (FedBN)
```bash
python scripts/run_federated.py \
    --algorithm fedbn \
    --rounds 5 \
    --clients 5 \
    --local-epochs 1 \
    --batch-size 32 \
    --seed 42
```

### E. Run Privacy-Preserving Federated Learning (DP-FedAvg)
```bash
python scripts/run_federated.py \
    --algorithm dp_fedavg \
    --rounds 5 \
    --clients 5 \
    --local-epochs 1 \
    --batch-size 32 \
    --dp-clip 1.0 \
    --dp-noise 0.05 \
    --seed 42
```

### F. Run Prediction & Grad-CAM Explainability (CLI)
```bash
python scripts/predict_image.py \
    --image data/raw/pcam/train/00001b2b5609af42ab0ab276dd4cd41c3e7745b5.tif \
    --model fedprox \
    --output-dir artifacts/predictions/
```

### G. Start Django REST API & Interactive Demonstration UI
```bash
python backend/manage.py runserver 0.0.0.0:8000
```
Open **`http://localhost:8000/api/demo/`** in your browser.

### H. Run Full Unit & API Test Suite (87 Tests)
```bash
pytest -v
```

---

## 6. Storage & Artifact Structure

```
artifacts/
├── checkpoints/
│   └── pcam_resnet18_best.pt        # Centralized baseline model (94.13% acc)
├── experiments/
│   ├── fedavg/                      # FedAvg checkpoints and metrics
│   ├── fedprox/                     # FedProx checkpoints and metrics
│   ├── fedbn/                       # FedBN checkpoints and metrics
│   └── dp_fedavg/                   # DP-FedAvg checkpoints and metrics
├── federated/
│   ├── best_global_model.pt         # Global federated checkpoint
│   └── final_global_model.pt        # Final federated checkpoint
└── reports/
    ├── centralized_metrics.json     # Centralized baseline test metrics
    ├── fedavg_metrics.json          # FedAvg multi-round metrics
    ├── fedprox_metrics.json         # FedProx metrics
    ├── fedbn_metrics.json           # FedBN metrics
    ├── dp_fedavg_metrics.json       # DP FedAvg metrics
    ├── comparison_report.json       # Comparative summary JSON
    ├── comparison_report.md         # Comparative summary Markdown
    └── figures/
        ├── val_accuracy_curves.png  # Accuracy progression across rounds
        ├── val_loss_curves.png      # Loss convergence across rounds
        ├── client_data_distribution.png # Non-IID class skew across sites
        └── model_comparison_metrics.png # Comparative bar chart
```
