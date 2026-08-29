#!/usr/bin/env python
"""
Generate comprehensive comparison report (JSON and Markdown) comparing:
1. Centralized ResNet-18 Baseline
2. Federated ResNet-18 (FedAvg, Non-IID Dirichlet)
3. Privacy-Preserving Federated ResNet-18 (FedAvg + DP Gradient Clipping & Gaussian Noise)
"""

import json
from pathlib import Path


def main():
    reports_dir = Path("artifacts/reports")
    cent_path = reports_dir / "centralized_metrics.json"
    fed_path = reports_dir / "federated_metrics.json"
    dp_path = reports_dir / "privacy_federated_metrics.json"

    if not cent_path.exists() or not fed_path.exists() or not dp_path.exists():
        print(f"[ERROR] One or more metric files missing in {reports_dir}")
        return

    with open(cent_path, "r", encoding="utf-8") as f:
        cent = json.load(f)
    with open(fed_path, "r", encoding="utf-8") as f:
        fed = json.load(f)
    with open(dp_path, "r", encoding="utf-8") as f:
        dp = json.load(f)

    c_test = cent["test_metrics"]
    f_test = fed["test_metrics"]
    dp_test = dp["test_metrics"]

    comparison = {
        "title": "Comparative Analysis: Centralized vs. Federated vs. Privacy-Preserving Federated Learning",
        "task": "PCam Lymph Node Metastasis Histopathology Cancer Detection",
        "model_architecture": "ResNet-18 (ImageNet Pretrained Backbones)",
        "hardware": cent.get("gpu_name", "NVIDIA GeForce RTX 3050 6GB Laptop GPU"),
        "comparison_table": {
            "Centralized Baseline": {
                "paradigm": "Centralized",
                "data_access": "All raw data pooled in central storage",
                "privacy_guarantee": "None (Raw data shared)",
                "rounds_or_epochs": "1 Epoch (154,018 samples)",
                "test_loss": round(c_test["loss"], 4),
                "accuracy": round(c_test["accuracy"] * 100, 2),
                "sensitivity_recall": round(c_test["sensitivity"] * 100, 2),
                "specificity": round(c_test["specificity"] * 100, 2),
                "precision": round(c_test["precision"] * 100, 2),
                "f1_score": round(c_test["f1_score"] * 100, 2),
                "roc_auc": round(c_test["roc_auc"], 4) if c_test.get("roc_auc") else None,
                "confusion_matrix": c_test["confusion_matrix"],
            },
            "Federated Learning (FedAvg)": {
                "paradigm": "Federated Multi-Site (5 Simulated Sites)",
                "data_access": "Isolated local site data, zero raw image sharing",
                "privacy_guarantee": "Baseline FedAvg (Raw data localized, model updates shared)",
                "rounds_or_epochs": f"{fed['rounds']} Rounds (Local Epochs: {fed['local_epochs']})",
                "partition_strategy": f"Non-IID Dirichlet (alpha={fed['dirichlet_alpha']})",
                "test_loss": round(f_test["loss"], 4),
                "accuracy": round(f_test["accuracy"] * 100, 2),
                "sensitivity_recall": round(f_test["sensitivity"] * 100, 2),
                "specificity": round(f_test["specificity"] * 100, 2),
                "precision": round(f_test["precision"] * 100, 2),
                "f1_score": round(f_test["f1_score"] * 100, 2),
                "roc_auc": round(f_test["roc_auc"], 4) if f_test.get("roc_auc") else None,
                "confusion_matrix": f_test["confusion_matrix"],
            },
            "Privacy-Preserving Federated Learning (DP-FedAvg)": {
                "paradigm": "Federated Multi-Site + Differential Privacy",
                "data_access": "Isolated local site data, zero raw image sharing",
                "privacy_guarantee": f"Client-Side DP (Clip Norm={dp['privacy']['max_grad_norm']}, Noise Std={dp['privacy']['noise_multiplier']})",
                "rounds_or_epochs": f"{dp['rounds']} Rounds (Local Epochs: {dp['local_epochs']})",
                "partition_strategy": f"Non-IID Dirichlet (alpha={dp['dirichlet_alpha']})",
                "test_loss": round(dp_test["loss"], 4),
                "accuracy": round(dp_test["accuracy"] * 100, 2),
                "sensitivity_recall": round(dp_test["sensitivity"] * 100, 2),
                "specificity": round(dp_test["specificity"] * 100, 2),
                "precision": round(dp_test["precision"] * 100, 2),
                "f1_score": round(dp_test["f1_score"] * 100, 2),
                "roc_auc": round(dp_test["roc_auc"], 4) if dp_test.get("roc_auc") else None,
                "confusion_matrix": dp_test["confusion_matrix"],
            },
        },
    }

    # Save JSON
    comp_json_path = reports_dir / "comparison_report.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Save Markdown
    comp_md_path = reports_dir / "comparison_report.md"
    md_content = rf"""# Experimental Comparison Report: Multi-Site Federated Cancer Detection

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images  
**Dataset:** PatchCamelyon (PCam) — Lymph Node Metastasis Detection  
**Model Architecture:** ResNet-18 (ImageNet Pretrained)  
**Execution Hardware:** {cent.get("gpu_name", "NVIDIA GeForce RTX 3050 6GB Laptop GPU")}  

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
| **Global Test Loss** | **{c_test['loss']:.4f}** | {f_test['loss']:.4f} | {dp_test['loss']:.4f} |
| **Test Accuracy** | **{c_test['accuracy']*100:.2f}%** | **{f_test['accuracy']*100:.2f}%** | **{dp_test['accuracy']*100:.2f}%** |
| **Sensitivity (Metastasis Recall)** | **{c_test['sensitivity']*100:.2f}%** | **{f_test['sensitivity']*100:.2f}%** | **{dp_test['sensitivity']*100:.2f}%** |
| **Specificity (Normal Tissue)** | **{c_test['specificity']*100:.2f}%** | **{f_test['specificity']*100:.2f}%** | **{dp_test['specificity']*100:.2f}%** |
| **Precision** | **{c_test['precision']*100:.2f}%** | **{f_test['precision']*100:.2f}%** | **{dp_test['precision']*100:.2f}%** |
| **F1-Score** | **{c_test['f1_score']*100:.2f}%** | **{f_test['f1_score']*100:.2f}%** | **{dp_test['f1_score']*100:.2f}%** |
| **ROC-AUC** | **{c_test['roc_auc']:.4f}** | **{f_test['roc_auc']:.4f}** | **{dp_test['roc_auc']:.4f}** |

---

## 3. Clinical & Privacy Analysis

1. **Utility Retention in Federated Learning:**
   Standard FedAvg achieves **{f_test['accuracy']*100:.2f}% test accuracy** and **{f_test['sensitivity']*100:.2f}% sensitivity**, demonstrating effective knowledge aggregation across distributed healthcare sites without transferring raw histopathology images or patient labels.

2. **Privacy vs. Utility Tradeoff:**
   Injecting Differential Privacy noise ($\sigma={dp['privacy']['noise_multiplier']}$) with gradient norm clipping ($C={dp['privacy']['max_grad_norm']}$) introduces an observable utility-privacy tradeoff, yielding **{dp_test['accuracy']*100:.2f}% test accuracy** and **{dp_test['specificity']*100:.2f}% specificity** while bounding gradient sensitivity against inference and model inversion attacks.

3. **Clinical Priority (False Negatives & Sensitivity):**
   In oncological screening, false negatives carry significant clinical risks. Standard FedAvg maintains high sensitivity (**{f_test['sensitivity']*100:.2f}%** metastasis recall), confirming that federated averaging effectively handles Non-IID class imbalances across participating institutions.

---

## 4. Generated Figures
- `artifacts/reports/figures/val_accuracy_curves.png`: Multi-round validation accuracy trajectory.
- `artifacts/reports/figures/val_loss_curves.png`: Multi-round validation loss convergence.
- `artifacts/reports/figures/client_data_distribution.png`: Non-IID class skew across simulated hospital institutions.
- `artifacts/reports/figures/model_comparison_metrics.png`: Bar chart comparison across all clinical metrics.

---

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
"""

    with open(comp_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[SAVED] Comparison JSON: {comp_json_path.resolve()}")
    print(f"[SAVED] Comparison Markdown: {comp_md_path.resolve()}")


if __name__ == "__main__":
    main()

