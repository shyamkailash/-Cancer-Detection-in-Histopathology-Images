#!/usr/bin/env python
"""
CLI Script for Safe, Controlled Fine-Tuning of Centralized ResNet-18 on PCam.
Evaluates baseline and fine-tuned candidates on the exact same held-out test set.
Enforces validation-only model selection and conservative quality gate before promotion.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from ml.data.manifest import DatasetManifest
from ml.datasets.dataloaders import PCamDataset
from ml.preprocessing.transforms import get_train_transforms, get_eval_transforms
from ml.models.resnet import create_resnet18
from ml.training.finetuner import CentralizedFineTuner
from ml.evaluation.metrics import evaluate_model
from ml.federated.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safe, Controlled Fine-Tuning of Centralized ResNet-18 on PCam."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/checkpoints/pcam_resnet18_best.pt",
        help="Path to initial baseline checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/finetuning/centralized",
        help="Directory to store fine-tuning experiment artifacts.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="data/metadata/manifests/pcam_manifest.jsonl",
        help="Path to dataset manifest.",
    )
    parser.add_argument("--stage1-epochs", type=int, default=1, help="Stage 1 (Head only) epochs (default: 1).")
    parser.add_argument("--stage2-epochs", type=int, default=2, help="Stage 2 (Layer4 + Head) epochs (default: 2).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--head-lr", type=float, default=3e-4, help="Learning rate for classifier head (default: 3e-4).")
    parser.add_argument("--layer4-lr", type=float, default=3e-5, help="Learning rate for layer4 features (default: 3e-5).")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Reserved parameter for shallow backbone (default: 1e-5).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4).")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (default: 2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0).")
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="Subset fraction (default: 1.0).")
    parser.add_argument("--auto-promote", action="store_true", help="Promote candidate if quality gate passes.")
    return parser.parse_args()


def draw_training_curves(history: list, output_path: Path):
    """Draw simple training and validation curves using PIL."""
    if not history:
        return
    width, height = 800, 450
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin_top, margin_bottom, margin_left, margin_right = 60, 60, 80, 50
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    draw.text((width // 2 - 160, 20), "Centralized Fine-Tuning Curves", fill=(30, 30, 30))

    epochs = [h["epoch"] for h in history]
    val_accs = [h["val_accuracy"] * 100 for h in history]
    train_accs = [h["train_accuracy"] * 100 for h in history]

    num_pts = len(epochs)
    if num_pts < 2:
        return

    # Draw gridlines (80% to 100%)
    for pct in [80, 85, 90, 95, 100]:
        y = margin_top + chart_h - int(((pct - 80) / 20.0) * chart_h)
        draw.line([(margin_left, y), (margin_left + chart_w, y)], fill=(230, 230, 230), width=1)
        draw.text((margin_left - 45, y - 7), f"{pct}%", fill=(100, 100, 100))

    # Train curve (blue)
    train_pts = []
    val_pts = []
    for i, (t_acc, v_acc) in enumerate(zip(train_accs, val_accs)):
        x = margin_left + int((i / (num_pts - 1)) * chart_w)
        y_t = margin_top + chart_h - int((max(80.0, min(100.0, t_acc)) - 80.0) / 20.0 * chart_h)
        y_v = margin_top + chart_h - int((max(80.0, min(100.0, v_acc)) - 80.0) / 20.0 * chart_h)
        train_pts.append((x, y_t))
        val_pts.append((x, y_v))

    for i in range(num_pts - 1):
        draw.line([train_pts[i], train_pts[i + 1]], fill=(52, 152, 219), width=3)
        draw.line([val_pts[i], val_pts[i + 1]], fill=(46, 204, 113), width=3)

    for x, y in train_pts:
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=(52, 152, 219))
    for x, y in val_pts:
        draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=(46, 204, 113))

    # Legend
    draw.rectangle([(margin_left, height - 25), (margin_left + 12, height - 13)], fill=(52, 152, 219))
    draw.text((margin_left + 18, height - 25), "Train Accuracy", fill=(50, 50, 50))
    draw.rectangle([(margin_left + 150, height - 25), (margin_left + 162, height - 13)], fill=(46, 204, 113))
    draw.text((margin_left + 168, height - 25), "Val Accuracy", fill=(50, 50, 50))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def generate_markdown_report(
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    config: Dict[str, Any],
    ft_res: Dict[str, Any],
    decision: str,
    output_path: Path,
):
    """Generate comprehensive human-readable Markdown report."""
    base_auc = baseline_metrics.get("roc_auc") or 0.0
    cand_auc = candidate_metrics.get("roc_auc") or 0.0

    diff_acc = (candidate_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100.0
    diff_sens = (candidate_metrics["sensitivity"] - baseline_metrics["sensitivity"]) * 100.0
    diff_spec = (candidate_metrics["specificity"] - baseline_metrics["specificity"]) * 100.0
    diff_prec = (candidate_metrics["precision"] - baseline_metrics["precision"]) * 100.0
    diff_f1 = (candidate_metrics["f1_score"] - baseline_metrics["f1_score"]) * 100.0
    diff_auc = cand_auc - base_auc
    diff_loss = candidate_metrics["loss"] - baseline_metrics["loss"]

    md = f"""# Centralized ResNet-18 Fine-Tuning Report

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images
**Dataset:** PatchCamelyon (PCam) — 220,025 samples
**Target:** Centralized ResNet-18 Transfer Learning Fine-Tuning
**Hardware:** {config.get('gpu_name', 'NVIDIA GeForce RTX 3050 6GB Laptop GPU')}
**Decision:** **[{decision}]**

---

## 1. Executive Summary

A 2-stage transfer learning fine-tuning experiment was conducted on the centralized ResNet-18 baseline:
- **Stage 1 (Head Calibration):** Classifier head trained with backbone frozen (`lr={config['head_lr']}`).
- **Stage 2 (Deep Feature Fine-Tuning):** Layer 4 and classifier head unfrozen with discriminative learning rates (`layer4_lr={config['layer4_lr']}`, `head_lr={config['head_lr']*0.5}`) and Cosine Annealing scheduler.

---

## 2. Comparative Results on Held-Out Test Set (33,003 Samples)

| Metric | Baseline | Fine-Tuned Candidate | Absolute Difference | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | **{baseline_metrics['accuracy']*100:.2f}%** | **{candidate_metrics['accuracy']*100:.2f}%** | **{diff_acc:+.2f}%** | {diff_acc/baseline_metrics['accuracy']:+.2f}% |
| **Sensitivity (Metastasis Recall)** | **{baseline_metrics['sensitivity']*100:.2f}%** | **{candidate_metrics['sensitivity']*100:.2f}%** | **{diff_sens:+.2f}%** | {diff_sens/baseline_metrics['sensitivity']:+.2f}% |
| **Specificity (Normal Tissue)** | **{baseline_metrics['specificity']*100:.2f}%** | **{candidate_metrics['specificity']*100:.2f}%** | **{diff_spec:+.2f}%** | {diff_spec/baseline_metrics['specificity']:+.2f}% |
| **Precision** | **{baseline_metrics['precision']*100:.2f}%** | **{candidate_metrics['precision']*100:.2f}%** | **{diff_prec:+.2f}%** | {diff_prec/baseline_metrics['precision']:+.2f}% |
| **F1-Score** | **{baseline_metrics['f1_score']*100:.2f}%** | **{candidate_metrics['f1_score']*100:.2f}%** | **{diff_f1:+.2f}%** | {diff_f1/baseline_metrics['f1_score']:+.2f}% |
| **ROC-AUC** | **{base_auc:.4f}** | **{cand_auc:.4f}** | **{diff_auc:+.4f}** | {diff_auc/base_auc*100:+.2f}% |
| **Test Loss** | **{baseline_metrics['loss']:.4f}** | **{candidate_metrics['loss']:.4f}** | **{diff_loss:+.4f}** | - |

---

## 3. Confusion Matrix Breakdown

### Baseline:
- True Positives (Metastasis): `{baseline_metrics['confusion_matrix']['tp']}`
- True Negatives (Normal): `{baseline_metrics['confusion_matrix']['tn']}`
- False Positives: `{baseline_metrics['confusion_matrix']['fp']}`
- False Negatives: `{baseline_metrics['confusion_matrix']['fn']}`

### Fine-Tuned Candidate:
- True Positives (Metastasis): `{candidate_metrics['confusion_matrix']['tp']}`
- True Negatives (Normal): `{candidate_metrics['confusion_matrix']['tn']}`
- False Positives: `{candidate_metrics['confusion_matrix']['fp']}`
- False Negatives: `{candidate_metrics['confusion_matrix']['fn']}`

---

## 4. Hyperparameters & Training Configuration

- **Random Seed:** `{config['seed']}`
- **Batch Size:** `{config['batch_size']}`
- **Stage 1 Epochs:** `{config['stage1_epochs']}` (Head LR: `{config['head_lr']}`)
- **Stage 2 Epochs:** `{config['stage2_epochs']}` (Layer4 LR: `{config['layer4_lr']}`, Head LR: `{config['head_lr']*0.5}`)
- **Scheduler:** `CosineAnnealingLR (eta_min=1e-6)`
- **Total Training Duration:** `{ft_res['total_duration_seconds']:.2f}s`
- **Best Validation Epoch:** `{ft_res['best_epoch']}` (Val Acc: `{ft_res['best_val_accuracy']*100:.2f}%`, Val Loss: `{ft_res['best_val_loss']:.4f}`)

---

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def check_validation_quality_gate(
    init_val_metrics: Dict[str, Any],
    candidate_val_metrics: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Evaluate validation-only quality gate.
    Returns (passed, explanation).
    Criteria:
    - Validation Accuracy must not decrease materially (>= init - 0.002)
    - Validation Sensitivity must not decrease materially (>= init - 0.005)
    - Validation ROC-AUC must not decrease materially (>= init - 0.002)
    - At least one metric (accuracy, sensitivity, or ROC-AUC) must strictly improve.
    """
    base_acc = init_val_metrics.get("accuracy", 0.0)
    cand_acc = candidate_val_metrics.get("accuracy", 0.0)

    base_sens = init_val_metrics.get("sensitivity", 0.0)
    cand_sens = candidate_val_metrics.get("sensitivity", 0.0)

    base_auc = init_val_metrics.get("roc_auc", 0.0) or 0.0
    cand_auc = candidate_val_metrics.get("roc_auc", 0.0) or 0.0

    not_regressed = (
        cand_acc >= (base_acc - 0.002)
        and cand_sens >= (base_sens - 0.005)
        and cand_auc >= (base_auc - 0.002)
    )
    strictly_improved = (
        cand_acc > base_acc
        or cand_sens > base_sens
        or cand_auc > base_auc
    )

    passed = not_regressed and strictly_improved
    explanation = (
        f"Val Acc: {cand_acc*100:.2f}% (vs {base_acc*100:.2f}%), "
        f"Val Sens: {cand_sens*100:.2f}% (vs {base_sens*100:.2f}%), "
        f"Val AUC: {cand_auc:.4f} (vs {base_auc:.4f})"
    )
    return passed, explanation


def promote_candidate_checkpoint(
    candidate_checkpoint_path: Path,
    baseline_checkpoint_path: Path,
    val_metrics: Dict[str, Any],
    candidate_test_metrics: Dict[str, Any],
    config: Dict[str, Any],
    best_epoch: int,
    train_history: list,
    num_samples_train: int,
    num_samples_val: int,
    num_samples_test: int,
    device_str: str,
    gpu_name: Optional[str] = None,
    report_path: Optional[Path] = None,
) -> bool:
    """
    Safely backup original checkpoint and promote candidate checkpoint to production.
    Does NOT reference undefined variables.
    """
    if not candidate_checkpoint_path.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found at {candidate_checkpoint_path}")

    # 1. Backup original baseline checkpoint
    backup_path = baseline_checkpoint_path.parent / f"{baseline_checkpoint_path.stem}.backup.pt"
    if baseline_checkpoint_path.exists():
        shutil.copy2(baseline_checkpoint_path, backup_path)
        print(f"[BACKUP] Created backup at: {backup_path}")

    # 2. Load candidate model state
    cand_ckpt = torch.load(candidate_checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = cand_ckpt.get("model_state_dict", cand_ckpt)

    # 3. Calculate num_parameters safely from state_dict
    num_params = sum(p.numel() for p in state_dict.values() if hasattr(p, "numel"))

    # 4. Save promoted checkpoint
    torch.save(
        {
            "epoch": best_epoch,
            "model_state_dict": state_dict,
            "metrics": {
                "train_loss": train_history[-1]["train_loss"] if train_history else 0.0,
                "train_accuracy": train_history[-1]["train_accuracy"] if train_history else 0.0,
                "val_loss": val_metrics.get("loss", 0.0),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
            },
            "promoted_from_finetuning": True,
        },
        baseline_checkpoint_path,
    )
    print(f"[PROMOTED] Baseline checkpoint successfully updated: {baseline_checkpoint_path}")

    # 5. Update centralized metrics report
    centralized_report_path = report_path or Path("artifacts/reports/centralized_metrics.json")
    centralized_report_path.parent.mkdir(parents=True, exist_ok=True)
    centralized_report_data = {
        "experiment_name": "centralized_resnet18_baseline_finetuned",
        "device": device_str,
        "gpu_name": gpu_name,
        "num_parameters": num_params,
        "epochs": config.get("stage1_epochs", 1) + config.get("stage2_epochs", 2),
        "batch_size": config.get("batch_size", 64),
        "train_samples": num_samples_train,
        "val_samples": num_samples_val,
        "test_samples": num_samples_test,
        "train_metrics": {
            "loss": train_history[-1]["train_loss"] if train_history else 0.0,
            "accuracy": train_history[-1]["train_accuracy"] if train_history else 0.0,
        },
        "val_metrics": val_metrics,
        "test_metrics": candidate_test_metrics,
        "training_time_seconds": config.get("total_duration_seconds", 0.0),
    }
    with open(centralized_report_path, "w", encoding="utf-8") as f:
        json.dump(centralized_report_data, f, indent=2)
    print(f"[SAVED] Updated centralized metrics: {centralized_report_path}")
    return True


def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Centralized ResNet-18 Fine-Tuning Experiment")
    print("=" * 65)
    print(f"Device:                 {device}")
    if torch.cuda.is_available():
        print(f"GPU:                    {torch.cuda.get_device_name(0)}")
    print(f"Baseline Checkpoint:    {args.checkpoint}")
    print(f"Stage 1 Epochs (Head):  {args.stage1_epochs}")
    print(f"Stage 2 Epochs (L4+Head): {args.stage2_epochs}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Head LR / Layer4 LR:    {args.head_lr} / {args.layer4_lr}")
    print(f"Patience:               {args.patience}")
    print(f"Deterministic Seed:     {args.seed}")

    # 1. Load manifest and DataLoaders
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"\nLoading PCam dataset manifest...")
    manifest = DatasetManifest.load_jsonl(manifest_path)

    if args.subset_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        sub_samples = [s for s in manifest.samples if rng.random() < args.subset_fraction]
        manifest = DatasetManifest(samples=sub_samples, source_id=manifest.source_id)
        print(f"[SUBSET MODE] Using {len(manifest)} samples ({args.subset_fraction * 100:.1f}% subset).")

    train_dataset = PCamDataset(manifest=manifest, split="train", transform=get_train_transforms())
    val_dataset = PCamDataset(manifest=manifest, split="val", transform=get_eval_transforms())
    test_dataset = PCamDataset(manifest=manifest, split="test", transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples:      {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples:       {len(test_dataset):,}")

    # 2. Load and evaluate original baseline model on test set (for reporting)
    base_ckpt_path = Path(args.checkpoint)
    baseline_model = create_resnet18(num_classes=2, pretrained=True).to(device)

    if base_ckpt_path.exists():
        print(f"\n[INFO] Loading initial checkpoint: {base_ckpt_path}")
        ckpt_data = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt_data:
            baseline_model.load_state_dict(ckpt_data["model_state_dict"])
        elif isinstance(ckpt_data, dict):
            baseline_model.load_state_dict(ckpt_data)

    print("\n--- Evaluating Original Baseline on Global Test Set ---")
    baseline_test_metrics = evaluate_model(baseline_model, test_loader, device=device)
    print(f"Baseline Test Loss:        {baseline_test_metrics['loss']:.4f}")
    print(f"Baseline Test Accuracy:    {baseline_test_metrics['accuracy'] * 100:.2f}%")
    print(f"Baseline Test Sensitivity: {baseline_test_metrics['sensitivity'] * 100:.2f}%")
    print(f"Baseline Test Specificity: {baseline_test_metrics['specificity'] * 100:.2f}%")
    print(f"Baseline Test Precision:   {baseline_test_metrics['precision'] * 100:.2f}%")
    print(f"Baseline Test F1-Score:    {baseline_test_metrics['f1_score'] * 100:.2f}%")
    if baseline_test_metrics.get("roc_auc"):
        print(f"Baseline Test ROC-AUC:     {baseline_test_metrics['roc_auc']:.4f}")

    # 3. Instantiate model for fine-tuning
    model_to_finetune = create_resnet18(num_classes=2, pretrained=True).to(device)
    if base_ckpt_path.exists():
        ckpt_data = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt_data:
            model_to_finetune.load_state_dict(ckpt_data["model_state_dict"])
        elif isinstance(ckpt_data, dict):
            model_to_finetune.load_state_dict(ckpt_data)

    finetuner = CentralizedFineTuner(
        model=model_to_finetune,
        device=device,
        output_dir=str(out_dir),
        max_grad_norm=1.0,
        amp=True,
    )

    # 4. Run fine-tuning (selection based purely on validation metrics)
    ft_res = finetuner.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        head_lr=args.head_lr,
        layer4_lr=args.layer4_lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    # 5. Evaluate fine-tuned candidate on global test set (for reporting/comparison)
    print("\n" + "=" * 65)
    print("Evaluating Fine-Tuned Candidate on Global Test Set")
    print("=" * 65)
    candidate_test_metrics = evaluate_model(finetuner.model, test_loader, device=device)
    print(f"Candidate Test Loss:        {candidate_test_metrics['loss']:.4f}")
    print(f"Candidate Test Accuracy:    {candidate_test_metrics['accuracy'] * 100:.2f}%")
    print(f"Candidate Test Sensitivity: {candidate_test_metrics['sensitivity'] * 100:.2f}%")
    print(f"Candidate Test Specificity: {candidate_test_metrics['specificity'] * 100:.2f}%")
    print(f"Candidate Test Precision:   {candidate_test_metrics['precision'] * 100:.2f}%")
    print(f"Candidate Test F1-Score:    {candidate_test_metrics['f1_score'] * 100:.2f}%")
    if candidate_test_metrics.get("roc_auc"):
        print(f"Candidate Test ROC-AUC:     {candidate_test_metrics['roc_auc']:.4f}")

    # 6. Save experiment artifacts
    config_dict = {
        "experiment": "centralized_resnet18_finetuning",
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": args.batch_size,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "head_lr": args.head_lr,
        "layer4_lr": args.layer4_lr,
        "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "subset_fraction": args.subset_fraction,
        "total_duration_seconds": ft_res["total_duration_seconds"],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(ft_res["history"], f, indent=2)

    metrics_dict = {
        "baseline_test_metrics": baseline_test_metrics,
        "candidate_test_metrics": candidate_test_metrics,
        "best_epoch": ft_res["best_epoch"],
        "best_val_accuracy": ft_res["best_val_accuracy"],
        "total_duration_seconds": ft_res["total_duration_seconds"],
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    # Save summary metrics in artifacts/finetuning/
    summary_metrics_path = Path("artifacts/finetuning/centralized_finetune_metrics.json")
    summary_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": config_dict,
            "baseline": baseline_test_metrics,
            "finetuned": candidate_test_metrics,
            "best_epoch": ft_res["best_epoch"],
            "best_val_accuracy": ft_res["best_val_accuracy"],
        }, f, indent=2)

    draw_training_curves(ft_res["history"], out_dir / "training_curves.png")

    # 7. Model Quality Gate Analysis (Validation-only)
    diff_acc = (candidate_test_metrics["accuracy"] - baseline_test_metrics["accuracy"]) * 100.0
    diff_sens = (candidate_test_metrics["sensitivity"] - baseline_test_metrics["sensitivity"]) * 100.0
    diff_spec = (candidate_test_metrics["specificity"] - baseline_test_metrics["specificity"]) * 100.0
    diff_prec = (candidate_test_metrics["precision"] - baseline_test_metrics["precision"]) * 100.0
    diff_f1 = (candidate_test_metrics["f1_score"] - baseline_test_metrics["f1_score"]) * 100.0
    base_auc = baseline_test_metrics.get("roc_auc") or 0.0
    cand_auc = candidate_test_metrics.get("roc_auc") or 0.0
    diff_auc = cand_auc - base_auc

    print("\n" + "=" * 65)
    print("COMPARISON: Baseline vs. Fine-Tuned Candidate (Held-Out Test Set)")
    print("=" * 65)
    print(f"{'Metric':<18} | {'Baseline':<12} | {'Fine-Tuned':<12} | {'Difference':<12}")
    print("-" * 65)
    print(f"{'Accuracy':<18} | {baseline_test_metrics['accuracy']*100:>10.2f}% | {candidate_test_metrics['accuracy']*100:>10.2f}% | {diff_acc:>+10.2f}%")
    print(f"{'Sensitivity':<18} | {baseline_test_metrics['sensitivity']*100:>10.2f}% | {candidate_test_metrics['sensitivity']*100:>10.2f}% | {diff_sens:>+10.2f}%")
    print(f"{'Specificity':<18} | {baseline_test_metrics['specificity']*100:>10.2f}% | {candidate_test_metrics['specificity']*100:>10.2f}% | {diff_spec:>+10.2f}%")
    print(f"{'Precision':<18} | {baseline_test_metrics['precision']*100:>10.2f}% | {candidate_test_metrics['precision']*100:>10.2f}% | {diff_prec:>+10.2f}%")
    print(f"{'F1-Score':<18} | {baseline_test_metrics['f1_score']*100:>10.2f}% | {candidate_test_metrics['f1_score']*100:>10.2f}% | {diff_f1:>+10.2f}%")
    print(f"{'ROC-AUC':<18} | {base_auc:>10.4f}  | {cand_auc:>10.4f}  | {diff_auc:>+10.4f}")
    print(f"{'Test Loss':<18} | {baseline_test_metrics['loss']:>10.4f}  | {candidate_test_metrics['loss']:>10.4f}  | {candidate_test_metrics['loss'] - baseline_test_metrics['loss']:>+10.4f}")

    # Validation Quality Gate check
    init_val = ft_res.get("init_val_metrics", {})
    best_val = ft_res.get("best_val_metrics", {})
    gate_passed, gate_reason = check_validation_quality_gate(init_val, best_val)

    decision = "ACCEPTED" if gate_passed else "REJECTED"
    print(f"\nModel Quality Gate Decision: [{decision}] — {gate_reason}")

    # Generate Markdown report
    report_md_path = Path("artifacts/finetuning/centralized_finetune_report.md")
    generate_markdown_report(
        baseline_metrics=baseline_test_metrics,
        candidate_metrics=candidate_test_metrics,
        config=config_dict,
        ft_res=ft_res,
        decision=decision,
        output_path=report_md_path,
    )
    print(f"[SAVED] Markdown report: {report_md_path.resolve()}")

    if gate_passed and args.auto_promote:
        promote_candidate_checkpoint(
            candidate_checkpoint_path=Path(ft_res["candidate_checkpoint"]),
            baseline_checkpoint_path=base_ckpt_path,
            val_metrics=best_val,
            candidate_test_metrics=candidate_test_metrics,
            config=config_dict,
            best_epoch=ft_res["best_epoch"],
            train_history=ft_res["history"],
            num_samples_train=len(train_dataset),
            num_samples_val=len(val_dataset),
            num_samples_test=len(test_dataset),
            device_str=str(device),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )
    else:
        if not gate_passed:
            print("\nFine-tuning rejected — baseline remains the production model.")
        else:
            print("\nCandidate accepted but --auto-promote flag was not set. Baseline checkpoint left untouched.")

    return 0


def main():
    args = parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
