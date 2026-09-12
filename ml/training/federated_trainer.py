"""
Unified Federated Training Runner for Multi-Site Histopathology Cancer Detection.
Supports FedAvg, FedProx, FedBN, and Privacy-Preserving DP-FedAvg.
Enforces baseline checkpoint initialization, non-IID partitioning, validation monitoring, and test reporting.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.manifest import DatasetManifest
from ml.datasets.dataloaders import PCamDataset
from ml.preprocessing.transforms import get_train_transforms, get_eval_transforms
from ml.models.resnet import create_resnet18
from ml.federated.partition import FederatedPartitioner
from ml.federated.privacy import PrivacyConfig
from ml.federated.client import FederatedClient
from ml.federated.fedprox import FedProxClient
from ml.federated.fedbn import FedBNClient, get_bn_state
from ml.federated.server import FederatedServer
from ml.federated.utils import set_seed, estimate_model_size_mb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Multi-Site Federated Learning Runner (FedAvg, FedProx, FedBN, DP-FedAvg)."
    )

    # Federated algorithm selection
    parser.add_argument(
        "--algorithm",
        "--method",
        dest="algorithm",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "fedbn", "dp_fedavg", "dp", "all"],
        help="Federated algorithm: 'fedavg', 'fedprox', 'fedbn', 'dp_fedavg', or 'all' (default: fedavg).",
    )

    # Communication and training hyperparameters
    parser.add_argument("--rounds", type=int, default=5, help="Number of global federated rounds (default: 5).")
    parser.add_argument("--clients", "--num-clients", dest="num_clients", type=int, default=5, help="Number of simulated healthcare sites (default: 5).")
    parser.add_argument("--local-epochs", type=int, default=1, help="Number of local epochs per round (default: 1).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for client and server (default: 64).")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4, help="Learning rate for local optimizers (default: 1e-4).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for local optimizers (default: 1e-4).")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (default: 0).")
    parser.add_argument("--client-fraction", type=float, default=1.0, help="Fraction of clients participating per round (default: 1.0).")

    # Algorithm specific hyperparameters
    parser.add_argument("--mu", type=float, default=0.01, help="Proximal coefficient for FedProx (default: 0.01).")
    parser.add_argument("--privacy", type=str, default="none", choices=["none", "dp"], help="Privacy mechanism: 'none' or 'dp'.")
    parser.add_argument("--max-grad-norm", "--dp-clip", dest="max_grad_norm", type=float, default=1.0, help="Max gradient L2 norm clipping threshold for DP (default: 1.0).")
    parser.add_argument("--noise-multiplier", "--dp-noise", dest="noise_multiplier", type=float, default=1.0, help="Gaussian noise multiplier for DP (default: 1.0).")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target privacy delta for DP (default: 1e-5).")

    # Partitioning & Experiment Configuration
    parser.add_argument("--partition", type=str, default="noniid", choices=["iid", "noniid"], help="Partition strategy: 'iid' or 'noniid' (default: noniid).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for non-IID partition (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP) if CUDA available.")
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="Fraction of training data to use (e.g. 0.20 for 20 percent training subset, default: 1.0).")

    # Checkpoint and manifest paths
    parser.add_argument("--manifest-path", type=str, default="data/metadata/manifests/pcam_manifest.jsonl", help="Path to PCam dataset manifest.")
    parser.add_argument("--baseline-checkpoint", "--checkpoint", dest="baseline_checkpoint", type=str, default="artifacts/checkpoints/pcam_resnet18_best.pt", help="Path to baseline checkpoint for initialization.")
    parser.add_argument("--output-base-dir", type=str, default="artifacts/federated", help="Base output directory for federated artifacts.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save algorithm checkpoints.")
    parser.add_argument("--report-path", type=str, default=None, help="Output path for experiment metrics JSON report.")

    return parser.parse_args()


def generate_algorithm_markdown_report(
    algo: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    partitions: Dict[str, Any],
    output_path: Path,
):
    """Generate human-readable markdown report for a single federated experiment."""
    test_metrics = results["test_metrics"]
    best_round = results["best_round"]
    best_val_acc = results["best_val_accuracy"]

    total_samples = sum(p["sample_count"] for p in partitions.values())

    md = f"""# Federated Learning Experiment Report: {algo.upper()}

**Semester Project:** Privacy-Preserving Federated Learning for Multi-Site Cancer Detection in Histopathology Images  
**Algorithm:** `{algo.upper()}`  
**Training Subset:** `{config.get('subset_fraction', 1.0) * 100:.1f}%` ({total_samples:,} Training Samples, Deterministic Seed = `{config['seed']}`)  
**Global Held-Out Test Set:** `33,003` Samples | **Global Validation Set:** `33,004` Samples  
**Partition Strategy:** `{config['partition_strategy']}` (Dirichlet alpha = `{config['dirichlet_alpha']}`)  
**Simulated Sites (Clients):** `{config['num_clients']}` (100% Client Participation)  
**Rounds:** `{config['rounds']}` | **Local Epochs:** `{config['local_epochs']}` | **Batch Size:** `{config['batch_size']}`  
**Hardware / Device:** `{config['device']}` ({config.get('gpu_name', 'CPU')})  

---

## 1. Held-Out Global Test Performance (33,003 Samples)

| Metric | Result |
| :--- | :---: |
| **Accuracy** | **{test_metrics['accuracy'] * 100:.2f}%** |
| **Sensitivity (Metastasis Recall)** | **{test_metrics['sensitivity'] * 100:.2f}%** |
| **Specificity (Normal Tissue)** | **{test_metrics['specificity'] * 100:.2f}%** |
| **Precision** | **{test_metrics['precision'] * 100:.2f}%** |
| **F1-Score** | **{test_metrics['f1_score'] * 100:.2f}%** |
| **ROC-AUC** | **{test_metrics.get('roc_auc', 0.0):.4f}** |
| **Test Loss** | **{test_metrics['loss']:.4f}** |

---

## 2. Training Dynamics & Round Progression

- **Selected Best Round:** Round `{best_round}` (Validation Accuracy: `{best_val_acc * 100:.2f}%`)
- **Total Training Duration:** `{results['total_training_time_seconds']:.2f}s`

| Round | Duration (s) | Val Loss | Val Accuracy | Val Sensitivity | Val Specificity | Val ROC-AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results["history"]:
        v = r["global_val_metrics"]
        v_auc = v.get("roc_auc") or 0.0
        md += f"| {r['round']} | {r['duration_seconds']:.1f} | {v['loss']:.4f} | {v['accuracy']*100:.2f}% | {v['sensitivity']*100:.2f}% | {v['specificity']*100:.2f}% | {v_auc:.4f} |\n"

    md += f"""
---

## 3. Simulated Healthcare Site Distribution

| Site ID | Site Name | Samples | Normal (Class 0) | Metastasis (Class 1) | Metastasis Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for cid, p in partitions.items():
        n_cnt = p['normal_count']
        m_cnt = p['metastasis_count']
        ratio_str = f"{m_cnt / n_cnt:.3f}" if n_cnt > 0 else "inf"
        md += f"| `{cid}` | {p['site_name']} | {p['sample_count']:,} | {n_cnt:,} ({p['normal_percentage']}%) | {m_cnt:,} ({p['metastasis_percentage']}%) | {ratio_str} |\n"

    if algo == "dp_fedavg":
        md += f"""
---

## 4. Privacy Configuration & Accounting

- **Differential Privacy Mechanism:** Gaussian DP on Local Gradients
- **Gradient Clipping Norm ($C$):** `{config['privacy']['max_grad_norm']}`
- **Gaussian Noise Multiplier (\\sigma):** `{config['privacy']['noise_multiplier']}`
- **Target Delta (\\delta):** `{config['privacy']['delta']}`
- **Formal Epsilon Accounting:** {config['privacy']['formal_epsilon_status']}
"""

    md += f"""
---

> **Academic Disclaimer:** This software is an educational and research prototype. It is not approved or certified for clinical diagnosis, patient triage, or medical decision-making.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


def run_single_federated_experiment(algo: str, args) -> Dict[str, Any]:
    """Execute complete federated learning experiment for a single algorithm."""
    set_seed(args.seed)

    # Normalize algorithm name
    algo = algo.lower()
    if algo == "dp":
        algo = "dp_fedavg"

    is_dp = (algo == "dp_fedavg" or args.privacy == "dp")

    # Determine checkpoint and experiment directories
    base_out = Path(args.output_base_dir)
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = base_out / algo
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print(f"Privacy-Preserving Federated Learning Experiment: [{algo.upper()}]")
    print("=" * 65)
    print(f"Algorithm:            {algo.upper()}")
    print(f"Device:               {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:             {torch.cuda.get_device_name(0)}")
    print(f"Baseline Checkpoint:  {args.baseline_checkpoint}")
    print(f"Federated Rounds:     {args.rounds}")
    print(f"Simulated Sites:      {args.num_clients}")
    print(f"Local Epochs:         {args.local_epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Partition Strategy:   {args.partition} (alpha={args.alpha})")
    if algo == "fedprox":
        print(f"FedProx Proximal mu:  {args.mu}")
    if is_dp:
        print(f"Privacy Mechanism:    DP (clip={args.max_grad_norm}, noise={args.noise_multiplier}, delta={args.delta})")
    else:
        print(f"Privacy Mechanism:    NONE (Raw weights transmitted, images isolated)")
    print(f"Client Participation: {args.client_fraction * 100:.0f}% ({args.num_clients}/{args.num_clients} sites)")
    print(f"Deterministic Seed:   {args.seed}")

    # 1. Load dataset manifest
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path.resolve()}")

    print(f"\nLoading manifest from {manifest_path}...")
    manifest = DatasetManifest.load_jsonl(manifest_path)

    # 2. Extract splits: strictly subset ONLY the training split, keeping full validation and test sets intact
    train_samples = [s for s in manifest.samples if s.split == "train"]
    val_samples = [s for s in manifest.samples if s.split == "val"]
    test_samples = [s for s in manifest.samples if s.split == "test"]

    if args.subset_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        num_subset = int(round(len(train_samples) * args.subset_fraction))
        selected_indices = sorted(rng.choice(len(train_samples), size=num_subset, replace=False))
        sub_train_samples = [train_samples[i] for i in selected_indices]
        manifest = DatasetManifest(samples=sub_train_samples + val_samples + test_samples, source_id=manifest.source_id)

        # Persist deterministic training subset manifest
        train_sub_manifest = DatasetManifest(samples=sub_train_samples, source_id="pcam_train_subset")
        sub_manifest_path = base_out / "client_partition" / "train_subset_manifest.jsonl"
        sub_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        train_sub_manifest.save_jsonl(sub_manifest_path)

        print(f"\n[REPRODUCIBLE TRAINING SUBSET]")
        print(f"  Total original training samples: {len(train_samples):,}")
        print(f"  Selected subset fraction:        {args.subset_fraction * 100:.1f}%")
        print(f"  Deterministic subset samples:    {len(sub_train_samples):,} (seed={args.seed})")
        print(f"  Validation split (held-out):     {len(val_samples):,} samples (100% full)")
        print(f"  Test split (held-out global):    {len(test_samples):,} samples (100% full)")
        print(f"  [SAVED] Training subset manifest: {sub_manifest_path.resolve()}")

    # 3. Partition training samples among simulated healthcare sites
    partitioner = FederatedPartitioner(num_clients=args.num_clients, seed=args.seed)
    partitions = partitioner.partition_manifest(
        manifest,
        partition_type=args.partition,
        alpha=args.alpha,
        target_split="train",
    )

    # Save partition metadata in client_partition directory
    partition_meta_path = base_out / "client_partition" / "site_partitions.json"
    partitioner.save_partitions(
        partitions,
        partition_meta_path,
        metadata={
            "partition_type": args.partition,
            "alpha": args.alpha,
            "num_clients": args.num_clients,
            "seed": args.seed,
            "subset_fraction": args.subset_fraction,
            "total_subset_samples": sum(p.sample_count for p in partitions.values()),
        },
    )
    # Also save to data/metadata/federated
    partitioner.save_partitions(
        partitions,
        Path("data/metadata/federated/site_partitions.json"),
        metadata={
            "partition_type": args.partition,
            "alpha": args.alpha,
            "num_clients": args.num_clients,
            "seed": args.seed,
            "subset_fraction": args.subset_fraction,
            "total_subset_samples": sum(p.sample_count for p in partitions.values()),
        },
    )
    print(f"[SAVED] Partition metadata saved to: {partition_meta_path}")

    # Print comprehensive client partition statistics
    total_subset_samples = sum(p.sample_count for p in partitions.values())
    print("\n" + "=" * 100)
    print("SIMULATED HEALTHCARE SITES: NON-IID DATA DISTRIBUTION (SEED=42)")
    print("=" * 100)
    print(f"Total Training Subset Samples: {total_subset_samples:,}")
    print(f"{'Site ID':<10} | {'Site Name':<30} | {'Samples':<8} | {'Normal (0)':<16} | {'Metastasis (1)':<18} | {'Ratio (M:N)':<10}")
    print("-" * 100)
    for cid, p in partitions.items():
        ratio_str = f"{p.metastasis_count / p.normal_count:.3f}" if p.normal_count > 0 else "inf"
        print(f"{cid:<10} | {p.site_name:<30} | {p.sample_count:<8} | {p.normal_count:,} ({p.normal_percentage:.1f}%) | {p.metastasis_count:,} ({p.metastasis_percentage:.1f}%) | {ratio_str:<10}")
    print("=" * 100)

    # 4. Load baseline model initialization strictly from baseline checkpoint
    base_ckpt_path = Path(args.baseline_checkpoint)
    initial_model = create_resnet18(num_classes=2, pretrained=True).to(device)
    if base_ckpt_path.exists():
        print(f"\n[INFO] Initializing global model from baseline checkpoint: {base_ckpt_path.resolve()}")
        ckpt_data = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt_data:
            initial_model.load_state_dict(ckpt_data["model_state_dict"])
        elif isinstance(ckpt_data, dict):
            initial_model.load_state_dict(ckpt_data)
    else:
        print(f"[WARNING] Baseline checkpoint not found at {base_ckpt_path}. Starting from ImageNet pretrained weights.")

    # 5. Build client datasets and instantiate corresponding client types
    privacy_cfg = PrivacyConfig(
        enabled=is_dp,
        max_grad_norm=args.max_grad_norm,
        noise_multiplier=args.noise_multiplier,
    )

    clients: List[FederatedClient] = []
    for cid, part in partitions.items():
        client_manifest = partitioner.get_client_manifest(manifest, part)
        client_dataset = PCamDataset(
            manifest=client_manifest,
            split="train",
            transform=get_train_transforms(),
        )

        if algo == "fedprox":
            client = FedProxClient(
                client_id=cid,
                site_name=part.site_name,
                dataset=client_dataset,
                device=device,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                mu=args.mu,
                privacy_config=privacy_cfg,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                amp=args.amp,
            )
        elif algo == "fedbn":
            client = FedBNClient(
                client_id=cid,
                site_name=part.site_name,
                dataset=client_dataset,
                device=device,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                privacy_config=privacy_cfg,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                amp=args.amp,
            )
            client.local_bn_state = get_bn_state(initial_model)
        else:
            # Standard FedAvg / DP-FedAvg
            client = FederatedClient(
                client_id=cid,
                site_name=part.site_name,
                dataset=client_dataset,
                device=device,
                batch_size=args.batch_size,
                local_epochs=args.local_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                privacy_config=privacy_cfg,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                amp=args.amp,
            )
        clients.append(client)

    # 6. Global Validation and Test Loaders (Full held-out splits)
    val_dataset = PCamDataset(manifest=manifest, split="val", transform=get_eval_transforms())
    test_dataset = PCamDataset(manifest=manifest, split="test", transform=get_eval_transforms())

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

    print(f"\nGlobal Validation Samples: {len(val_dataset):,}")
    print(f"Global Test Samples:       {len(test_dataset):,}")

    # 7. Instantiate and run FederatedServer
    server = FederatedServer(
        clients=clients,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        initial_model=initial_model,
        client_fraction=args.client_fraction,
        algorithm=algo,
        seed=args.seed,
        checkpoint_dir=str(ckpt_dir),
    )

    results = server.fit(rounds=args.rounds)

    # 8. Save experiment config JSON
    config_data = {
        "algorithm": algo,
        "seed": args.seed,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_clients": args.num_clients,
        "client_fraction": args.client_fraction,
        "partition_strategy": args.partition,
        "dirichlet_alpha": args.alpha,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mu": args.mu if algo == "fedprox" else None,
        "privacy": {
            **privacy_cfg.to_dict(),
            "delta": args.delta if is_dp else None,
            "formal_epsilon_status": "Formal epsilon accounting not implemented/verified (empirical gradient clipping and calibrated Gaussian perturbation applied)" if is_dp else "N/A",
        },
        "subset_fraction": args.subset_fraction,
        "total_training_samples": total_subset_samples,
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
    }
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Save metrics data JSON
    metrics_data = {
        "algorithm": algo,
        "best_round": results["best_round"],
        "best_val_accuracy": results["best_val_accuracy"],
        "best_val_metrics": results["best_val_metrics"],
        "init_val_metrics": results.get("init_val_metrics"),
        "total_training_time_seconds": results["total_training_time"],
        "communication_stats": results.get("communication_stats", {}),
        "test_metrics": results["test_metrics"],
        "history": results["history"],
    }
    with open(ckpt_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Generate Markdown report
    report_md_path = ckpt_dir / "report.md"
    generate_algorithm_markdown_report(
        algo=algo,
        config=config_data,
        results=metrics_data,
        partitions={cid: p.to_dict() for cid, p in partitions.items()},
        output_path=report_md_path,
    )
    print(f"[SAVED] Algorithm Markdown report: {report_md_path.resolve()}")

    # Full report output for artifacts/reports/
    report_data = {
        "experiment_name": f"federated_pcam_{algo}_{args.partition}",
        **config_data,
        "partitions": {cid: p.to_dict() for cid, p in partitions.items()},
        **metrics_data,
    }

    if args.report_path:
        report_file = Path(args.report_path)
    else:
        report_file = Path(f"artifacts/reports/{algo}_metrics.json")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"[SAVED] Experiment metrics JSON: {report_file.resolve()}")
    return report_data


def run_federated_experiment(args) -> Dict[str, Any]:
    """Top-level dispatcher supporting single algorithm or 'all' algorithms."""
    algo = args.algorithm.lower()
    if algo == "dp":
        algo = "dp_fedavg"

    if algo == "all":
        algorithms = ["fedavg", "fedprox", "fedbn", "dp_fedavg"]
        all_results = {}
        for a in algorithms:
            print(f"\n{'='*70}\nSTARTING BENCHMARK: {a.upper()}\n{'='*70}")
            res = run_single_federated_experiment(a, args)
            all_results[a] = res
        return all_results
    else:
        return run_single_federated_experiment(algo, args)


def main():
    args = parse_args()
    run_federated_experiment(args)


if __name__ == "__main__":
    main()
