"""
Unified Federated Training Runner for Multi-Site Histopathology Cancer Detection.
Supports FedAvg, FedProx, FedBN, and Privacy-Preserving DP-FedAvg.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml.data.manifest import DatasetManifest
from ml.datasets.dataloaders import PCamDataset
from ml.preprocessing.transforms import get_train_transforms, get_eval_transforms
from ml.federated.partition import FederatedPartitioner
from ml.federated.privacy import PrivacyConfig
from ml.federated.client import FederatedClient
from ml.federated.fedprox import FedProxClient
from ml.federated.fedbn import FedBNClient
from ml.federated.server import FederatedServer
from ml.federated.utils import set_seed, estimate_model_size_mb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Multi-Site Federated Learning Runner (FedAvg, FedProx, FedBN, DP-FedAvg)."
    )

    # Federated algorithm selection
    parser.add_argument(
        "--algorithm",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "fedbn", "dp_fedavg", "dp"],
        help="Federated algorithm: 'fedavg', 'fedprox', 'fedbn', or 'dp_fedavg' (default: fedavg).",
    )

    # Communication and training hyperparameters
    parser.add_argument("--rounds", type=int, default=5, help="Number of global federated rounds (default: 5).")
    parser.add_argument("--clients", "--num-clients", dest="num_clients", type=int, default=5, help="Number of simulated healthcare sites (default: 5).")
    parser.add_argument("--local-epochs", type=int, default=1, help="Number of local epochs per round (default: 1).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for client and server (default: 32).")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-4, help="Learning rate for local optimizers (default: 1e-4).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for local optimizers (default: 1e-4).")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers (default: 0).")
    parser.add_argument("--client-fraction", type=float, default=1.0, help="Fraction of clients participating per round (default: 1.0).")

    # Algorithm specific hyperparameters
    parser.add_argument("--mu", type=float, default=0.01, help="Proximal coefficient for FedProx (default: 0.01).")
    parser.add_argument("--privacy", type=str, default="none", choices=["none", "dp"], help="Privacy mechanism: 'none' or 'dp' (differential privacy).")
    parser.add_argument("--max-grad-norm", "--dp-clip", dest="max_grad_norm", type=float, default=1.0, help="Max gradient L2 norm clipping threshold for DP (default: 1.0).")
    parser.add_argument("--noise-multiplier", "--dp-noise", dest="noise_multiplier", type=float, default=0.05, help="Gaussian noise multiplier for DP (default: 0.05).")

    # Partitioning & Experiment Configuration
    parser.add_argument("--partition", type=str, default="noniid", choices=["iid", "noniid"], help="Partition strategy: 'iid' or 'noniid' (default: noniid).")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter for non-IID partition (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (AMP) if CUDA available.")
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="Fraction of data to use (e.g. 0.10 for fast evaluation, default: 1.0).")

    # Output paths
    parser.add_argument("--manifest-path", type=str, default="data/metadata/manifests/pcam_manifest.jsonl", help="Path to PCam dataset manifest.")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save global checkpoints.")
    parser.add_argument("--report-path", type=str, default=None, help="Output path for experiment metrics JSON report.")

    return parser.parse_args()


def run_federated_experiment(args) -> Dict[str, Any]:
    """Execute complete federated learning experiment for the requested algorithm."""
    set_seed(args.seed)

    # Normalize algorithm name
    algo = args.algorithm.lower()
    if algo == "dp":
        algo = "dp_fedavg"

    is_dp = (algo == "dp_fedavg" or args.privacy == "dp")

    # Determine checkpoint and experiment directories
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = Path(f"artifacts/experiments/{algo}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"Privacy-Preserving Federated Learning Experiment: [{algo.upper()}]")
    print("=" * 65)
    print(f"Algorithm:            {algo.upper()}")
    print(f"Device:               {device}")
    if torch.cuda.is_available():
        print(f"GPU Name:             {torch.cuda.get_device_name(0)}")
    print(f"Federated Rounds:     {args.rounds}")
    print(f"Simulated Sites:      {args.num_clients}")
    print(f"Local Epochs:         {args.local_epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Partition Strategy:   {args.partition} (alpha={args.alpha})")
    if algo == "fedprox":
        print(f"FedProx Proximal mu:  {args.mu}")
    if is_dp:
        print(f"Privacy Mechanism:    DP (clip={args.max_grad_norm}, noise={args.noise_multiplier})")
    else:
        print(f"Privacy Mechanism:    NONE (Raw weights transmitted, images isolated)")
    print(f"Client Fraction:      {args.client_fraction}")
    print(f"Deterministic Seed:   {args.seed}")

    # Load dataset manifest
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found at: {manifest_path.resolve()}")

    print(f"\nLoading manifest from {manifest_path}...")
    manifest = DatasetManifest.load_jsonl(manifest_path)

    # Optional subset filtering for fast testing
    if args.subset_fraction < 1.0:
        rng = np.random.default_rng(args.seed)
        sub_samples = []
        for s in manifest.samples:
            if rng.random() < args.subset_fraction:
                sub_samples.append(s)
        manifest = DatasetManifest(samples=sub_samples, source_id=manifest.source_id)
        print(f"[SUBSET MODE] Using {len(manifest)} samples ({args.subset_fraction * 100:.1f}% subset).")

    # Partition training samples among simulated healthcare sites
    partitioner = FederatedPartitioner(num_clients=args.num_clients, seed=args.seed)
    partitions = partitioner.partition_manifest(
        manifest,
        partition_type=args.partition,
        alpha=args.alpha,
        target_split="train",
    )

    # Save partition metadata
    partition_meta_path = Path("data/metadata/federated/site_partitions.json")
    partitioner.save_partitions(
        partitions,
        partition_meta_path,
        metadata={
            "partition_type": args.partition,
            "alpha": args.alpha,
            "num_clients": args.num_clients,
            "seed": args.seed,
        },
    )
    print(f"[SAVED] Partition metadata saved to: {partition_meta_path}")

    # Print partition breakdown
    print("\nSimulated Site Data Distributions:")
    for cid, p in partitions.items():
        print(f"  - {cid} ({p.site_name}): {p.sample_count} samples | Normal: {p.normal_count} ({p.normal_percentage}%) | Metastasis: {p.metastasis_count} ({p.metastasis_percentage}%)")

    # Build client datasets and instantiate corresponding client types
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

    # Global Validation and Test Loaders
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

    print(f"\nGlobal Validation Samples: {len(val_dataset)}")
    print(f"Global Test Samples:       {len(test_dataset)}")

    # Instantiate and run FederatedServer
    server = FederatedServer(
        clients=clients,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        algorithm=algo,
        client_fraction=args.client_fraction,
        seed=args.seed,
        checkpoint_dir=str(ckpt_dir),
    )

    results = server.fit(rounds=args.rounds)

    # Save experiment config JSON
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
        "privacy": privacy_cfg.to_dict(),
        "subset_fraction": args.subset_fraction,
    }
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # Save experiment metrics JSON
    metrics_data = {
        "algorithm": algo,
        "best_round": results["best_round"],
        "best_val_accuracy": results["best_val_accuracy"],
        "total_training_time_seconds": results["total_training_time"],
        "communication_stats": results.get("communication_stats", {}),
        "test_metrics": results["test_metrics"],
        "history": results["history"],
    }
    with open(ckpt_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Full report output
    report_data = {
        "experiment_name": f"federated_pcam_{algo}_{args.partition}",
        **config_data,
        "partitions": {cid: p.to_dict() for cid, p in partitions.items()},
        **metrics_data,
    }

    # Save report to artifacts/reports/
    if args.report_path:
        report_file = Path(args.report_path)
    else:
        report_file = Path(f"artifacts/reports/{algo}_metrics.json")

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[SAVED] Experiment metrics report: {report_file.resolve()}")
    return report_data


def main():
    args = parse_args()
    run_federated_experiment(args)


if __name__ == "__main__":
    main()
