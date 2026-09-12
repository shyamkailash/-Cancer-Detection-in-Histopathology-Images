#!/usr/bin/env python
"""
Unified CLI Experiment Runner for Federated Learning Benchmark.
Supports:
  ./venv/bin/python scripts/run_federated_experiments.py --method fedavg
  ./venv/bin/python scripts/run_federated_experiments.py --method fedprox
  ./venv/bin/python scripts/run_federated_experiments.py --method fedbn
  ./venv/bin/python scripts/run_federated_experiments.py --method dp_fedavg
  ./venv/bin/python scripts/run_federated_experiments.py --method all
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from ml.training.federated_trainer import run_federated_experiment
from scripts.generate_comparison_report import generate_federated_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Federated Learning Benchmark Runner (FedAvg, FedProx, FedBN, DP-FedAvg)."
    )
    parser.add_argument(
        "--method",
        "--algorithm",
        dest="algorithm",
        type=str,
        default="all",
        choices=["fedavg", "fedprox", "fedbn", "dp_fedavg", "dp", "all"],
        help="Method to benchmark: 'fedavg', 'fedprox', 'fedbn', 'dp_fedavg', or 'all' (default: all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/federated_experiment.yaml",
        help="Path to YAML configuration file (default: configs/federated_experiment.yaml).",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Number of global rounds override.")
    parser.add_argument("--num-clients", "--clients", dest="num_clients", type=int, default=None, help="Number of simulated sites override.")
    parser.add_argument("--local-epochs", type=int, default=None, help="Number of local epochs override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay override.")
    parser.add_argument("--mu", type=float, default=None, help="FedProx proximal mu override.")
    parser.add_argument("--noise-multiplier", type=float, default=None, help="DP noise multiplier override.")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="DP max grad norm override.")
    parser.add_argument("--delta", type=float, default=None, help="DP delta override.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--subset-fraction", type=float, default=None, help="Subset fraction override.")
    parser.add_argument("--output-base-dir", type=str, default=None, help="Output base directory override.")
    parser.add_argument("--generate-report", action="store_true", default=True, help="Generate comparative report and plots after run.")

    return parser.parse_args()


def load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    exp_cfg = cfg.get("experiment", {})
    algo_cfgs = cfg.get("algorithms", {})

    # Populate defaults from YAML config if CLI arg was not explicitly provided
    if args.rounds is None:
        args.rounds = exp_cfg.get("rounds", 3)
    if args.subset_fraction is None:
        args.subset_fraction = float(exp_cfg.get("subset_fraction", 0.20))
    if args.num_clients is None:
        args.num_clients = exp_cfg.get("num_clients", 5)
    if args.local_epochs is None:
        args.local_epochs = exp_cfg.get("local_epochs", 1)
    if args.batch_size is None:
        args.batch_size = exp_cfg.get("batch_size", 64)
    if args.learning_rate is None:
        args.learning_rate = float(exp_cfg.get("learning_rate", 1e-4))
    if args.weight_decay is None:
        args.weight_decay = float(exp_cfg.get("weight_decay", 1e-4))
    if args.seed is None:
        args.seed = exp_cfg.get("seed", 42)
    if args.mu is None:
        args.mu = float(algo_cfgs.get("fedprox", {}).get("mu", 0.01))
    if args.noise_multiplier is None:
        args.noise_multiplier = float(algo_cfgs.get("dp_fedavg", {}).get("noise_multiplier", 1.0))
    if args.max_grad_norm is None:
        args.max_grad_norm = float(algo_cfgs.get("dp_fedavg", {}).get("max_grad_norm", 1.0))
    if args.delta is None:
        args.delta = float(algo_cfgs.get("dp_fedavg", {}).get("delta", 1e-5))
    if args.output_base_dir is None:
        args.output_base_dir = exp_cfg.get("output_base_dir", "artifacts/federated")

    args.partition = exp_cfg.get("partition_type", "noniid")
    args.alpha = float(exp_cfg.get("dirichlet_alpha", 0.5))
    args.manifest_path = exp_cfg.get("dataset_manifest", "data/metadata/manifests/pcam_manifest.jsonl")
    args.baseline_checkpoint = exp_cfg.get("baseline_checkpoint", "artifacts/checkpoints/pcam_resnet18_best.pt")
    args.client_fraction = float(exp_cfg.get("client_fraction", 1.0))
    args.amp = True
    args.num_workers = 0
    args.privacy = "dp" if args.algorithm in ["dp", "dp_fedavg"] else "none"
    args.checkpoint_dir = None
    args.report_path = None

    # Run experiments
    results = run_federated_experiment(args)

    # Generate comparative analysis and plots
    if args.generate_report:
        print("\n" + "=" * 65)
        print("Generating Comparative Summary Tables & Visualizations...")
        print("=" * 65)
        generate_federated_comparison(output_dir=Path(args.output_base_dir) / "comparison")

    return results


if __name__ == "__main__":
    main()

