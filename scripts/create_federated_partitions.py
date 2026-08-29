#!/usr/bin/env python
"""
CLI Script for generating and inspecting simulated healthcare site dataset partitions (IID / Non-IID).
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.manifest import DatasetManifest
from ml.federated.partition import FederatedPartitioner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Partition PCam dataset across simulated healthcare sites (IID / Non-IID Dirichlet)."
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="data/metadata/manifests/pcam_manifest.jsonl",
        help="Path to PCam dataset manifest.",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=5,
        help="Number of simulated healthcare sites (default: 5).",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="noniid",
        choices=["iid", "noniid"],
        help="Partition strategy: 'iid' or 'noniid' (default: noniid).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter for non-IID partition (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic partitioning (default: 42).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/metadata/federated/site_partitions.json",
        help="Path to save partition metadata JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path.resolve()}", file=sys.stderr)
        return 1

    print(f"Loading manifest from {manifest_path}...")
    manifest = DatasetManifest.load_jsonl(manifest_path)

    partitioner = FederatedPartitioner(num_clients=args.num_clients, seed=args.seed)
    partitions = partitioner.partition_manifest(
        manifest,
        partition_type=args.partition,
        alpha=args.alpha,
        target_split="train",
    )

    out_path = Path(args.output_path)
    partitioner.save_partitions(
        partitions,
        out_path,
        metadata={
            "partition_type": args.partition,
            "alpha": args.alpha,
            "num_clients": args.num_clients,
            "seed": args.seed,
        },
    )

    print("\n" + "=" * 70)
    print(f"Simulated Multi-Site Healthcare Partitioning ({args.partition.upper()}, alpha={args.alpha}, seed={args.seed})")
    print("=" * 70)
    for cid, p in partitions.items():
        print(f"  {cid:<8} | {p.site_name:<30} | Total: {p.sample_count:<6} | Normal: {p.normal_count:<6} ({p.normal_percentage:5.1f}%) | Metastasis: {p.metastasis_count:<6} ({p.metastasis_percentage:5.1f}%)")

    print(f"\n[SAVED] Partition metadata: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

