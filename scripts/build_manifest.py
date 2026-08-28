#!/usr/bin/env python
"""
CLI Script for building standardized dataset manifests and creating leakage-safe splits.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.registry import DatasetRegistry
from ml.data.manifest import DatasetManifest
from ml.data.split import DatasetSplitter
from ml.sources.pcam import PCamSource
from ml.sources.breakhis import BreakHisSource
from ml.reports.dataset_report import DatasetReportGenerator


SOURCES = {
    "pcam": PCamSource,
    "breakhis": BreakHisSource,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build standardized dataset manifest and generate leakage-safe splits."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=list(SOURCES.keys()),
        help="Dataset source identifier (e.g. 'pcam', 'breakhis').",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset directory. Defaults to registered default_dir.",
    )
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Perform reproducible train/val/test split assignment.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of data for training split (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for validation split (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of data for test split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        choices=["auto", "patient", "slide", "stratified"],
        help="Split strategy ('auto', 'patient', 'slide', or 'stratified').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/metadata/manifests",
        help="Directory to save generated manifests.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["jsonl", "csv", "both"],
        help="Manifest output format (jsonl, csv, or both).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_cls = SOURCES[args.source]
    adapter = source_cls()
    registry = DatasetRegistry()

    try:
        source_meta = registry.get_source(args.source)
        default_dir = Path(source_meta.get("default_dir", f"data/raw/{args.source}"))
    except KeyError:
        default_dir = Path(f"data/raw/{args.source}")

    data_dir = Path(args.data_dir) if args.data_dir else default_dir
    print(f"Building manifest for '{adapter.dataset_name}' from: {data_dir.resolve()}...")

    if not data_dir.exists():
        print(f"[ERROR] Directory does not exist: {data_dir.resolve()}", file=sys.stderr)
        return 1

    samples = adapter.discover_samples(data_dir)
    if not samples:
        print(f"[WARNING] No samples discovered in: {data_dir.resolve()}")
        return 0

    manifest = DatasetManifest(samples=samples, source_id=args.source)

    if args.create_splits:
        print(f"Applying '{args.strategy}' splitting strategy (seed={args.seed}, ratios={args.train_ratio}/{args.val_ratio}/{args.test_ratio})...")
        splitter = DatasetSplitter(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        manifest = splitter.split(manifest, strategy=args.strategy)

        is_leak_free, leak_issues = splitter.verify_no_leakage(manifest)
        if is_leak_free:
            print("  [LEAKAGE AUDIT] PASSED: No cross-split patient/slide/hash leakage detected.")
        else:
            print("  [LEAKAGE AUDIT] WARNINGS:")
            for issue in leak_issues:
                print(f"    - {issue}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"{args.source}_manifest.jsonl"
    csv_path = out_dir / f"{args.source}_manifest.csv"

    if args.format in ("jsonl", "both"):
        manifest.save_jsonl(jsonl_path)
        print(f"[SAVED] Manifest JSONL: {jsonl_path.resolve()}")

    if args.format in ("csv", "both"):
        manifest.save_csv(csv_path)
        print(f"[SAVED] Manifest CSV: {csv_path.resolve()}")

    # Print summary
    summary = manifest.get_summary()
    print(f"\nManifest built successfully: {summary['total_samples']} samples ({summary['valid_samples']} valid, {summary['invalid_samples']} invalid).")
    print(f"Split breakdown: {summary['split_distribution']}")
    print(f"Label breakdown: {summary['label_distribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

