#!/usr/bin/env python
"""
CLI Script for downloading or displaying acquisition instructions for histopathology datasets.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.registry import DatasetRegistry
from ml.sources.pcam import PCamSource
from ml.sources.breakhis import BreakHisSource


SOURCES = {
    "pcam": PCamSource,
    "breakhis": BreakHisSource,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Acquire or view acquisition instructions for histopathology datasets."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=list(SOURCES.keys()),
        help="Dataset source identifier (e.g. 'pcam', 'breakhis').",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=None,
        help="Target destination directory for raw dataset. Defaults to data/raw/<source>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_cls = SOURCES[args.source]
    adapter = source_cls()
    registry = DatasetRegistry()

    print("=" * 70)
    print(f"Cancer Detection Histopathology - Dataset Acquisition: {adapter.dataset_name}")
    print("=" * 70)

    try:
        source_meta = registry.get_source(args.source)
        default_dir = Path(source_meta.get("default_dir", f"data/raw/{args.source}"))
    except KeyError:
        default_dir = Path(f"data/raw/{args.source}")

    dest_dir = Path(args.dest_dir) if args.dest_dir else default_dir

    if adapter.is_available(dest_dir):
        print(f"\n[INFO] Dataset '{args.source}' already exists locally at: {dest_dir.resolve()}")
        print("To re-validate or generate manifest, run:")
        print(f"  python scripts/validate_dataset.py --source {args.source}")
        print(f"  python scripts/build_manifest.py --source {args.source} --create-splits")
        return 0

    print(f"\n[ACQUISITION STRATEGY] for {adapter.dataset_name}:")
    download_info = adapter.download(dest_dir)

    for step in download_info.get("instructions", []):
        print(f"  {step}")

    print(f"\nOfficial URL: {download_info.get('official_url', 'N/A')}")
    print(f"License: {download_info.get('license', 'N/A')}")
    print(f"Target Directory: {dest_dir.resolve()}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
