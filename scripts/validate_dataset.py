#!/usr/bin/env python
"""
CLI Script for validating histopathology image integrity, labels, and duplicates.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.registry import DatasetRegistry
from ml.data.manifest import DatasetManifest
from ml.sources.pcam import PCamSource
from ml.sources.breakhis import BreakHisSource
from ml.reports.dataset_report import DatasetReportGenerator


SOURCES = {
    "pcam": PCamSource,
    "breakhis": BreakHisSource,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate histopathology dataset images, labels, and detect duplicates."
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
        "--output-dir",
        type=str,
        default="reports/generated",
        help="Directory to save validation reports.",
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
    print(f"Scanning and validating '{adapter.dataset_name}' in: {data_dir.resolve()}...")

    if not data_dir.exists():
        print(f"[ERROR] Directory does not exist: {data_dir.resolve()}", file=sys.stderr)
        print(f"Please acquire the dataset first using: python scripts/download_dataset.py --source {args.source}", file=sys.stderr)
        return 1

    samples = adapter.discover_samples(data_dir)
    if not samples:
        print(f"[WARNING] No supported image files discovered in: {data_dir.resolve()}")
        return 0

    manifest = DatasetManifest(samples=samples, source_id=args.source)
    report_gen = DatasetReportGenerator()
    report = report_gen.generate_report(manifest, dataset_name=adapter.dataset_name, source_id=args.source)

    out_dir = Path(args.output_dir)
    json_path = out_dir / f"{args.source}_validation_report.json"
    md_path = out_dir / f"{args.source}_validation_report.md"

    report_gen.save_json(report, json_path)
    report_gen.save_markdown(report, md_path)

    print("\n" + report_gen.generate_markdown(report))
    print(f"\n[SAVED] JSON Report: {json_path.resolve()}")
    print(f"[SAVED] Markdown Report: {md_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
