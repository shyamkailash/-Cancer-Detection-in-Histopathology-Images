"""
Dataset Quality and Manifest Reporting in JSON and Markdown formats.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from ..data.manifest import DatasetManifest
from ..data.split import DatasetSplitter
from ..validation.duplicates import DuplicateDetector


class DatasetReportGenerator:
    """
    Produces comprehensive dataset quality, split balance, and data leakage audit reports.
    """

    def __init__(self, splitter: Optional[DatasetSplitter] = None, dup_detector: Optional[DuplicateDetector] = None):
        self.splitter = splitter or DatasetSplitter()
        self.dup_detector = dup_detector or DuplicateDetector()

    def generate_report(
        self,
        manifest: DatasetManifest,
        dataset_name: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive report dictionary from a dataset manifest."""
        summary = manifest.get_summary()
        dup_report = self.dup_detector.generate_duplicate_report(manifest.samples)
        is_leak_free, leak_issues = self.splitter.verify_no_leakage(manifest)

        # Calculate dimension statistics
        resolutions: Dict[str, int] = {}
        for s in manifest.samples:
            if s.image_width and s.image_height:
                res_key = f"{s.image_width}x{s.image_height}"
                resolutions[res_key] = resolutions.get(res_key, 0) + 1

        # Validation errors breakdown
        error_counts: Dict[str, int] = {}
        for s in manifest.samples:
            for err in s.validation_errors:
                error_counts[err] = error_counts.get(err, 0) + 1

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_id": source_id or manifest.source_id or "unknown",
            "dataset_name": dataset_name or "Histopathology Dataset",
            "summary": summary,
            "resolutions": resolutions,
            "validation_errors": error_counts,
            "duplicate_audit": {
                "unique_duplicate_hashes": dup_report["unique_duplicate_hashes"],
                "total_duplicate_instances": dup_report["total_duplicate_instances"],
                "redundant_samples_count": dup_report["redundant_samples_count"],
                "cross_source_duplicate_hashes": dup_report["cross_source_duplicate_hashes"],
            },
            "data_leakage_audit": {
                "is_leakage_free": is_leak_free,
                "leakage_issues": leak_issues,
            },
        }
        return report

    def generate_markdown(self, report: Dict[str, Any]) -> str:
        """Convert report dictionary to a formatted Markdown string."""
        s = report["summary"]
        dup = report["duplicate_audit"]
        leak = report["data_leakage_audit"]

        md_lines = [
            f"# Dataset Quality Report: {report['dataset_name']}",
            f"**Source ID:** `{report['source_id']}` | **Generated At:** `{report['generated_at']}`\n",
            "## 1. Overview Statistics",
            f"- **Total Samples Discovered:** {s['total_samples']}",
            f"- **Valid Samples:** {s['valid_samples']} ({s['valid_samples'] / max(1, s['total_samples']) * 100:.1f}%)",
            f"- **Invalid / Corrupted Samples:** {s['invalid_samples']}",
            f"- **Unique Patients:** {s['unique_patients']}",
            f"- **Unique Slides:** {s['unique_slides']}",
            f"- **Unique Image Hashes:** {s['unique_file_hashes']}",
            "",
            "## 2. Class Distribution",
            "| Class / Label | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ]

        total_valid = max(1, s["valid_samples"])
        for lbl, count in s["label_distribution"].items():
            pct = (count / total_valid) * 100
            md_lines.append(f"| `{lbl}` | {count} | {pct:.1f}% |")

        md_lines.extend([
            "",
            "## 3. Split Distribution",
            "| Split | Count | Percentage |",
            "| :--- | :--- | :--- |",
        ])

        total_samples = max(1, s["total_samples"])
        for split_name, count in s["split_distribution"].items():
            pct = (count / total_samples) * 100
            md_lines.append(f"| `{split_name}` | {count} | {pct:.1f}% |")

        md_lines.extend([
            "",
            "## 4. Duplicate & Integrity Audit",
            f"- **Unique Duplicate Clusters:** {dup['unique_duplicate_hashes']}",
            f"- **Total Redundant Images:** {dup['redundant_samples_count']}",
            f"- **Cross-Source Duplicate Hashes:** {dup['cross_source_duplicate_hashes']}",
            "",
            "## 5. Data Leakage Verification",
            f"- **Leakage Free:** {'✅ PASS' if leak['is_leakage_free'] else '❌ FAIL'}",
        ])

        if not leak["is_leakage_free"]:
            md_lines.append("\n**Leakage Warnings:**")
            for issue in leak["leakage_issues"]:
                md_lines.append(f"- ⚠️ {issue}")

        if report["resolutions"]:
            md_lines.extend([
                "",
                "## 6. Image Resolutions & Formats",
                "| Resolution (WxH) | Count |",
                "| :--- | :--- |",
            ])
            for res, count in report["resolutions"].items():
                md_lines.append(f"| `{res}` | {count} |")

        if report["validation_errors"]:
            md_lines.extend([
                "",
                "## 7. Validation Errors Breakdown",
                "| Error Description | Count |",
                "| :--- | :--- |",
            ])
            for err, count in report["validation_errors"].items():
                md_lines.append(f"| `{err}` | {count} |")

        return "\n".join(md_lines)

    def save_json(self, report: Dict[str, Any], filepath: Path) -> None:
        """Save report to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def save_markdown(self, report: Dict[str, Any], filepath: Path) -> None:
        """Save report to a Markdown file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        md_text = self.generate_markdown(report)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md_text)

