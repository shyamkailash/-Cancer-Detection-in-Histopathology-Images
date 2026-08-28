"""
Tests for CLI scripts execution and argument parsing.
"""

import subprocess
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def create_sample_dataset(root: Path):
    """Create miniature synthetic dataset for CLI integration test."""
    p0 = root / "0"
    p1 = root / "1"
    p0.mkdir(parents=True, exist_ok=True)
    p1.mkdir(parents=True, exist_ok=True)

    arr = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p0 / "patch_0.png")
    Image.fromarray(arr + 1).save(p1 / "patch_1.png")


def test_cli_help_options():
    """Verify all CLI scripts support --help without error."""
    scripts = [
        "scripts/download_dataset.py",
        "scripts/validate_dataset.py",
        "scripts/build_manifest.py",
    ]
    for script in scripts:
        res = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "usage:" in res.stdout.lower() or "options:" in res.stdout.lower()


def test_build_manifest_and_validate_cli_execution(tmp_path):
    """Test full CLI lifecycle with synthetic data."""
    data_dir = tmp_path / "raw_pcam"
    create_sample_dataset(data_dir)

    out_dir = tmp_path / "manifests"
    reports_dir = tmp_path / "reports"

    # 1. Test build_manifest.py
    cmd_build = [
        sys.executable,
        "scripts/build_manifest.py",
        "--source", "pcam",
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--create-splits",
        "--format", "both",
    ]
    res_build = subprocess.run(cmd_build, capture_output=True, text=True)
    assert res_build.returncode == 0
    assert (out_dir / "pcam_manifest.jsonl").exists()
    assert (out_dir / "pcam_manifest.csv").exists()

    # 2. Test validate_dataset.py
    cmd_val = [
        sys.executable,
        "scripts/validate_dataset.py",
        "--source", "pcam",
        "--data-dir", str(data_dir),
        "--output-dir", str(reports_dir),
    ]
    res_val = subprocess.run(cmd_val, capture_output=True, text=True)
    assert res_val.returncode == 0
    assert (reports_dir / "pcam_validation_report.json").exists()
    assert (reports_dir / "pcam_validation_report.md").exists()

