#!/usr/bin/env python3
"""
Model Checkpoint Synchronization and Verification Script.

Ensures all 6 production model checkpoints exist locally in the `artifacts/` hierarchy.
If checkpoints are missing, downloads them from `CHECKPOINTS_BASE_URL` (e.g. GitHub Releases, AWS S3, Cloudflare R2).
Supports verification-only mode and fallback synthetic weight creation for CI/staging builds.
"""

import argparse
import os
from pathlib import Path
import sys
import urllib.request
import urllib.error
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mapping of model ID to relative checkpoint path
REQUIRED_CHECKPOINTS = {
    "centralized": "artifacts/checkpoints/pcam_resnet18_best.pt",
    "centralized_finetuned": "artifacts/finetuning/centralized/best_model.pt",
    "fedavg": "artifacts/federated/fedavg/best_global_model.pt",
    "fedprox": "artifacts/federated/fedprox/best_global_model.pt",
    "fedbn": "artifacts/federated/fedbn/best_global_model.pt",
    "dp_fedavg": "artifacts/federated/dp_fedavg/best_global_model.pt",
}


def verify_checkpoint(path: Path) -> bool:
    """Verify that file exists and can be loaded as a PyTorch state dict."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            sd = data.get("model_state_dict") or data.get("state_dict") or data
            return isinstance(sd, dict) and len(sd) > 0
        return False
    except Exception:
        return False


def download_file(url: str, dest: Path) -> bool:
    """Download a file with progress reporting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(".tmp")
    print(f"Downloading {url} -> {dest}...")
    try:
        urllib.request.urlretrieve(url, str(tmp_dest))
        tmp_dest.rename(dest)
        print(f"Successfully downloaded {dest} ({dest.stat().st_size / (1024 * 1024):.2f} MB)")
        return True
    except Exception as exc:
        print(f"Failed to download from {url}: {exc}", file=sys.stderr)
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False


def create_synthetic_checkpoint(dest: Path) -> None:
    """Create a synthetic ResNet-18 checkpoint for testing environments only."""
    from ml.models.resnet import create_resnet18

    dest.parent.mkdir(parents=True, exist_ok=True)
    model = create_resnet18(num_classes=2, pretrained=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ci_synthetic": True,
            "epoch": 1,
            "metrics": {"val_acc": 0.50},
        },
        str(dest),
    )
    print(f"[TEST FIXTURE] Generated synthetic checkpoint at {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure production model checkpoints are available.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify that checkpoints exist and are valid without downloading.",
    )
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow creating synthetic checkpoints if external downloads are not configured (testing only).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CHECKPOINTS_BASE_URL", ""),
        help="Base URL for remote checkpoint storage (e.g. https://.../releases/download/v1.0/).",
    )
    args = parser.parse_args()

    missing = []
    verified = []

    print(f"Checking model checkpoints in {PROJECT_ROOT}...")

    for model_id, rel_path in REQUIRED_CHECKPOINTS.items():
        abs_path = (PROJECT_ROOT / rel_path).resolve()
        if abs_path.exists() and verify_checkpoint(abs_path):
            size_mb = abs_path.stat().st_size / (1024 * 1024)
            verified.append((model_id, rel_path, size_mb))
            print(f"  [OK] {model_id}: {rel_path} ({size_mb:.2f} MB)")
        else:
            missing.append((model_id, rel_path, abs_path))
            print(f"  [MISSING] {model_id}: {rel_path}")

    if not missing:
        print(f"\nAll {len(verified)} model checkpoints are present and valid.")
        return 0

    if args.verify_only:
        print(f"\nError: {len(missing)} model checkpoints are missing in verify-only mode.", file=sys.stderr)
        return 1

    # Attempt to download missing checkpoints
    base_url = args.base_url.rstrip("/") if args.base_url else ""
    allow_synthetic = args.allow_synthetic or (os.environ.get("ALLOW_SYNTHETIC_CHECKPOINTS", "").lower() in ("1", "true", "yes"))

    if base_url:
        print(f"\nAttempting to download missing checkpoints from {base_url}...")
        for model_id, rel_path, abs_path in missing:
            filename = Path(rel_path).name
            download_url = f"{base_url}/{filename}"
            if not download_file(download_url, abs_path):
                if allow_synthetic:
                    print(f"Download failed; falling back to synthetic checkpoint for {model_id}...")
                    create_synthetic_checkpoint(abs_path)
                else:
                    print(f"Error: Failed to obtain checkpoint for {model_id} from {download_url}", file=sys.stderr)
                    return 1
    elif allow_synthetic:
        print("\nNo CHECKPOINTS_BASE_URL specified. ALLOW_SYNTHETIC_CHECKPOINTS is active; creating test checkpoints...")
        for model_id, rel_path, abs_path in missing:
            create_synthetic_checkpoint(abs_path)
    else:
        print("\nError: Model checkpoints are missing and no CHECKPOINTS_BASE_URL was provided.", file=sys.stderr)
        print("To resolve:", file=sys.stderr)
        print("1. Set CHECKPOINTS_BASE_URL in your environment pointing to remote storage (e.g. GitHub Releases/S3), OR", file=sys.stderr)
        print("2. Commit the pruned <45 MB checkpoints directly to the repository, OR", file=sys.stderr)
        print("3. For test/staging environments, set ALLOW_SYNTHETIC_CHECKPOINTS=1.", file=sys.stderr)
        return 1

    print("\nCheckpoint verification complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
