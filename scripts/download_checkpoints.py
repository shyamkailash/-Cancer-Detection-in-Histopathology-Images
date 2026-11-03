#!/usr/bin/env python3
"""
Model Checkpoint Synchronization and Verification Script.

Ensures all 6 production model checkpoints exist locally in the `artifacts/` hierarchy.
If checkpoints are missing, downloads them from `CHECKPOINTS_BASE_URL` or per-model
environment variables (e.g. GitHub Releases, AWS S3, Cloudflare R2).

Features:
- Disambiguated remote candidate URLs to avoid collisions on shared filenames (`best_global_model.pt`).
- Strict PyTorch architecture validation (verifies weights, layer names, and 2-class fc shape).
- Safe atomic download to temporary files before moving to destination.
- Strict verify-only mode for CI pipelines.
- Transparent reporting of missing checkpoints.
"""

import argparse
import os
from pathlib import Path
import sys
import urllib.request
import urllib.error
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Comprehensive checkpoint specification with disambiguated remote filenames
CHECKPOINT_SPEC = {
    "centralized": {
        "rel_path": "artifacts/checkpoints/pcam_resnet18_best.pt",
        "remote_candidates": [
            "artifacts/checkpoints/pcam_resnet18_best.pt",
            "checkpoints/pcam_resnet18_best.pt",
            "centralized_pcam_resnet18_best.pt",
            "centralized.pt",
            "pcam_resnet18_best.pt",
        ],
        "env_var": "CHECKPOINT_URL_CENTRALIZED",
    },
    "centralized_finetuned": {
        "rel_path": "artifacts/finetuning/centralized/best_model.pt",
        "remote_candidates": [
            "artifacts/finetuning/centralized/best_model.pt",
            "finetuning/centralized/best_model.pt",
            "centralized_finetuned_best_model.pt",
            "centralized_finetuned.pt",
        ],
        "env_var": "CHECKPOINT_URL_CENTRALIZED_FINETUNED",
    },
    "fedavg": {
        "rel_path": "artifacts/federated/fedavg/best_global_model.pt",
        "remote_candidates": [
            "artifacts/federated/fedavg/best_global_model.pt",
            "federated/fedavg/best_global_model.pt",
            "fedavg_best_global_model.pt",
            "fedavg.pt",
        ],
        "env_var": "CHECKPOINT_URL_FEDAVG",
    },
    "fedprox": {
        "rel_path": "artifacts/federated/fedprox/best_global_model.pt",
        "remote_candidates": [
            "artifacts/federated/fedprox/best_global_model.pt",
            "federated/fedprox/best_global_model.pt",
            "fedprox_best_global_model.pt",
            "fedprox.pt",
        ],
        "env_var": "CHECKPOINT_URL_FEDPROX",
    },
    "fedbn": {
        "rel_path": "artifacts/federated/fedbn/best_global_model.pt",
        "remote_candidates": [
            "artifacts/federated/fedbn/best_global_model.pt",
            "federated/fedbn/best_global_model.pt",
            "fedbn_best_global_model.pt",
            "fedbn.pt",
        ],
        "env_var": "CHECKPOINT_URL_FEDBN",
    },
    "dp_fedavg": {
        "rel_path": "artifacts/federated/dp_fedavg/best_global_model.pt",
        "remote_candidates": [
            "artifacts/federated/dp_fedavg/best_global_model.pt",
            "federated/dp_fedavg/best_global_model.pt",
            "dp_fedavg_best_global_model.pt",
            "dp_fedavg.pt",
        ],
        "env_var": "CHECKPOINT_URL_DP_FEDAVG",
    },
}


def verify_checkpoint(path: Path) -> tuple[bool, str]:
    """Verify that file exists, is readable, and contains a valid ResNet-18 cancer detection state dict."""
    if not path.is_file():
        return False, "File does not exist or is not a regular file."
    if path.stat().st_size < 1_000_000:
        return False, f"File size too small ({path.stat().st_size} bytes, expected >1MB)."

    try:
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            return False, "Loaded checkpoint object is not a dictionary."

        sd = data.get("model_state_dict") or data.get("state_dict") or data
        if not isinstance(sd, dict) or len(sd) < 100:
            return False, f"State dict has {len(sd) if isinstance(sd, dict) else 0} keys (expected >=100 for ResNet-18)."

        # Normalize module. prefixes
        cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

        required_layers = ["conv1.weight", "layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        for layer in required_layers:
            if layer not in cleaned:
                return False, f"Missing required layer '{layer}' in state dict."

        # Verify output layer shape matches binary classification (2 classes)
        fc_w = cleaned["fc.weight"]
        fc_b = cleaned["fc.bias"]
        if tuple(fc_w.shape) != (2, 512):
            return False, f"Unexpected fc.weight shape {fc_w.shape} (expected (2, 512))."
        if tuple(fc_b.shape) != (2,):
            return False, f"Unexpected fc.bias shape {fc_b.shape} (expected (2,))."

        return True, f"Valid ResNet-18 binary classification checkpoint ({len(cleaned)} weights)"
    except Exception as exc:
        return False, f"PyTorch load error: {exc}"


def download_file(url: str, dest: Path) -> bool:
    """Download a file atomically to a temporary destination and verify before finalizing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f"{dest.name}.download.tmp")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "OncoLens-Checkpoint-Downloader/1.0"},
        )
        print(f"  --> Connecting: {url}")
        with urllib.request.urlopen(req, timeout=90) as response:
            if response.status != 200:
                print(f"  [HTTP {response.status}] Failed to download from {url}", file=sys.stderr)
                return False

            with open(tmp_dest, "wb") as f_out:
                bytes_downloaded = 0
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    bytes_downloaded += len(chunk)

        is_valid, reason = verify_checkpoint(tmp_dest)
        if is_valid:
            tmp_dest.replace(dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [SUCCESS] Verified and installed: {dest.name} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"  [INVALID] Downloaded file failed verification: {reason}", file=sys.stderr)
            if tmp_dest.exists():
                tmp_dest.unlink()
            return False
    except Exception as exc:
        print(f"  [ERROR] Download failed: {exc}", file=sys.stderr)
        if tmp_dest.exists():
            tmp_dest.unlink()
        return False


def create_synthetic_checkpoint(dest: Path) -> None:
    """Create a synthetic ResNet-18 checkpoint for test/staging environments only."""
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
    canonical_release_url = "https://github.com/shyamkailash/-Cancer-Detection-in-Histopathology-Images/releases/download/v1.0.0"
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CHECKPOINTS_BASE_URL", canonical_release_url),
        help="Base URL for remote checkpoint storage (default: GitHub Releases v1.0.0).",
    )
    args = parser.parse_args()

    missing = []
    verified = []

    print(f"Checking model checkpoints in {PROJECT_ROOT}...")

    for model_id, spec in CHECKPOINT_SPEC.items():
        rel_path = spec["rel_path"]
        abs_path = (PROJECT_ROOT / rel_path).resolve()
        is_valid, reason = verify_checkpoint(abs_path)
        if is_valid:
            size_mb = abs_path.stat().st_size / (1024 * 1024)
            verified.append((model_id, rel_path, size_mb))
            print(f"  [OK] {model_id:22}: {rel_path} ({size_mb:.2f} MB)")
        else:
            missing.append((model_id, spec, abs_path))
            print(f"  [MISSING] {model_id:22}: {rel_path} ({reason})")

    if not missing:
        print(f"\nAll {len(verified)} model checkpoints are present, verified, and operational.")
        return 0

    if args.verify_only:
        print(f"\nError: {len(missing)} model checkpoints are missing or invalid in verify-only mode.", file=sys.stderr)
        return 1

    base_url = args.base_url.rstrip("/") if args.base_url else ""
    allow_synthetic = args.allow_synthetic or (
        os.environ.get("ALLOW_SYNTHETIC_CHECKPOINTS", "").lower() in ("1", "true", "yes")
    )

    failed_models = []
    for model_id, spec, abs_path in missing:
        rel_path = spec["rel_path"]
        env_var_name = spec.get("env_var", "")
        specific_url = os.environ.get(env_var_name, "").strip() if env_var_name else ""

        downloaded = False

        # 1. Try model-specific URL if configured
        if specific_url:
            print(f"\nAttempting download for '{model_id}' via {env_var_name}...")
            if download_file(specific_url, abs_path):
                downloaded = True

        # 2. Try candidate URLs with base_url
        if not downloaded and base_url:
            print(f"\nAttempting download for '{model_id}' from {base_url}...")
            for cand in spec["remote_candidates"]:
                cand_url = f"{base_url}/{cand}"
                if download_file(cand_url, abs_path):
                    downloaded = True
                    break

        # 3. Fallback or record failure
        if not downloaded:
            if allow_synthetic:
                print(f"  [FALLBACK] ALLOW_SYNTHETIC_CHECKPOINTS is active; creating test fixture for {model_id}...")
                create_synthetic_checkpoint(abs_path)
            else:
                failed_models.append((model_id, rel_path))

    if failed_models:
        print(f"\nError: Failed to obtain valid checkpoints for {len(failed_models)} model(s):", file=sys.stderr)
        for mid, rpath in failed_models:
            print(f"  - {mid} (expected at {rpath})", file=sys.stderr)
        print("\nTo resolve in production:", file=sys.stderr)
        print("  Set CHECKPOINTS_BASE_URL (or CHECKPOINT_URL_<MODEL_ID>) in your Render dashboard.", file=sys.stderr)
        print("  Example: CHECKPOINTS_BASE_URL=https://github.com/shyamkailash/-Cancer-Detection-in-Histopathology-Images/releases/download/v1.0.0", file=sys.stderr)
        return 1

    print("\nCheckpoint synchronization complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
