#!/usr/bin/env bash
"""Prune unneeded optimizer buffers from trained checkpoints for production deployment."""

from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prune_checkpoint(
    input_path: Path,
    output_path: Path,
    keep_keys=("model_state_dict", "state_dict", "epoch", "metrics"),
) -> None:
    if not input_path.exists():
        print(f"Error: {input_path} does not exist.")
        sys.exit(1)

    print(f"Loading {input_path} ({input_path.stat().st_size / (1024 * 1024):.2f} MB)...")
    checkpoint = torch.load(str(input_path), map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        print(f"Error: checkpoint {input_path} is not a dictionary.")
        sys.exit(1)

    pruned = {k: v for k, v in checkpoint.items() if k in keep_keys}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pruned, str(output_path))
    print(f"Saved pruned checkpoint to {output_path} ({output_path.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    src = PROJECT_ROOT / "artifacts" / "checkpoints" / "pcam_resnet18_best.pt"
    dest = src
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    if len(sys.argv) > 2:
        dest = Path(sys.argv[2])

    prune_checkpoint(src, dest)
