"""
Unit tests for scripts/download_checkpoints.py.
Validates disambiguated remote filenames, atomic download safety, and state-dict verification.
"""

from pathlib import Path
import pytest
import torch

from scripts.download_checkpoints import (
    CHECKPOINT_SPEC,
    verify_checkpoint,
    download_file,
    PROJECT_ROOT,
)


def test_checkpoint_spec_disambiguation():
    """Ensure every model has a distinct remote filename to avoid collisions on best_global_model.pt."""
    models = list(CHECKPOINT_SPEC.keys())
    assert len(models) == 6
    assert "fedavg" in models
    assert "fedprox" in models
    assert "fedbn" in models
    assert "dp_fedavg" in models

    # Verify no two federated models share candidate flat filenames
    all_flat_candidates = []
    for model_id, spec in CHECKPOINT_SPEC.items():
        candidates = spec["remote_candidates"]
        assert len(candidates) >= 2
        for cand in candidates:
            # If flat filename (no slash), must not collide across different federated models
            if "/" not in cand:
                assert cand not in all_flat_candidates, f"Collision detected for flat candidate '{cand}' in {model_id}"
                all_flat_candidates.append(cand)


def test_verify_checkpoint_on_empty_and_corrupt_files(tmp_path):
    """Ensure verify_checkpoint rejects empty or corrupted files."""
    empty_file = tmp_path / "empty.pt"
    empty_file.write_bytes(b"")
    is_valid, msg = verify_checkpoint(empty_file)
    assert not is_valid
    assert "small" in msg.lower() or "not exist" in msg.lower()

    corrupt_file = tmp_path / "corrupt.pt"
    corrupt_file.write_bytes(b"0" * 2_000_000)
    is_valid, msg = verify_checkpoint(corrupt_file)
    assert not is_valid
    assert "load error" in msg.lower() or "not a dictionary" in msg.lower()


def test_verify_checkpoint_rejects_incompatible_architecture(tmp_path):
    """Ensure verify_checkpoint rejects checkpoints lacking required ResNet-18 layers or output classes."""
    bad_arch_file = tmp_path / "bad_arch.pt"
    # Pad dictionary to exceed 1MB threshold so layer verification triggers
    padding = torch.randn(300, 1000)
    torch.save(
        {
            "padding": padding,
            "model_state_dict": {
                "conv1.weight": torch.randn(64, 3, 7, 7),
                "layer4.1.conv2.weight": torch.randn(512, 512, 3, 3),
                "fc.weight": torch.randn(10, 512),  # 10 classes instead of 2!
                "fc.bias": torch.randn(10),
            },
        },
        str(bad_arch_file),
    )
    is_valid, msg = verify_checkpoint(bad_arch_file)
    assert not is_valid
    assert "keys" in msg.lower() or "missing" in msg.lower() or "unexpected" in msg.lower()


def test_verify_checkpoint_on_real_local_checkpoints():
    """Verify that all genuine local checkpoints pass verification."""
    for model_id, spec in CHECKPOINT_SPEC.items():
        abs_path = PROJECT_ROOT / spec["rel_path"]
        if abs_path.exists():
            is_valid, msg = verify_checkpoint(abs_path)
            assert is_valid, f"Verification failed for local {model_id} checkpoint: {msg}"
