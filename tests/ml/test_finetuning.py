"""
Unit and integration tests for Centralized Fine-Tuning functionality, CLI, and auto-promotion.
"""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ml.models.resnet import create_resnet18
from ml.training.finetuner import (
    freeze_backbone,
    apply_eval_to_frozen_bn,
    get_discriminative_param_groups,
    CentralizedFineTuner,
)
from scripts.finetune_centralized import (
    parse_args,
    check_validation_quality_gate,
    promote_candidate_checkpoint,
)


def create_synthetic_data(n_samples: int = 16):
    x = torch.randn(n_samples, 3, 96, 96)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(x, y)


def test_cli_help_and_defaults(monkeypatch):
    """Test CLI argument parsing and default values."""
    monkeypatch.setattr("sys.argv", ["finetune_centralized.py"])
    args = parse_args()

    assert args.stage1_epochs == 1
    assert args.stage2_epochs == 2
    assert args.batch_size == 64
    assert args.head_lr == 3e-4
    assert args.layer4_lr == 3e-5
    assert args.backbone_lr == 1e-5
    assert args.seed == 42
    assert args.auto_promote is False


def test_stage1_layer_freezing():
    """Verify Stage 1 freezes all convolutional layers except fc (1,026 params)."""
    model = create_resnet18(num_classes=2, pretrained=False)
    freeze_backbone(model, freeze=True, unfreeze_layer4=False, unfreeze_all=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 1026  # 512 * 2 + 2 = 1026

    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_stage2_layer4_and_fc_trainable_and_layer123_frozen():
    """Verify Stage 2 unfreezes layer4 and fc while keeping layer1/2/3 frozen (8,394,754 params)."""
    model = create_resnet18(num_classes=2, pretrained=False)
    freeze_backbone(model, freeze=True, unfreeze_layer4=True, unfreeze_all=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == 8394754

    for name, param in model.named_parameters():
        if "fc" in name or "layer4" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_backbone_lr_greater_than_zero_does_not_unfreeze_backbone():
    """Verify that backbone_lr > 0 does NOT unfreeze layer1-layer3 in standard Stage 2."""
    model = create_resnet18(num_classes=2, pretrained=False)
    # Standard Stage 2 unfreezing explicitly sets unfreeze_all=False
    freeze_backbone(model, freeze=True, unfreeze_layer4=True, unfreeze_all=False)

    # Check layer1, layer2, layer3 parameters are strictly frozen
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ["conv1", "bn1", "layer1", "layer2", "layer3"]):
            assert param.requires_grad is False
        elif any(name.startswith(p) for p in ["layer4", "fc"]):
            assert param.requires_grad is True

    # Check discriminative param groups only includes fc and layer4
    groups = get_discriminative_param_groups(
        model,
        head_lr=3e-4,
        layer4_lr=3e-5,
        backbone_lr=1e-5,
    )
    assert len(groups) == 2
    assert groups[0]["lr"] == 3e-4  # fc
    assert groups[1]["lr"] == 3e-5  # layer4


def test_frozen_batchnorm_behavior():
    """Verify BatchNorm layers in frozen blocks remain in eval mode during training."""
    model = create_resnet18(num_classes=2, pretrained=False)

    # Stage 1: All backbone frozen
    freeze_backbone(model, freeze=True, unfreeze_layer4=False, unfreeze_all=False)
    model.train()
    apply_eval_to_frozen_bn(model)

    assert model.bn1.training is False
    assert model.layer1[0].bn1.training is False
    assert model.layer2[0].bn1.training is False
    assert model.layer3[0].bn1.training is False
    assert model.layer4[0].bn1.training is False
    assert model.fc.training is True

    # Stage 2: Layer4 + FC trainable, Layer 1-3 frozen
    freeze_backbone(model, freeze=True, unfreeze_layer4=True, unfreeze_all=False)
    model.train()
    apply_eval_to_frozen_bn(model)

    assert model.bn1.training is False
    assert model.layer1[0].bn1.training is False
    assert model.layer2[0].bn1.training is False
    assert model.layer3[0].bn1.training is False
    assert model.layer4[0].bn1.training is True
    assert model.fc.training is True


def test_validation_quality_gate():
    """Verify validation quality gate criteria."""
    init_val = {
        "accuracy": 0.9427,
        "sensitivity": 0.8980,
        "roc_auc": 0.9825,
    }

    # Improved metrics -> Pass
    better_val = {
        "accuracy": 0.9520,
        "sensitivity": 0.9400,
        "roc_auc": 0.9880,
    }
    passed, reason = check_validation_quality_gate(init_val, better_val)
    assert passed is True

    # Regressed accuracy -> Fail
    regressed_val = {
        "accuracy": 0.9300,
        "sensitivity": 0.8900,
        "roc_auc": 0.9700,
    }
    passed, reason = check_validation_quality_gate(init_val, regressed_val)
    assert passed is False


def test_auto_promotion_path_safety_and_no_undefined_name(tmp_path, monkeypatch):
    """Verify promote_candidate_checkpoint executes safely without NameError."""
    base_ckpt_path = tmp_path / "checkpoints" / "pcam_resnet18_best.pt"
    base_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_ckpt_path = tmp_path / "finetuning" / "best_model.pt"
    candidate_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_model = create_resnet18(num_classes=2, pretrained=False)
    torch.save({"model_state_dict": dummy_model.state_dict(), "epoch": 1}, base_ckpt_path)
    torch.save({"model_state_dict": dummy_model.state_dict(), "epoch": 2}, candidate_ckpt_path)

    # Monkeypatch centralized report path to temp directory
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "centralized_metrics.json"

    val_metrics = {"accuracy": 0.955, "loss": 0.130, "sensitivity": 0.945, "roc_auc": 0.988}
    test_metrics = {"accuracy": 0.957, "loss": 0.125, "sensitivity": 0.948, "roc_auc": 0.989}
    config = {"stage1_epochs": 1, "stage2_epochs": 2, "batch_size": 64, "total_duration_seconds": 120.0}

    # Execute promote_candidate_checkpoint with test report path
    success = promote_candidate_checkpoint(
        candidate_checkpoint_path=candidate_ckpt_path,
        baseline_checkpoint_path=base_ckpt_path,
        val_metrics=val_metrics,
        candidate_test_metrics=test_metrics,
        config=config,
        best_epoch=2,
        train_history=[{"train_loss": 0.14, "train_accuracy": 0.95}],
        num_samples_train=100,
        num_samples_val=20,
        num_samples_test=20,
        device_str="cpu",
        gpu_name=None,
        report_path=report_file,
    )

    assert success is True
    # Verify backup exists
    backup_path = tmp_path / "checkpoints" / "pcam_resnet18_best.backup.pt"
    assert backup_path.exists()
    # Verify promoted checkpoint exists and has metadata
    promoted_data = torch.load(base_ckpt_path, map_location="cpu", weights_only=False)
    assert promoted_data["promoted_from_finetuning"] is True
    assert promoted_data["epoch"] == 2


def test_centralized_finetuner_cpu_and_artifacts(tmp_path):
    """Test full 2-stage fine-tuning execution and artifact saving on CPU."""
    device = torch.device("cpu")
    model = create_resnet18(num_classes=2, pretrained=False)

    train_loader = DataLoader(create_synthetic_data(12), batch_size=4)
    val_loader = DataLoader(create_synthetic_data(8), batch_size=4)

    finetuner = CentralizedFineTuner(
        model=model,
        device=device,
        output_dir=str(tmp_path / "finetuning"),
        amp=False,
    )

    results = finetuner.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        stage1_epochs=1,
        stage2_epochs=1,
        head_lr=1e-3,
        layer4_lr=1e-4,
        backbone_lr=1e-5,
    )

    assert "best_epoch" in results
    assert "best_val_accuracy" in results
    assert len(results["history"]) == 2
    assert (tmp_path / "finetuning" / "best_stage1.pt").exists()
    assert (tmp_path / "finetuning" / "best_stage2.pt").exists()
    assert (tmp_path / "finetuning" / "best_model.pt").exists()
