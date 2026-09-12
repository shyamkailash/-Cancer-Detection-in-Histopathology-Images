"""
Centralized Fine-Tuning Engine for PCam ResNet-18.
Supports 2-stage transfer learning fine-tuning:
- Stage 1: Frozen backbone with classifier head calibration (Backbone BN in eval mode).
- Stage 2: Discriminative layer fine-tuning for Layer 4 + Head (Layer 1-3 remain frozen in eval mode).
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.evaluation.metrics import evaluate_model


def freeze_backbone(
    model: nn.Module,
    freeze: bool = True,
    unfreeze_layer4: bool = False,
    unfreeze_all: bool = False,
) -> None:
    """
    Freeze or unfreeze backbone layers while keeping classification head trainable.

    Args:
        model: PyTorch ResNet-18 model.
        freeze: If True, freeze backbone layers.
        unfreeze_layer4: If True, keep layer4 trainable along with fc.
        unfreeze_all: If True, make all model parameters trainable.
    """
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        elif unfreeze_layer4 and "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = not freeze


def apply_eval_to_frozen_bn(model: nn.Module) -> None:
    """
    Set BatchNorm layers to eval() mode if their parameters are frozen.
    This prevents running mean and running variance statistics from updating
    during training when backbone feature extractors are frozen.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
            if not any(p.requires_grad for p in module.parameters()):
                module.eval()


def get_discriminative_param_groups(
    model: nn.Module,
    head_lr: float = 3e-4,
    layer4_lr: float = 3e-5,
    backbone_lr: float = 1e-5,
    weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """
    Build parameter groups with layer-specific discriminative learning rates.
    For Stage 2 (where only layer4 and fc have requires_grad=True), returns
    groups for fc and layer4.
    """
    fc_params = []
    layer4_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            fc_params.append(param)
        elif "layer4" in name:
            layer4_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if fc_params:
        groups.append({"params": fc_params, "lr": head_lr, "weight_decay": weight_decay})
    if layer4_params:
        groups.append({"params": layer4_params, "lr": layer4_lr, "weight_decay": weight_decay})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})

    return groups


class CentralizedFineTuner:
    """
    Manages safe fine-tuning, validation monitoring, and checkpoint selection.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = "artifacts/finetuning/centralized",
        max_grad_norm: float = 1.0,
        amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        self.amp = amp and device.type == "cuda"
        self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Train model for a single epoch with frozen BatchNorm protection."""
        self.model.train()
        apply_eval_to_frozen_bn(self.model)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = labels.size(0)

            optimizer.zero_grad(set_to_none=True)

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type=dev_type, enabled=self.amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=self.amp)

            with autocast_ctx:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.amp and scaler is not None:
                scaler.scale(loss).backward()
                if self.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()

            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        }

    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        stage1_epochs: int = 1,
        stage2_epochs: int = 2,
        head_lr: float = 3e-4,
        layer4_lr: float = 3e-5,
        backbone_lr: float = 1e-5,
        weight_decay: float = 1e-4,
        patience: int = 2,
    ) -> Dict[str, Any]:
        """
        Execute full 2-stage fine-tuning schedule.
        """
        total_start = time.time()
        history: List[Dict[str, Any]] = []

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        best_val_metrics = None

        # Evaluate initial model state before training
        print("\n--- Initial Validation Before Fine-Tuning ---")
        init_val = evaluate_model(self.model, val_loader, criterion=self.criterion, device=self.device)
        print(f"Initial Val Loss: {init_val['loss']:.4f} | Val Acc: {init_val['accuracy'] * 100:.2f}% | Sensitivity: {init_val['sensitivity'] * 100:.2f}% | ROC-AUC: {init_val.get('roc_auc', 0):.4f}")
        best_val_acc = init_val["accuracy"]
        best_val_loss = init_val["loss"]
        best_val_metrics = init_val

        global_epoch = 0

        # =====================================================================
        # Stage 1: Fine-tune classification head with frozen backbone
        # =====================================================================
        if stage1_epochs > 0:
            print("\n" + "=" * 65)
            print(f"Stage 1: Classifier Head Calibration ({stage1_epochs} Epochs, Backbone Frozen)")
            print("=" * 65)

            freeze_backbone(self.model, freeze=True, unfreeze_layer4=False, unfreeze_all=False)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable Parameters (Head only): {trainable_params:,}")

            opt_stage1 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=head_lr,
                weight_decay=weight_decay,
            )

            for epoch in range(1, stage1_epochs + 1):
                global_epoch += 1
                t0 = time.time()
                print(f"\n[Stage 1] Epoch {global_epoch} (Stage Epoch {epoch}/{stage1_epochs})")

                train_metrics = self.train_one_epoch(train_loader, opt_stage1, scaler)
                val_metrics = evaluate_model(self.model, val_loader, criterion=self.criterion, device=self.device)
                duration = time.time() - t0

                gpu_mem = None
                if torch.cuda.is_available() and self.device.type == "cuda":
                    gpu_mem = round(torch.cuda.memory_allocated(self.device) / (1024 ** 2), 1)

                print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy'] * 100:.2f}%")
                print(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy'] * 100:.2f}%")
                print(f"Val Sens:   {val_metrics['sensitivity'] * 100:.2f}% | Val Spec:  {val_metrics['specificity'] * 100:.2f}% | Val AUC: {val_metrics.get('roc_auc', 0):.4f}")
                print(f"Epoch Time: {duration:.2f}s | GPU Mem: {gpu_mem} MB" if gpu_mem else f"Epoch Time: {duration:.2f}s")

                rec = {
                    "stage": 1,
                    "epoch": global_epoch,
                    "learning_rate": head_lr,
                    "duration_seconds": duration,
                    "gpu_memory_mb": gpu_mem,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_sensitivity": val_metrics["sensitivity"],
                    "val_specificity": val_metrics["specificity"],
                    "val_precision": val_metrics["precision"],
                    "val_f1_score": val_metrics["f1_score"],
                    "val_roc_auc": val_metrics.get("roc_auc"),
                }
                history.append(rec)

                # Check for improvement
                if val_metrics["accuracy"] >= best_val_acc or val_metrics["loss"] < best_val_loss:
                    if val_metrics["accuracy"] > best_val_acc or (val_metrics["accuracy"] == best_val_acc and val_metrics["loss"] < best_val_loss):
                        best_val_acc = val_metrics["accuracy"]
                        best_val_loss = val_metrics["loss"]
                        best_epoch = global_epoch
                        best_val_metrics = val_metrics
                        best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        print(f"[*] New best model at Epoch {global_epoch} (Val Acc: {best_val_acc * 100:.2f}%, Val Loss: {best_val_loss:.4f})")

            # Save Stage 1 best checkpoint
            stage1_ckpt_path = self.output_dir / "best_stage1.pt"
            torch.save(
                {
                    "epoch": best_epoch,
                    "stage": 1,
                    "model_state_dict": best_state_dict,
                    "val_metrics": best_val_metrics,
                },
                stage1_ckpt_path,
            )
            print(f"[SAVED] Stage 1 checkpoint: {stage1_ckpt_path}")

        # =====================================================================
        # Stage 2: Discriminative fine-tuning of deep features (layer4 + fc)
        # =====================================================================
        if stage2_epochs > 0:
            print("\n" + "=" * 65)
            print(f"Stage 2: Discriminative Fine-Tuning ({stage2_epochs} Epochs, Layer4 + Classifier)")
            print("=" * 65)

            # Unfreeze only layer4 and fc (layer1, layer2, layer3 remain frozen)
            freeze_backbone(self.model, freeze=True, unfreeze_layer4=True, unfreeze_all=False)
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Trainable Parameters (Layer4 + Head): {trainable_params:,}")

            param_groups = get_discriminative_param_groups(
                self.model,
                head_lr=head_lr * 0.5,
                layer4_lr=layer4_lr,
                backbone_lr=backbone_lr,
                weight_decay=weight_decay,
            )

            opt_stage2 = torch.optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_stage2,
                T_max=stage2_epochs,
                eta_min=1e-6,
            )

            patience_counter = 0

            for epoch in range(1, stage2_epochs + 1):
                global_epoch += 1
                t0 = time.time()
                current_lrs = [pg["lr"] for pg in opt_stage2.param_groups]
                print(f"\n[Stage 2] Epoch {global_epoch} (Stage Epoch {epoch}/{stage2_epochs}) | LRs: {[f'{lr:.2e}' for lr in current_lrs]}")

                train_metrics = self.train_one_epoch(train_loader, opt_stage2, scaler)
                val_metrics = evaluate_model(self.model, val_loader, criterion=self.criterion, device=self.device)
                scheduler.step()
                duration = time.time() - t0

                gpu_mem = None
                if torch.cuda.is_available() and self.device.type == "cuda":
                    gpu_mem = round(torch.cuda.memory_allocated(self.device) / (1024 ** 2), 1)

                print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy'] * 100:.2f}%")
                print(f"Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy'] * 100:.2f}%")
                print(f"Val Sens:   {val_metrics['sensitivity'] * 100:.2f}% | Val Spec:  {val_metrics['specificity'] * 100:.2f}% | Val AUC: {val_metrics.get('roc_auc', 0):.4f}")
                print(f"Epoch Time: {duration:.2f}s | GPU Mem: {gpu_mem} MB" if gpu_mem else f"Epoch Time: {duration:.2f}s")

                rec = {
                    "stage": 2,
                    "epoch": global_epoch,
                    "learning_rate": current_lrs[0],
                    "duration_seconds": duration,
                    "gpu_memory_mb": gpu_mem,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_sensitivity": val_metrics["sensitivity"],
                    "val_specificity": val_metrics["specificity"],
                    "val_precision": val_metrics["precision"],
                    "val_f1_score": val_metrics["f1_score"],
                    "val_roc_auc": val_metrics.get("roc_auc"),
                }
                history.append(rec)

                # Check for improvement
                improved = False
                if val_metrics["accuracy"] > best_val_acc:
                    improved = True
                elif val_metrics["accuracy"] == best_val_acc and val_metrics["loss"] < best_val_loss:
                    improved = True

                if improved:
                    best_val_acc = val_metrics["accuracy"]
                    best_val_loss = val_metrics["loss"]
                    best_epoch = global_epoch
                    best_val_metrics = val_metrics
                    best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                    print(f"[*] New best model at Epoch {global_epoch} (Val Acc: {best_val_acc * 100:.2f}%, Val Loss: {best_val_loss:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\n[EARLY STOPPING] Validation did not improve for {patience} consecutive epochs.")
                        break

            # Save Stage 2 best checkpoint
            stage2_ckpt_path = self.output_dir / "best_stage2.pt"
            torch.save(
                {
                    "epoch": best_epoch,
                    "stage": 2,
                    "model_state_dict": best_state_dict,
                    "val_metrics": best_val_metrics,
                },
                stage2_ckpt_path,
            )
            print(f"[SAVED] Stage 2 checkpoint: {stage2_ckpt_path}")

        total_duration = time.time() - total_start

        # Save candidate best model
        candidate_path = self.output_dir / "best_model.pt"
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": best_state_dict,
                "val_metrics": best_val_metrics,
                "history": history,
                "total_training_time_seconds": total_duration,
            },
            candidate_path,
        )
        print(f"\n[SAVED] Fine-tuned candidate model: {candidate_path.resolve()}")

        # Load best model back into instance for testing
        self.model.load_state_dict(best_state_dict)

        return {
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_acc,
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_val_metrics,
            "init_val_metrics": init_val,
            "total_duration_seconds": total_duration,
            "history": history,
            "candidate_checkpoint": str(candidate_path),
        }
