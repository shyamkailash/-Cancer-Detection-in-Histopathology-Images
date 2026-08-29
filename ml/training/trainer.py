"""
Training and evaluation utilities for PCam ResNet-18 classification.
"""

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


class Trainer:
    """Train and evaluate a PyTorch image classification model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_one_epoch(self, loader) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Dict[str, float],
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )