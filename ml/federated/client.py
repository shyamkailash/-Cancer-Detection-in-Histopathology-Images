"""
Local Healthcare Site Federated Client for distributed training.
Trains local ResNet-18 model on isolated site data and transmits only model updates.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml.models.resnet import create_resnet18
from .privacy import DifferentialPrivacyEngine, PrivacyConfig


class FederatedClient:
    """
    Simulated healthcare institution client (e.g. Hospital site).
    Performs local training on its isolated dataset without sharing raw histopathology images or patient labels.
    """

    def __init__(
        self,
        client_id: str,
        site_name: str,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 32,
        local_epochs: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        privacy_config: Optional[PrivacyConfig] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        amp: bool = False,
    ):
        self.client_id = client_id
        self.site_name = site_name
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.privacy_config = privacy_config or PrivacyConfig()
        self.dp_engine = DifferentialPrivacyEngine(self.privacy_config)
        self.num_workers = num_workers
        self.pin_memory = pin_memory and device.type == "cuda"
        self.amp = amp and device.type == "cuda"

        # Local model instance
        self.model = create_resnet18(num_classes=2, pretrained=False).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Train local model starting from the global model state_dict for local_epochs.
        Returns:
            Dictionary containing updated model state_dict, sample count, and training loss/accuracy.
            Raw images and patient records are never transmitted.
        """
        # Load global model weights
        self.model.load_state_dict(global_state_dict)
        self.model.train()

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        dev_type = "cuda" if self.device.type == "cuda" else "cpu"

        for epoch in range(self.local_epochs):
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

                if self.amp:
                    scaler.scale(loss).backward()
                    if self.privacy_config.enabled:
                        scaler.unscale_(optimizer)
                        self.dp_engine.process_gradients(self.model)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.privacy_config.enabled:
                        self.dp_engine.process_gradients(self.model)
                    optimizer.step()

                total_loss += loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Move state dict to CPU before returning to aggregator
        cpu_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        return {
            "client_id": self.client_id,
            "site_name": self.site_name,
            "state_dict": cpu_state_dict,
            "num_samples": len(self.dataset),
            "train_loss": float(avg_loss),
            "train_accuracy": float(accuracy),
            "privacy_enabled": self.privacy_config.enabled,
        }
