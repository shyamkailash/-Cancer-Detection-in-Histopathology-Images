"""
FedProx (Federated Optimization with Proximal Regularization) implementation.
Mitigates client drift in heterogeneous Non-IID multi-site healthcare settings.
Reference: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .client import FederatedClient
from .privacy import PrivacyConfig


def compute_proximal_loss(
    model: nn.Module,
    global_params: Dict[str, torch.Tensor],
    mu: float = 0.01,
) -> torch.Tensor:
    """
    Compute the proximal regularization term: (mu / 2) * sum(||w - w_global||^2).
    Ensures local updates do not drift too far from the global model.

    Args:
        model: Local PyTorch neural network
        global_params: Dictionary of frozen global model parameters
        mu: Proximal regularization coefficient (mu >= 0)

    Returns:
        Scalar proximal loss tensor
    """
    if mu <= 0.0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    proximal_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and name in global_params:
            g_param = global_params[name]
            proximal_loss = proximal_loss + torch.sum((param - g_param) ** 2)

    return (mu / 2.0) * proximal_loss


class FedProxClient(FederatedClient):
    """
    FedProx Client incorporating proximal regularization during local site training.
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
        mu: float = 0.01,
        privacy_config: Optional[PrivacyConfig] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        amp: bool = False,
    ):
        super().__init__(
            client_id=client_id,
            site_name=site_name,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            privacy_config=privacy_config,
            num_workers=num_workers,
            pin_memory=pin_memory,
            amp=amp,
        )
        self.mu = float(mu)

    def train(self, global_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Train local model with proximal regularization for local_epochs.
        """
        # Load global model weights
        self.model.load_state_dict(global_state_dict)
        self.model.train()

        # Create immutable snapshot of global parameters on the local device for proximal calculation
        global_params = {
            name: param.detach().clone().to(self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

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
        total_cls_loss = 0.0
        total_prox_loss = 0.0
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
                    cls_loss = self.criterion(outputs, labels)
                    prox_loss = compute_proximal_loss(self.model, global_params, self.mu)
                    loss = cls_loss + prox_loss

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
                total_cls_loss += cls_loss.item() * batch_size
                total_prox_loss += prox_loss.item() * batch_size
                preds = outputs.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_cls_loss = total_cls_loss / total_samples if total_samples > 0 else 0.0
        avg_prox_loss = total_prox_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        cpu_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        return {
            "client_id": self.client_id,
            "site_name": self.site_name,
            "state_dict": cpu_state_dict,
            "num_samples": len(self.dataset),
            "train_loss": float(avg_loss),
            "train_cls_loss": float(avg_cls_loss),
            "train_prox_loss": float(avg_prox_loss),
            "train_accuracy": float(accuracy),
            "algorithm": "fedprox",
            "mu": self.mu,
            "privacy_enabled": self.privacy_config.enabled,
        }

