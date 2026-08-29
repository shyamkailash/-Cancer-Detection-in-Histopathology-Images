"""
FedBN (Federated Learning with Local Batch Normalization) implementation.
Keeps BatchNorm parameters and running statistics strictly local to each healthcare site
to mitigate domain and staining shift across hospital slide scanners.
Reference: Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization", ICLR 2021.
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .client import FederatedClient
from .privacy import PrivacyConfig


def get_bn_module_names(model: nn.Module) -> Set[str]:
    """Identify all BatchNorm module names in a PyTorch model dynamically."""
    bn_names = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_names.add(name)
    return bn_names


def is_bn_parameter_or_buffer(key: str, bn_module_names: Optional[Set[str]] = None) -> bool:
    """
    Check if a state_dict key corresponds to a BatchNorm parameter or buffer.
    """
    if bn_module_names is not None:
        parts = key.split(".")
        # Check all possible prefix paths
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix in bn_module_names:
                return True

    # Generic heuristics for standard torchvision models (e.g. ResNet)
    key_lower = key.lower()
    if any(bn_kw in key_lower for bn_kw in [".bn", "bn1.", "bn2.", "downsample.1.", "running_mean", "running_var", "num_batches_tracked"]):
        return True

    return False


def get_bn_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract all BatchNorm parameters and tracking buffers from model."""
    bn_modules = get_bn_module_names(model)
    full_state = model.state_dict()
    return {
        k: v.cpu().clone()
        for k, v in full_state.items()
        if is_bn_parameter_or_buffer(k, bn_modules)
    }


def get_non_bn_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract all non-BatchNorm parameters from model."""
    bn_modules = get_bn_module_names(model)
    full_state = model.state_dict()
    return {
        k: v.cpu().clone()
        for k, v in full_state.items()
        if not is_bn_parameter_or_buffer(k, bn_modules)
    }


def aggregate_fedbn(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    base_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    bn_module_names: Optional[Set[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate only non-BatchNorm parameters across client updates using sample weighting.
    BatchNorm parameters and running statistics are NOT aggregated globally.

    Args:
        client_updates: List of (client_state_dict, num_samples)
        base_state_dict: Optional base global state dict to retain global BatchNorm state
        bn_module_names: Set of module names representing BatchNorm layers

    Returns:
        Aggregated state_dict where non-BN parameters are averaged and BN parameters are retained.
    """
    if not client_updates:
        raise ValueError("Cannot aggregate empty client updates list.")

    total_samples = sum(samples for _, samples in client_updates)
    if total_samples <= 0:
        raise ValueError(f"Invalid total samples for aggregation: {total_samples}")

    first_state, _ = client_updates[0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key, val in first_state.items():
        if is_bn_parameter_or_buffer(key, bn_module_names):
            # Do NOT average BatchNorm parameters or running statistics
            if base_state_dict and key in base_state_dict:
                aggregated_state[key] = base_state_dict[key].clone()
            else:
                aggregated_state[key] = val.clone()
        else:
            # Weighted average for convolutional and linear weights
            if torch.is_floating_point(val):
                weighted_sum = torch.zeros_like(val, dtype=torch.float32)
                for client_state, num_samples in client_updates:
                    weight = float(num_samples) / float(total_samples)
                    weighted_sum += client_state[key].to(val.device, dtype=torch.float32) * weight
                aggregated_state[key] = weighted_sum.to(dtype=val.dtype)
            else:
                aggregated_state[key] = val.clone()

    return aggregated_state


class FedBNClient(FederatedClient):
    """
    FedBN Client that retains its own local BatchNorm parameters and statistics
    across communication rounds to preserve site-specific domain adaptation.
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
        self.bn_module_names = get_bn_module_names(self.model)
        # Initialize persistent local BatchNorm state
        self.local_bn_state: Dict[str, torch.Tensor] = get_bn_state(self.model)

    def train(self, global_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Load global non-BN weights, restore local site-specific BN parameters, and train locally.
        """
        current_state = self.model.state_dict()

        # Update non-BN weights from global server, while keeping local BN parameters
        merged_state = {}
        for k, v in global_state_dict.items():
            if is_bn_parameter_or_buffer(k, self.bn_module_names):
                # Keep local client's BatchNorm parameters and running statistics
                merged_state[k] = self.local_bn_state.get(k, current_state[k]).to(self.device)
            else:
                # Load globally synchronized representation
                merged_state[k] = v.to(self.device)

        self.model.load_state_dict(merged_state)
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

        # Save updated site-specific BN state
        self.local_bn_state = get_bn_state(self.model)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        cpu_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        return {
            "client_id": self.client_id,
            "site_name": self.site_name,
            "state_dict": cpu_state_dict,
            "num_samples": len(self.dataset),
            "train_loss": float(avg_loss),
            "train_accuracy": float(accuracy),
            "algorithm": "fedbn",
            "privacy_enabled": self.privacy_config.enabled,
        }

