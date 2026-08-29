"""
Differential Privacy and Client-Side Gradient Perturbation for Federated Learning.
Implements gradient norm clipping and calibrated Gaussian noise injection.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


@dataclass
class PrivacyConfig:
    """Configuration for client-side differential privacy perturbation."""
    enabled: bool = False
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_grad_norm": self.max_grad_norm,
            "noise_multiplier": self.noise_multiplier,
            "mechanism": "Gaussian DP Prototype (Gradient Clipping + Gaussian Noise)" if self.enabled else "None (Baseline FL)",
        }


class DifferentialPrivacyEngine:
    """
    Client-side Differential Privacy Engine.
    Applies gradient clipping to bound sensitivity and injects Gaussian noise
    before the optimizer updates local model weights.

    Mathematical Mechanism:
      1. Total gradient L2 norm: G = sqrt( sum( ||grad_p||_2^2 ) )
      2. Clipping coefficient: S = min(1.0, max_grad_norm / (G + 1e-6))
      3. Clipped gradient: g_clipped = g * S
      4. Gaussian Noise: n ~ Normal(0, (noise_multiplier * max_grad_norm)^2)
      5. Perturbed gradient: g_final = g_clipped + n
    """

    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()

    def process_gradients(self, model: nn.Module) -> Dict[str, float]:
        """
        Clip gradients and inject calibrated Gaussian noise into model parameter gradients in-place.
        Returns gradient diagnostics (original norm, clipped norm, noise scale).
        """
        if not self.config.enabled:
            return {"grad_norm": 0.0, "noise_scale": 0.0}

        parameters = [p for p in model.parameters() if p.grad is not None]
        if not parameters:
            return {"grad_norm": 0.0, "noise_scale": 0.0}

        device = parameters[0].grad.device
        dtype = parameters[0].grad.dtype

        # 1. Calculate total L2 norm of all gradients
        total_norm_sq = 0.0
        for p in parameters:
            param_norm = p.grad.detach().norm(2).item()
            total_norm_sq += param_norm ** 2
        total_norm = total_norm_sq ** 0.5

        # 2. Compute clipping coefficient
        max_norm = self.config.max_grad_norm
        clip_coef = min(1.0, max_norm / (total_norm + 1e-6))

        # 3. Clip and add Gaussian noise
        noise_std = self.config.noise_multiplier * max_norm

        for p in parameters:
            # Clip
            p.grad.detach().mul_(clip_coef)

            # Add Gaussian noise
            if noise_std > 0:
                noise = torch.randn(p.grad.shape, device=device, dtype=dtype) * noise_std
                p.grad.detach().add_(noise)

        return {
            "original_grad_norm": float(total_norm),
            "clipped_grad_norm": float(total_norm * clip_coef),
            "noise_std": float(noise_std),
        }

