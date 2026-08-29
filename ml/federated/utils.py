"""
Federated Learning utilities for reproducibility, seed management, and efficiency metrics.
"""

import os
import random
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> int:
    """
    Set deterministic seeds across Python, NumPy, and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimate model size in megabytes based on float32 parameter counts."""
    num_params = sum(p.numel() for p in model.parameters())
    num_buffers = sum(b.numel() for b in model.buffers())
    total_elements = num_params + num_buffers
    return float(total_elements * 4 / (1024 * 1024))  # 4 bytes per float32


def estimate_communication_volume(
    num_rounds: int,
    num_participating_clients_per_round: int,
    model_size_mb: float,
) -> Dict[str, Any]:
    """
    Estimate the bidirectional communication volume (Downlink + Uplink).
    Downlink: Server dispatches global model to clients.
    Uplink: Clients send updated model back to server.
    """
    # Downlink: server -> N clients, Uplink: N clients -> server
    # Total per round = 2 * N * model_size_mb
    per_round_mb = 2.0 * num_participating_clients_per_round * model_size_mb
    total_volume_mb = num_rounds * per_round_mb
    return {
        "model_size_mb": round(model_size_mb, 2),
        "rounds": num_rounds,
        "clients_per_round": num_participating_clients_per_round,
        "per_round_mb": round(per_round_mb, 2),
        "total_communication_mb": round(total_volume_mb, 2),
        "total_communication_gb": round(total_volume_mb / 1024.0, 3),
        "note": "Estimated bidirectional network payload (float32 parameter transfers, excluding protocol overhead).",
    }

