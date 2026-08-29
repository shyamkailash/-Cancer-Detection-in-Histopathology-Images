"""
Federated Averaging (FedAvg) aggregation implementation.
"""

from typing import List, Tuple, Dict
import torch


def federated_averaging(
    client_updates: List[Tuple[Dict[str, torch.Tensor], int]]
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted Federated Averaging (FedAvg) over client model state dictionaries.

    Formula:
        w_global = sum( (n_k / N) * w_k for k in clients )

    Args:
        client_updates: List of tuples (client_state_dict, client_sample_count).

    Returns:
        Aggregated global state_dict.
    """
    if not client_updates:
        raise ValueError("Cannot perform FedAvg on an empty list of client updates.")

    total_samples = sum(num_samples for _, num_samples in client_updates)
    if total_samples <= 0:
        raise ValueError(f"Total client sample count must be positive. Got {total_samples}")

    # Use first client's state dict keys as template
    first_state_dict, _ = client_updates[0]
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key, template_tensor in first_state_dict.items():
        if template_tensor.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            # Weighted sum for floating point parameters and buffers
            weighted_sum = torch.zeros_like(template_tensor, dtype=template_tensor.dtype)
            for state_dict, num_samples in client_updates:
                weight = float(num_samples) / float(total_samples)
                tensor_val = state_dict[key].to(template_tensor.device)
                weighted_sum.add_(tensor_val, alpha=weight)
            aggregated_state[key] = weighted_sum
        else:
            # For non-floating point buffers (e.g. num_batches_tracked), copy from first update
            aggregated_state[key] = template_tensor.clone()

    return aggregated_state

