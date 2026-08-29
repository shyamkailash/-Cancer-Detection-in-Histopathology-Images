"""
Privacy-Preserving Federated Learning package for Multi-Site Histopathology Cancer Detection.
"""

from .partition import FederatedPartitioner, SitePartition
from .privacy import DifferentialPrivacyEngine, PrivacyConfig
from .client import FederatedClient
from .aggregation import federated_averaging
from .server import FederatedServer
from .fedprox import FedProxClient, compute_proximal_loss
from .fedbn import (
    FedBNClient,
    aggregate_fedbn,
    get_bn_state,
    get_non_bn_state,
    get_bn_module_names,
    is_bn_parameter_or_buffer,
)
from .utils import set_seed, estimate_model_size_mb, estimate_communication_volume

__all__ = [
    "FederatedPartitioner",
    "SitePartition",
    "DifferentialPrivacyEngine",
    "PrivacyConfig",
    "FederatedClient",
    "federated_averaging",
    "FederatedServer",
    "FedProxClient",
    "compute_proximal_loss",
    "FedBNClient",
    "aggregate_fedbn",
    "get_bn_state",
    "get_non_bn_state",
    "get_bn_module_names",
    "is_bn_parameter_or_buffer",
    "set_seed",
    "estimate_model_size_mb",
    "estimate_communication_volume",
]
