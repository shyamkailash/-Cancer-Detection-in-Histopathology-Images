"""
Privacy-Preserving Federated Learning package for Multi-Site Histopathology Cancer Detection.
"""

from .partition import FederatedPartitioner, SitePartition
from .privacy import DifferentialPrivacyEngine, PrivacyConfig
from .client import FederatedClient
from .aggregation import federated_averaging
from .server import FederatedServer

__all__ = [
    "FederatedPartitioner",
    "SitePartition",
    "DifferentialPrivacyEngine",
    "PrivacyConfig",
    "FederatedClient",
    "federated_averaging",
    "FederatedServer",
]

