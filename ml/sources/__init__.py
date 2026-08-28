"""
Dataset Source Adapters for histopathology data ingestion.
"""

from .base import DatasetSource
from .pcam import PCamSource
from .breakhis import BreakHisSource

__all__ = [
    "DatasetSource",
    "PCamSource",
    "BreakHisSource",
]
