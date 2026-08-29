"""
Evaluation metrics package for histopathology cancer detection.
"""

from .metrics import calculate_classification_metrics, evaluate_model

__all__ = ["calculate_classification_metrics", "evaluate_model"]

