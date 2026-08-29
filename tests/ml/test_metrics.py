"""
Tests for Clinical and Statistical Evaluation Metrics.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ml.evaluation.metrics import calculate_classification_metrics, evaluate_model, compute_roc_auc


def test_calculate_classification_metrics_perfect():
    """Test metrics on perfectly predicted targets."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1]
    y_probs = [0.1, 0.2, 0.15, 0.9, 0.85, 0.95]

    metrics = calculate_classification_metrics(y_true, y_pred, y_probs)

    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["tp"] == 3
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_calculate_classification_metrics_with_errors():
    """Test metrics with known false positives and false negatives."""
    # TN=2, FP=1, FN=1, TP=2
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 1]

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 4 / 6
    assert metrics["sensitivity"] == 2 / 3  # TP / (TP + FN)
    assert metrics["specificity"] == 2 / 3  # TN / (TN + FP)
    assert metrics["precision"] == 2 / 3    # TP / (TP + FP)
    assert round(metrics["f1_score"], 4) == round(2 * (2/3) * (2/3) / (4/3), 4)


def test_compute_roc_auc_corner_cases():
    """Test ROC-AUC computation with inverted and single-class cases."""
    # Inverted predictions
    y_true = [0, 0, 1, 1]
    y_score = [0.9, 0.8, 0.2, 0.1]
    auc = compute_roc_auc(np.array(y_true), np.array(y_score))
    assert auc == 0.0

    # Single class only
    auc_single = compute_roc_auc(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert auc_single is None


def test_evaluate_model_with_dummy_model():
    """Test evaluate_model helper with dummy PyTorch model and DataLoader."""
    # Simple linear model mapping 2 input features to 2 output classes
    model = nn.Sequential(nn.Linear(4, 2))

    x = torch.randn(10, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5)

    metrics = evaluate_model(model, loader)

    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "sensitivity" in metrics
    assert "specificity" in metrics
    assert "precision" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["total_samples"] == 10

