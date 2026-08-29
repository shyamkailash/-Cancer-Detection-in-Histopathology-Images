"""
Clinical and Statistical Evaluation Metrics for Histopathology Cancer Detection.
"""

from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
    using exact rank-sum / Mann-Whitney U formulation.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]

    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    if n_pos == 0 or n_neg == 0:
        return None

    # Rank scores with tie-handling
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Average ranks for ties
    sorted_scores = y_score[order]
    unique_scores, inverse, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    tie_indices = np.where(counts > 1)[0]
    for tie in tie_indices:
        tie_ranks = np.where(inverse == tie)[0]
        mean_rank = np.mean(tie_ranks + 1)
        ranks[order[tie_ranks]] = mean_rank

    sum_pos_ranks = np.sum(ranks[pos_indices])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def calculate_classification_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive medical classification metrics for cancer detection:
    - Accuracy
    - Sensitivity / Recall (Metastasis detection rate: TP / (TP + FN))
    - Specificity (Normal tissue detection rate: TN / (TN + FP))
    - Precision (Positive predictive value: TP / (TP + FP))
    - F1-Score (Harmonic mean of precision and sensitivity)
    - ROC-AUC (Area under ROC curve from metastasis probability)
    - Confusion Matrix (TN, FP, FN, TP)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if y_probs is not None and isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
            "roc_auc": None,
            "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
            "total_samples": 0,
        }

    # Binary confusion matrix components
    # Class 0: Normal, Class 1: Metastasis/Malignant
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = (2.0 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    roc_auc = None
    if y_probs is not None:
        y_probs = np.asarray(y_probs, dtype=float).ravel()
        roc_auc = compute_roc_auc(y_true, y_probs)

    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1_score),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "total_samples": total,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluate a PyTorch classification model over a DataLoader.
    Computes loss and complete classification/medical metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        # Probability for class 1 (Metastasis)
        if probs.shape[1] > 1:
            all_probs.extend(probs[:, 1].cpu().tolist())
        else:
            all_probs.extend(probs[:, 0].cpu().tolist())

    metrics = calculate_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        y_probs=all_probs,
    )

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    metrics["loss"] = float(avg_loss)
    return metrics

