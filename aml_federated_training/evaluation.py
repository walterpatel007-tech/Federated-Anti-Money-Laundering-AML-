"""Evaluation utilities for AML models."""

from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> Dict[str, float]:
    """Compute a suite of binary classification metrics."""

    labels_np = labels.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()
    binary_preds = (preds_np >= 0.5).astype(int)
    metrics = {
        "precision": precision_score(labels_np, binary_preds, zero_division=0),
        "recall": recall_score(labels_np, binary_preds, zero_division=0),
        "f1": f1_score(labels_np, binary_preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels_np, preds_np)
    except ValueError:
        metrics["roc_auc"] = 0.5
    return metrics
