"""Metrics helpers."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "f1_macro": float(f1_score(y_true_np, y_pred_np, average="macro")),
    }


def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.tensor(confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy()))
