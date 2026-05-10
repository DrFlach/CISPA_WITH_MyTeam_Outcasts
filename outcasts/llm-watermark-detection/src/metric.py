"""Competition metric: True Positive Rate at False Positive Rate = 1%.

The server splits ids into 70% public / 30% held-out via deterministic MD5 hash
and computes TPR @ FPR=0.01 on each. We replicate the metric for local CV.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> float:
    """Largest TPR achievable while keeping FPR <= target_fpr.

    Matches sklearn's roc_curve-based computation used by the evaluator.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not mask.any():
        # No threshold meets FPR <= target_fpr — happens with very few negatives
        return 0.0
    return float(tpr[mask].max())
