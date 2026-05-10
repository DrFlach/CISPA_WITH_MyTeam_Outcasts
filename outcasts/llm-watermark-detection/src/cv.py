"""Stratified K-Fold split. Always returns the same splits given the same input length."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import config


def get_folds(y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, valid_idx) tuples for N_FOLDS stratified splits."""
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    return list(skf.split(np.zeros(len(y)), y))


def fold_assignments(y: np.ndarray) -> np.ndarray:
    """Return a vector of fold IDs (0..N_FOLDS-1) per row.

    Useful when you want to materialize the split once and slice features by fold ID.
    """
    folds = np.full(len(y), -1, dtype=np.int8)
    for fold_id, (_, valid_idx) in enumerate(get_folds(y)):
        folds[valid_idx] = fold_id
    assert (folds >= 0).all(), "every row should be assigned to exactly one fold"
    return folds
