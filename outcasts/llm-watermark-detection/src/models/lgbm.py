"""LightGBM trainer with stratified CV.

Returns out-of-fold (OOF) predictions for the train set and averaged test
predictions. Per-fold boosters are saved to disk so you can inspect feature
importance or re-predict without retraining.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib

from .. import config, cv
from ..metric import tpr_at_fpr


def train_cv(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame,
    feature_set_name: str = "default",
) -> dict:
    """Train N_FOLDS LightGBM boosters; return OOF preds, test preds, models, importances."""
    assert list(X_train.columns) == list(X_test.columns), \
        "train/test columns must match exactly"

    folds = cv.get_folds(y_train)
    oof = np.zeros(len(y_train), dtype=np.float64)
    test_preds = np.zeros(len(X_test), dtype=np.float64)
    boosters = []
    fold_metrics = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        Xtr, ytr = X_train.iloc[tr_idx], y_train[tr_idx]
        Xva, yva = X_train.iloc[va_idx], y_train[va_idx]
        dtr = lgb.Dataset(Xtr, label=ytr)
        dva = lgb.Dataset(Xva, label=yva, reference=dtr)

        booster = lgb.train(
            params=config.LGBM_PARAMS,
            train_set=dtr,
            num_boost_round=config.LGBM_NUM_BOOST_ROUND,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(config.LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        va_pred = booster.predict(Xva, num_iteration=booster.best_iteration)
        oof[va_idx] = va_pred
        test_preds += booster.predict(X_test, num_iteration=booster.best_iteration) / len(folds)
        boosters.append(booster)

        fold_metrics.append({
            "fold": fold_id,
            "best_iter": booster.best_iteration,
            "logloss": log_loss(yva, va_pred, labels=[0, 1]),
            "auc": roc_auc_score(yva, va_pred),
            "acc": accuracy_score(yva, (va_pred > 0.5).astype(int)),
            "tpr@fpr0.01": tpr_at_fpr(yva, va_pred, 0.01),
        })
        m = fold_metrics[-1]
        print(f"  fold {fold_id}: auc={m['auc']:.4f}  tpr@fpr1%={m['tpr@fpr0.01']:.4f}  "
              f"logloss={m['logloss']:.4f}  acc={m['acc']:.4f}  best_iter={m['best_iter']}")

    cv_metrics = {
        "oof_logloss": log_loss(y_train, oof, labels=[0, 1]),
        "oof_auc": roc_auc_score(y_train, oof),
        "oof_acc": accuracy_score(y_train, (oof > 0.5).astype(int)),
        "oof_tpr@fpr0.01": tpr_at_fpr(y_train, oof, 0.01),
    }
    print(f"  OOF: auc={cv_metrics['oof_auc']:.4f}  "
          f"tpr@fpr1%={cv_metrics['oof_tpr@fpr0.01']:.4f}  "
          f"logloss={cv_metrics['oof_logloss']:.4f}  "
          f"acc={cv_metrics['oof_acc']:.4f}")

    # Save artifacts
    out_dir = config.MODELS_DIR / f"lgbm_{feature_set_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(boosters):
        b.save_model(str(out_dir / f"fold_{i}.txt"))
    np.save(out_dir / "oof.npy", oof)
    np.save(out_dir / "test_preds.npy", test_preds)
    pd.DataFrame(fold_metrics).to_csv(out_dir / "fold_metrics.csv", index=False)

    # Aggregate feature importance across folds
    fi = pd.DataFrame({
        "feature": boosters[0].feature_name(),
        **{f"gain_fold_{i}": b.feature_importance(importance_type="gain")
           for i, b in enumerate(boosters)},
    })
    fi["gain_mean"] = fi[[c for c in fi.columns if c.startswith("gain_fold_")]].mean(axis=1)
    fi = fi.sort_values("gain_mean", ascending=False).reset_index(drop=True)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)

    return {
        "oof": oof,
        "test_preds": test_preds,
        "boosters": boosters,
        "fold_metrics": fold_metrics,
        "cv_metrics": cv_metrics,
        "feature_importance": fi,
        "out_dir": out_dir,
    }
