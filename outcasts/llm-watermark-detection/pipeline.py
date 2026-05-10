"""End-to-end pipeline orchestrator.

Run modes:
    python pipeline.py stat                # CPU only — fast baseline
    python pipeline.py logprobs            # GPU — strong features
    python pipeline.py all                 # combine everything

This is the file you run during normal iteration. main.py is just a thin
argparse wrapper around it.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from src import config, data, submit
from src.features import statistical
from src.models import lgbm


def _stat_features(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    cache = config.FEATURES_DIR
    out = {}
    for name, df in splits.items():
        path = cache / f"{name}_stat.parquet"
        if path.exists():
            print(f"[pipeline] loading cached stat features for {name} from {path}")
            out[name] = pd.read_parquet(path)
        else:
            print(f"[pipeline] extracting stat features for {name} ({len(df)} rows)")
            feats = statistical.extract_features(df, show_progress=True)
            feats.to_parquet(path)
            out[name] = feats
    return out


def _logprob_features(
    splits: dict[str, pd.DataFrame],
    model_name: str | None = None,
    model_tag: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Lazy import — only requires torch/transformers when actually called."""
    from src.features.llm_logprobs import extract_or_load
    model_name = model_name or config.BASE_LM_NAME
    model_tag = model_tag or config.BASE_LM_TAG
    out = {}
    for name, df in splits.items():
        path = config.FEATURES_DIR / f"{name}_lp_{model_tag}.parquet"
        out[name] = extract_or_load(df["text"].tolist(), path, model_name=model_name)
    return out


def run(mode: str = "stat") -> None:
    print(f"\n=== pipeline mode: {mode} ===\n")
    splits = data.load_all()
    for name, df in splits.items():
        print(f"  {name}: {len(df)} rows" + (f", positives={df.get('label', pd.Series()).sum()}"
              if 'label' in df.columns else ""))
    print()

    feature_dfs: dict[str, list[pd.DataFrame]] = {k: [] for k in splits}

    if mode in ("stat", "all"):
        stat_feats = _stat_features(splits)
        for k in splits:
            feature_dfs[k].append(stat_feats[k])

    if mode in ("logprobs", "all"):
        lp_feats = _logprob_features(splits)
        for k in splits:
            feature_dfs[k].append(lp_feats[k])

    # Concatenate feature blocks per split
    X = {k: pd.concat(parts, axis=1) for k, parts in feature_dfs.items()}
    print(f"\n[pipeline] feature matrix shapes: " +
          ", ".join(f"{k}={v.shape}" for k, v in X.items()))

    # We train CV on train (360 rows) and use valid (180 rows) as a held-out
    # sanity check that's NOT used for early stopping or model selection.
    y_train = splits["train"]["label"].to_numpy()
    y_valid = splits["valid"]["label"].to_numpy()

    print(f"\n[pipeline] training LightGBM CV on {len(y_train)} train rows ...")
    res = lgbm.train_cv(
        X_train=X["train"], y_train=y_train,
        X_test=X["test"],
        feature_set_name=mode,
    )

    # Held-out validation score (using mean of fold predictors)
    valid_pred = np.mean([
        b.predict(X["valid"], num_iteration=b.best_iteration) for b in res["boosters"]
    ], axis=0)
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    from src.metric import tpr_at_fpr
    print(f"\n[pipeline] held-out valid: "
          f"auc={roc_auc_score(y_valid, valid_pred):.4f}  "
          f"tpr@fpr1%={tpr_at_fpr(y_valid, valid_pred, 0.01):.4f}  "
          f"logloss={log_loss(y_valid, valid_pred, labels=[0,1]):.4f}  "
          f"acc={accuracy_score(y_valid, (valid_pred > 0.5).astype(int)):.4f}")

    # Top features
    print("\n[pipeline] top-15 features by mean gain:")
    print(res["feature_importance"][["feature", "gain_mean"]].head(15).to_string(index=False))

    # Build submission
    submit.build_submission(
        test_ids=splits["test"]["id"],
        test_scores=res["test_preds"],
        name_suffix=f"lgbm_{mode}_oofTPR{res['cv_metrics']['oof_tpr@fpr0.01']:.4f}",
    )


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "stat"
    run(mode)
