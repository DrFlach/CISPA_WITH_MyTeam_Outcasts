"""Build and validate CSV submissions for the leaderboard server.

Server spec (from submission_template.py):
- header: id,score (exactly)
- 2250 rows, ids 1..2250 (each exactly once)
- score: float in [0.0, 1.0], finite, higher = more likely watermarked
- max 10 MB
- metric: TPR @ FPR = 1%  (so we submit continuous probabilities, never binarize)
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from . import config


N_EXPECTED_IDS = 2250
EXPECTED_HEADER = ["id", "score"]
MAX_FILE_BYTES = 10 * 1024 * 1024


def build_submission(
    test_ids: pd.Series,
    test_scores: np.ndarray,
    name_suffix: str = "",
) -> Path:
    """Write a submission CSV in the exact format the server requires.

    Args:
        test_ids: integer ids from 1..2250 (one per test row)
        test_scores: float in [0,1], higher = more likely watermarked
        name_suffix: appended to filename for traceability
    """
    assert len(test_ids) == len(test_scores)

    # Clip into [0, 1] in case a model returned 1.0000001 due to float rounding
    scores = np.clip(np.asarray(test_scores, dtype=np.float64), 0.0, 1.0)

    df = pd.DataFrame({"id": np.asarray(test_ids, dtype=np.int64), "score": scores})
    df = df.sort_values("id").reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{name_suffix}" if name_suffix else ""
    path = config.SUBMISSIONS_DIR / f"submission_{ts}{suffix}.csv"
    df.to_csv(path, index=False, float_format="%.6f")

    # Pre-flight validation — catches issues before the server does
    validate(path)

    print(f"[submit] wrote {len(df)} rows -> {path}")
    print(f"[submit] score stats: min={scores.min():.4f} mean={scores.mean():.4f} "
          f"max={scores.max():.4f} std={scores.std():.4f}")
    return path


def validate(path: Path) -> None:
    """Run the same checks the server does. Raises on any failure."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Extension must be .csv, got {path.suffix}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"File too large: {size} bytes (limit {MAX_FILE_BYTES})")

    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_HEADER:
        raise ValueError(f"Header must be exactly {EXPECTED_HEADER}, got {list(df.columns)}")

    if df["id"].duplicated().any():
        dups = df.loc[df["id"].duplicated(), "id"].head().tolist()
        raise ValueError(f"Duplicate ids found, e.g. {dups}")

    expected = set(range(1, N_EXPECTED_IDS + 1))
    actual = set(df["id"].astype(int).tolist())
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"Missing {len(missing)} ids, first: {sorted(missing)[:5]}")
    if extra:
        raise ValueError(f"Unexpected {len(extra)} ids, first: {sorted(extra)[:5]}")

    s = df["score"]
    if s.isna().any():
        raise ValueError("score contains NaN values")
    if not np.isfinite(s).all():
        raise ValueError("score contains non-finite values")
    if (s < 0).any() or (s > 1).any():
        bad = s[(s < 0) | (s > 1)].head().tolist()
        raise ValueError(f"score must be within [0, 1]; bad values e.g. {bad}")

    print(f"[validate] OK: {len(df)} rows, header correct, ids 1..{N_EXPECTED_IDS}, "
          f"scores in [{s.min():.4f}, {s.max():.4f}]")
