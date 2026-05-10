"""Load JSONL splits into pandas DataFrames with consistent labels."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from . import config


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{path} missing required keys 'id'/'text'")
    return df


def load_train() -> pd.DataFrame:
    """Combined train: 360 rows with column `label` (0=clean, 1=watermarked).

    Adds a unique `row_id` because original `id` collides between wm and clean.
    """
    wm = _load_jsonl(config.DATA_DIR / "train_wm.jsonl").assign(label=1, source="train_wm")
    clean = _load_jsonl(config.DATA_DIR / "train_clean.jsonl").assign(label=0, source="train_clean")
    df = pd.concat([wm, clean], ignore_index=True)
    df["row_id"] = df.index
    return df


def load_valid() -> pd.DataFrame:
    """Held-out validation, same structure as train."""
    wm = _load_jsonl(config.DATA_DIR / "valid_wm.jsonl").assign(label=1, source="valid_wm")
    clean = _load_jsonl(config.DATA_DIR / "valid_clean.jsonl").assign(label=0, source="valid_clean")
    df = pd.concat([wm, clean], ignore_index=True)
    df["row_id"] = df.index
    return df


def load_test() -> pd.DataFrame:
    """Test set: keep original `id` (it's the submission key)."""
    df = _load_jsonl(config.DATA_DIR / "test.jsonl").assign(source="test")
    df["row_id"] = df.index
    return df


def load_all() -> dict[str, pd.DataFrame]:
    return {"train": load_train(), "valid": load_valid(), "test": load_test()}


if __name__ == "__main__":
    splits = load_all()
    for name, df in splits.items():
        print(f"{name}: {len(df)} rows, columns={list(df.columns)}")
        if "label" in df.columns:
            print(f"  label balance: {df['label'].value_counts().to_dict()}")
