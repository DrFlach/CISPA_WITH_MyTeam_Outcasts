import csv
import random
import zipfile
import requests
from pathlib import Path
import pandas as pd
import json

# ----------------------------
# CONFIG
# ----------------------------
ZIP_FILE = "Dataset.zip"          # Path to dataset zip file
DATASET_DIR = Path("dataset")     # Folder after extraction
SUBMISSION_FILE = "submission.csv"


# ----------------------------
# UNZIP DATASET
# ----------------------------
if not DATASET_DIR.exists():
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)
else:
    print("Dataset already extracted.")


# ----------------------------
# LOAD TEXT DATASETS
# ----------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

print("Loading text datasets...")

train_clean = load_jsonl(DATASET_DIR / "train_clean.jsonl")
train_wm = load_jsonl(DATASET_DIR / "train_wm.jsonl")
val_clean = load_jsonl(DATASET_DIR / "valid_clean.jsonl")
val_wm = load_jsonl(DATASET_DIR / "valid_wm.jsonl")
test_set = load_jsonl(DATASET_DIR / "test.jsonl")   # only id + text

print(
    f"Train clean: {len(train_clean)} | "
    f"Train wm: {len(train_wm)} | "
    f"Val clean: {len(val_clean)} | "
    f"Val wm: {len(val_wm)} | "
    f"Test: {len(test_set)}"
)

# ----------------------------
# DUMMY MODEL (RANDOM SCORES)
# ----------------------------
print("Generating random prediction scores for submission...")

preds = []
for sample in test_set:
    text_id = sample["id"]
    score = round(random.random(), 4)  # random float in [0,1]
    preds.append([text_id, score])

# ----------------------------
# SAVE SUBMISSION CSV
# ----------------------------
with open(SUBMISSION_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "score"])
    writer.writerows(preds)

print(f"Saved submission file to {SUBMISSION_FILE}")
print("Format: id,score | Allowed scores: [0,1]")

