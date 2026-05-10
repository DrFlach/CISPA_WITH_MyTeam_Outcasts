"""Upload a submission CSV to the leaderboard server.

Usage:
    python upload.py                                    # uploads the latest CSV in output/submissions/
    python upload.py output/submissions/foo.csv         # uploads a specific file
    python upload.py --dry-run                          # validate only, do not upload

Override credentials via environment variables if you don't want to commit them:
    export WM_API_KEY=...
    export WM_BASE_URL=http://...
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import requests

from src import config
from src.submit import validate

# --- Server config (taken from submission_template.py) ---
BASE_URL = os.environ.get("WM_BASE_URL", "http://35.192.205.84:80")
API_KEY = os.environ.get("WM_API_KEY", "31fd8c57481049a79ce9e526df488d56")
TASK_ID = os.environ.get("WM_TASK_ID", "13-llm-watermark-detection")


def find_latest_submission() -> Path:
    files = sorted(config.SUBMISSIONS_DIR.glob("submission_*.csv"))
    # exclude *.probs.csv legacy files just in case
    files = [f for f in files if not f.name.endswith(".probs.csv")]
    if not files:
        sys.exit(f"No submissions found in {config.SUBMISSIONS_DIR}")
    return files[-1]


def upload(path: Path) -> dict:
    print(f"[upload] sending {path} ({path.stat().st_size} bytes) "
          f"to {BASE_URL}/submit/{TASK_ID}")
    with open(path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (path.name, f, "csv")},
            timeout=(10, 120),
        )

    if resp.status_code == 413:
        sys.exit("Upload rejected: file too large (HTTP 413)")

    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if not resp.ok:
        print(f"[upload] HTTP {resp.status_code}", file=sys.stderr)
        print(f"[upload] body: {body}", file=sys.stderr)
        resp.raise_for_status()

    return body


def main():
    parser = argparse.ArgumentParser(description="Upload submission to leaderboard")
    parser.add_argument("path", nargs="?", default=None,
                        help="path to CSV (default: latest in output/submissions/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate but do not upload")
    args = parser.parse_args()

    path = Path(args.path) if args.path else find_latest_submission()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    print(f"[upload] target: {path}")
    validate(path)

    if args.dry_run:
        print("[upload] dry run — not sending")
        return

    body = upload(path)
    print(f"[upload] success.")
    print(f"[upload] server response: {body}")
    if "submission_id" in body:
        print(f"[upload] submission_id: {body['submission_id']}")
    for key in ("score", "score_held_out"):
        if key in body:
            print(f"[upload] {key}: {body[key]}")


if __name__ == "__main__":
    main()
