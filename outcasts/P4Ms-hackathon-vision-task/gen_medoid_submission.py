#!/usr/bin/env python3
"""
gen_mode_submission.py — Improved baseline using character-wise mode
instead of medoid. Validation says:
  CREDIT: medoid 0.2581 → mode '2290 0161 0669 2299' = 0.2692 (+0.011)
  PHONE:  medoid 0.3902 → mode '+15056437849'      = 0.3955 (+0.005)

Runs in <5 seconds.
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("TEST_N", "0")

import pandas as pd
from collections import Counter
import main as M


def compute_modes(val_records):
    cards = []
    phones = []
    for r in val_records:
        if r.get("pii_type") == "CREDIT" and r.get("gt"):
            m = re.search(r"\d{4}\s\d{4}\s\d{4}\s\d{4}", r["gt"])
            if m: cards.append(m.group(0))
        elif r.get("pii_type") == "PHONE" and r.get("gt"):
            m = re.search(r"\+\d{10,12}", r["gt"])
            if m: phones.append(m.group(0))

    # mode card
    pos = [Counter() for _ in range(19)]
    for c in cards:
        for i, ch in enumerate(c):
            pos[i][ch] += 1
    mode_card = "".join(p.most_common(1)[0][0] for p in pos) if cards else "0000 0000 0000 0000"

    # mode phone (12 chars)
    pos = [Counter() for _ in range(12)]
    for p in phones:
        for i, ch in enumerate(p[:12]):
            pos[i][ch] += 1
    mode_phone = "".join(c.most_common(1)[0][0] for c in pos) if phones else "+10000000000"

    return mode_card, mode_phone


def main():
    M.log("=" * 70)
    M.log("MODE-BASED submission (improved over medoid)")
    M.log("=" * 70)

    M.maybe_extract_standalone()
    M.setup_standalone_path()

    task_records = M.load_task_records()
    val_records = M.load_validation_records()
    val_stats = M.profile_validation(val_records)

    mode_card, mode_phone = compute_modes(val_records)
    M.log(f"[mode] card:  {mode_card!r}")
    M.log(f"[mode] phone: {mode_phone!r}")

    # Override the medoid in val_stats (so _fallback uses mode-based values)
    val_stats["medoid_card"] = mode_card
    val_stats["medoid_phone"] = mode_phone

    M.log(f"[gen] generating predictions for {len(task_records)} users")

    rows = []
    for rec in task_records:
        uid = rec["id"]
        name = rec.get("name")
        for ptype in ("EMAIL", "CREDIT", "PHONE"):
            fb = M._fallback(ptype, rec, name, val_stats)
            rows.append({"id": uid, "pii_type": ptype, "pred": fb["pred"]})

    df = pd.DataFrame(rows)
    df["pred"] = df["pred"].astype(str).str.strip()
    df["pred"] = df["pred"].apply(
        lambda s: s if 10 <= len(s) <= 100 else (s + "x" * (10 - len(s)) if len(s) < 10 else s[:100]))

    out_path = M.OUT_DIR / "submission_mode.csv"
    df.to_csv(out_path, index=False)
    M.log(f"[gen] wrote {out_path} rows={len(df)}")

    assert len(df) == 3000
    per_id = df.groupby("id")["pii_type"].apply(set)
    assert all(s == {"EMAIL", "CREDIT", "PHONE"} for s in per_id)
    assert df["pred"].str.len().between(10, 100).all()
    M.log("[gen] submission validation OK")
    M.log(f"\nSubmit:")
    M.log(f"  curl -X POST http://35.192.205.84:80/submit/27-p4ms \\")
    M.log(f"    -H 'X-API-Key: 31fd8c57481049a79ce9e526df488d56' \\")
    M.log(f"    -F 'file=@{out_path}'")


if __name__ == "__main__":
    main()