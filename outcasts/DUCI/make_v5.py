import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_IDS = ["00", "01", "02", "10", "11", "12", "20", "21", "22"]

# From W3 (cross-model ensemble)
ensemble = {
    "00": 0.060, "01": 0.330, "02": 0.470,
    "10": 0.020, "11": 0.117, "12": 0.397,
    "20": 0.000, "21": 0.327, "22": 0.400,
}

# Per-arch reference: model XXX with z near 0 (no overfit on MIXED) anchors p=0
# 00: z=-4.7   (slight signal)
# 10: z=-1.96  (borderline, almost no signal)
# 20: z=+0.14  (no signal -> p=0 confirmed)

# Final strategy: 
# - Trust raw ensemble for low-signal models (00, 10, 20).
# - Apply scale*1.7 to mid models, push to ~1.0 for high models (per-arch).

def make_arch_pattern_continuous(low, mid, high):
    """Per-arch ranking by ensemble value."""
    out = {}
    for arch in ["0", "1", "2"]:
        group = [m for m in MODEL_IDS if m[0] == arch]
        sorted_group = sorted(group, key=lambda m: ensemble[m])
        out[sorted_group[0]] = low(sorted_group[0])
        out[sorted_group[1]] = mid(sorted_group[1])
        out[sorted_group[2]] = high(sorted_group[2])
    return out

# Final: low = raw ensemble, mid = ensemble*1.7, high = 1.0
final = make_arch_pattern_continuous(
    low=lambda m: ensemble[m],
    mid=lambda m: float(np.clip(ensemble[m] * 1.7, 0.0, 1.0)),
    high=lambda m: 1.0,
)

df = pd.DataFrame({"model_id": MODEL_IDS,
                   "proportion": [final[m] for m in MODEL_IDS]})
df.to_csv(os.path.join(OUT_DIR, "sub_FINAL.csv"), index=False)
print(df)