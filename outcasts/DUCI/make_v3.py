import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_IDS = ["00", "01", "02", "10", "11", "12", "20", "21", "22"]
raw = {
    "00": 0.090, "01": 0.348, "02": 0.470,
    "10": 0.030, "11": 0.120, "12": 0.370,
    "20": 0.000, "21": 0.210, "22": 0.391,
}

def make_arch_pattern(low, mid, high):
    """Assigns per-arch by ranking raw values."""
    out = {}
    for arch in ["0", "1", "2"]:
        group = [m for m in MODEL_IDS if m[0] == arch]
        sorted_group = sorted(group, key=lambda m: raw[m])
        out[sorted_group[0]] = low
        out[sorted_group[1]] = mid
        out[sorted_group[2]] = high
    return out

def save(name, vals):
    df = pd.DataFrame({"model_id": MODEL_IDS,
                       "proportion": [vals[m] for m in MODEL_IDS]})
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"\n{name}")
    print(df.to_string(index=False))

# Variants of arch pattern
save("sub_X1_0_05_09.csv",  make_arch_pattern(0.0, 0.5, 0.9))
save("sub_X2_0_04_08.csv",  make_arch_pattern(0.0, 0.4, 0.8))
save("sub_X3_01_05_09.csv", make_arch_pattern(0.1, 0.5, 0.9))
save("sub_X4_01_04_07.csv", make_arch_pattern(0.1, 0.4, 0.7))
save("sub_X5_0_06_10.csv",  make_arch_pattern(0.0, 0.6, 1.0))

# Hybrid: blend H3 with raw*2.0 (H4)
h3 = make_arch_pattern(0.0, 0.5, 1.0)
h4 = {m: float(np.clip(raw[m] * 2.0, 0.0, 1.0)) for m in MODEL_IDS}
hybrid = {m: 0.5 * h3[m] + 0.5 * h4[m] for m in MODEL_IDS}
save("sub_X6_hybrid_h3_h4.csv", hybrid)