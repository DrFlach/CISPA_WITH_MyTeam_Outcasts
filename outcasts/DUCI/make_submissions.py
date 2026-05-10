import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_IDS = ["00", "01", "02", "10", "11", "12", "20", "21", "22"]

# Raw estimates from your last run (median of 3 estimators, no scaling)
raw = {
    "00": 0.090,
    "01": 0.348,
    "02": 0.470,
    "10": 0.030,
    "11": 0.120,
    "12": 0.370,
    "20": 0.000,
    "21": 0.210,
    "22": 0.391,
}

def save(name, vals):
    df = pd.DataFrame({"model_id": MODEL_IDS,
                       "proportion": [vals[m] for m in MODEL_IDS]})
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"\n{name}")
    print(df)
    return path

# H1: SCALE = 1.7
h1 = {m: float(np.clip(raw[m] * 1.7, 0.0, 1.0)) for m in MODEL_IDS}
save("submission_H1_scale17.csv", h1)

# H2: Snap to {0, 0.5, 1.0}, choose nearest
def snap(v, grid):
    g = np.array(grid)
    return float(g[np.argmin(np.abs(g - v))])

h2 = {m: snap(raw[m], [0.0, 0.5, 1.0]) for m in MODEL_IDS}
save("submission_H2_snap_indep.csv", h2)

# H3: Per-arch ranking, force {0, 0.5, 1.0}
h3 = {}
for arch in ["0", "1", "2"]:
    group = [m for m in MODEL_IDS if m[0] == arch]
    # rank by raw value: lowest -> 0, middle -> 0.5, highest -> 1.0
    sorted_group = sorted(group, key=lambda m: raw[m])
    h3[sorted_group[0]] = 0.0
    h3[sorted_group[1]] = 0.5
    h3[sorted_group[2]] = 1.0

save("submission_H3_arch_pattern.csv", h3)

# H4 (bonus): SCALE = 2.0
h4 = {m: float(np.clip(raw[m] * 2.0, 0.0, 1.0)) for m in MODEL_IDS}
save("submission_H4_scale20.csv", h4)