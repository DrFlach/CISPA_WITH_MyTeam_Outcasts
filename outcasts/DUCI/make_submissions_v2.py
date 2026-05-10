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

def save(name, vals):
    df = pd.DataFrame({"model_id": MODEL_IDS,
                       "proportion": [vals[m] for m in MODEL_IDS]})
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"\n{name}")
    print(df.to_string(index=False))

# A: SCALE 2.3
a = {m: float(np.clip(raw[m] * 2.3, 0.0, 1.0)) for m in MODEL_IDS}
save("submission_A_scale23.csv", a)

# B: SCALE 2.5
b = {m: float(np.clip(raw[m] * 2.5, 0.0, 1.0)) for m in MODEL_IDS}
save("submission_B_scale25.csv", b)

# C: arch pattern 0/0.5/1.0 (H3 from before)
c = {}
for arch in ["0", "1", "2"]:
    group = [m for m in MODEL_IDS if m[0] == arch]
    sorted_group = sorted(group, key=lambda m: raw[m])
    c[sorted_group[0]] = 0.0
    c[sorted_group[1]] = 0.5
    c[sorted_group[2]] = 1.0
save("submission_C_archpattern.csv", c)