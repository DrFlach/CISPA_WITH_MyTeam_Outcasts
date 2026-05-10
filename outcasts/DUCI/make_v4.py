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

h3 = make_arch_pattern(0.0, 0.5, 1.0)
h4 = {m: float(np.clip(raw[m] * 2.0, 0.0, 1.0)) for m in MODEL_IDS}
h1 = {m: float(np.clip(raw[m] * 1.7, 0.0, 1.0)) for m in MODEL_IDS}

# Y1: raw для слабых, H3 для сильных
y1 = {m: raw[m] for m in MODEL_IDS}
for arch in ["0", "1", "2"]:
    group = [m for m in MODEL_IDS if m[0] == arch]
    high_m = max(group, key=lambda m: raw[m])
    y1[high_m] = 1.0
save("sub_Y1_raw_with_high1.csv", y1)

# Y2: raw для слабых+средних, mid=0.5 high=1.0 для верха
y2 = {m: raw[m] for m in MODEL_IDS}
for arch in ["0", "1", "2"]:
    group = sorted([m for m in MODEL_IDS if m[0] == arch], key=lambda m: raw[m])
    y2[group[1]] = 0.5
    y2[group[2]] = 1.0
save("sub_Y2_raw_low_mid05_high10.csv", y2)

# Y3: 0.3*H3 + 0.7*H4
y3 = {m: 0.3 * h3[m] + 0.7 * h4[m] for m in MODEL_IDS}
save("sub_Y3_blend_03_07.csv", y3)

# Y4: 0.7*H3 + 0.3*H4
y4 = {m: 0.7 * h3[m] + 0.3 * h4[m] for m in MODEL_IDS}
save("sub_Y4_blend_07_03.csv", y4)

# Y5: H3 для верхних, raw для нижних, raw*1.5 для средних
y5 = {}
for arch in ["0", "1", "2"]:
    group = sorted([m for m in MODEL_IDS if m[0] == arch], key=lambda m: raw[m])
    y5[group[0]] = raw[group[0]]
    y5[group[1]] = float(np.clip(raw[group[1]] * 1.5, 0.0, 1.0))
    y5[group[2]] = 1.0
save("sub_Y5_mixed_scaling.csv", y5)

# Y6: variant of X6 with stronger raw component
y6 = {m: 0.7 * raw[m] * 2.0 + 0.3 * h3[m] for m in MODEL_IDS}
y6 = {m: float(np.clip(v, 0.0, 1.0)) for m, v in y6.items()}
save("sub_Y6_h4heavy.csv", y6)

# Y7: variant of X6 with slightly more H3
y7 = {m: 0.4 * h4[m] + 0.6 * h3[m] for m in MODEL_IDS}
save("sub_Y7_blend_04_06.csv", y7)