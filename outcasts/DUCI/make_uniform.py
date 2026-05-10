import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_IDS = ["00", "01", "02", "10", "11", "12", "20", "21", "22"]

submission = pd.DataFrame({
    "model_id": MODEL_IDS,
    "proportion": [1.0 / 9] * 9,
})

out_path = os.path.join(OUT_DIR, "sample_submission.csv")
submission.to_csv(out_path, index=False)

print(submission)
print(f"Saved to: {out_path}")