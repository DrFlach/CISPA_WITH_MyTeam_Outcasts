import os
import gc
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models


# =========================================================
# CONFIG
# =========================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

MIXED_DIR = os.path.join(ROOT, "DATA", "MIXED")
POP_DIR = os.path.join(ROOT, "DATA", "POPULATION")
MODELS_DIR = os.path.join(ROOT, "MODELS")
OUT_DIR = os.path.join(ROOT, "output")

os.makedirs(OUT_DIR, exist_ok=True)

MODEL_IDS = [
    "00", "01", "02",
    "10", "11", "12",
    "20", "21", "22"
]

NUM_CLASSES = 100

ARCH_MAP = {
    "0": models.resnet18,
    "1": models.resnet50,
    "2": models.resnet152,
}


# =========================================================
# MODEL LOADING
# =========================================================

def load_model(model_id, device):

    model_fn = ARCH_MAP[model_id[0]]

    path = None

    for ext in [".pkl", ".pth", ".pt"]:

        candidate = os.path.join(
            MODELS_DIR,
            f"model_{model_id}{ext}"
        )

        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            f"weights not found for {model_id}"
        )

    print(f"\nLoading {model_id}")

    model = model_fn(
        weights=None,
        num_classes=NUM_CLASSES
    )

    try:

        with open(path, "rb") as f:
            payload = pickle.load(f)

    except Exception:

        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

    if isinstance(payload, dict):

        state_dict = payload.get(
            "state_dict",
            payload
        )

    else:
        state_dict = payload

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


# =========================================================
# DATA
# =========================================================

def find_xy(folder):

    x = None
    y = None

    for f in os.listdir(folder):

        fl = f.lower()

        if fl.startswith("x") and fl.endswith(".npy"):
            x = os.path.join(folder, f)

        if fl.startswith("y") and fl.endswith(".npy"):
            y = os.path.join(folder, f)

    if x is None or y is None:
        raise RuntimeError(
            f"missing npy files in {folder}"
        )

    return x, y


def load_data(folder):
    x_path, y_path = find_xy(folder)
    X = np.load(x_path)
    Y = np.load(y_path)

    X = torch.tensor(X).float()
    if X.max() > 2:
        X = X / 255.0
    if X.ndim == 4 and X.shape[-1] == 3:
        X = X.permute(0, 3, 1, 2)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
    X = (X - mean) / std

    Y = torch.tensor(Y).long()

    # ====== ДОБАВЬ ЭТО ======
    print(f"  folder={folder}")
    print(f"  X shape={tuple(X.shape)} dtype={X.dtype} "
          f"min={X.min().item():.3f} max={X.max().item():.3f}")
    print(f"  Y shape={tuple(Y.shape)} dtype={Y.dtype} "
          f"unique_labels={len(torch.unique(Y))} "
          f"min={Y.min().item()} max={Y.max().item()}")
    # ========================

    return X, Y


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(
    model,
    X,
    Y,
    device,
    batch_size=256
):

    loader = DataLoader(
        TensorDataset(X, Y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    probs_all = []
    preds_all = []
    confs_all = []
    margins_all = []
    entropy_all = []
    losses_all = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb).float()

            probs = F.softmax(
                logits,
                dim=1
            )

            top2 = torch.topk(
                probs,
                k=2,
                dim=1
            ).values

            margin = top2[:, 0] - top2[:, 1]

            entropy = -(
                probs *
                torch.log(probs + 1e-12)
            ).sum(dim=1)

            conf = probs.max(dim=1).values

            pred = probs.argmax(dim=1)

            loss = F.cross_entropy(
                logits,
                yb,
                reduction="none"
            )

            probs_all.append(
                probs.cpu()
            )

            preds_all.append(
                pred.cpu()
            )

            confs_all.append(
                conf.cpu()
            )

            margins_all.append(
                margin.cpu()
            )

            entropy_all.append(
                entropy.cpu()
            )

            losses_all.append(
                loss.cpu()
            )

    probs = torch.cat(probs_all).numpy()
    preds = torch.cat(preds_all).numpy()

    confs = torch.cat(confs_all).numpy()
    margins = torch.cat(margins_all).numpy()
    entropies = torch.cat(entropy_all).numpy()
    losses = torch.cat(losses_all).numpy()

    labels = Y.numpy()

    return {
        "probs": probs,
        "preds": preds,
        "confs": confs,
        "margins": margins,
        "entropies": entropies,
        "losses": losses,
        "labels": labels,
    }


# =========================================================
# SIGNAL BUILDING
# =========================================================

def symmetric_kl(p, q):

    p = p.astype(np.float64)
    q = q.astype(np.float64)

    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)

    kl1 = np.sum(
        p * np.log((p + 1e-12) / (q + 1e-12))
    )

    kl2 = np.sum(
        q * np.log((q + 1e-12) / (p + 1e-12))
    )

    return 0.5 * (kl1 + kl2)


def build_score(mix, pop):

    mix_preds = mix["preds"]
    pop_preds = pop["preds"]

    mix_probs = mix["probs"]
    pop_probs = pop["probs"]

    labels = mix["labels"]

    # =====================================================
    # 1. CLASS HISTOGRAM SHIFT
    # =====================================================

    hist_mix = np.bincount(
        mix_preds,
        minlength=NUM_CLASSES
    ).astype(np.float64)

    hist_pop = np.bincount(
        pop_preds,
        minlength=NUM_CLASSES
    ).astype(np.float64)

    hist_mix /= hist_mix.sum()
    hist_pop /= hist_pop.sum()

    hist_score = symmetric_kl(
        hist_mix,
        hist_pop
    )

    # =====================================================
    # 2. TRUE CLASS PROBABILITY SHIFT
    # =====================================================

    true_mix = mix_probs[
        np.arange(len(labels)),
        labels
    ]

    true_pop = pop_probs[
        np.arange(len(labels)),
        labels
    ]

    true_score = np.mean(
        true_mix - true_pop
    )

    # =====================================================
    # 3. CONFIDENCE SHIFT
    # =====================================================

    conf_score = (
        mix["confs"].mean()
        - pop["confs"].mean()
    )

    # =====================================================
    # 4. MARGIN SHIFT
    # =====================================================

    margin_score = (
        mix["margins"].mean()
        - pop["margins"].mean()
    )

    # =====================================================
    # 5. ENTROPY SHIFT
    # =====================================================

    entropy_score = (
        pop["entropies"].mean()
        - mix["entropies"].mean()
    )

    # =====================================================
    # 6. LOSS IMPROVEMENT
    # =====================================================

    loss_score = (
        pop["losses"].mean()
        - mix["losses"].mean()
    )

    # =====================================================
    # 7. HIGH CONFIDENCE RATE
    # =====================================================

    hc_mix = (
        mix["confs"] > 0.90
    ).mean()

    hc_pop = (
        pop["confs"] > 0.90
    ).mean()

    high_conf_score = hc_mix - hc_pop

    # =====================================================
    # FINAL SIGNAL
    # =====================================================

    final_score = (
        3.5 * true_score
        + 2.5 * conf_score
        + 2.0 * margin_score
        + 1.5 * entropy_score
        + 2.5 * loss_score
        + 3.0 * high_conf_score
        + 0.5 * hist_score
    )

    return float(final_score)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_scores(raw_scores):

    vals = np.array(raw_scores)

    vals = vals - vals.min()

    vals = np.power(
        vals + 1e-12,
        1.35
    )

    vals = vals / (
        vals.sum() + 1e-12
    )

    vals = np.clip(
        vals,
        0.001,
        0.95
    )

    vals = vals / vals.sum()

    return vals


# =========================================================
# MAIN
# =========================================================

def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"DEVICE = {device}")

    print("\nLoading datasets...")

    X_mix, Y_mix = load_data(MIXED_DIR)
    X_pop, Y_pop = load_data(POP_DIR)

    raw_scores = []

    print("\nRunning inference...")

    for model_id in MODEL_IDS:

        model = load_model(
            model_id,
            device
        )

        mix_features = extract_features(
            model,
            X_mix,
            Y_mix,
            device
        )

        pop_features = extract_features(
            model,
            X_pop,
            Y_pop,
            device
        )

        score = build_score(
            mix_features,
            pop_features
        )

        raw_scores.append(score)

        print(
            f"{model_id} "
            f"raw_score={score:.6f} "
            f"mix_loss={mix_features['losses'].mean():.4f} "
            f"pop_loss={pop_features['losses'].mean():.4f}"
        )

        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    proportions = normalize_scores(
        raw_scores
    )

    submission = pd.DataFrame({
        "model_id": MODEL_IDS,
        "proportion": proportions
    })

    out_path = os.path.join(
        OUT_DIR,
        "sample_submission.csv"
    )

    submission.to_csv(
        out_path,
        index=False
    )

    print("\nSubmission:")
    print(submission)

    print("\nSaved to:")
    print(out_path)


if __name__ == "__main__":
    main()
