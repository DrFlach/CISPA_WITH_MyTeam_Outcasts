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
# PATHS
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

ARCHITECTURE_MAP = {
    "0": models.resnet18,
    "1": models.resnet50,
    "2": models.resnet152,
}


# =========================================================
# MODEL LOADING
# =========================================================

def load_model(model_id, device):

    model_fn = ARCHITECTURE_MAP[model_id[0]]

    weights_path = None

    for ext in [".pkl", ".pth", ".pt"]:
        candidate = os.path.join(
            MODELS_DIR,
            f"model_{model_id}{ext}"
        )

        if os.path.exists(candidate):
            weights_path = candidate
            break

    if weights_path is None:
        raise FileNotFoundError(
            f"Weights not found for model {model_id}"
        )

    print(f"Loading model {model_id}")

    model = model_fn(
        weights=None,
        num_classes=NUM_CLASSES
    )

    try:

        with open(weights_path, "rb") as f:
            payload = pickle.load(f)

    except Exception:

        payload = torch.load(
            weights_path,
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

def find_npy(folder):

    x_file = None
    y_file = None

    for f in os.listdir(folder):

        if f.lower().startswith("x") and f.endswith(".npy"):
            x_file = os.path.join(folder, f)

        if f.lower().startswith("y") and f.endswith(".npy"):
            y_file = os.path.join(folder, f)

    if x_file is None or y_file is None:
        raise FileNotFoundError(
            f"Missing X/Y npy files in {folder}"
        )

    return x_file, y_file


def load_data(folder):

    x_path, y_path = find_npy(folder)

    X = np.load(x_path)
    Y = np.load(y_path)

    X = torch.tensor(X).float()

    if X.max() > 2:
        X = X / 255.0

    if X.ndim == 4 and X.shape[-1] == 3:
        X = X.permute(0, 3, 1, 2)

    mean = torch.tensor(
        [0.4914, 0.4822, 0.4465]
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.2470, 0.2435, 0.2616]
    ).view(1, 3, 1, 1)

    X = (X - mean) / std

    Y = torch.tensor(Y).long()

    return X, Y


# =========================================================
# METRICS
# =========================================================

def compute_metrics(
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

    metrics = {
        "loss": [],
        "confidence": [],
        "entropy": [],
        "margin": [],
        "true_logit": [],
        "energy": [],
        "correct": [],
    }

    eps = 1e-12

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb).float()

            probs = F.softmax(logits, dim=1)

            loss = F.cross_entropy(
                logits,
                yb,
                reduction="none"
            )

            confidence = probs.max(dim=1).values

            entropy = -(
                probs *
                torch.log(probs + eps)
            ).sum(dim=1)

            top2 = torch.topk(
                logits,
                k=2,
                dim=1
            ).values

            margin = top2[:, 0] - top2[:, 1]

            true_logit = logits.gather(
                1,
                yb.unsqueeze(1)
            ).squeeze(1)

            energy = torch.logsumexp(
                logits,
                dim=1
            )

            pred = logits.argmax(dim=1)

            correct = (
                pred == yb
            ).float()

            metrics["loss"].append(
                loss.cpu().float()
            )

            metrics["confidence"].append(
                confidence.cpu().float()
            )

            metrics["entropy"].append(
                entropy.cpu().float()
            )

            metrics["margin"].append(
                margin.cpu().float()
            )

            metrics["true_logit"].append(
                true_logit.cpu().float()
            )

            metrics["energy"].append(
                energy.cpu().float()
            )

            metrics["correct"].append(
                correct.cpu().float()
            )

    for k in metrics:
        metrics[k] = torch.cat(metrics[k])

    return metrics


# =========================================================
# HELPERS
# =========================================================

def robust_rank(
    values,
    reference,
    higher_better=True
):

    ref = np.sort(reference)

    ranks = np.searchsorted(
        ref,
        values,
        side="right"
    )

    ranks = ranks.astype(np.float64)

    ranks /= len(ref)

    if higher_better:
        return ranks

    return 1.0 - ranks


def robust_zscore(values, reference):

    med = np.median(reference)

    mad = np.median(
        np.abs(reference - med)
    ) + 1e-8

    return (
        values - med
    ) / (1.4826 * mad)


# =========================================================
# ESTIMATION
# =========================================================

def estimate_proportion(
    mixed_metrics,
    pop_metrics
):

    eps = 1e-8

    # ---------------------------------------------
    # normalize metrics
    # ---------------------------------------------

    metric_names = [
        "true_logit",
        "margin",
        "confidence",
        "energy",
        "loss"
    ]

    mixed = {}
    pop = {}

    for k in metric_names:

        m = mixed_metrics[k].numpy()
        p = pop_metrics[k].numpy()

        mixed[k] = robust_zscore(m, p)
        pop[k] = robust_zscore(p, p)

    # ---------------------------------------------
    # ensemble score
    # ---------------------------------------------

    mix_score = (
        0.30 * robust_rank(
            mixed["true_logit"],
            pop["true_logit"],
            True
        )
        +
        0.25 * robust_rank(
            mixed["margin"],
            pop["margin"],
            True
        )
        +
        0.20 * robust_rank(
            mixed["confidence"],
            pop["confidence"],
            True
        )
        +
        0.15 * robust_rank(
            mixed["energy"],
            pop["energy"],
            True
        )
        +
        0.10 * robust_rank(
            mixed["loss"],
            pop["loss"],
            False
        )
    )

    pop_score = (
        0.30 * robust_rank(
            pop["true_logit"],
            pop["true_logit"],
            True
        )
        +
        0.25 * robust_rank(
            pop["margin"],
            pop["margin"],
            True
        )
        +
        0.20 * robust_rank(
            pop["confidence"],
            pop["confidence"],
            True
        )
        +
        0.15 * robust_rank(
            pop["energy"],
            pop["energy"],
            True
        )
        +
        0.10 * robust_rank(
            pop["loss"],
            pop["loss"],
            False
        )
    )

    # ---------------------------------------------
    # tail estimation
    # ---------------------------------------------

    quantiles = np.linspace(
        0.70,
        0.95,
        25
    )

    estimates = []

    for q in quantiles:

        tau = np.quantile(
            pop_score,
            q
        )

        p_pop = (
            pop_score >= tau
        ).mean()

        p_mix = (
            mix_score >= tau
        ).mean()

        excess = p_mix - p_pop

        alpha_q = excess / (
            1.0 - p_pop + eps
        )

        estimates.append(alpha_q)

    estimates = np.array(estimates)

    estimates = estimates[
        np.isfinite(estimates)
    ]

    estimates = np.clip(
        estimates,
        0.0,
        1.0
    )

    if len(estimates) == 0:
        return 0.0

    lo = np.percentile(estimates, 20)
    hi = np.percentile(estimates, 80)

    estimates = estimates[
        (estimates >= lo) &
        (estimates <= hi)
    ]

    alpha = np.median(estimates)

    # ---------------------------------------------
    # accuracy calibration
    # ---------------------------------------------

    mix_acc = mixed_metrics[
        "correct"
    ].mean().item()

    pop_acc = pop_metrics[
        "correct"
    ].mean().item()

    acc_gap = max(
        mix_acc - pop_acc,
        0.0
    )

    acc_estimate = np.clip(
        acc_gap * 3.5,
        0.0,
        1.0
    )

    alpha = (
        0.75 * alpha +
        0.25 * acc_estimate
    )

    alpha = np.clip(
        alpha,
        0.0,
        1.0
    )

    return float(alpha)


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

    X_mixed, Y_mixed = load_data(MIXED_DIR)
    X_pop, Y_pop = load_data(POP_DIR)

    proportions = {}

    print("\nRunning inference...\n")

    for model_id in MODEL_IDS:

        model = load_model(
            model_id,
            device
        )

        mixed_metrics = compute_metrics(
            model,
            X_mixed,
            Y_mixed,
            device
        )

        pop_metrics = compute_metrics(
            model,
            X_pop,
            Y_pop,
            device
        )

        mix_acc = mixed_metrics[
            "correct"
        ].mean().item()

        pop_acc = pop_metrics[
            "correct"
        ].mean().item()

        alpha = estimate_proportion(
            mixed_metrics,
            pop_metrics
        )

        proportions[model_id] = alpha

        print(
            f"{model_id} | "
            f"mix_acc={mix_acc:.4f} | "
            f"pop_acc={pop_acc:.4f} | "
            f"alpha={alpha:.4f}"
        )

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =====================================================
    # SAVE SUBMISSION
    # =====================================================

    submission = pd.DataFrame({
        "model_id": MODEL_IDS,
        "proportion": [
            float(proportions[m])
            for m in MODEL_IDS
        ]
    })

    out_path = os.path.join(
        OUT_DIR,
        "sample_submission.csv"
    )

    submission.to_csv(
        out_path,
        index=False
    )

    print("\nSaved submission:")
    print(out_path)

    print("\nSubmission preview:")
    print(submission)


if __name__ == "__main__":
    main()