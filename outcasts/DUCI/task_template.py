# import os
# import sys
# import requests

# BASE_URL  = "http://35.192.205.84"
# API_KEY   = "YOUR_KEY"
# TASK_ID   = "11-duci"

# FILE_PATH = "output/sample_submission.csv"


# """
# MODEL_ID = model_{0i} ---> ResNet18
# MODEL_ID = model_{1i} ---> ResNet50
# MODEL_ID = model_{2i} ---> ResNet152
# """

# # ── Model loading ──────────────────────────────────────────────────────────────
# import torch
# import torchvision.models as models

# NUM_CLASSES = 100

# ARCHITECTURE_MAP = {
#     "0": models.resnet18,
#     "1": models.resnet50,
#     "2": models.resnet152,
# }

# def build_model(model_id: str) -> torch.nn.Module:
#     """
#     Build a model from a MODEL_ID string like 'model_00', 'model_12', etc.

#     Naming convention:
#         model_{arch}{instance}
#         arch     0 → ResNet18  |  1 → ResNet50  |  2 → ResNet152
#         instance 0, 1, 2  (three independently trained seeds per architecture)

#     Examples:
#         model_00, model_01, model_02  → ResNet18
#         model_10, model_11, model_12  → ResNet50
#         model_20, model_21, model_22  → ResNet152
#     """
#     # strip optional "model_" prefix
#     key = model_id.removeprefix("model_")   # e.g. "12"
#     arch_digit = key[0]                      # first digit encodes architecture

#     if arch_digit not in ARCHITECTURE_MAP:
#         raise ValueError(
#             f"Unknown architecture digit '{arch_digit}' in model_id '{model_id}'. "
#             f"Expected 0 (ResNet18), 1 (ResNet50), or 2 (ResNet152)."
#         )

#     model_fn = ARCHITECTURE_MAP[arch_digit]
#     model = model_fn(weights=None, num_classes=NUM_CLASSES)
#     return model


# def load_model(model_id: str, weights_path: str,
#                device: str = "cpu") -> torch.nn.Module:
#     """
#     Instantiate the correct architecture for *model_id* and load weights
#     from *weights_path*.

#     Args:
#         model_id:     e.g. 'model_00', 'model_12', 'model_21'
#         weights_path: path to the saved state-dict (.pth / .pt file)
#         device:       'cpu', 'cuda', 'cuda:0', etc.

#     Returns:
#         model in eval mode with weights loaded.
#     """
#     model = build_model(model_id)
#     state_dict = torch.load(weights_path, map_location=device)
#     # support both raw state-dicts and checkpoint dicts
#     if isinstance(state_dict, dict) and "state_dict" in state_dict:
#         state_dict = state_dict["state_dict"]
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model


# # ── Example usage ──────────────────────────────────────────────────────────────
# # Replace the paths below with the actual locations of your weight files.
# #
# # MODEL_WEIGHTS = {
# #     "model_00": "/path/to/weights/model_00.pth",
# #     "model_01": "/path/to/weights/model_01.pth",
# #     "model_02": "/path/to/weights/model_02.pth",
# #     "model_10": "/path/to/weights/model_10.pth",
# #     "model_11": "/path/to/weights/model_11.pth",
# #     "model_12": "/path/to/weights/model_12.pth",
# #     "model_20": "/path/to/weights/model_20.pth",
# #     "model_21": "/path/to/weights/model_21.pth",
# #     "model_22": "/path/to/weights/model_22.pth",
# # }
# #
# # for model_id, path in MODEL_WEIGHTS.items():
# #     model = load_model(model_id, path, device="cpu")
# #     print(f"Loaded {model_id}: {model.__class__.__name__}")
# # ──────────────────────────────────────────────────────────────────────────────



# def die(msg):
#     print(f"{msg}", file=sys.stderr)
#     sys.exit(1)


# if not os.path.isfile(FILE_PATH):
#     die(f"File not found: {FILE_PATH}")

# try:
#     with open(FILE_PATH, "rb") as f:
#         files = {
#             "file": (os.path.basename(FILE_PATH), f, "csv"),
#         }
#         resp = requests.post(
#             f"{BASE_URL}/submit/{TASK_ID}",
#             headers={"X-API-Key": API_KEY},
#             files=files,
#             timeout=(10, 120),
#         )
#     try:
#         body = resp.json()
#     except Exception:
#         body = {"raw_text": resp.text}

#     if resp.status_code == 413:
#         die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

#     resp.raise_for_status()

#     submission_id = body.get("submission_id")
#     print("Successfully submitted.")
#     print("Server response:", body)
#     if submission_id:
#         print(f"Submission ID: {submission_id}")

# except requests.exceptions.RequestException as e:
#     detail = getattr(e, "response", None)
#     print(f"Submission error: {e}")
#     if detail is not None:
#         try:
#             print("Server response:", detail.json())
#         except Exception:
#             print("Server response (text):", detail.text)
#     sys.exit(1)

import os
import sys
import requests

BASE_URL  = "http://35.192.205.84"
API_KEY   = "31fd8c57481049a79ce9e526df488d56"
TASK_ID   = "11-duci"

FILE_PATH = "output/sample_submission.csv"

def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def submit():
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }

            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),
            )

        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large")

        resp.raise_for_status()

        print("Successfully submitted.")
        print("Server response:", body)

    except Exception as e:
        print("Submission error:", e)

if __name__ == "__main__":
    submit()