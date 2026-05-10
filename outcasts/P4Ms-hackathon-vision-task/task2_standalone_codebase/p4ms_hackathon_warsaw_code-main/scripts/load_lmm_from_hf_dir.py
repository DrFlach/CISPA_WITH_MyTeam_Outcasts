"""
Load the target/shadow LMM from HuggingFace.

The directory should contain (at minimum):
  - saved_config.json   (training args snapshot produced by the trainer)
  - non_lora_trainables.bin  (checkpoint with non-LoRA trainable weights)
  - config.json          (HF model config, optional — base model is used if absent)

Usage examples:

  # From cache (after downloading, use when running stuff on Juelich compute nodes).
  python scripts/load_lmm_from_hf_dir.py \
      --model_dir $HOME/.cache/huggingface/hub/models--SprintML--target_lmm/snapshots/737c2514c95ec1cb4c7630ac6218d4fba968e15d

  # From a HuggingFace Hub repo (initial download, do this on the login node):
  python scripts/load_lmm_from_hf_dir.py \
      --model_dir SprintML/target_lmm
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append(os.getcwd())

import torch

from src.lmms.configs.unified_config import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from src.lmms.models import get_model_tokenizer
from src.lmms.models.unified_arch import UnifiedMetaForCausalLM
from transformers import AutoTokenizer
from src.lmms.utils.util import set_seed


# ── helpers ──────────────────────────────────────────────────────────────────


def resolve_model_dir(model_dir: str) -> Path:
    """
    Resolve *model_dir* to a local path.

    If it looks like a local path that exists, return it directly.
    Otherwise, treat it as a HuggingFace Hub repo-id and download
    a snapshot with ``huggingface_hub.snapshot_download``.
    """
    local = Path(model_dir)
    if local.is_dir():
        return local

    # Looks like a HF Hub identifier (e.g. "SprintML/shadow_1mm")
    try:
        from huggingface_hub import snapshot_download

        downloaded = snapshot_download(repo_id=model_dir)
        return Path(downloaded)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve model_dir='{model_dir}' as a local path "
            f"or as a HuggingFace Hub repo: {exc}"
        ) from exc


def load_saved_config(model_dir: Path) -> dict:
    """Load the ``saved_config.json`` produced during training."""
    cfg_path = model_dir / "saved_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"saved_config.json not found in {model_dir}. "
            "This file is required to reconstruct model/data/training args."
        )
    with open(cfg_path, "r") as f:
        return json.load(f)


def _filter_keys(cls, d: dict) -> dict:
    """Keep only keys that ``cls`` actually accepts."""
    from dataclasses import fields as dc_fields

    valid = {f.name for f in dc_fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


def args_from_saved_config(
    saved_config: dict,
    model_dir: Path,
    ckpt_filename: str = "non_lora_trainables.bin",
) -> tuple:
    """
    Reconstruct ``ModelArguments``, ``DataArguments``, and
    ``TrainingArguments`` from a ``saved_config.json`` dict.

    Overrides ``pretrained_ckpt_path`` to point at the checkpoint file
    inside *model_dir* so that weights are loaded automatically.
    """
    model_kwargs = _filter_keys(ModelArguments, saved_config.get("model_args", {}))
    data_kwargs = _filter_keys(DataArguments, saved_config.get("data_args", {}))
    training_kwargs = _filter_keys(
        TrainingArguments, saved_config.get("training_args", {})
    )

    model_kwargs["pretrained_ckpt_path"] = str(model_dir / ckpt_filename)
    model_kwargs["visual_proj_ckpt_dir"] = None
    training_kwargs.pop("_n_gpu", None)

    model_args = ModelArguments(**model_kwargs)
    data_args = DataArguments(**data_kwargs)
    training_args = TrainingArguments(**training_kwargs)

    return model_args, data_args, training_args


# ── main ─────────────────────────────────────────────────────────────────────


def load_lmm(
    model_dir: str,
    device: str = "cuda",
    dtype: str = "bf16",
    seed: int = 42,
    do_not_setup_inference: bool = False,
) -> tuple[
    UnifiedMetaForCausalLM,
    AutoTokenizer,
    ModelArguments,
    DataArguments,
    TrainingArguments,
]:
    """
    High-level entry point: load the full LMM from a HuggingFace-style
    directory and return ``(model, tokenizer, model_args, data_args,
    training_args)``.

    Parameters
    ----------
    model_dir : str
        Path to a local directory **or** a HuggingFace Hub repo-id.
    device : str
        Target device (``cuda``, ``cuda:0``, ``cpu``, …).
    dtype : str
        Compute dtype shorthand: ``bf16``, ``fp16``, ``fp32``.
    seed : int
        Random seed.
    """
    set_seed(seed)

    resolved_dir = resolve_model_dir(model_dir)
    print(f"[load_lmm] Resolved model dir → {resolved_dir}")

    saved_config = load_saved_config(resolved_dir)
    model_args, data_args, training_args = args_from_saved_config(
        saved_config, resolved_dir
    )

    # Override compute dtype via training_args flags
    training_args.bf16 = dtype == "bf16"
    training_args.fp16 = dtype == "fp16"

    # Disable deepspeed for single-GPU inference
    training_args.deepspeed = None
    training_args.local_rank = -1

    print(f"[load_lmm] model_args: {model_args}")
    print(f"[load_lmm] data_args:  {data_args}")

    model, tokenizer = get_model_tokenizer(model_args, training_args)

    device = torch.device(device)
    model = model.to(device)
    if not do_not_setup_inference:
        model_setup_inference(model, tokenizer)
    print(f"[load_lmm] Model loaded on {device} with dtype={dtype}")

    return model, tokenizer, model_args, data_args, training_args


def model_setup_inference(model, tokenizer):
    try:
        model.base_model.set_mode("inference")
    except Exception:
        pass

    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.use_cache = True
    model.generation_config.max_new_tokens = 25
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    print("Model ready for inference.")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Load a trained LMM from a HuggingFace-style directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "Path to a local HuggingFace-style directory "
            "or a HuggingFace Hub repo-id (e.g. SprintML/shadow_1mm)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Target device (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Compute dtype (default: bf16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--do_not_setup_inference",
        action="store_true",
        help="Do not setup model for inference.",
    )
    args = parser.parse_args()

    model, tokenizer, model_args, data_args, training_args = load_lmm(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        do_not_setup_inference=args.do_not_setup_inference,
    )

    print(f"\n✓ Model loaded successfully {args.model_dir}")
    print(f"  LLM base: {model_args.model_name_or_path}")
    print(f"  Pretrained ckpt: {model_args.pretrained_ckpt_path}")


if __name__ == "__main__":
    main()
