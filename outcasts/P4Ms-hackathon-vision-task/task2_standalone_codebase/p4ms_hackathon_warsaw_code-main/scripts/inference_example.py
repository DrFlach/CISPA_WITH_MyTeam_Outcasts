"""
Example: load a trained LMM and compute per-token losses on a single dataset sample.

Usage:
  python scripts/inference_example.py \
      --model_dir SprintML/target_lmm \
      --dataset_name p4ms_vqa \
      --sample_index 0

  # Or from cache:
  python scripts/inference_example.py \
      --model_dir $HOME/.cache/huggingface/hub/models--SprintML--target_lmm/snapshots/737c2514c95ec1cb4c7630ac6218d4fba968e15d \
      --dataset_name p4ms_vqa \
      --sample_index 42
"""

import os
import sys
import argparse

sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

import torch
import torch.nn.functional as F

from load_lmm_from_hf_dir import load_lmm
from src.lmms.dataset.multitask_dataset import get_dataset_collator
from src.lmms.utils.util import prepare_sample


def get_per_token_losses(
    model,
    sample: dict,
    device: torch.device,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a forward pass on a single collated sample and return per-token
    cross-entropy losses.

    Parameters
    ----------
    model : UnifiedMetaForCausalLM
        The loaded LMM.
    sample : dict
        A collated batch dict with keys ``batch_input_ids``,
        ``batch_labels``, ``batch_X_modals``.
    device : torch.device
        Target device.
    ignore_index : int
        Label value that marks non-target positions (default: -100).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - Per-token losses of shape ``(T_expanded - 1,)``. Positions with
          ``ignore_index`` labels (instruction / image / padding) are 0.
        - Shifted labels of shape ``(T_expanded - 1,)`` (post-expansion,
          aligned with the losses). Use ``labels != -100`` to get the
          mask of positions that actually contribute to the loss.
    """
    sample = prepare_sample(sample, device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Build multimodal inputs (encodes image/audio, interleaves embeddings)
        inputs = model.prepare_multimodal_inputs(
            batch_input_ids=sample["batch_input_ids"],
            batch_labels=sample["batch_labels"],
            batch_X_modals=sample["batch_X_modals"],
        )

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            inputs_embeds=inputs["inputs_embeds"],
            labels=inputs["labels"],
        )

        # Shift logits and labels (standard causal LM convention)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = inputs["labels"][:, 1:].contiguous()

        B, T, V = shift_logits.shape
        per_token_loss = F.cross_entropy(
            shift_logits.view(-1, V).float(),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=ignore_index,
        ).view(B, T)

    # Return only the first (and only) sample in the batch
    return per_token_loss[0].cpu(), shift_labels[0].cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single dataset sample and print per-token losses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to a local HF-style directory or a HuggingFace Hub repo-id.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="p4ms_vqa",
        help="Dataset task name (as defined in config/datasets.yaml).",
    )
    parser.add_argument(
        "--sample_index",
        type=int,
        default=0,
        help="Index of the sample to run inference on.",
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
    args = parser.parse_args()

    # ── 1. Load model ────────────────────────────────────────────────────
    model, tokenizer, model_args, data_args, training_args = load_lmm(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
    )
    device = torch.device(args.device)

    # ── 2. Build dataset & collator ──────────────────────────────────────
    image_processor = (
        model.get_model().visual_encoder.image_processor
        if training_args.visual_branch
        else None
    )
    data_args.tasks = args.dataset_name

    dataset, collator = get_dataset_collator(
        data_args=data_args,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    # ── 3. Grab a single sample and collate it ───────────────────────────
    idx = args.sample_index
    assert idx < len(dataset), f"sample_index {idx} >= dataset length {len(dataset)}"
    raw_sample = dataset[idx]
    batch = collator([raw_sample])

    # ── 4. Forward pass → per-token losses ───────────────────────────────
    token_losses, expanded_labels = get_per_token_losses(model, batch, device)

    # ── 5. Print results ─────────────────────────────────────────────────
    #   NOTE: token_losses and expanded_labels are in the *expanded* space
    #   (image tokens inflated to embeddings), so their length differs from
    #   the raw input_ids.  Use expanded_labels for masking.
    labeled_mask = expanded_labels != -100
    n_labeled = labeled_mask.sum().item()
    labeled_losses = token_losses[labeled_mask]
    mean_loss = labeled_losses.mean().item() if n_labeled > 0 else float("nan")

    print(f"\n{'═' * 70}")
    print(f"Dataset: {args.dataset_name}  |  Sample index: {idx}")
    print(f"User ID: {raw_sample.get('user_id', 'N/A')}")
    print(f"Sequence length (raw): {len(batch['batch_input_ids'][0])} tokens")
    print(f"Sequence length (expanded, with image embeds): {len(token_losses) + 1}")
    print(f"{'═' * 70}")

    # Show conversation text
    for turn in raw_sample["conversation"]:
        instr_preview = turn["instruction"][:120].replace("\n", " ")
        out_preview = turn["output"][:120].replace("\n", " ")
        print(f"  Instruction: {instr_preview}...")
        print(f"  Output:      {out_preview}...")

    print(f"\n{'─' * 70}")
    print(f"  Labeled tokens: {n_labeled}")
    print(f"  Mean loss (labeled): {mean_loss:.4f}")
    print(f"  Total tokens with loss > 0: {(token_losses > 0).sum().item()}")
    print(f"{'─' * 70}")

    # Print a few tokens with highest loss
    if n_labeled > 0:
        n_show = min(10, n_labeled)
        # Only consider labeled positions for the top-k
        labeled_indices = labeled_mask.nonzero(as_tuple=True)[0]
        labeled_vals = token_losses[labeled_indices]
        top_within = labeled_vals.argsort(descending=True)[:n_show]
        top_indices = labeled_indices[top_within]

        # Decode the labeled tokens for display
        labeled_token_ids = expanded_labels[labeled_indices]
        print(f"\n  Top-{n_show} highest-loss labeled positions:")
        for rank, ti in enumerate(top_indices):
            ti = ti.item()
            tok_id = expanded_labels[ti].item()
            tok_str = tokenizer.decode([tok_id])
            print(
                f"    {rank + 1:>2}. expanded_pos={ti:>4}  "
                f"loss={token_losses[ti]:.4f}  "
                f"token='{tok_str}'"
            )

    print(f"\n  Per-token loss vector shape: {token_losses.shape}")
    print(f"  Labeled loss vector shape:   {labeled_losses.shape}")
    print(f"  (Use get_per_token_losses() programmatically for downstream analysis)")
    print()


if __name__ == "__main__":
    main()
