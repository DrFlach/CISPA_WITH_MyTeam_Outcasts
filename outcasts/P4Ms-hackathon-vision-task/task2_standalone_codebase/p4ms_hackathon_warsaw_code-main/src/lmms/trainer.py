# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os, math
from typing import List, Optional
from tqdm import tqdm
import torch, torch.distributed as dist
from torch.utils.data import Sampler
from contextlib import nullcontext

from transformers import Trainer
from transformers.trainer import has_length

from src.lmms.dataset.multitask_dataset import get_dataset_collator
from src.lmms.utils.util import prepare_sample
from torch.optim.lr_scheduler import LambdaLR
from torch.func import functional_call


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks


class UnifiedTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        ddp = super()._wrap_model(model, training=training, dataloader=dataloader)
        if isinstance(ddp, DDP):
            pass
            # try:
            #     ddp._set_static_graph()  # PyTorch static-graph mode
            # except Exception:
            #     pass
        return ddp

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        keys_to_match = self.args.save_modules.split(",")

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            self.model.named_parameters(), keys_to_match
        )
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            torch.save(
                weight_to_save, os.path.join(output_dir, f"finetune_weights.bin")
            )
            super().save_state()

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(UnifiedTrainer, self)._save(output_dir, state_dict)


class NoDeepSpeedTrainer(UnifiedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.optimizer is None:
            self.optimizer = optimizer

        if self.lr_scheduler is None:
            max_lr = self.args.learning_rate  # 1e-5
            min_lr = 1e-6  # 1e-6

            min_ratio = min_lr / max_lr
            warmup_steps = self.args.get_warmup_steps(num_training_steps)

            def lr_lambda(current_step):
                # Phase 1: Logarithmic Warmup
                if current_step < warmup_steps:
                    # Formula: log(current + 1) / log(warmup + 1)
                    # We add 1 to avoid log(0)
                    log_progress = math.log(current_step + 1) / math.log(
                        max(1, warmup_steps) + 1
                    )

                    # Map [0, 1] -> [min_ratio, 1.0]
                    return min_ratio + log_progress * (1.0 - min_ratio)

                # Phase 2: Linear Decay (Standard)
                decay_progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                return max(0.0, 1.0 - decay_progress * (1.0 - min_ratio))

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        return self.lr_scheduler

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
        else:
            loss = super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch
            )
            outputs = None
        if self.args.world_size > 1:
            loss = loss / self.args.world_size

        return (loss, outputs) if return_outputs else loss


class GlobalAdaptiveClippingTrainer(UnifiedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        self._ga_buffer = []  # Buffer for gradient accumulation micro-batches

    def training_step(self, model, inputs, num_items_in_batch):
        model.train()
        rank = dist.get_rank()
        inputs = self._prepare_inputs(inputs)

        # Buffer this micro-batch
        self._ga_buffer.append(inputs)

        gas = self.args.gradient_accumulation_steps
        # If not yet at the accumulation boundary, return zero loss (no backward)
        if len(self._ga_buffer) < gas:
            return torch.tensor(0.0, device=model.device)

        # --- We have all micro-batches. Process them. ---
        buffered_inputs = self._ga_buffer
        self._ga_buffer = []

        inner_model = model.module if hasattr(model, "module") else model

        # Pre-compute labels for all micro-batches
        all_batch_labels = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
            for inp in buffered_inputs:
                labels = inner_model.prepare_multimodal_inputs(**inp)["labels"].cpu()
                all_batch_labels.append(labels)

        params_and_buffers = {k: v for k, v in inner_model.named_parameters()}
        params_and_buffers.update({k: v for k, v in inner_model.named_buffers()})
        trainable_param_names = [
            n for n, p in inner_model.named_parameters() if p.requires_grad
        ]

        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        # --- PASS 1: Compute per-sample norms across ALL micro-batches ---
        all_sample_norms = []  # stored on CPU to save VRAM

        # Temporarily set eval mode for Pass 1 — gradient checkpointing uses
        # torch.utils.checkpoint which is incompatible with torch.autograd.grad().
        inner_model.eval()

        for inp, batch_labels in tqdm(
            zip(buffered_inputs, all_batch_labels), desc="Pass 1", disable=rank != 0
        ):
            # Create fresh shadow params per micro-batch so the graph can be freed after
            shadow_state = {}
            shadow_trainables = {}
            for k, v in params_and_buffers.items():
                if k in trainable_param_names:
                    shadow_v = v.detach().requires_grad_(True)
                    shadow_state[k] = shadow_v
                    shadow_trainables[k] = shadow_v
                else:
                    shadow_state[k] = v

            batch_inputs = prepare_sample(inp, model.device)

            # Functional Forward
            output = functional_call(
                inner_model, shadow_state, args=(), kwargs=batch_inputs
            )

            # Loss Calculation
            logits = output.logits[..., :-1, :].contiguous()
            del output  # free non-logit activations
            labels = batch_labels[..., 1:].to(model.device).contiguous()
            losses = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            del logits  # free logits tensor
            losses = losses.view(labels.size(0), -1).mean(dim=1)

            # Per-sample grad norms — use retain_graph=False on the last sample
            n_samples = losses.size(0)
            for s_idx in range(n_samples):
                retain = s_idx < n_samples - 1  # free graph on last sample
                grads = torch.autograd.grad(
                    losses[s_idx], shadow_trainables.values(), retain_graph=retain
                )
                total_norm = torch.norm(
                    torch.stack([torch.norm(g, 2) for g in grads]), 2
                )
                all_sample_norms.append(total_norm.item())  # move to CPU scalar
                del grads

            del shadow_state, shadow_trainables, losses

        # Restore training mode for Pass 2 (re-enables gradient checkpointing)
        model.train()
        torch.cuda.empty_cache()

        # Compute global min norm across all processes
        if not all_sample_norms:
            local_min_norm = torch.tensor(1e9, device=model.device)
        else:
            local_min_norm = torch.tensor(min(all_sample_norms), device=model.device)
        if world_size > 1:
            dist.all_reduce(local_min_norm, op=dist.ReduceOp.MIN)

        global_min_norm = local_min_norm + 1e-6
        current_norms = torch.tensor(all_sample_norms, device=model.device)
        loss_scales = torch.clamp(global_min_norm / (current_norms + 1e-6), max=1.0)

        # --- PASS 2: Real forward-backward for each micro-batch with loss scaling ---
        total_loss_val = 0.0
        norm_offset = 0
        for inp, batch_labels in tqdm(
            zip(buffered_inputs, all_batch_labels),
            desc=f"Pass 2, ls_mean: {loss_scales.mean().item():.4f}, ls_min: {loss_scales.min().item():.4f}, ls_max: {loss_scales.max().item():.4f}",
            disable=rank != 0,
        ):
            batch_inputs = prepare_sample(inp, model.device)
            output = inner_model(**batch_inputs)

            logits = output.logits[..., :-1, :].contiguous()
            labels = batch_labels[..., 1:].to(model.device).contiguous()
            losses = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            losses = losses.view(labels.size(0), -1).mean(dim=1)

            n_samples = losses.size(0)
            micro_scales = loss_scales[norm_offset : norm_offset + n_samples]
            norm_offset += n_samples

            weighted_loss = (losses * micro_scales).mean()
            total_loss_val += losses.mean().item()

            kwargs = {"scale_wrt_gas": False}
            self.accelerator.backward(weighted_loss, **kwargs)

        avg_loss = torch.tensor(total_loss_val, device=model.device)
        if world_size > 1:
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss / world_size

        return avg_loss.detach() * len(loss_scales)


def get_trainer(model, tokenizer, data_args, training_args) -> UnifiedTrainer:
    image_processor = (
        model.get_model().visual_encoder.image_processor
        if training_args.visual_branch
        else None
    )
    dataset, collator = get_dataset_collator(
        data_args=data_args, tokenizer=tokenizer, image_processor=image_processor
    )

    if training_args.global_adaptive_clipping:
        trainer_class = GlobalAdaptiveClippingTrainer
    elif training_args.deepspeed is None:
        trainer_class = NoDeepSpeedTrainer
    else:
        trainer_class = UnifiedTrainer

    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    return trainer
