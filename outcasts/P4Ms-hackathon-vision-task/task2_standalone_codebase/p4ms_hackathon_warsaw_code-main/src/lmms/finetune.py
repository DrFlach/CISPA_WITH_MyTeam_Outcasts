import os, sys

sys.path.append(os.getcwd())
import pathlib
import torch

from src.lmms.configs.unified_config import parse_args
from src.lmms.trainer import get_trainer

from src.lmms.utils.util import set_seed
from src.lmms.utils.deepspeed_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)

from src.lmms.models import (
    write_model_info,
    get_model_tokenizer,
)


def setup_modules_to_save(model, training_args):
    save_modules = training_args.save_modules
    if training_args.local_rank in [-1, 0]:
        print(f"save_modules: {save_modules}")
    save_modules = save_modules.split(",")
    for name, param in model.named_parameters():
        require_grad = False
        for target in save_modules:
            if target in name:
                require_grad = True
                break
        param.requires_grad_(require_grad)


def train(model, trainer, training_args):
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and training_args.resume_from_checkpoint
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True


def save(model, training_args):
    state_dict = get_peft_state_maybe_zero_3(model.named_parameters())
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        torch.save(
            non_lora_state_dict,
            os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
        )


def main():
    set_seed(42)
    model_args, data_args, training_args = parse_args()
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    model, tokenizer = get_model_tokenizer(model_args, training_args)
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model.config.use_cache = False
    setup_modules_to_save(model, training_args)

    if training_args.local_rank == 0:
        write_model_info(training_args, model)

    trainer = get_trainer(model, tokenizer, data_args, training_args)
    train(model, trainer, training_args)
    save(model, training_args)


if __name__ == "__main__":
    main()
