from transformers import CLIPImageProcessor, PreTrainedTokenizer
from torch.utils.data import ConcatDataset
from src.lmms.dataset import datasets
import numpy as np
import torch
from hydra.utils import instantiate


def build_dataset(
    args,
    video_processor: CLIPImageProcessor = None,
    tokenizer: PreTrainedTokenizer = None,
) -> ConcatDataset:
    return ConcatDataset(
        [
            instantiate(datasets[task], _partial_=True)(
                args=args, video_processor=video_processor, tokenizer=tokenizer
            )
            for task in args.tasks.split(",")
        ]
    )


class DataCollatorForMultiTaskDataset:
    def __init__(
        self,
        tokenizer,
        is_test: bool = False,
        instruction_to_label: bool = False,
        mm_input_only: bool = False,
    ):
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.instruction_to_label = instruction_to_label
        self.mm_input_only = mm_input_only

    def remove_question(self, instruction: str) -> str:
        for token in ["<image_end>", "<audio_end>"]:
            if token in instruction:
                idx = instruction.find(token)
                instruction = instruction[: idx + len(token)]
                break
        return instruction

    def conversation_to_ids_labels(self, conversation):
        if self.is_test:
            assert (
                len(conversation) == 1
            ), "Test conversation should have only one message."
        input_ids = []
        labels = []

        for message in conversation:
            instruction = message["instruction"]
            output = message["output"]
            if self.mm_input_only:
                instruction = self.remove_question(instruction)
                output = ""
            instruction_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(instruction)
            )
            output_ids = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(output)
            )
            label_ids = (
                [-100] * len(instruction_ids)
                if not self.instruction_to_label
                else instruction_ids.copy()
            )
            if not self.is_test:
                instruction_ids += output_ids
                label_ids += output_ids
            input_ids += instruction_ids
            labels += label_ids
            if self.mm_input_only:
                break
        return input_ids, labels

    def __call__(self, instances):
        batch_input_ids = []
        batch_labels = []
        batch_X_modals = []

        for instance in instances:
            conversation = instance["conversation"]
            input_ids, labels = self.conversation_to_ids_labels(conversation)

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

            X_modals = {}
            image = instance.get("image", None)
            if image is not None:
                X_modals["<image>"] = image

            audio = instance.get("audio", None)
            if audio is not None:
                X_modals["<audio>"] = audio

            batch_X_modals.append(X_modals)

        return {
            "batch_input_ids": batch_input_ids,
            "batch_labels": batch_labels,
            "batch_X_modals": batch_X_modals,
        }


def get_dataset_collator(
    data_args,
    tokenizer,
    image_processor,
    instruction_to_label: bool = False,
    mm_input_only: bool = False,
):
    dataset = build_dataset(
        video_processor=image_processor,
        tokenizer=tokenizer,
        args=data_args,
    )
    data_collator = DataCollatorForMultiTaskDataset(
        tokenizer=tokenizer,
        is_test=data_args.is_test,
        instruction_to_label=instruction_to_label,
        mm_input_only=mm_input_only,
    )

    return dataset, data_collator
