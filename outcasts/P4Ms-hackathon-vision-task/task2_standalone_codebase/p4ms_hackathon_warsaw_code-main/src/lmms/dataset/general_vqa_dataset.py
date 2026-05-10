import json, os, random

from PIL import Image
from src.lmms.dataset import VQADataset, get_formatted_question, sample_to_chat_template
from datasets import load_dataset, load_from_disk
import numpy as np


class LLaVAVQADataset(VQADataset):
    def add_samples(self):
        metadata = json.load(open(self.metadata_paths[0], "r"))

        llava_image_id_to_image_path = dict()
        for entry in metadata:
            llava_image_id = entry["image"].split("_")[-1].split(".")[0]
            llava_image_id_to_image_path[llava_image_id] = os.path.join(
                self.media_path, entry["image"]
            )

        llava_annots = json.load(open(self.annot_path, "r"))
        for annot in llava_annots:
            conversations = annot["conversations"]
            conv_formatted = []
            for idx in range(len(conversations) // 2):
                assert conversations[idx * 2]["from"] == "human", conversations[idx * 2]
                assert conversations[idx * 2 + 1]["from"] == "gpt", conversations[
                    idx * 2 + 1
                ]
                # if one question per sample insert everytime, else only on the first question
                insert_modality = (idx == 0) or not self.pack_conversations
                instruction = get_formatted_question(
                    conversations[idx * 2]["value"],
                    "image",
                    insert_modality=insert_modality,
                )
                output = conversations[idx * 2 + 1]["value"]
                conv_formatted.append(
                    {
                        "instruction": instruction,
                        "output": output,
                    }
                )

            new_samples = self.get_samples_from_convs(
                {"image_path": llava_image_id_to_image_path[annot["id"]]},
                conv_formatted,
            )
            self.samples.extend(new_samples)
            self.tot += len(conv_formatted)

        self.trunc_dataset()


class LLaVACC3MPretrainDataset(VQADataset):
    def add_samples(self):
        llava_annots = json.load(open(self.annot_path, "r"))
        for annot in llava_annots:
            conversations = annot["conversations"]
            for idx in range(len(conversations) // 2):
                assert conversations[idx * 2]["from"] == "human", conversations[idx * 2]
                assert conversations[idx * 2 + 1]["from"] == "gpt", conversations[
                    idx * 2 + 1
                ]
                instruction = get_formatted_question(
                    conversations[idx * 2]["value"], "image"
                )
                output = conversations[idx * 2 + 1]["value"]

                new_samples = self.get_samples_from_convs(
                    {"image_path": f"{self.media_path}/{annot['image']}"},
                    [
                        {
                            "instruction": instruction,
                            "output": output,
                        }
                    ],
                )
                self.samples.extend(new_samples)
                self.tot += 1

        self.trunc_dataset()


class TextCapsDataset(VQADataset):
    def add_samples(self):
        with open(self.annot_path, "r") as f:
            annots = json.load(f)["data"]

        for annot in annots:
            instruction = get_formatted_question(
                "Describe me what you see in the image.", "image"
            )
            output = random.choice(annot["reference_strs"])
            if output == "":
                continue

            new_samples = self.get_samples_from_convs(
                {"image_path": f"{self.media_path}/{annot['image_id']}.jpg"},
                [
                    {
                        "instruction": instruction,
                        "output": output,
                    }
                ],
            )
            self.samples.extend(new_samples)
            self.tot += 1

        self.trunc_dataset()


class ArxivQADataset(VQADataset):
    def _parse_question(self, question: str, options: list) -> str:
        question_text = question + "\n"
        for idx, option in enumerate(options[:4]):
            option = chr(ord("A") + idx) + ". " + option
            question_text += option + "\n"
        question_text += "Please provide the correct option (A, B, C, or D). Then, provide rationale for your choice."
        return question_text

    def _parse_answer(self, answer: str, rationale: str) -> str:
        return f"The correct answer is {answer[0]}. Rationale: {rationale}"

    def add_samples(self):
        annots = load_dataset("MMInstruction/ArxivQA", split="train", streaming=False)

        for annot in annots:
            try:
                if annot["label"][0].upper() not in list("ABCD"):
                    continue
            except Exception:
                print(f"Skipping invalid answer: {annot['label']}")
                # print(f"{annot=}")
                continue
            instruction = self._parse_question(annot["question"], annot["options"])
            instruction = get_formatted_question(instruction, "image")
            output = self._parse_answer(annot["label"], annot["rationale"])

            new_samples = self.get_samples_from_convs(
                {"image_path": f"{self.media_path}/{annot['image']}"},
                [
                    {
                        "instruction": instruction,
                        "output": output,
                    }
                ],
            )
            self.samples.extend(new_samples)
            self.tot += 1


class SynthDogENDataset(VQADataset):
    def add_samples(self):
        self.samples = load_dataset(
            "naver-clova-ix/synthdog-en",
            split="train",
            cache_dir=self.kwargs.get("cache_dir", None),
        )
        self.tot = len(self.samples)
        self.trunc_dataset()

    def trunc_dataset(self):
        if self.num_samples is not None:
            indices = np.random.permutation(len(self.samples))[: self.num_samples]
            self.samples = self.samples.select(indices)
            self.tot = self.num_samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = (
            "What is written in the image? Write down the text exactly as it appears."
        )
        answer = (
            "The text is " + eval(sample["ground_truth"])["gt_parse"]["text_sequence"]
        )
        if answer.strip() == "":
            print("Warning: Empty answer in synthdog-en")
            return

        conversation = {
            "instruction": get_formatted_question(instruction, "image"),
            "output": answer,
        }
        image = self.video_processor.preprocess(sample["image"], return_tensors="pt")[
            "pixel_values"
        ]
        new_sample = self.get_samples_from_convs(dict(), [conversation])[0]

        output_sample = sample_to_chat_template(new_sample, self.tokenizer)
        output_sample["image"] = image
        return output_sample


class HFP4MsVQADataset(VQADataset):
    def add_samples(self):
        self.hf_dataset = load_dataset(
            self.kwargs["hf_repo"],
            self.kwargs.get("hf_config", None),
            split=self.kwargs.get("hf_split", "train"),
            cache_dir=self.kwargs.get("cache_dir", None),
        )

        text_cols = [col for col in self.hf_dataset.column_names if col != "path"]
        for hf_idx, item in enumerate(self.hf_dataset.select_columns(text_cols)):
            conversation = item["conversation"]
            conv_formatted = []
            for idx, conv in enumerate(conversation):
                insert_modality = True
                instruction = get_formatted_question(
                    conv["instruction"],
                    "image",
                    insert_modality=insert_modality,
                )
                output = conv["output"]
                conv_formatted.append(
                    {
                        "instruction": instruction,
                        "output": output,
                    }
                )

            user_id = item.get("user_id", "unknown")

            new_samples = self.get_samples_from_convs(
                {
                    "user_id": user_id,
                    "hf_idx": hf_idx,  # index for lazy loading the image
                },
                conv_formatted,
            )
            self.samples.extend(new_samples)
            self.tot += 1

    def __getitem__(self, index):
        sample = self.samples[index]
        data = sample_to_chat_template(sample, self.tokenizer)

        # Lazy image preprocessing (runs in DataLoader workers)
        hf_idx = sample["hf_idx"]
        pil_image: Image.Image = self.hf_dataset[hf_idx]["path"]
        pil_image = pil_image.convert("RGB").resize(
            (self.args.data_image_size, self.args.data_image_size),
            Image.Resampling.BILINEAR,
        )
        data["image"] = self.video_processor.preprocess(pil_image, return_tensors="pt")[
            "pixel_values"
        ]
        data["user_id"] = sample["user_id"]
        return data
