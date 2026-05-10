import json, random, webdataset as wds

from src.lmms.dataset import (
    IterTaskDataset,
    VQADataset,
    sample_to_chat_template,
    get_formatted_question,
)
from datasets import load_dataset
from tenacity import retry, wait_exponential, stop_after_attempt
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")


class OCRVQADataset(IterTaskDataset):
    def add_samples(self):
        self.ocrvqa_split = self.kwargs.get("split", "train")
        self.parse_annotations()

    def parse_annotations(self):
        self.annotations_raw = json.load(open(self.annot_path, "r"))
        self.annotations = dict()
        for v in self.annotations_raw.values():
            indices = [
                idx
                for idx, a in enumerate(v["answers"])
                if a.lower() not in ["no", "yes"]
            ]
            questions = [v["questions"][i] for i in indices]
            answers = [v["answers"][i] for i in indices]
            self.annotations[v["imageURL"]] = {
                "questions": questions,
                "answers": answers,
                "split": v["split"],
            }

    def to_dict(self, sample):
        key = sample[1]["url"]
        annot = self.annotations.get(key, None)
        if annot is None:
            print(f"Warning: {key} not in annotations")
            return None
        if len(annot["questions"]) == 0:
            print(f"Warning: {key} has no more questions")
            return None
        return {
            "image": sample[0],
            "key": key,
            "split": {1: "train", 2: "val", 3: "test"}[annot["split"]],
            "success": sample[1]["status"] == "success",
        }

    def get_iterable_dataset(self):
        dataset = wds.DataPipeline(
            wds.ResampledShards(self.media_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )
        return wds.split_by_worker(wds.split_by_node(dataset))

    def __len__(self):
        assert (
            self.num_samples is not None
        ), "num_samples must be specified in config/datasets.yaml"
        return self.num_samples

    def _put_logic(self, q, sample, produced: int) -> None:
        try:
            image = self.video_processor.preprocess(
                sample["image"], return_tensors="pt"
            )["pixel_values"]
        except Exception as e:
            print(f"Warning: Failed to preprocess image for {sample['key']}: {e}")
            return

        if any(
            [
                sample is None,
                sample["split"] != self.ocrvqa_split,
                not sample["success"],
            ]
        ):
            return
        key = sample["key"]
        annot = self.annotations[key]

        turn_idx = 0
        conversations = []
        for idx in range(len(annot["questions"])):
            instruction = annot["questions"][idx]
            answer = annot["answers"][idx]
            if answer.strip() == "":
                print(f"Warning: Empty answer for {sample['key']}, question idx {idx}")
                continue
            if instruction.strip() == "":
                print(
                    f"Warning: Empty question for {sample['key']}, question idx {idx}"
                )
                continue
            # Only add modality tags for the first turn if packing conversations
            insert_modality = (turn_idx == 0) or not self.pack_conversations
            conversation = {
                "instruction": get_formatted_question(
                    instruction, "image", insert_modality
                ),
                "output": answer,
            }
            conversations.append(conversation)
            turn_idx += 1

        new_samples = self.get_samples_from_convs(dict(), conversations)

        for sample in new_samples:
            output_sample = sample_to_chat_template(sample, self.tokenizer)
            output_sample["image"] = image

            q.put(output_sample, timeout=0.2)
            produced += 1


class SynthDogENDataset(IterTaskDataset):
    def add_samples(self): ...

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(6)
    )
    def get_iterable_dataset(self):
        dataset = load_dataset(
            "naver-clova-ix/synthdog-en", split="train", streaming=True
        )
        return self.shard_dataset(dataset)

    def _put_logic(self, q, sample, produced: int) -> None:
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
        q.put(output_sample, timeout=0.2)
        produced += 1


class TextOCRDataset(VQADataset):
    def add_samples(self):
        with open(self.annot_path, "r") as f:
            for line in f:
                annot = json.loads(line)
                instruction = get_formatted_question(
                    "Describe me the text you see in the image. Write down the text exactly as it appears.",
                    "image",
                )
                output = annot["caption_image"] + "\n" + annot["caption_text"]
                conversation = {
                    "instruction": instruction,
                    "output": output,
                }
                new_sample = self.get_samples_from_convs(
                    {"image_path": f"{self.media_path}/{annot['filename']}"},
                    [conversation],
                )[0]

                self.samples.append(new_sample)
                self.tot += 1
