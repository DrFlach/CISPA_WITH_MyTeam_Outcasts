from typing import List
from transformers import CLIPImageProcessor, PreTrainedTokenizer
from torch.utils.data import Dataset
from src.lmms.configs.unified_config import DataArguments
from PIL import Image
from typing import Any, Dict, List

import whisper, torch, multiprocessing as mp, numpy as np, os, random
from datasets.iterable_dataset import ShufflingConfig
from time import sleep
from whisper.audio import SAMPLE_RATE


def sample_to_chat_template(sample, tokenizer):
    conversation = sample["conversation"]
    conv_formatted = []
    for i, turn in enumerate(conversation):
        messages = []
        if i == 0:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )

        messages.append({"role": "user", "content": turn["instruction"]})
        conv_formatted.append(
            {
                "instruction": tokenizer.apply_chat_template(
                    conversation=messages, add_generation_prompt=True, tokenize=False
                ),
                "output": turn["output"],
            }
        )
    return {"conversation": conv_formatted}


class TaskDataset(Dataset):
    def __init__(
        self,
        args: DataArguments,
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        media_path: str = None,
        annot_path: str = None,
        metadata_paths: List[str] = None,
        num_samples: int = None,
        **kwargs,
    ):
        self.samples = []
        self.tot = 0
        self.args = args
        self.video_processor = video_processor
        self.tokenizer = tokenizer
        self.media_path = media_path
        self.annot_path = annot_path
        self.metadata_paths = metadata_paths
        self.num_samples = num_samples
        self.kwargs = kwargs

        self.add_samples()
        self.maybe_duplicate_samples()

    def add_samples(self):
        raise NotImplementedError

    def maybe_duplicate_samples(self):
        n_duplicates = self.kwargs.get("n_duplicates", False)
        if n_duplicates and n_duplicates > 1:
            self.samples = self.samples * n_duplicates
            print(
                f"Duplicating {n_duplicates} times, total: {len(self.samples)} samples"
            )
            self.tot = len(self.samples)

    def __len__(self):
        return len(self.samples) if self.num_samples is None else self.num_samples

    def trunc_dataset(self):
        if self.num_samples is not None:
            self.samples = random.choices(self.samples, k=self.num_samples)
            self.tot = self.num_samples

    def get_samples_from_convs(
        self, modality_data: Dict[str, Any], conversations: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        samples = []
        for conv in conversations:
            sample = modality_data.copy()
            sample.update({"conversation": [conv]})
            samples.append(sample)

        return samples


class VQADataset(TaskDataset):
    def load_image(self, sample) -> torch.Tensor:
        image_path = sample["image_path"]
        image = (
            Image.open(image_path)
            .convert("RGB")
            .resize(
                (self.args.data_image_size, self.args.data_image_size),
                Image.Resampling.BILINEAR,
            )
        )
        image = self.video_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        return image.unsqueeze(0)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        data = sample_to_chat_template(sample, self.tokenizer)
        data["image"] = self.load_image(sample)
        return data


class IterTaskDataset(TaskDataset):
    SENTINEL = "__SENTINEL__"

    def __init__(self, buffer_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctx = mp.get_context("spawn")
        self._q: mp.Queue = self._ctx.Queue(maxsize=buffer_size)
        self._stop = self._ctx.Event()

        self._producer = self._ctx.Process(
            target=self._producer_loop,
            args=(
                self.get_iterable_dataset,
                self._q,
                self._stop,
                self.num_samples,
                self.SENTINEL,
                self._put_logic,
            ),
            daemon=True,
        )
        self._producer.start()

    def get_iterable_dataset(self):
        raise NotImplementedError

    def shard_dataset(self, dataset):
        shuffling = ShufflingConfig(np.random.default_rng(42), _original_seed=42)
        dataset._shuffling = shuffling
        dataset._ex_iterable = dataset._ex_iterable.shuffle_data_sources(
            dataset._effective_generator()
        )

        num_gpus = int(os.environ.get("WORLD_SIZE", 1))
        gpu_idx = int(os.environ.get("RANK", 0))
        shard = dataset.shard(num_shards=num_gpus, index=gpu_idx)
        return shard

    def _put_logic(self, q: mp.Queue, item, produced: int) -> None:
        q.put(item, timeout=0.2)
        produced += 1

    @staticmethod
    def _producer_loop(
        dataset_getter,
        q: mp.Queue,
        stop_event,
        max_items: int,
        sentinel,
        put_logic=None,
    ):
        try:
            dataset = dataset_getter()
            it = iter(dataset)
            produced = 0
            while not stop_event.is_set() and produced < max_items:
                try:
                    sample = next(it)
                except StopIteration:
                    break
                while not stop_event.is_set():
                    try:
                        put_logic(q, sample, produced)
                        break
                    except Exception:
                        continue
        except Exception as e:
            # Propagate error to consumer.
            try:
                q.put(("__EXC__", repr(e)))
            except Exception:
                pass
        finally:
            # Signal completion.
            try:
                q.put(sentinel)
            except Exception:
                pass

    def __getitem__(self, index: int):
        while True:
            try:
                item = self._q.get(timeout=1)  # blocking read
            except mp.queues.Empty:
                continue
            if item is self.SENTINEL:
                # If producer finished early, stop here.
                raise IndexError
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__EXC__":
                # Re-raise producer exception in the consumer process.
                raise RuntimeError(f"Producer failed: {item[1]} {self}")
            if item is not None:
                return item

    def __len__(self):
        return self.num_samples

    def close(self):
        try:
            self._stop.set()
            if self._producer.is_alive():
                self._producer.join(timeout=1.0)
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass

    def __del__(self):
        self.close()


def get_formatted_question(
    question: str,
    modality: str,
    insert_modality: bool = True,
    optional_string: str = "",
) -> str:
    question = question.strip().replace("<image>", "").replace("<audio>", "")
    formatted_question = ""
    if insert_modality:
        modality_list = modality.split("+")
        for modality_name in modality_list:
            if modality_name == "image":
                formatted_question += "<image_start><image><image_end>"
            elif modality_name == "audio":
                formatted_question += "<audio_start><audio><audio_end>"
            elif modality_name == "text":
                assert (
                    optional_string != "" and optional_string is not None
                ), "optional_string must be provided for text modality"
                formatted_question += (
                    f"You are given the following text passage: {optional_string}"
                )
            else:
                raise ValueError(f"Unsupported modality: {modality_name}")
            formatted_question += "\n"
    formatted_question += "<question_start>" + question + "<question_end>"
    return formatted_question
