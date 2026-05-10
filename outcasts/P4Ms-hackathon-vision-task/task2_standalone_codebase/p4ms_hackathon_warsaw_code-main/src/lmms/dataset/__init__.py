from src.lmms.dataset.task_dataset import (
    get_formatted_question,
    sample_to_chat_template,
    TaskDataset,
    VQADataset,
    IterTaskDataset,
)
import yaml


with open("config/datasets.yaml", mode="r") as fd:
    datasets = yaml.safe_load(fd)
