# Implementation Details & Starter Code

This document covers the technical details of the codebase provided for the hackathon, including how to set up the environment, load the models and datasets, run inference, and understand the model architecture.

## Repository Structure

```
.
├── README.md                          # Main Hackathon README
├── pyproject.toml                     # Project dependencies (Python 3.10)
│
├── config/
│   ├── datasets.yaml                  # Dataset registry (task names → classes & HF repos)
│   ├── default_args.json              # Default model/data/training arguments
│   ├── vision.yaml                    # Vision encoder configurations (LLaVA-HR variants)
│   └── models/
│       ├── blueprints.yaml            # Shared model blueprints (e.g., vqa_1b)
│       └── fft.yaml                   # Full fine-tuning model configs (target & shadow)
│
├── scripts/
│   ├── load_lmm_from_hf_dir.py       # ⭐ Load target/shadow LMM from HuggingFace
│   └── inference_example.py           # ⭐ End-to-end inference example (per-token losses)
│
├── src/
│   ├── __init__.py
│   └── lmms/                          # Core LMM implementation
│       ├── configs/
│       │   └── unified_config.py      # ModelArguments, DataArguments, TrainingArguments
│       ├── dataset/
│       │   ├── __init__.py            # Loads datasets.yaml registry
│       │   ├── task_dataset.py        # Base TaskDataset, VQADataset classes
│       │   ├── general_vqa_dataset.py # Dataset classes incl. HFP4MsVQADataset
│       │   ├── multitask_dataset.py   # Dataset builder, DataCollator, get_dataset_collator()
│       │   └── ocr_vqa_dataset.py     # OCR-VQA dataset variants
│       ├── models/
│       │   ├── __init__.py            # Model registry & get_model_tokenizer()
│       │   ├── unified_mllm.py       # Dynamic model factory (model_factory)
│       │   ├── unified_arch.py        # UnifiedMetaModel, multimodal input preparation
│       │   ├── llava_hr_vision/       # LLaVA-HR vision encoder + projector
│       │   ├── llms/                  # Custom LLM modeling files (OLMo-2, LLaMA)
│       │   ├── adaptations/           # Adapter modules
│       │   ├── peft_hyper/            # PEFT/LoRA hyperparameter modules
│       │   ├── speech/                # Speech encoder (not used in this task)
│       │   └── utils/                 # Model utilities (checkpoint key mapping, etc.)
│       ├── utils/
│       │   ├── util.py                # prepare_sample(), set_seed(), checkpoint loading
│       │   └── deepspeed_utils.py     # DeepSpeed ZeRO-3 weight gathering utilities
│       ├── deepspeed/
│       │   └── stage2.json            # DeepSpeed ZeRO Stage-2 config
│       ├── finetune.py                # Training entry point
│       └── trainer.py                 # Custom HF Trainer classes
│
├── bash_scripts/
│   ├── judec_run.bash                 # SLURM job script (Jülich HPC)
│   └── train/
│       ├── vqa.bash                   # Training script for the target model
│       └── vqa_shadow.bash            # Training script for the shadow model
│
└── notebooks/
    └── hf_dataset_demo.ipynb          # Interactive demo: load model + dataset, run inference
```

## Environment Setup

### Prerequisites

- **Python 3.10** (required)
- **CUDA 12.x** with a GPU that supports `bfloat16` (e.g., A100, H100)
- [**`uv`**](https://docs.astral.sh/uv/) (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd multimodal_memorization

# Create a virtual environment and install dependencies
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

> [!NOTE]
> The `flash-attn` wheel is pinned to CUDA 12.2 + PyTorch 2.3. If your setup differs, you may need to install a compatible flash-attention build manually.

### HuggingFace Authentication

The models and datasets are hosted on HuggingFace. Make sure you're authenticated:

```bash
pip install huggingface_hub
huggingface-cli login
```

### Pre-downloading Models (Recommended)

On login/head nodes (with internet access), pre-download the model snapshots so compute nodes can load them offline:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('SprintML/target_lmm')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('SprintML/shadow_lmm')"
```

## Loading the Models

The `scripts/load_lmm_from_hf_dir.py` module provides the `load_lmm()` function — the **primary entry point** for loading either the target or shadow model.

### From Python

```python
import sys, os
sys.path.append(os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

from load_lmm_from_hf_dir import load_lmm

# Load the target model from HuggingFace
model, tokenizer, model_args, data_args, training_args = load_lmm(
    model_dir="SprintML/target_lmm",  # or SprintML/shadow_lmm
    device="cuda",
    dtype="bf16",
)
```

### From the Command Line

```bash
# Download + load (first run, requires internet)
python scripts/load_lmm_from_hf_dir.py --model_dir SprintML/target_lmm

# Load from local cache (offline, e.g., on compute nodes)
python scripts/load_lmm_from_hf_dir.py \
    --model_dir $HOME/.cache/huggingface/hub/models--SprintML--target_lmm/snapshots/<hash>
```

### What `load_lmm()` Returns

| Return Value | Type | Description |
|---|---|---|
| `model` | `UnifiedMetaForCausalLM` | The full multimodal model (LLM + vision encoder + projector). |
| `tokenizer` | `AutoTokenizer` | OLMo-2 tokenizer with added multimodal special tokens. |
| `model_args` | `ModelArguments` | Model configuration (LLM name, encoder type, checkpoint path, etc.). |
| `data_args` | `DataArguments` | Data configuration (image size, task names, test mode, etc.). |
| `training_args` | `TrainingArguments` | Training configuration (dtype, batch size, etc.). |

## Loading the Datasets

All datasets are registered in `config/datasets.yaml` and loaded from HuggingFace. The three relevant dataset configs for this hackathon are:

| Config Name | Description | HF Config | Samples |
|---|---|---|---|
| `p4ms_vqa_hf_task` | **Scrubbed evaluation set** — PII redacted from images and text | `task` | 1000 × 3 |
| `p4ms_vqa_hf_validation_w_pii_image_and_text` | **Validation set** — original PII in images and text | `validation_pii` | 280 × 3 |
| `p4ms_vqa_hf_validation_wo_pii_image_w_pii_text` | **Validation set** — PII in text only, redacted from images | `validation_pii_txt_only` | 280 × 3 |

All three are hosted under `SprintML/P4Ms-hackathon-vision-task` on HuggingFace.

### Programmatic Access

```python
from src.lmms.dataset.multitask_dataset import get_dataset_collator

# Set the task to the evaluation dataset
data_args.tasks = "p4ms_vqa_hf_task"

image_processor = model.get_model().visual_encoder.image_processor

dataset, collator = get_dataset_collator(
    data_args=data_args,
    tokenizer=tokenizer,
    image_processor=image_processor,
)

# Access a single sample
sample = dataset[0]
print(sample["conversation"])  # List of dicts with "instruction" and "output" keys
print(sample["user_id"])       # User identifier for submission
```

### Direct HuggingFace Access

You can also load the raw dataset directly without the codebase wrapper:

```python
from datasets import load_dataset

# Scrubbed evaluation set
ds = load_dataset("SprintML/P4Ms-hackathon-vision-task", "task", split="train")

# Validation set with PII
ds_val = load_dataset("SprintML/P4Ms-hackathon-vision-task", "validation_pii", split="train")
```

Each row contains:
- `path`: A PIL image
- `user_id`: String identifier
- `conversation`: List of `{"instruction": ..., "output": ...}` dicts

## Running Inference

### Quick Start: Per-Token Losses

The `scripts/inference_example.py` script demonstrates end-to-end inference on a single sample:

```bash
python scripts/inference_example.py \
    --model_dir SprintML/target_lmm \
    --dataset_name p4ms_vqa_hf_task \
    --sample_index 0
```

This will:
1. Load the target model
2. Build the dataset and collator
3. Run a forward pass on the specified sample
4. Print per-token cross-entropy losses, showing which tokens the model is most/least confident about

### Programmatic Inference

```python
import torch
import torch.nn.functional as F
from src.lmms.utils.util import prepare_sample

# Assume model, tokenizer, dataset, collator are already loaded (see above)

idx = 42
raw_sample = dataset[idx]
batch = collator([raw_sample])
batch = prepare_sample(batch, torch.device("cuda"))

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # Build multimodal inputs (encodes image, interleaves embeddings)
    inputs = model.prepare_multimodal_inputs(
        batch_input_ids=batch["batch_input_ids"],
        batch_labels=batch["batch_labels"],
        batch_X_modals=batch["batch_X_modals"],
    )

    # Forward pass
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        position_ids=inputs["position_ids"],
        inputs_embeds=inputs["inputs_embeds"],
        labels=inputs["labels"],
    )

    # Per-token losses
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = inputs["labels"][:, 1:].contiguous()
    B, T, V = shift_logits.shape

    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(B, T)
```

### Generating Text (Autoregressive Decoding)

```python
with torch.no_grad():
    generated_ids = model.generate(
        batch_input_ids=batch["batch_input_ids"],
        batch_labels=batch["batch_labels"],
        batch_X_modals=batch["batch_X_modals"],
        max_new_tokens=50,
    )
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

## Understanding the Architecture

### Model Stack

```
┌──────────────────────────────────┐
│     OLMo-2 1B (Causal LM)       │  ← Base language model (allenai/OLMo-2-0425-1B-Instruct)
├──────────────────────────────────┤
│     MLP Vision-Language          │  ← 2-layer MLP projector (1024 → 2048)
│     Projector                    │
├──────────────────────────────────┤
│     LLaVA-HR Vision Encoder     │  ← Dual-path: CLIP ViT-L/14@336 + ConvNeXt-L@1024
│     (MultiPath CLIP)             │
└──────────────────────────────────┘
```

- **Image resolution:** 1024 × 1024 pixels
- **Image tokens:** The vision encoder produces a grid of visual tokens that are projected into the LLM's embedding space
- **Special tokens:** `<image_start>`, `<image>`, `<image_end>`, `<question_start>`, `<question_end>`

### Input Format

Each sample is formatted using a chat template:

```
<|system|>You are a helpful assistant.<|end|>
<|user|><image_start><image><image_end>
<question_start>[QUESTION TEXT]<question_end><|end|>
<|assistant|>[ANSWER TEXT]
```

The `<image>` token is replaced at runtime with the encoded visual embeddings from the vision encoder.

### Key Classes

| Class | File | Purpose |
|---|---|---|
| `UnifiedMetaForCausalLM` | `src/lmms/models/unified_arch.py` | Main model interface — `prepare_multimodal_inputs()`, `generate()` |
| `UnifiedMetaModel` | `src/lmms/models/unified_arch.py` | Inner model — holds vision encoder, projectors, `load_pretrained_weights()` |
| `HFP4MsVQADataset` | `src/lmms/dataset/general_vqa_dataset.py` | Hackathon dataset loader (HuggingFace-backed, lazy image loading) |
| `DataCollatorForMultiTaskDataset` | `src/lmms/dataset/multitask_dataset.py` | Tokenizes conversations, builds `batch_input_ids`, `batch_labels`, `batch_X_modals` |
| `VQADataset` | `src/lmms/dataset/task_dataset.py` | Base class for all VQA datasets |
| `ModelArguments` / `DataArguments` / `TrainingArguments` | `src/lmms/configs/unified_config.py` | Configuration dataclasses |

## Training Pipeline (Reference Only)

> [!NOTE]
> The training pipeline is provided for **reference purposes** — to help you understand how the target model was trained. You are not required to retrain any models.

The target model was trained by fine-tuning the full model (FFT — full fine-tuning, no LoRA) on a mixture of VQA datasets:

```
p4ms_vqa (unknown to you), llava_vqa, synthdog_en, ocrvqa, text_ocr, text_caps
```

The shadow model was trained on the same mixture but with `p4ms_vqa_shadow` instead of `p4ms_vqa`.

Key training details visible from the bash scripts:
- **Base LLM:** `allenai/OLMo-2-0425-1B-Instruct`
- **Optimizer:** AdamW
- **Precision:** bf16
- **Parallelism:** DeepSpeed ZeRO Stage-2
- **Epochs:** 1
- **Effective batch size:** 256

### Running Training (if needed)

```bash
bash bash_scripts/train/vqa.bash          # Target model
bash bash_scripts/train/vqa_shadow.bash   # Shadow model
```
