from src.lmms.models.unified_mllm import model_factory
from src.lmms.models.utils import (
    get_adapter,
    write_model_info,
    get_compute_type,
    enable_gradient_checkpointing,
)

from transformers import (
    AutoTokenizer,
    Olmo2Config,
    Olmo2Model,
    Olmo2ForCausalLM,
    LlamaForCausalLM,
    LlamaConfig,
    LlamaModel,
    Gemma3ForCausalLM,
    Gemma3Config,
    Gemma3TextModel,
    AutoConfig,
)
from src.lmms.models.llms.modeling_olmo2 import (
    Olmo2Model as Olmo2MMaskModel,
    Olmo2ForCausalLM as Olmo2MMaskForCausalLM,
    Olmo2Config as Olmo2MMaskConfig,
)
from src.lmms.models.llms.modeling_llama import (
    LlamaModel as Llama32Model,
    LlamaForCausalLM as Llama32ForCausalLM,
    LlamaConfig as Llama32Config,
)
from src.lmms.configs.unified_config import ModelArguments, TrainingArguments
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")


model_to_cfg = {
    "olmo2_mmask": Olmo2MMaskConfig,
    "llama32_mask": Llama32Config,
    "default_olmo2": Olmo2Config,
    "default_llama32": LlamaConfig,
    "default_gemma": AutoConfig,
    "default_falcon": AutoConfig,
}

model_to_llm_cls = {
    "olmo2_mmask": Olmo2MMaskModel,
    "llama32_mask": Llama32Model,
    "default_olmo2": Olmo2Model,
    "default_llama32": LlamaModel,
    "default_gemma": Gemma3TextModel,
    "default_falcon": LlamaModel,
}

model_to_llm_for_causal_cls = {
    "olmo2_mmask": Olmo2MMaskForCausalLM,
    "llama32_mask": Llama32ForCausalLM,
    "default_olmo2": Olmo2ForCausalLM,
    "default_llama32": LlamaForCausalLM,
    "default_gemma": Gemma3ForCausalLM,
    "default_falcon": LlamaForCausalLM,
}

model_to_tokenizer = {
    "olmo2_mmask": AutoTokenizer,
    "llama32_mask": AutoTokenizer,
    "default_olmo2": AutoTokenizer,
    "default_llama32": AutoTokenizer,
    "default_gemma": AutoTokenizer,
    "default_falcon": AutoTokenizer,
}


def get_model_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    llm_name = model_args.llm_name
    config_cls = model_to_cfg[llm_name]
    llm_cls = model_to_llm_cls[llm_name]
    llm_for_causal_cls = model_to_llm_for_causal_cls[llm_name]
    is_llama = "llama" in llm_name.lower()

    config = config_cls.from_pretrained(
        model_args.model_name_or_path, local_files_only=False
    )

    tokenizer = model_to_tokenizer.get(llm_name, AutoTokenizer).from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        use_fast=True,
        cache_dir=cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = model_factory(config_cls, llm_cls, llm_for_causal_cls).from_pretrained(
        model_args.model_name_or_path,
        config=config,
        inputs_embeds_with_mmask=model_args.inputs_embeds_with_mmask,
        attn_implementation="eager" if not is_llama else None,
        torch_dtype=get_compute_type(training_args),
        cache_dir=cache_dir,
    )
    model.config._attn_implementation = None
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    if model_args.adapter_type is not None:
        model = get_adapter(
            training_args.lora_trainable.split(","), model_args.adapter_type, model
        )

    model.get_model().pad_token_id = tokenizer.pad_token_id
    model.initialize_MM_tokenizer(tokenizer)

    model.get_model().init_multimodal_modules(
        args=model_args,
        visual_branch=training_args.visual_branch,
        speech_branch=training_args.speech_branch,
    )

    model.get_model().load_pretrained_weights(model_args.pretrained_ckpt_path)

    return model, tokenizer
