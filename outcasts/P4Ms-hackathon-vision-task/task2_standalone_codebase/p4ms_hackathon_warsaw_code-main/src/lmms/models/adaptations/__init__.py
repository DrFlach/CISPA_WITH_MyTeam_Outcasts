from src.lmms.models.peft_hyper.peft_model import (
    PeftModelForCausalLM,
)
from src.lmms.models.adaptations.base import AdapterConfig, AdapterModel
from src.lmms.models.adaptations.lora_moe import LoRAMoEModel

adaptation_models = {
    "lora_moe": LoRAMoEModel,
}


def _prepare_adapter_config(peft_config, model_config):
    if len(peft_config.target_modules) == 1:
        peft_config.fan_in_fan_out = True
        peft_config.enable_lora = [True, False, True]
    return peft_config


def get_peft_model(model, peft_config):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    adapter_model_cls = adaptation_models.get(peft_config.adapter_type, AdapterModel)
    model_config = model.config.to_dict()
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    peft_config = _prepare_adapter_config(peft_config, model_config)
    return PeftModelForCausalLM(model, peft_config, adapter_model_cls)
