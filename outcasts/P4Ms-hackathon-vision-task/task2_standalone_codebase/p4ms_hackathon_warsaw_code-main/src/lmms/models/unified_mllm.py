import torch
from torch import nn
from typing import Optional, List
from transformers import AutoConfig, AutoModelForCausalLM

from src.lmms.models.unified_arch import (
    UnifiedMetaModel,
    UnifiedMetaForCausalLM,
)


def model_factory(config_cls, llm_cls, llm_for_causal_cls):

    class UnifiedConfig(config_cls):
        model_type = "unified_llm"

    class UnifiedModel(UnifiedMetaModel, llm_cls):
        config_class = UnifiedConfig

        def __init__(self, config: AutoConfig, inputs_embeds_with_mmask=False):
            super(UnifiedModel, self).__init__(config)
            self.config = config
            self.inputs_embeds_with_mmask = inputs_embeds_with_mmask

    class UnifiedForCausalLM(llm_for_causal_cls, UnifiedMetaForCausalLM):
        config_class = UnifiedConfig

        def __init__(self, config: AutoConfig, **kwargs):
            super().__init__(config)
            self.config = config
            self.model = UnifiedModel(config, **kwargs)
            self.pretraining_tp = getattr(config, "pretraining_tp", None)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            # Initialize weights and apply pal processing
            self.post_init()

        def get_model(self) -> UnifiedMetaModel:
            return self.model

        def forward(
            self,
            batch_input_ids=None,
            batch_labels=None,
            batch_X_modals=None,
            # used for inference
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            if input_ids is not None and input_ids.shape[1] == 1:
                inputs_embeds = self.get_model().embed_tokens(input_ids)
                input_ids = None

            elif inputs_embeds is None and batch_input_ids is not None:
                inputs = self.prepare_multimodal_inputs(
                    batch_input_ids=batch_input_ids,
                    batch_labels=batch_labels,
                    batch_X_modals=batch_X_modals,
                )
                input_ids = inputs["input_ids"]
                inputs_embeds = inputs["inputs_embeds"]
                attention_mask = inputs["attention_mask"]
                labels = inputs["labels"]
                position_ids = inputs["position_ids"]

            output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Add phantom loss from dummy encoder forwards (set during
            # prepare_multimodal_inputs for absent modalities).
            if self.training and output.loss is not None:
                phantom = getattr(self, "_phantom_loss", None)
                if phantom is not None:
                    output.loss = output.loss + phantom

            return output

        @torch.no_grad()
        def generate(
            self,
            batch_input_ids,
            batch_labels,
            batch_X_modals,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            **kwargs,
        ):
            if inputs_embeds is None:
                inputs = self.prepare_multimodal_inputs(
                    batch_input_ids=batch_input_ids,
                    batch_labels=batch_labels,
                    batch_X_modals=batch_X_modals,
                )
                inputs_embeds = inputs["inputs_embeds"]

            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
            )
            inputs_embeds = inputs["inputs_embeds"]
            output_hidden_states = kwargs.pop("output_hidden_states", False)
            return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
            temperature = kwargs.pop("temperature", 1.0)
            top_p = kwargs.pop("top_p", 1.0)
            num_beams = kwargs.pop("num_beams", 1)
            use_cache = kwargs.pop("use_cache", True)

            return super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict_in_generate=return_dict_in_generate,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                use_cache=use_cache,
                **kwargs,
            )

        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
        ):
            images = kwargs.pop("images", None)
            _inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            if images is not None:
                _inputs["images"] = images
            return _inputs

        @property
        def device(self):
            return list(self.parameters())[0].device

    AutoConfig.register("unified_llm", UnifiedConfig)
    AutoModelForCausalLM.register(UnifiedConfig, UnifiedForCausalLM)

    return UnifiedForCausalLM
