import torch.nn as nn, re
from functools import partial
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

from src.lmms.models.peft_hyper.utils import PeftConfig, PeftType
from hydra.utils import instantiate
import yaml

with open("config/adapters.yaml", mode="r") as fd:
    adapters = yaml.safe_load(fd)


@dataclass
class AdapterConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].
    Args:
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        adapter_type (`str`): Type of adapter.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
    """

    adapter_type: str = field(default=None, metadata={"help": "Type of adapter"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    merge_weights: bool = field(
        default=False,
        metadata={"help": "Merge weights of the original model and the Lora model"},
    )
    enable_lora: Optional[List[bool]] = field(
        default=None, metadata={"help": "Used with `lora.MergedLinear`."}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class AdapterModel(nn.Module):
    def __init__(self, config: AdapterConfig, model):
        super().__init__()
        self.peft_config = config
        self.model = model
        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_4bit or loaded_in_8bit:
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in self.peft_config.target_modules
                )
            if target_module_found:  # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if (
                    isinstance(target, nn.Linear)
                    and self.peft_config.enable_lora is None
                ):
                    new_module = instantiate(
                        adapters[self.peft_config.adapter_type],
                        _partial_=True,
                    )(target.in_features, target.out_features, bias=bias)

                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {
            k: v.value if isinstance(v, Enum) else v
            for k, v in asdict(self.peft_config).items()
        }
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.lora_enable = enabled

    def set_mode(self, mode="train"):
        assert mode in [
            "inference",
            "train",
        ], "Mode should be either 'inference' or 'train'"
        if mode == "train":
            inference_mode = False
        else:
            inference_mode = True
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(
                    key.endswith(target_key)
                    for target_key in self.peft_config.target_modules
                )
            if target_module_found:  # here
                _, target, _ = self._get_submodules(key)

                if isinstance(target, LoRALinear):
                    target.inference_mode = inference_mode


class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r: int = 4,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        nn.Linear.__init__(self, self.in_features, self.out_features)

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.merged = False
        self.merge_weights = merge_weights

        self.blc_alpha = blc_alpha
        self.blc_weight = blc_weight

        self.lora_alpha = lora_alpha
        self.fan_in_fan_out = fan_in_fan_out
        self.inference_mode = False

        self.transpose = partial(
            lambda x, fan_in_fan_out: x.T if fan_in_fan_out else x,
            fan_in_fan_out=self.fan_in_fan_out,
        )
        self.kwargs = kwargs
        self.setup_weights()

    def setup_weights(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def train(self, mode: bool = True):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def eval(self):
        raise NotImplementedError("This method should be implemented in subclasses.")


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALinear) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
