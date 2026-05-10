# based on https://arxiv.org/abs/2503.01743

from typing import List
import math, torch.nn as nn, torch.nn.functional as F
from torch import Tensor as T
from src.lmms.models.adaptations.base import LoRALinear


class PhiLinear(LoRALinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r: int = 4,
        modalities: str = "vision",
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs,
    ):
        self.modalities = modalities.split(",")
        assert (
            "text" not in self.modalities
        ), "Text modality should not have LoRA in PhiLinear."
        LoRALinear.__init__(
            self,
            in_features,
            out_features,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            blc_alpha=blc_alpha,
            blc_weight=blc_weight,
            fan_in_fan_out=fan_in_fan_out,
            merge_weights=merge_weights,
            **kwargs,
        )

    def setup_weights(self):
        self.scaling = dict()

        for _, modality in enumerate(self.modalities):
            setattr(
                self,
                f"lora_A_{modality}",
                nn.Linear(self.in_features, self.lora_r, bias=False),
            )
            self.scaling[modality] = self.lora_alpha / self.lora_r
            self.weight.requires_grad = False

            setattr(
                self,
                f"lora_B_{modality}",
                nn.Linear(self.lora_r, self.out_features, bias=False),
            )

        self.reset_parameters()
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A_vision") or hasattr(self, "lora_A_audio"):
            for modality in self.modalities:
                nn.init.kaiming_uniform_(
                    getattr(self, f"lora_A_{modality}").weight, a=math.sqrt(5)
                )
                nn.init.zeros_(getattr(self, f"lora_B_{modality}").weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        for modality in self.modalities:
            getattr(self, f"lora_A_{modality}").train(mode)
            getattr(self, f"lora_B_{modality}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        for modality in self.modalities:
            getattr(self, f"lora_A_{modality}").eval()
            getattr(self, f"lora_B_{modality}").eval()

    def _forward_without_modalities(self, x: T):
        return F.linear(x, self.transpose(self.weight), bias=self.bias)

    def _forward_with_modalities(self, x: T, modality_mask: List[T] = None):
        result = F.linear(x, self.transpose(self.weight), bias=self.bias)

        modality_masks = {
            "vision": modality_mask[1],
            "audio": modality_mask[2],
        }

        inputs_A = {modality: x * mask for modality, mask in modality_masks.items()}
        outputs_A = {
            modality: getattr(self, f"lora_A_{modality}")(
                self.lora_dropout(inputs_A[modality])
            )
            * self.scaling[modality]
            for modality in self.modalities
        }

        outputs_B = []
        for modality in self.modalities:
            output_B = getattr(self, f"lora_B_{modality}")(outputs_A[modality])
            outputs_B.append(output_B)
        return sum(outputs_B) + result

    def forward(self, x: T, modality_mask: List[T] = None):
        if self.inference_mode and x.size(1) == 1:
            return self._forward_without_modalities(x)
        return self._forward_with_modalities(x, modality_mask)
