# based on https://arxiv.org/abs/2311.02684

from typing import List
import math, torch.nn as nn, torch.nn.functional as F, torch
from torch import Tensor as T
from src.lmms.models.adaptations.base import (
    LoRALinear,
    AdapterModel,
    AdapterConfig,
    adapters,
)


class Top2Gating(nn.Module):
    MIN_EXPERT_CAPACITY = 4

    def __init__(
        self,
        dim,
        num_gates,
        eps=1e-9,
        outer_expert_dims=tuple(),
        second_policy_train="random",
        second_policy_eval="random",
        second_threshold_train=0.2,
        second_threshold_eval=0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
    ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    @staticmethod
    def top1(tensor):
        values, index = tensor.topk(k=1, dim=-1)
        values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
        return values, index

    def forward(self, x, reduce_token=False):
        *_, b, group_size, dim = x.shape
        if reduce_token:
            group_size = 1
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval

        raw_gates = torch.einsum("...bnd,...de->...bne", x, self.w_gating)
        if reduce_token:
            raw_gates = raw_gates.mean(dim=1).unsqueeze(dim=1)
        raw_gates = raw_gates.softmax(dim=-1)

        # FIND TOP 2 EXPERTS PER POSITON
        # Find the top expert for each position. shape=[batch, group]

        gate_1, index_1 = self.top1(raw_gates)
        mask_1 = F.one_hot(index_1, num_gates).float()

        gates_without_top_1 = raw_gates * (1.0 - mask_1)

        gate_2, index_2 = self.top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, num_gates).float()

        # normalize top2 gate scores
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom

        # Depending on the policy in the hparams, we may drop out some of the
        # second-place experts.
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0.0, 1.0)
            mask_2 *= (
                (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
            )
        else:
            raise ValueError(f"Unknown policy {policy}")

        if reduce_token:
            soft_gate = torch.zeros(b, num_gates).to(gate_1.device)
            soft_gate = soft_gate.to(gate_1.dtype)
            soft_gate.scatter_(1, index_1, gate_1)
            soft_gate = soft_gate.to(gate_2.dtype)
            soft_gate.scatter_(1, index_2, gate_2)

        else:
            soft_gate = torch.zeros(b * group_size, num_gates).to(gate_1.device)
            soft_gate = soft_gate.to(gate_1.dtype)
            soft_gate.scatter_(1, index_1.view(-1, 1), gate_1.view(-1, 1))
            soft_gate = soft_gate.to(gate_2.dtype)
            soft_gate.scatter_(1, index_2.view(-1, 1), gate_2.view(-1, 1))
            soft_gate = soft_gate.reshape(b, group_size, num_gates).contiguous()

        return soft_gate


class LoRAMoEModel(AdapterModel):
    def __init__(self, config: AdapterConfig, model):
        super().__init__(config, model)
        self.num_experts = adapters["lora_moe"]["num_experts"]
        self.d_model = model.config.hidden_size
        self.gating_network = Top2Gating(self.d_model, self.num_experts)
        # TODO make new forward to pass inputs_embeds to gating network

    def moe_set_gate(self, inputs_embeds: List[T]):
        soft_gate = self.gating_network(inputs_embeds, reduce_token=True)
        for _, module in self.model.named_modules():
            if isinstance(module, LoRAMoELinear):
                module.set_gate(soft_gate)
        return


class LoRAMoELinear(LoRALinear):
    # TODO integrate multiple experts
    # TODO integrate gate input
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
        ), "Text modality should not have LoRA in LoRAMoELinear."
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
