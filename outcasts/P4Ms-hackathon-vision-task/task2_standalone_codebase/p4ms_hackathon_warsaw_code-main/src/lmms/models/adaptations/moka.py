# based on https://arxiv.org/abs/2506.05191

from typing import List
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor as T
from src.lmms.models.adaptations.base import LoRALinear


class MokALinear(LoRALinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r: int = 4,
        modalities: str = "text,vision",
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        blc_alpha: float = 0.0,
        blc_weight: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = False,
        **kwargs,
    ):
        self.modalities = modalities.split(",")
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

        setattr(self, "lora_B", nn.Linear(self.lora_r, self.out_features, bias=False))

        self.reset_parameters()
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A_text"):
            for modality in self.modalities:
                nn.init.kaiming_uniform_(
                    getattr(self, f"lora_A_{modality}").weight, a=math.sqrt(5)
                )
            nn.init.zeros_(getattr(self, f"lora_B").weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

        for modality in self.modalities:
            getattr(self, f"lora_A_{modality}").train(mode)
        getattr(self, "lora_B").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        for modality in self.modalities:
            getattr(self, f"lora_A_{modality}").eval()
        getattr(self, "lora_B").eval()

    def cross_attn(
        self, modality_tokens: T, question_tokens: T, modality_mask: T, question_mask: T
    ) -> T:
        # modality_tokens, question_tokens: (B, T, D)
        # modality_mask, question_mask: (B, T, 1) [bool or 0/1]
        B, T, D = modality_tokens.shape
        dtype = modality_tokens.dtype
        device = modality_tokens.device

        Q = modality_tokens
        K = question_tokens.to(dtype)
        V = question_tokens.to(dtype)

        # sqrt(#valid keys) with floor at 1 to keep scale finite and graph identical
        Nt = question_mask.sum(dim=1).to(dtype)  # (B, 1)
        scale = Nt.clamp_min(1.0).sqrt().unsqueeze(-1)  # (B, 1, 1)

        # ---- append a dummy always-valid zero key/value ----
        K_dummy = torch.zeros(B, 1, D, dtype=dtype, device=device)
        V_dummy = torch.zeros(B, 1, D, dtype=dtype, device=device)
        K_ext = torch.cat([K, K_dummy], dim=1)  # (B, T+1, D)
        V_ext = torch.cat([V, V_dummy], dim=1)  # (B, T+1, D)

        qmask = question_mask.to(torch.bool).squeeze(-1)  # (B, T)
        qmask_ext = torch.cat(
            [qmask, torch.ones(B, 1, dtype=torch.bool, device=device)], dim=1
        )  # (B, T+1)

        # logits and masking (no control flow; identical on all ranks)
        logits = torch.matmul(Q, K_ext.transpose(-2, -1))  # (B, T, T+1)
        logits = logits / scale
        logits = logits.masked_fill(~qmask_ext.unsqueeze(1), torch.finfo(dtype).min)

        attn = torch.softmax(logits, dim=-1).to(dtype)  # (B, T, T+1)
        out = torch.matmul(attn, V_ext)  # (B, T, D)

        out = out * modality_mask  # (B, T, D)
        return modality_tokens + out * self.blc_weight

    def _forward_without_modalities(self, x: T):
        result = F.linear(x, self.transpose(self.weight), bias=self.bias)
        output_A = (
            getattr(self, "lora_A_text")(self.lora_dropout(x)) * self.scaling["text"]
        )
        output_B = getattr(self, "lora_B")(output_A)
        return output_B + result

    def _forward_with_modalities(self, x: T, modality_mask: List[T] = None):
        result = F.linear(x, self.transpose(self.weight), bias=self.bias)

        modality_masks = {
            "text": modality_mask[0],
            "vision": modality_mask[1],
            "audio": modality_mask[2],
            "question": modality_mask[3],
        }

        inputs_A = {
            modality: x * mask
            for modality, mask in modality_masks.items()
            if modality != "question"
        }

        outputs_A = {
            modality: getattr(self, f"lora_A_{modality}")(
                self.lora_dropout(inputs_A[modality])
            )
            * self.scaling[modality]
            for modality in self.modalities
        }

        inputs_B = [outputs_A["text"]]
        for modality in self.modalities:
            if modality == "text":
                continue
            input_B = self.cross_attn(
                modality_tokens=outputs_A[modality],
                question_tokens=outputs_A["text"] * modality_masks["question"],
                modality_mask=modality_masks[modality],
                question_mask=modality_masks["question"],
            )
            inputs_B.append(input_B)

        input_B = sum(inputs_B)
        output_B = getattr(self, "lora_B")(input_B)
        return output_B + result

    def forward(self, x: T, modality_mask: List[T] = None):
        if self.inference_mode and x.size(1) == 1:
            return self._forward_without_modalities(x)
        return self._forward_with_modalities(x, modality_mask)
