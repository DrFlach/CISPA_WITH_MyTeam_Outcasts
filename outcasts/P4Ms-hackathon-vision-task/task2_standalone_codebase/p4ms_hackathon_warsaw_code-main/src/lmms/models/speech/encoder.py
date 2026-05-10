from torch import nn
import torch
from transformers import WhisperModel
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")


class SpeechEncoder(nn.Module):
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(
            ckpt_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir
        ).encoder

    def encode_audio(self, audio):
        audio_embeds = self.encoder(audio.permute(0, 2, 1)).last_hidden_state.float()
        return audio_embeds

    def forward(self, audio):
        return self.encode_audio(audio)
