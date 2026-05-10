import torch.nn as nn, json, torch


class SpeechMLPProjector(nn.Module):
    def __init__(self, llm_dim):
        super().__init__()
        self.llm_dim = llm_dim

        from transformers import Blip2QFormerConfig

        configuration = Blip2QFormerConfig()
        # (encoder维度)1280->2048(llm维度)   3B
        # 1280->5120    32B
        if self.llm_dim <= 1536:
            self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)
        elif self.llm_dim <= 2560:
            self.linear1 = nn.Linear(configuration.hidden_size, 1536)  # 从 768 -> 2560
            self.relu = nn.ReLU()  # 激活函数
            self.linear2 = nn.Linear(1536, self.llm_dim)  # 从 2560 -> 5120
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)  # 最终归一化
        else:
            self.linear1 = nn.Linear(configuration.hidden_size, 2560)  # 从 768 -> 2560
            self.relu = nn.ReLU()  # 激活函数
            self.linear2 = nn.Linear(2560, self.llm_dim)  # 从 2560 -> 5120
            self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)  # 最终归一化

    def forward(self, query_output):
        if self.llm_dim <= 1536:
            query_proj = self.norm(self.linear(query_output.last_hidden_state))
        else:
            x = self.linear1(query_output.last_hidden_state)  # 从 1280 -> 2560
            x = self.relu(x)  # 激活
            x = self.linear2(x)  # 从 2560 -> 5120
            query_proj = self.norm(x)  # LayerNorm 归一化

        return query_proj


class SLProjector(nn.Module):
    def __init__(
        self,
        bert_config_path: str,
        num_query_token: int = 80,
        d_model: int = 3584,
    ) -> None:
        from transformers import Blip2QFormerConfig, Blip2QFormerModel

        super().__init__()
        encoder_config = Blip2QFormerConfig.from_dict(
            json.load(open(bert_config_path, "r"))
        )
        self.Qformer = Blip2QFormerModel(config=encoder_config)

        self.num_query_token = num_query_token
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=1.0)
        self.speech_proj = SpeechMLPProjector(d_model)

    def forward(self, audio_feature):
        """
        audio_feature: b,t,n,d
        text_ids: b,L
        """
        device = audio_feature.device
        audio_atts = torch.ones(
            audio_feature.size()[:-1], dtype=torch.int32, device=device
        )  # bt,n

        query_tokens = self.query_tokens.expand(
            audio_feature.shape[0], -1, -1
        )  # bt,32,d
        query_output = self.Qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_feature,
            encoder_attention_mask=audio_atts,
            return_dict=True,
        )
        audio_embeds = self.speech_proj(query_output)
        return audio_embeds
