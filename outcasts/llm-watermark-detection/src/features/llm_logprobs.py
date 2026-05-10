"""GPU-side feature extractor: per-token log-probabilities under a base LM.

This is the *main* signal for watermark detection. Watermarking schemes
(Kirchenbauer green-list, Aaronson, SynthID) all leave statistical traces in
the distribution of token log-probs under the model that generated them. We
extract a wide set of summary statistics and let the gradient booster figure
out which ones matter.

USAGE (GPU):
    from src.features.llm_logprobs import LogprobFeatureExtractor
    from src import data, config

    extractor = LogprobFeatureExtractor(config.BASE_LM_NAME)
    train_df = data.load_train()
    feats = extractor.extract(train_df["text"].tolist())
    feats.to_parquet("output/features/train_logprobs_llama3.parquet")
"""
from __future__ import annotations
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import config


class LogprobFeatureExtractor:
    """Loads a causal LM once, extracts per-text logprob summary features."""

    def __init__(self, model_name: str = None, max_tokens: int = None,
                 device: str = None, dtype: torch.dtype | None = None):
        self.model_name = model_name or config.BASE_LM_NAME
        self.max_tokens = max_tokens or config.MAX_TOKENS
        self.device = device or config.DEVICE
        if dtype is None:
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Run on a GPU node or set config.DEVICE = 'cpu' for debugging.")
        print(f"[logprobs] loading {self.model_name} on {self.device} ({dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device_map = "auto" if self.device == "auto" else {"": self.device}
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=device_map
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def _features_for_one(self, text: str) -> dict[str, float]:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens,
            add_special_tokens=True,
        ).to(self.input_device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 2:
            return self._empty_features()

        logits = self.model(input_ids).logits  # [1, T, V]
        # log-probs of the actually-realized next token at each position
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target_ids = input_ids[:, 1:]                                  # [1, T-1]
        token_logp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, T-1]
        token_logp = token_logp[0].cpu().numpy()                       # (T-1,)

        # Full-distribution entropy at each position
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)[0].cpu().numpy()    # (T-1,)

        # GLTR-style rank: how many vocab entries have higher logp than the realized token?
        realized_logp = log_probs.gather(2, target_ids.unsqueeze(-1))  # [1, T-1, 1]
        ranks = (log_probs > realized_logp).sum(dim=-1) + 1            # [1, T-1]
        ranks = ranks[0].cpu().numpy()                                 # (T-1,)

        return self._summary_stats(token_logp, entropy, ranks)

    @staticmethod
    def _summary_stats(logp: np.ndarray, entropy: np.ndarray,
                       ranks: np.ndarray) -> dict[str, float]:
        n = len(logp)
        feats = {
            "lp_n_tokens": float(n),
            # log-prob distribution
            "lp_mean": float(np.mean(logp)),
            "lp_std": float(np.std(logp)),
            "lp_min": float(np.min(logp)),
            "lp_max": float(np.max(logp)),
            "lp_p10": float(np.percentile(logp, 10)),
            "lp_p25": float(np.percentile(logp, 25)),
            "lp_p50": float(np.percentile(logp, 50)),
            "lp_p75": float(np.percentile(logp, 75)),
            "lp_p90": float(np.percentile(logp, 90)),
            # perplexity
            "lp_ppl": float(np.exp(-np.mean(logp))),
            "lp_log_ppl": float(-np.mean(logp)),
            # burstiness — std of logp over rolling windows
            "lp_burstiness": float(np.std(_rolling_mean(logp, 16))) if n >= 16 else 0.0,
            # entropy of the next-token distribution (model uncertainty)
            "lp_ent_mean": float(np.mean(entropy)),
            "lp_ent_std": float(np.std(entropy)),
            "lp_ent_p25": float(np.percentile(entropy, 25)),
            "lp_ent_p75": float(np.percentile(entropy, 75)),
            # rank buckets (GLTR)
            "lp_rank_top10": float(np.mean(ranks <= 10)),
            "lp_rank_top100": float(np.mean(ranks <= 100)),
            "lp_rank_top1000": float(np.mean(ranks <= 1000)),
            "lp_rank_mean": float(np.mean(ranks)),
            "lp_rank_median": float(np.median(ranks)),
            # spike rate — fraction of tokens that are extremely unlikely
            "lp_spike_rate": float(np.mean(logp < -10)),
            # high-confidence rate — fraction of tokens with logp > log(0.5)
            "lp_high_conf_rate": float(np.mean(logp > np.log(0.5))),
        }
        return feats

    @staticmethod
    def _empty_features() -> dict[str, float]:
        keys = [
            "lp_n_tokens", "lp_mean", "lp_std", "lp_min", "lp_max",
            "lp_p10", "lp_p25", "lp_p50", "lp_p75", "lp_p90",
            "lp_ppl", "lp_log_ppl", "lp_burstiness",
            "lp_ent_mean", "lp_ent_std", "lp_ent_p25", "lp_ent_p75",
            "lp_rank_top10", "lp_rank_top100", "lp_rank_top1000",
            "lp_rank_mean", "lp_rank_median",
            "lp_spike_rate", "lp_high_conf_rate",
        ]
        return {k: 0.0 for k in keys}

    def extract(self, texts: list[str]) -> pd.DataFrame:
        rows = [self._features_for_one(t) for t in tqdm(texts, desc="lm logprobs")]
        return pd.DataFrame(rows)

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) < win:
        return x
    cs = np.cumsum(np.insert(x, 0, 0.0))
    return (cs[win:] - cs[:-win]) / win


FEATURE_PREFIX = "lp_"


# --- Convenience: cache features to parquet so re-runs are instant ---
def extract_or_load(texts: list[str], cache_path: Path,
                    model_name: str = None) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"[logprobs] loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)
    extractor = LogprobFeatureExtractor(model_name)
    feats = extractor.extract(texts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache_path)
    extractor.close()
    return feats
