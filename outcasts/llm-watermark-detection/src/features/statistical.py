"""CPU-only stylometric features.

These are weak on their own but cheap to compute and add useful diversity to the
ensemble. Watermarks tend to subtly shift token-frequency distributions, so
character-level entropy and n-gram repetition rates can pick up faint signals.
"""
from __future__ import annotations
import re
from collections import Counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_PUNCT_RE = re.compile(r"[.,;:!?\-—'\"()\[\]{}]")


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array(counts, dtype=np.float64) / total
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def extract_one(text: str) -> dict[str, float]:
    n_chars = len(text)
    words = _WORD_RE.findall(text.lower())
    n_words = len(words)
    n_unique = len(set(words))
    word_lens = [len(w) for w in words] or [0]

    # Character distribution entropy (Shannon, base 2)
    char_counts = Counter(text)
    char_entropy = _entropy(list(char_counts.values()))

    # Word-level entropy
    word_counts = Counter(words)
    word_entropy = _entropy(list(word_counts.values()))

    # Bigram repetition
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_counts = Counter(bigrams)
    n_bigrams = len(bigrams)
    repeat_bigrams = sum(c for c in bigram_counts.values() if c > 1)
    bigram_repeat_rate = repeat_bigrams / n_bigrams if n_bigrams else 0.0

    # Punctuation
    n_punct = len(_PUNCT_RE.findall(text))
    n_newlines = text.count("\n")
    n_uppercase = sum(1 for c in text if c.isupper())
    n_digits = sum(1 for c in text if c.isdigit())

    # Burstiness in sentence length (proxy: split by sentence terminators)
    sents = re.split(r"[.!?]+\s+", text.strip()) if text else []
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents if s.strip()]
    if len(sent_lens) >= 2:
        sent_burst = float(np.std(sent_lens) / (np.mean(sent_lens) + 1e-9))
    else:
        sent_burst = 0.0

    return {
        "stat_n_chars": n_chars,
        "stat_n_words": n_words,
        "stat_n_unique_words": n_unique,
        "stat_ttr": n_unique / n_words if n_words else 0.0,
        "stat_avg_word_len": float(np.mean(word_lens)),
        "stat_std_word_len": float(np.std(word_lens)),
        "stat_char_entropy": char_entropy,
        "stat_word_entropy": word_entropy,
        "stat_bigram_repeat_rate": bigram_repeat_rate,
        "stat_punct_rate": n_punct / n_chars if n_chars else 0.0,
        "stat_newline_rate": n_newlines / n_chars if n_chars else 0.0,
        "stat_uppercase_rate": n_uppercase / n_chars if n_chars else 0.0,
        "stat_digit_rate": n_digits / n_chars if n_chars else 0.0,
        "stat_n_sentences": len(sent_lens),
        "stat_sent_burstiness": sent_burst,
    }


def extract_features(df: pd.DataFrame, text_col: str = "text",
                     show_progress: bool = True) -> pd.DataFrame:
    """Apply extract_one to every row, return a feature DataFrame indexed like df."""
    iterator = tqdm(df[text_col], desc="stat features", disable=not show_progress)
    rows = [extract_one(t) for t in iterator]
    feat = pd.DataFrame(rows, index=df.index)
    return feat


FEATURE_PREFIX = "stat_"
