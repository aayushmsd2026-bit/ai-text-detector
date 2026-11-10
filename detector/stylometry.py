"""Writing-style features that need no model, only the text itself."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Phrases that show up far more often in assistant-written prose than in
# typical human writing. Kept short on purpose; the classifier learns how much
# weight to give this signal, so it never decides a result on its own.
STOCK_PHRASES = [
    "delve", "tapestry", "testament to", "in today's", "it's important to note",
    "it is important to note", "plays a crucial role", "plays a vital role",
    "navigate the", "landscape of", "multifaceted", "furthermore", "moreover",
    "in conclusion", "overall,", "additionally,", "seamless", "robust",
    "leverage", "foster", "holistic", "pivotal", "underscore", "showcase",
    "a myriad of", "ever-evolving", "not only", "nuanced", "realm",
]

TRANSITIONS = ["however", "therefore", "furthermore", "moreover", "additionally",
               "consequently", "nevertheless", "thus", "hence", "ultimately"]

FEATURE_NAMES = [
    "n_words", "sent_len_mean", "sent_len_std", "sent_len_cv",
    "type_token_ratio", "hapax_ratio", "avg_word_len",
    "stock_phrase_rate", "transition_rate", "comma_rate",
    "em_dash_rate", "colon_rate", "bullet_line_ratio",
    "repeated_trigram_rate", "first_person_rate", "contraction_rate",
]


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, sentence) spans, also splitting on blank lines."""
    spans = []
    for para in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        p_start, p_text = para.start(), para.group()
        pos = 0
        for piece in _SENT_SPLIT.split(p_text):
            idx = p_text.find(piece, pos)
            if piece.strip():
                spans.append((p_start + idx, p_start + idx + len(piece), piece))
            pos = idx + len(piece)
    return spans


def features(text: str) -> dict[str, float]:
    words = [w.lower() for w in _WORD.findall(text)]
    n = max(len(words), 1)
    sents = [s for _, _, s in split_sentences(text)]
    lens = np.array([len(_WORD.findall(s)) for s in sents if _WORD.findall(s)] or [0])
    counts = Counter(words)
    lower = text.lower()
    trigrams = list(zip(words, words[1:], words[2:]))
    tri_counts = Counter(trigrams)
    lines = [ln for ln in text.splitlines() if ln.strip()]

    per100 = lambda c: 100.0 * c / n  # noqa: E731
    mean = float(lens.mean())
    return {
        "n_words": float(len(words)),
        "sent_len_mean": mean,
        "sent_len_std": float(lens.std()),
        # Coefficient of variation: humans mix short and long sentences more.
        "sent_len_cv": float(lens.std() / mean) if mean else 0.0,
        "type_token_ratio": len(counts) / n,
        "hapax_ratio": sum(1 for c in counts.values() if c == 1) / n,
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "stock_phrase_rate": per100(sum(lower.count(p) for p in STOCK_PHRASES)),
        "transition_rate": per100(sum(counts[t] for t in TRANSITIONS)),
        "comma_rate": per100(text.count(",")),
        "em_dash_rate": per100(text.count("—") + text.count(" -- ")),
        "colon_rate": per100(text.count(":")),
        "bullet_line_ratio": (sum(1 for ln in lines if re.match(r"\s*([-*•]|\d+[.)])\s", ln)) / len(lines)) if lines else 0.0,
        "repeated_trigram_rate": (sum(c for c in tri_counts.values() if c > 1) / len(trigrams)) if trigrams else 0.0,
        "first_person_rate": per100(sum(counts[w] for w in ("i", "me", "my", "mine", "i'm", "i've", "i'd"))),
        "contraction_rate": per100(sum(1 for w in words if "'" in w)),
    }
