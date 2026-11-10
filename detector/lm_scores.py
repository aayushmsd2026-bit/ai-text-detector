"""Binoculars-style scoring with two small open language models.

Method (Hans et al., 2024, "Spotting LLMs With Binoculars"):
  log_ppl   = how surprised the *performer* model is by the actual tokens
  x_ppl     = how surprised the performer is, on average, by what the
              *observer* model expected (cross-perplexity)
  score     = log_ppl / x_ppl

Human text tends to be surprising relative to what models expect (higher
score). Machine text tends to be exactly what models expect (lower score).

Both models must share a tokenizer. The defaults are a small base model and
its instruction-tuned sibling so the app runs on a free CPU host. Override
with the OBSERVER_MODEL / PERFORMER_MODEL environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

DEFAULT_OBSERVER = os.environ.get("OBSERVER_MODEL", "Qwen/Qwen2.5-0.5B")
DEFAULT_PERFORMER = os.environ.get("PERFORMER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_TOKENS_PER_CHUNK = int(os.environ.get("MAX_TOKENS_PER_CHUNK", "512"))


@dataclass
class TokenScores:
    """Per-token signals for one text, plus character offsets for each token."""

    log_ppl: np.ndarray  # performer cross-entropy on the real next token
    x_ppl: np.ndarray  # observer→performer cross-entropy over the vocabulary
    offsets: list[tuple[int, int]] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if len(self.log_ppl) == 0:
            return {"binoculars": float("nan"), "log_ppl": float("nan"),
                    "x_ppl": float("nan"), "log_ppl_std": float("nan")}
        lp, xp = float(self.log_ppl.mean()), float(self.x_ppl.mean())
        return {
            "binoculars": lp / xp if xp > 0 else float("nan"),
            "log_ppl": lp,
            "x_ppl": xp,
            # "burstiness": human writing swings between predictable and
            # surprising tokens more than model writing does.
            "log_ppl_std": float(self.log_ppl.std()),
        }

    def span_score(self, start: int, end: int) -> float | None:
        """Binoculars score restricted to tokens inside [start, end)."""
        idx = [i for i, (s, e) in enumerate(self.offsets) if s >= start and e <= end and e > s]
        if len(idx) < 3:
            return None
        xp = self.x_ppl[idx].sum()
        return float(self.log_ppl[idx].sum() / xp) if xp > 0 else None


class BinocularsScorer:
    """Loads the two models lazily and scores text token by token."""

    def __init__(self, observer: str = DEFAULT_OBSERVER, performer: str = DEFAULT_PERFORMER,
                 device: str | None = None):
        self.observer_name = observer
        self.performer_name = performer
        self.device = device
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        perf_tok = AutoTokenizer.from_pretrained(self.performer_name)
        if perf_tok.get_vocab() != self.tokenizer.get_vocab():
            raise ValueError("Observer and performer models must share a tokenizer.")
        self.observer = AutoModelForCausalLM.from_pretrained(self.observer_name, torch_dtype=dtype).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(self.performer_name, torch_dtype=dtype).to(self.device).eval()
        self._loaded = True

    def score(self, text: str) -> TokenScores:
        self._load()
        torch = self.torch
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
        log_ppl, x_ppl, kept_offsets = [], [], []

        # Score in chunks so long documents fit in memory. Each chunk's first
        # token has no context, so it is dropped (standard next-token shift).
        for start in range(0, len(ids), MAX_TOKENS_PER_CHUNK):
            chunk = ids[start:start + MAX_TOKENS_PER_CHUNK]
            if len(chunk) < 2:
                continue
            x = torch.tensor([chunk], device=self.device)
            with torch.no_grad():
                obs_logits = self.observer(x).logits[0, :-1].float()
                perf_logits = self.performer(x).logits[0, :-1].float()
            targets = x[0, 1:]
            perf_logp = torch.log_softmax(perf_logits, dim=-1)
            lp = -perf_logp.gather(-1, targets[:, None]).squeeze(-1)
            obs_p = torch.softmax(obs_logits, dim=-1)
            xp = -(obs_p * perf_logp).sum(-1)
            log_ppl.append(lp.cpu().numpy())
            x_ppl.append(xp.cpu().numpy())
            kept_offsets.extend(offsets[start + 1:start + len(chunk)])

        if not log_ppl:
            return TokenScores(np.array([]), np.array([]), [])
        return TokenScores(np.concatenate(log_ppl), np.concatenate(x_ppl), kept_offsets)
