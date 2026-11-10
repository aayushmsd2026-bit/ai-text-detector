"""Binoculars scoring in Python, using the same ONNX file as the website.

Method (Hans et al., 2024, "Spotting LLMs With Binoculars"):
  log_ppl   = how surprised the *performer* model is by the actual tokens
  x_ppl     = how surprised the performer is, on average, by what the
              *observer* model expected (cross-perplexity)
  score     = log_ppl / x_ppl

Human text tends to be surprising relative to what models expect (higher
score). Machine text tends to be exactly what models expect (lower score).

The models are created by scripts/export_models.py into site/models/.
The browser version (site/js/detector.js) mirrors this file line for line.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "site" / "models"))
CHUNK = 512  # must match CHUNK in site/js/detector.js


@dataclass
class TokenScores:
    """Per-token signals for one text, plus character offsets for each token."""

    log_ppl: np.ndarray
    x_ppl: np.ndarray
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
            # "Burstiness": human writing swings between predictable and
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


def token_offsets(ids: list[int], decode) -> list[tuple[int, int]]:
    """Character span of each token, found by decoding tokens in order.

    Same algorithm as tokenOffsets() in site/js/detector.js. Leading
    whitespace is excluded so a token belongs to the sentence it starts.
    """
    offsets, pos, group = [], 0, []
    for i in ids:
        group.append(i)
        piece = decode(group)
        if piece.endswith("\ufffd") and len(group) < 4:  # partial UTF-8 character
            continue
        lead = len(piece) - len(piece.lstrip())
        offsets.extend([(pos + lead, pos + len(piece))] * len(group))
        pos += len(piece)
        group = []
    offsets.extend([(pos, pos)] * len(group))
    return offsets


class BinocularsScorer:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.observer_name = meta.get("observer", "?")
        self.performer_name = meta.get("performer", "?")
        self._session = None

    def _load(self):
        if self._session is None:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            path = self.model_dir / "binoculars.onnx"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found. Run: python scripts/export_models.py")
            self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._tok = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

    def score(self, text: str) -> TokenScores:
        self._load()
        ids = self._tok.encode(text, add_special_tokens=False).ids
        offsets = token_offsets(ids, lambda g: self._tok.decode(g, skip_special_tokens=False))
        lps, xps, kept = [], [], []
        # Score in chunks so long documents fit in memory. Each chunk's first
        # token has no context, so it gets no score (next-token shift).
        for start in range(0, len(ids), CHUNK):
            chunk = ids[start:start + CHUNK]
            if len(chunk) < 2:
                continue
            lp, xp = self._session.run(None, {"input_ids": np.array([chunk], dtype=np.int64)})
            lps.append(lp)
            xps.append(xp)
            kept.extend(offsets[start + 1:start + len(chunk)])
        if not lps:
            return TokenScores(np.array([]), np.array([]), [])
        return TokenScores(np.concatenate(lps), np.concatenate(xps), kept)
