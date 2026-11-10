import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.lm_scores import BinocularsScorer, TokenScores  # noqa: E402


def fake_score(self, text):
    """Deterministic stand-in for the language models (no downloads in CI).

    Texts containing 'Furthermore' get a low (AI-like) Binoculars score.
    """
    offsets, pos = [], 0
    for w in text.split():
        i = text.index(w, pos)
        offsets.append((i, i + len(w)))
        pos = i + len(w)
    n = len(offsets)
    ai = "furthermore" in text.lower()
    rng = np.random.default_rng(len(text))
    lp = rng.normal(2.0 if ai else 3.2, 0.2 if ai else 0.8, n).clip(0.1)
    xp = rng.normal(2.4, 0.1, n).clip(0.1)
    return TokenScores(lp, xp, offsets)


@pytest.fixture(autouse=True)
def no_models(request, monkeypatch):
    """Use the fake scorer everywhere except tests marked real_models."""
    if request.node.get_closest_marker("real_models"):
        return
    monkeypatch.setattr(BinocularsScorer, "score", fake_score)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_models: run the real ONNX scorer instead of the fake one")
