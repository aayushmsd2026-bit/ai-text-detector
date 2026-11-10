"""Combine language-model scores and style features into one verdict."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import stylometry
from .lm_scores import BinocularsScorer, TokenScores

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("CLASSIFIER_PATH", ROOT / "model" / "classifier.joblib"))
LM_FEATURES = ["binoculars", "log_ppl", "x_ppl", "log_ppl_std"]
ALL_FEATURES = LM_FEATURES + stylometry.FEATURE_NAMES

# Used only until a classifier has been trained. Below this Binoculars score
# text leans AI. Small models sit higher than the paper's Falcon-7B value
# (0.90), and the right number depends on the model pair, so treat untrained
# results as rough. `scripts/train.py` replaces this with a learned model.
FALLBACK_THRESHOLD = float(os.environ.get("BINOCULARS_THRESHOLD", "0.95"))
FALLBACK_SCALE = 0.05
MIN_WORDS = 60


@dataclass
class SentenceResult:
    start: int
    end: int
    text: str
    ai_probability: float | None


@dataclass
class Result:
    ai_probability: float
    label: str
    calibrated: bool
    features: dict[str, float]
    sentences: list[SentenceResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_info: dict = field(default_factory=dict)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _bino_to_prob(score: float) -> float:
    return _sigmoid((FALLBACK_THRESHOLD - score) / FALLBACK_SCALE)


def label_for(p: float) -> str:
    if p >= 0.8:
        return "Likely AI-generated"
    if p >= 0.6:
        return "Possibly AI-generated / heavily AI-assisted"
    if p > 0.4:
        return "Uncertain"
    if p > 0.2:
        return "Probably human-written"
    return "Likely human-written"


class Detector:
    def __init__(self, scorer: BinocularsScorer | None = None, model_path: Path = MODEL_PATH):
        self.scorer = scorer or BinocularsScorer()
        self.clf = None
        self.meta: dict = {}
        if model_path.exists():
            import joblib
            bundle = joblib.load(model_path)
            self.clf, self.meta = bundle["model"], bundle.get("meta", {})

    def feature_vector(self, text: str, tokens: TokenScores | None = None) -> dict[str, float]:
        tokens = tokens if tokens is not None else self.scorer.score(text)
        return {**tokens.summary(), **stylometry.features(text)}

    def _prob(self, feats: dict[str, float]) -> float:
        if self.clf is not None:
            x = np.array([[feats.get(k, 0.0) for k in self.meta.get("features", ALL_FEATURES)]])
            x = np.nan_to_num(x)
            return float(self.clf.predict_proba(x)[0, 1])
        b = feats.get("binoculars", float("nan"))
        return 0.5 if math.isnan(b) else _bino_to_prob(b)

    def analyze(self, text: str) -> Result:
        text = text.strip()
        warnings = []
        tokens = self.scorer.score(text)
        feats = self.feature_vector(text, tokens)
        if feats["n_words"] < MIN_WORDS:
            warnings.append(f"Only {int(feats['n_words'])} words. Results under {MIN_WORDS} words are unreliable.")
        if self.clf is None:
            warnings.append("No trained classifier found: using the uncalibrated Binoculars threshold only.")

        p = self._prob(feats)
        sentences = []
        for s, e, sent in stylometry.split_sentences(text):
            bs = tokens.span_score(s, e)
            sentences.append(SentenceResult(s, e, sent, None if bs is None else _bino_to_prob(bs)))

        info = {"observer": self.scorer.observer_name, "performer": self.scorer.performer_name}
        info.update({k: v for k, v in self.meta.items() if k != "features"})
        return Result(p, label_for(p), self.clf is not None, feats, sentences, warnings, info)


def result_to_json(r: Result) -> str:
    return json.dumps({
        "ai_probability": round(r.ai_probability, 4), "label": r.label, "calibrated": r.calibrated,
        "warnings": r.warnings, "model_info": r.model_info,
        "features": {k: round(v, 4) for k, v in r.features.items()},
        "sentences": [{"text": s.text, "ai_probability": None if s.ai_probability is None else round(s.ai_probability, 3)}
                      for s in r.sentences],
    }, indent=2)
