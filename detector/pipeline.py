"""Combine language-model scores and style features into one verdict.

Mirrors site/js/detector.js: same features, same classifier formula, same
fallback, so the training job and the website agree.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import stylometry
from .lm_scores import ROOT, BinocularsScorer, TokenScores

CLASSIFIER_PATH = Path(os.environ.get("CLASSIFIER_PATH", ROOT / "site" / "model.json"))
LM_FEATURES = ["binoculars", "log_ppl", "x_ppl", "log_ppl_std"]
ALL_FEATURES = LM_FEATURES + stylometry.FEATURE_NAMES

# Used only until a classifier has been trained: below this Binoculars score
# text leans AI. The right value depends on the model pair, so untrained
# results are rough. Training also stores a fitted threshold for the
# sentence view in model.json.
FALLBACK_THRESHOLD = 0.95
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


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50), -50)))


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


def load_classifier(path: Path = CLASSIFIER_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def classifier_prob(clf: dict, feats: dict[str, float]) -> float:
    z = clf["intercept"]
    for name, mean, scale, coef in zip(clf["features"], clf["mean"], clf["scale"], clf["coef"]):
        v = feats.get(name, 0.0)
        v = 0.0 if v is None or math.isnan(v) else v
        z += coef * (v - mean) / (scale or 1.0)
    return sigmoid(z)


class Detector:
    def __init__(self, scorer: BinocularsScorer | None = None, classifier_path: Path = CLASSIFIER_PATH):
        self.scorer = scorer or BinocularsScorer()
        self.clf = load_classifier(classifier_path)

    def feature_vector(self, text: str, tokens: TokenScores | None = None) -> dict[str, float]:
        tokens = tokens if tokens is not None else self.scorer.score(text)
        return {**tokens.summary(), **stylometry.features(text)}

    def _sentence_prob(self, bino: float) -> float:
        thr = (self.clf or {}).get("sentence_threshold", FALLBACK_THRESHOLD)
        scale = (self.clf or {}).get("sentence_scale", FALLBACK_SCALE)
        return sigmoid((thr - bino) / scale)

    def analyze(self, text: str) -> Result:
        text = text.strip()
        warnings = []
        tokens = self.scorer.score(text)
        feats = self.feature_vector(text, tokens)
        if feats["n_words"] < MIN_WORDS:
            warnings.append(f"Only {int(feats['n_words'])} words. Results under {MIN_WORDS} words are unreliable.")
        if self.clf is not None:
            p = classifier_prob(self.clf, feats)
        else:
            warnings.append("No trained classifier found: using the uncalibrated Binoculars threshold only.")
            b = feats["binoculars"]
            p = 0.5 if math.isnan(b) else self._sentence_prob(b)

        sentences = []
        for s, e, sent in stylometry.split_sentences(text):
            bs = tokens.span_score(s, e)
            sentences.append(SentenceResult(s, e, sent, None if bs is None else self._sentence_prob(bs)))
        return Result(p, label_for(p), self.clf is not None, feats, sentences, warnings)


def result_to_json(r: Result) -> str:
    return json.dumps({
        "ai_probability": round(r.ai_probability, 4), "label": r.label, "calibrated": r.calibrated,
        "warnings": r.warnings,
        "features": {k: round(v, 4) for k, v in r.features.items()},
        "sentences": [{"text": s.text, "ai_probability": None if s.ai_probability is None else round(s.ai_probability, 3)}
                      for s in r.sentences],
    }, indent=2)
