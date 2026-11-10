"""The website (JavaScript) and the training job (Python) must compute the same
features and probabilities, or the trained classifier would be wrong in the browser."""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from detector import stylometry
from detector.lm_scores import TokenScores
from detector.pipeline import classifier_prob, label_for

SITE_JS = Path(__file__).resolve().parent.parent / "site" / "js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

TEXTS = [
    "I wasn't sure the bus would come. It did, eventually — twenty minutes late.\n\nMy sister laughed: \"Told you!\"",
    "Furthermore, renewable energy plays a crucial role. Moreover, it is important to note that it is robust.\n"
    "- First point here\n- Second point: done\n1. Numbered item\n\nIn conclusion, it's a testament to progress.",
    "short",
    "Dr. Smith arrived at 5 p.m. He said \"Hello.\" (Then left.) 3 things happened. Done -- really done.",
    "",
]


def run_node(script: str) -> object:
    out = subprocess.run(["node", "--input-type=module", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def close(a, b):
    if isinstance(a, float) and math.isnan(a):
        return b is None or (isinstance(b, float) and math.isnan(b))
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def test_stylometry_matches():
    js = run_node(f"""
        import {{ features, splitSentences, FEATURE_NAMES }} from "{(SITE_JS / 'stylometry.js').as_uri()}";
        const texts = {json.dumps(TEXTS)};
        console.log(JSON.stringify({{ names: FEATURE_NAMES,
          feats: texts.map(features), spans: texts.map(t => splitSentences(t).map(s => [s.start, s.end, s.text])) }}));
    """)
    assert js["names"] == stylometry.FEATURE_NAMES
    for text, jf, jspans in zip(TEXTS, js["feats"], js["spans"]):
        pf = stylometry.features(text)
        for k in stylometry.FEATURE_NAMES:
            assert close(pf[k], jf[k]), (k, text, pf[k], jf[k])
        assert [list(s) for s in stylometry.split_sentences(text)] == jspans


def test_scores_and_classifier_match():
    rng = np.random.default_rng(0)
    lp, xp = rng.uniform(0.5, 5, 40), rng.uniform(1, 4, 40)
    offsets = [(i * 5, i * 5 + 4) for i in range(40)]
    ts = TokenScores(lp, xp, offsets)
    feats = {**ts.summary(), **stylometry.features(TEXTS[1])}
    clf = {"features": list(feats), "mean": rng.normal(size=len(feats)).tolist(),
           "scale": rng.uniform(0.5, 2, len(feats)).tolist(), "coef": rng.normal(size=len(feats)).tolist(),
           "intercept": 0.3}
    js = run_node(f"""
        import {{ summary, spanScore, classifierProb, labelFor }} from "{(SITE_JS / 'detector.js').as_uri()}";
        const lp = {json.dumps(lp.tolist())}, xp = {json.dumps(xp.tolist())}, offsets = {json.dumps(offsets)};
        const feats = {json.dumps(feats)};
        const clf = {json.dumps(clf)};
        console.log(JSON.stringify({{ summary: summary(lp, xp), span: spanScore({{lp, xp, offsets}}, 10, 60),
          short: spanScore({{lp, xp, offsets}}, 0, 9), p: classifierProb(clf, feats),
          labels: [0.1, 0.3, 0.5, 0.7, 0.9].map(labelFor) }}));
    """)
    for k, v in ts.summary().items():
        assert close(v, js["summary"][k])
    assert close(ts.span_score(10, 60), js["span"])
    assert ts.span_score(0, 9) is None and js["short"] is None
    assert close(classifier_prob(clf, feats), js["p"])
    assert [label_for(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)] == js["labels"]


def test_token_offsets_match():
    from detector.lm_scores import token_offsets

    pieces = [" Hello", ",", " wor", "ld", "\n\n", "Next", " sentence", "."]
    # A multi-byte character split over two tokens: first half decodes to U+FFFD.
    decode_py = lambda g: "".join("é" if i == 99 else ("�" if i == 98 and g[-1] == 98 else pieces[i] if i < 98 else "") for i in g)  # noqa: E731
    ids = [0, 1, 2, 3, 98, 99, 4, 5, 6, 7]
    py = token_offsets(ids, decode_py)
    js = run_node(f"""
        import {{ tokenOffsets }} from "{(SITE_JS / 'detector.js').as_uri()}";
        const pieces = {json.dumps(pieces)};
        const decode = g => g.map(i => i === 99 ? "é" : (i === 98 && g[g.length-1] === 98 ? "\\uFFFD" : (i < 98 ? pieces[i] : ""))).join("");
        console.log(JSON.stringify(tokenOffsets({json.dumps(ids)}, decode)));
    """)
    assert [list(o) for o in py] == js
    assert py[0] == (1, 6)  # leading space excluded
