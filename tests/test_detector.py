import json
import random
from pathlib import Path

from detector import stylometry
from detector.pipeline import Detector, result_to_json

HUMAN = ("I wasn't sure the bus would come. It did, eventually, twenty minutes late and packed. "
         "My sister laughed at me for bringing an umbrella on a sunny day, but by noon it poured. ") * 6
AI = ("Furthermore, renewable energy plays a crucial role in fostering sustainable communities. "
      "Moreover, it is important to note that robust infrastructure underscores long-term resilience. ") * 6


def test_sentence_spans_round_trip():
    text = "First one. Second one!\n\nThird paragraph here? Yes."
    spans = stylometry.split_sentences(text)
    assert [s for _, _, s in spans] == ["First one.", "Second one!", "Third paragraph here?", "Yes."]
    for s, e, sent in spans:
        assert text[s:e] == sent


def test_style_features_differ():
    h, a = stylometry.features(HUMAN), stylometry.features(AI)
    assert set(h) == set(stylometry.FEATURE_NAMES)
    assert a["stock_phrase_rate"] > h["stock_phrase_rate"]
    assert h["contraction_rate"] > a["contraction_rate"]


def test_uncalibrated_detector(tmp_path):
    det = Detector(model_path=tmp_path / "missing.joblib")
    ra, rh = det.analyze(AI), det.analyze(HUMAN)
    assert ra.ai_probability > rh.ai_probability
    assert not ra.calibrated and any("uncalibrated" in w for w in ra.warnings)
    assert ra.sentences and all(s.ai_probability is not None for s in ra.sentences)
    json.loads(result_to_json(ra))


def test_short_text_warning(tmp_path):
    r = Detector(model_path=tmp_path / "x").analyze("Too short to judge.")
    assert any("unreliable" in w for w in r.warnings)


def test_training_end_to_end(tmp_path, monkeypatch):
    import scripts.train as train

    data = tmp_path / "data"
    rng = random.Random(0)
    for sub, base in (("human", HUMAN), ("ai", AI)):
        (data / sub).mkdir(parents=True)
        for i in range(30):
            sents = base.split(". ")
            rng.shuffle(sents)
            (data / sub / f"src__{i}.txt").write_text(". ".join(sents))
    monkeypatch.setattr(train, "DATA", data)
    monkeypatch.setattr(train, "CACHE", data / "features.csv")
    monkeypatch.setattr(train, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "model" / "clf.joblib")
    monkeypatch.setattr("sys.argv", ["train.py"])
    train.main()

    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["auc"] > 0.9
    det = Detector(model_path=tmp_path / "model" / "clf.joblib")
    assert det.analyze(AI).ai_probability > det.analyze(HUMAN).ai_probability
    assert det.analyze(AI).calibrated
