"""Train (or retrain) the classifier on everything in data/human and data/ai.

Features for each file are cached in data/features.csv by content hash, so a
monthly retrain only scores the new files.

    python scripts/train.py

Writes model/classifier.joblib, reports/metrics.json and reports/latest.md.
Refuses to replace the current model if the new one scores worse on the
held-out set (pass --force to override).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.pipeline import ALL_FEATURES, MODEL_PATH, Detector  # noqa: E402

DATA = ROOT / "data"
CACHE = DATA / "features.csv"
REPORTS = ROOT / "reports"


def load_files() -> list[tuple[Path, int]]:
    files = []
    for label, sub in ((0, "human"), (1, "ai")):
        for p in sorted((DATA / sub).rglob("*")):
            if p.suffix.lower() in {".txt", ".md"} and p.name.lower() != "readme.md":
                files.append((p, label))
    return files


def sub_of(p: Path) -> str:
    return "ai" if (DATA / "ai") in p.parents else "human"


def build_features(det: Detector, files) -> pd.DataFrame:
    cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["sha", *ALL_FEATURES])
    known = set(cache["sha"])
    rows, new = [], []
    for i, (p, label) in enumerate(files):
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text.split()) < 40:
            continue
        sha = hashlib.sha1(text.encode()).hexdigest()
        if sha not in known:
            print(f"[{i + 1}/{len(files)}] scoring {p.relative_to(DATA)}", flush=True)
            new.append({"sha": sha, **det.feature_vector(text)})
            known.add(sha)
        rows.append({"sha": sha, "label": label, "path": str(p.relative_to(DATA)),
                     "source": f"{sub_of(p)}:{p.name.split('__')[0] if '__' in p.name else 'your-files'}"})
    if new:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True)
        cache.to_csv(CACHE, index=False)
    return pd.DataFrame(rows).merge(cache.drop_duplicates("sha"), on="sha", how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.25)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split
    import joblib

    files = load_files()
    counts = {lab: sum(1 for _, l in files if l == lab) for lab in (0, 1)}
    print(f"Found {counts[0]} human and {counts[1]} AI files")
    if min(counts.values()) < 20:
        sys.exit("Need at least 20 files of each kind in data/human and data/ai to train.")

    det = Detector(model_path=Path("/nonexistent"))  # score with LM only, ignore any old classifier
    df = build_features(det, files)
    X = np.nan_to_num(df[ALL_FEATURES].to_numpy(dtype=float))
    y = df["label"].to_numpy()

    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df, test_size=args.test_size, stratify=y, random_state=42)
    if len(y_tr) >= 300:
        # Enough data for a non-linear model that can learn feature interactions.
        base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300)
    else:
        base = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    clf = CalibratedClassifierCV(base, cv=3, method="isotonic" if len(y_tr) > 400 else "sigmoid")
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, p))
    acc = float(accuracy_score(y_te, p >= 0.5))
    tn, fp, fn, tp = confusion_matrix(y_te, p >= 0.5).ravel()
    fpr = float(fp / max(fp + tn, 1))
    # Baseline: Binoculars score alone, to show what the classifier adds.
    bino_auc = float(roc_auc_score(y_te, -df_te["binoculars"].fillna(1.0)))

    old = json.loads((REPORTS / "metrics.json").read_text()) if (REPORTS / "metrics.json").exists() else {}
    if old.get("auc") and auc < old["auc"] - 0.02 and not args.force:
        sys.exit(f"New model AUC {auc:.3f} is worse than current {old['auc']:.3f}; keeping the old model.")

    # Refit on all data for the shipped model.
    clf.fit(X, y)
    meta = {"features": ALL_FEATURES, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "auc": auc, "accuracy": acc, "false_positive_rate": fpr,
            "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
            "model_type": type(base).__name__}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "meta": meta}, MODEL_PATH)

    REPORTS.mkdir(exist_ok=True)
    metrics = {**meta, "binoculars_only_auc": bino_auc, "n_test": int(len(y_te))}
    metrics.pop("features")
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    per_source = (df_te.assign(pred=(p >= 0.5).astype(int))
                  .groupby("source").apply(lambda g: (g["pred"] == g["label"]).mean(), include_groups=False))
    md = [f"# Training report ({meta['trained_at'][:10]})", "",
          f"Data: {meta['n_human']} human, {meta['n_ai']} AI documents. Held-out test set: {len(y_te)}.", "",
          "| Metric | Value |", "|---|---|",
          f"| AUC (classifier) | {auc:.3f} |", f"| AUC (Binoculars score alone) | {bino_auc:.3f} |",
          f"| Accuracy at 50% | {acc:.3f} |",
          f"| False-positive rate (human flagged as AI) | {fpr:.3f} |", "",
          "## Accuracy by source", "", "| Source | Accuracy |", "|---|---|",
          *[f"| {s} | {v:.3f} |" for s, v in per_source.items()]]
    (REPORTS / "latest.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
