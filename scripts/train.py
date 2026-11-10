"""Train (or retrain) the classifier on everything in data/human and data/ai.

Features for each file are cached in data/features-<models>.csv by content
hash, so a monthly retrain only scores the new files. Changing the model pair
starts a fresh cache automatically.

    python scripts/export_models.py   # once, creates site/models/
    python scripts/train.py

Writes site/model.json (read by the website), reports/metrics.json and
reports/latest.md. Refuses to replace the current model if the new one scores
worse on the held-out set (pass --force to override).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detector.pipeline import ALL_FEATURES, CLASSIFIER_PATH, Detector  # noqa: E402

DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODEL_OUT = CLASSIFIER_PATH


def load_files() -> list[tuple[Path, int]]:
    files = []
    for label, sub in ((0, "human"), (1, "ai")):
        for p in sorted((DATA / sub).rglob("*")):
            if p.suffix.lower() in {".txt", ".md"} and p.name.lower() != "readme.md":
                files.append((p, label))
    return files


def source_of(p: Path) -> str:
    kind = "ai" if (DATA / "ai") in p.parents else "human"
    return f"{kind}:{p.name.split('__')[0] if '__' in p.name else 'your-files'}"


def cache_path(det: Detector) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{det.scorer.observer_name}_{det.scorer.performer_name}").strip("-")
    return DATA / f"features-{slug}.csv"


def build_features(det: Detector, files) -> pd.DataFrame:
    cache_file = cache_path(det)
    cache = pd.read_csv(cache_file) if cache_file.exists() else pd.DataFrame(columns=["sha", *ALL_FEATURES])
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
        rows.append({"sha": sha, "label": label, "path": str(p.relative_to(DATA)), "source": source_of(p)})
    if new:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True)
        cache.to_csv(cache_file, index=False)
    return pd.DataFrame(rows).merge(cache.drop_duplicates("sha"), on="sha", how="left")


def fit(X, y, C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=5000).fit(scaler.transform(X), y)
    return scaler, lr


def predict(scaler, lr, X):
    return lr.predict_proba(scaler.transform(X))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--C", type=float, default=0.5, help="regularisation (smaller = simpler model)")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import train_test_split

    files = load_files()
    counts = {lab: sum(1 for _, l in files if l == lab) for lab in (0, 1)}
    print(f"Found {counts[0]} human and {counts[1]} AI files")
    if min(counts.values()) < 20:
        sys.exit("Need at least 20 files of each kind in data/human and data/ai to train.")

    det = Detector(classifier_path=Path("/nonexistent"))
    df = build_features(det, files)
    X = np.nan_to_num(df[ALL_FEATURES].to_numpy(dtype=float))
    y = df["label"].to_numpy()

    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df, test_size=args.test_size, stratify=y, random_state=42)
    scaler, lr = fit(X_tr, y_tr, args.C)
    p = predict(scaler, lr, X_te)
    auc = float(roc_auc_score(y_te, p))
    acc = float(accuracy_score(y_te, p >= 0.5))
    tn, fp, fn, tp = confusion_matrix(y_te, p >= 0.5, labels=[0, 1]).ravel()
    fpr = float(fp / max(fp + tn, 1))
    # Baseline: Binoculars score alone, to show what the classifier adds.
    bino_auc = float(roc_auc_score(y_te, -df_te["binoculars"].fillna(1.0)))

    old = json.loads((REPORTS / "metrics.json").read_text()) if (REPORTS / "metrics.json").exists() else {}
    if old.get("auc") and auc < old["auc"] - 0.02 and not args.force:
        sys.exit(f"New model AUC {auc:.3f} is worse than current {old['auc']:.3f}; keeping the old model.")

    # Refit on all data for the shipped model.
    scaler, lr = fit(X, y, args.C)
    # Sentence highlighting uses the Binoculars score alone (style features
    # mean little for one sentence). Fit a 1-D logistic curve for it.
    b = np.nan_to_num(df["binoculars"].to_numpy(dtype=float), nan=1.0)[:, None]
    one = LogisticRegression(max_iter=1000).fit(b, y)
    w, c = float(one.coef_[0, 0]), float(one.intercept_[0])
    sent_thr, sent_scale = (-c / w, -1.0 / w) if w < 0 else (0.95, 0.05)

    meta = {"trained_at": dt.datetime.now(dt.timezone.utc).isoformat(), "auc": auc, "accuracy": acc,
            "false_positive_rate": fpr, "binoculars_only_auc": bino_auc,
            "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()), "n_test": int(len(y_te)),
            "observer": det.scorer.observer_name, "performer": det.scorer.performer_name}
    model = {"features": ALL_FEATURES, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
             "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
             "sentence_threshold": sent_thr, "sentence_scale": sent_scale, "meta": meta}
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps(model, indent=1))

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(meta, indent=2))
    per_source = (df_te.assign(pred=(p >= 0.5).astype(int))
                  .groupby("source").apply(lambda g: (g["pred"] == g["label"]).mean(), include_groups=False))
    top = sorted(zip(ALL_FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1]))[:8]
    md = [f"# Training report ({meta['trained_at'][:10]})", "",
          f"Models: `{meta['observer']}` + `{meta['performer']}`. "
          f"Data: {meta['n_human']} human, {meta['n_ai']} AI documents. Held-out test set: {len(y_te)}.", "",
          "| Metric | Value |", "|---|---|",
          f"| AUC (classifier) | {auc:.3f} |", f"| AUC (Binoculars score alone) | {bino_auc:.3f} |",
          f"| Accuracy at 50% | {acc:.3f} |",
          f"| False-positive rate (human flagged as AI) | {fpr:.3f} |", "",
          "## Accuracy by source", "", "| Source | Accuracy |", "|---|---|",
          *[f"| {s} | {v:.3f} |" for s, v in per_source.items()], "",
          "## Strongest signals (positive = points to AI)", "", "| Feature | Weight |", "|---|---|",
          *[f"| {f} | {c:+.2f} |" for f, c in top]]
    (REPORTS / "latest.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
