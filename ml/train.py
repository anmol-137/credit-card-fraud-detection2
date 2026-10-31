"""Train the fraud-detection model and write everything the web app needs.

Usage:
    python ml/train.py --data data/creditcard.csv

Outputs
    models/fraud_model.joblib         deployed Random Forest + threshold + reference stats
    models/model_card.json            metrics, dataset facts, feature importance
    app/static/data/samples.json      real test-set transactions for the demo buttons
    app/static/img/{pr_roc,confusion,importance}.png
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
REQUIRED = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]


def load_clean(path: Path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    raw_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)          # duplicates leak between train and test
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)       # Time is seconds since the first transaction
    df = df.drop(columns=["Time"])
    return df, raw_rows


def best_f1_threshold(y_true, scores) -> float:
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f1)])


def evaluate(y_true, scores, threshold) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    args = ap.parse_args()

    df, raw_rows = load_clean(Path(args.data))
    features = [c for c in df.columns if c != "Class"]
    X, y = df[features], df["Class"]
    print(f"Rows {raw_rows:,} -> {len(df):,} after removing duplicates; fraud cases: {int(y.sum())}")

    # Stratified 60/20/20 split: train fits, validation picks the threshold, test is for reporting only.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    # Baseline: unsupervised anomaly detection (never sees labels)
    iso = IsolationForest(n_estimators=200, contamination=float(y_tr.mean()), random_state=SEED, n_jobs=-1)
    iso.fit(X_tr)
    iso_val, iso_te = -iso.score_samples(X_val), -iso.score_samples(X_te)
    iso_metrics = evaluate(y_te, iso_te, best_f1_threshold(y_val, iso_val))

    # Deployed model: Random Forest with class weights for the imbalance
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=2, max_features="sqrt",
                                max_samples=0.5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)
    rf_val, rf_te = rf.predict_proba(X_val)[:, 1], rf.predict_proba(X_te)[:, 1]
    threshold = best_f1_threshold(y_val, rf_val)
    rf_metrics = evaluate(y_te, rf_te, threshold)
    print("Random Forest   :", {k: round(v, 4) for k, v in rf_metrics.items()})
    print("Isolation Forest:", {k: round(v, 4) for k, v in iso_metrics.items()})

    # Reference statistics of legitimate training transactions (used for the "unusual signals" panel)
    legit = X_tr[y_tr == 0]
    median = legit.median()
    scale = (legit.quantile(0.75) - legit.quantile(0.25)) / 1.349
    scale = scale.where(scale > 0, legit.std()).where(lambda s: s > 0, 1.0)
    importance = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({
        "model": rf, "features": features, "threshold": threshold,
        "ref_median": median.to_dict(), "ref_scale": scale.to_dict(),
        "importance": importance.to_dict(),
    }, ROOT / "models" / "fraud_model.joblib", compress=3)

    card = {
        "deployed_model": "Random Forest",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows_raw": raw_rows, "rows_clean": int(len(df)), "duplicates_removed": int(raw_rows - len(df)),
            "fraud_cases": int(y.sum()), "fraud_rate_pct": round(float(y.mean() * 100), 4),
            "test_rows": int(len(y_te)), "test_fraud_cases": int(y_te.sum()),
        },
        "threshold": threshold,
        "metrics": {"Random Forest": rf_metrics, "Isolation Forest": iso_metrics},
        "features": features,
        "top_features": [{"feature": k, "importance": round(float(v), 4)} for k, v in importance.head(10).items()],
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    (ROOT / "models" / "model_card.json").write_text(json.dumps(card, indent=2))

    # Demo samples: real, unseen test transactions (2 caught fraud, 1 missed fraud if any, rest legit)
    rng = np.random.RandomState(7)
    s = pd.Series(rf_te, index=X_te.index)
    fraud_idx = list(y_te[y_te == 1].index)
    caught = [i for i in fraud_idx if s[i] >= threshold]
    missed = [i for i in fraud_idx if s[i] < threshold]
    legit_ok = [i for i in y_te[y_te == 0].index if s[i] < threshold]
    pick = list(rng.choice(caught, 2, replace=False))
    if missed:
        pick.append(rng.choice(missed))
    pick += list(rng.choice(legit_ok, 6 - len(pick), replace=False))
    rng.shuffle(pick)
    samples = [{
        "label": f"Sample {n}",
        "actual": "fraud" if int(y_te[i]) == 1 else "legit",
        "features": {f: round(float(X_te.loc[i, f]), 6) for f in features},
    } for n, i in enumerate(pick, 1)]
    out = ROOT / "app" / "static"
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "data" / "samples.json").write_text(json.dumps(samples))

    # Figures for the web page
    img = out / "img"
    img.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    scores = {"Random Forest": rf_te, "Isolation Forest": iso_te}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, sc in scores.items():
        p, r, _ = precision_recall_curve(y_te, sc)
        ax[0].plot(r, p, label=f"{name} (AP {average_precision_score(y_te, sc):.2f})")
        fpr, tpr, _ = roc_curve(y_te, sc)
        ax[1].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, sc):.3f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall"); ax[0].legend()
    ax[1].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC"); ax[1].legend()
    fig.tight_layout(); fig.savefig(img / "pr_roc.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    cm = np.array([[rf_metrics["tn"], rf_metrics["fp"]], [rf_metrics["fn"], rf_metrics["tp"]]])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred legit", "Pred fraud"], yticklabels=["Actual legit", "Actual fraud"])
    ax.set_title("Random Forest, test set")
    fig.tight_layout(); fig.savefig(img / "confusion.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    top = importance.head(12)[::-1]
    ax.barh(top.index, top.values, color="#3D5A80")
    ax.set_title("Random Forest: top features")
    fig.tight_layout(); fig.savefig(img / "importance.png", dpi=120); plt.close(fig)
    print("Done. Wrote models/, samples.json and figures.")


if __name__ == "__main__":
    main()
