"""Train, compare and evaluate SMS spam models. Writes models/ and reports/."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from spamfilter import data, model  # noqa: E402

SEED = 42
ROOT = Path(__file__).parent
REPORTS, MODELS = ROOT / "reports", ROOT / "models"


def scores(y, pred, proba) -> dict:
    return {
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
    }


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    raw = data.load(dedupe=False)
    df = data.load(dedupe=True)
    print(f"raw rows: {len(raw)}  after dedupe: {len(df)}  spam share: {df.label.mean():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df.text, df.label, test_size=0.2, stratify=df.label, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1) Compare candidates with 5-fold out-of-fold probabilities on the training split only.
    comparison = {}
    for name, pipe in model.candidates(SEED).items():
        t0 = time.perf_counter()
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr, f1_at_thr = model.best_threshold(y_tr.values, oof)
        comparison[name] = {
            "cv_f1_at_0.5": round(f1_score(y_tr, (oof >= 0.5).astype(int)), 4),
            "cv_pr_auc": round(average_precision_score(y_tr, oof), 4),
            "tuned_threshold": thr,
            "cv_f1_at_tuned": round(f1_at_thr, 4),
            "cv_seconds": round(time.perf_counter() - t0, 1),
        }
        print(name, comparison[name])

    best = max(comparison, key=lambda k: comparison[k]["cv_f1_at_tuned"])
    thr = comparison[best]["tuned_threshold"]
    print(f"selected: {best} (threshold {thr})")

    # 2) Fit the selected model on the full training split, evaluate once on the held-out test set.
    final = clone(model.candidates(SEED)[best]).fit(X_tr, y_tr)
    proba = final.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)
    test = scores(y_te, pred, proba)
    test_default = scores(y_te, (proba >= 0.5).astype(int), proba)
    cm = confusion_matrix(y_te, pred)

    t0 = time.perf_counter()
    for _ in range(5):
        final.predict_proba(X_te)
    ms_per_msg = (time.perf_counter() - t0) / (5 * len(X_te)) * 1000

    # 3) Leakage check: same pipeline, but split the raw data that still contains duplicates.
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        raw.text, raw.label, test_size=0.2, stratify=raw.label, random_state=SEED)
    overlap = int(Xr_te.isin(set(Xr_tr)).sum())
    leaky = clone(model.candidates(SEED)[best]).fit(Xr_tr, yr_tr)
    lp = leaky.predict_proba(Xr_te)[:, 1]
    leaky_scores = scores(yr_te, (lp >= thr).astype(int), lp)

    # 4) Error analysis on the clean test set.
    errors = []
    for text, y, p in zip(X_te, y_te, proba):
        if int(p >= thr) != y:
            errors.append({"text": text, "true": "spam" if y else "ham", "spam_probability": round(float(p), 3)})
    errors.sort(key=lambda e: abs(e["spam_probability"] - thr), reverse=True)

    metrics = {
        "dataset": {"raw_rows": len(raw), "deduped_rows": len(df), "spam_share": round(float(df.label.mean()), 4),
                    "train_rows": len(X_tr), "test_rows": len(X_te)},
        "cv_comparison": comparison,
        "selected_model": best,
        "threshold": thr,
        "test_at_tuned_threshold": test,
        "test_at_0.5": test_default,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "leakage_check": {"test_rows_also_in_train_when_not_deduped": overlap, "scores_with_duplicates": leaky_scores},
        "inference_ms_per_message": round(ms_per_msg, 3),
        "top_spam_terms": model.top_global_terms(final, 15),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS / "errors.json").write_text(json.dumps(errors, indent=2))
    joblib.dump({"pipeline": final, "threshold": thr, "name": best}, MODELS / "spam_model.joblib")

    # 5) Charts.
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test confusion matrix (threshold {thr})")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrix.png", dpi=160); plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_te, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rec, prec, lw=2)
    ax.scatter([test["recall"]], [test["precision"]], color="crimson", zorder=3, label=f"chosen threshold {thr}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.5, 1.02)
    ax.set_title(f"Precision-recall, test set (AP = {test['pr_auc']})"); ax.legend(loc="lower left"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(REPORTS / "pr_curve.png", dpi=160); plt.close(fig)

    names = list(comparison)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [comparison[n]["cv_f1_at_0.5"] for n in names], 0.4, label="F1 @ 0.5")
    ax.bar(x + 0.2, [comparison[n]["cv_f1_at_tuned"] for n in names], 0.4, label="F1 @ tuned threshold")
    ax.set_xticks(x, [n.replace("_", "\n") for n in names]); ax.set_ylim(0.8, 1.0)
    ax.set_title("5-fold CV F1 on the training split"); ax.legend(loc="lower right"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(REPORTS / "model_comparison.png", dpi=160); plt.close(fig)

    terms = metrics["top_spam_terms"][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh([t for t, _ in terms], [w for _, w in terms], color="#d9534f")
    ax.set_title("Most spam-indicative word features"); ax.set_xlabel("model weight")
    fig.tight_layout(); fig.savefig(REPORTS / "top_terms.png", dpi=160); plt.close(fig)

    print(json.dumps({k: metrics[k] for k in ("test_at_tuned_threshold", "confusion_matrix", "leakage_check",
                                              "inference_ms_per_message")}, indent=2))


if __name__ == "__main__":
    main()
