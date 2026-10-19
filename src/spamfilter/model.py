"""Candidate pipelines, threshold tuning and per-message explanations."""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def make_features() -> FeatureUnion:
    """Word 1-2 grams plus character 2-5 grams.

    Character n-grams catch obfuscated spam such as "fr33" or "txt" variants
    and things like currency symbols and long digit runs.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])


def candidates(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline([("features", make_features()), ("clf", MultinomialNB(alpha=0.1))]),
        "logistic_regression": Pipeline([
            ("features", make_features()),
            ("clf", LogisticRegression(C=10, class_weight="balanced", max_iter=2000, random_state=seed)),
        ]),
        "linear_svm_calibrated": Pipeline([
            ("features", make_features()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced", random_state=seed), cv=3)),
        ]),
    }


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Pick the probability cut-off that maximises F1 on out-of-fold predictions."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def _feature_weights(pipe: Pipeline) -> np.ndarray:
    """One weight per feature; positive values push a message towards spam."""
    clf = pipe.named_steps["clf"]
    if isinstance(clf, MultinomialNB):
        return clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    if isinstance(clf, CalibratedClassifierCV):
        return np.mean([c.estimator.coef_.ravel() for c in clf.calibrated_classifiers_], axis=0)
    return clf.coef_.ravel()


def feature_names(pipe: Pipeline) -> np.ndarray:
    return pipe.named_steps["features"].get_feature_names_out()


def top_global_terms(pipe: Pipeline, n: int = 15, prefix: str = "word__") -> list[tuple[str, float]]:
    """Most spam-indicative word features across the whole model."""
    names, weights = feature_names(pipe), _feature_weights(pipe)
    mask = np.char.startswith(names.astype(str), prefix)
    idx = np.where(mask)[0]
    order = idx[np.argsort(weights[idx])[::-1][:n]]
    return [(names[i].removeprefix(prefix), float(weights[i])) for i in order]


def explain(pipe: Pipeline, text: str, n: int = 8) -> list[tuple[str, float]]:
    """Word-level features present in `text`, ranked by contribution (tf-idf value x weight)."""
    feats = pipe.named_steps["features"]
    vec = feats.transform([text]).tocsr()
    names, weights = feature_names(pipe), _feature_weights(pipe)
    contrib = []
    for j, v in zip(vec.indices, vec.data):
        name = str(names[j])
        if name.startswith("word__"):
            contrib.append((name.removeprefix("word__"), float(v * weights[j])))
    contrib.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return contrib[:n]
