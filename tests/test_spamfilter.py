import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from spamfilter import data, model  # noqa: E402


def test_load_parses_and_dedupes(tmp_path):
    f = tmp_path / "sms.tsv"
    f.write_text("ham\thello there\nspam\tWIN cash now\nham\thello there\n\n", encoding="utf-8")
    raw = data.load(f, dedupe=False)
    clean = data.load(f, dedupe=True)
    assert len(raw) == 3 and len(clean) == 2
    assert clean.label.tolist() == [0, 1]


def test_best_threshold_finds_separating_cutoff():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr, f1 = model.best_threshold(y, p)
    assert 0.3 < thr <= 0.8 and f1 == 1.0


def test_pipeline_learns_and_explains():
    texts = ["win free cash prize now", "free prize claim txt", "urgent win cash",
             "see you at lunch", "are we meeting today", "call me when home"] * 5
    labels = [1, 1, 1, 0, 0, 0] * 5
    for name, pipe in model.candidates().items():
        pipe.fit(texts, labels)
        assert pipe.predict_proba(["free cash prize"])[0, 1] > pipe.predict_proba(["see you at lunch"])[0, 1], name
        terms = dict(model.explain(pipe, "free cash prize"))
        assert terms and max(terms.values()) > 0


def test_trained_model_if_present():
    path = ROOT / "models" / "spam_model.joblib"
    if not path.exists():
        return
    from predict import classify, load_bundle
    b = load_bundle(path)
    assert classify(b, "Congratulations! You won a FREE prize. Call 09061701461 to claim")["label"] == "spam"
    assert classify(b, "ok, I'll pick you up at 7")["label"] == "ham"
