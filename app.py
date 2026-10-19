"""Streamlit dashboard: try the model, inspect metrics and browse mistakes.

    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from predict import classify, load_bundle  # noqa: E402

st.set_page_config(page_title="SMS Spam Filter", layout="wide")
bundle = st.cache_resource(load_bundle)()
metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
errors = json.loads((ROOT / "reports" / "errors.json").read_text())

st.title("SMS Spam Filter")
st.caption(f"Model: {bundle['name']}  |  decision threshold: {bundle['threshold']}  |  "
           "trained on the UCI SMS Spam Collection (deduplicated)")

tab_try, tab_eval, tab_err = st.tabs(["Try it", "Evaluation", "Mistakes"])

with tab_try:
    text = st.text_area("Paste an SMS", "URGENT! You have won a 1 week FREE membership. Txt CLAIM to 81010 now")
    if text.strip():
        r = classify(bundle, text)
        c1, c2 = st.columns(2)
        c1.metric("Prediction", r["label"].upper())
        c2.metric("Spam probability", f"{r['spam_probability']:.1%}")
        st.progress(min(max(r["spam_probability"], 0.0), 1.0))
        if r["top_terms"]:
            st.write("Words that moved the score (positive = towards spam):")
            st.bar_chart(pd.DataFrame(r["top_terms"], columns=["term", "contribution"]).set_index("term"))

with tab_eval:
    t = metrics["test_at_tuned_threshold"]
    cols = st.columns(5)
    for col, k in zip(cols, ["precision", "recall", "f1", "roc_auc", "pr_auc"]):
        col.metric(k.upper().replace("_", "-"), f"{t[k]:.3f}")
    st.write(f"Held-out test set: {metrics['dataset']['test_rows']} messages. "
             f"Confusion matrix: {metrics['confusion_matrix']}")
    st.subheader("Model comparison (5-fold CV on training split)")
    st.dataframe(pd.DataFrame(metrics["cv_comparison"]).T)
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "reports" / "pr_curve.png"))
    c2.image(str(ROOT / "reports" / "confusion_matrix.png"))
    c1.image(str(ROOT / "reports" / "model_comparison.png"))
    c2.image(str(ROOT / "reports" / "top_terms.png"))
    lk = metrics["leakage_check"]
    st.info(f"Leakage check: without deduplication, {lk['test_rows_also_in_train_when_not_deduped']} test messages "
            f"were exact copies of training messages and F1 read {lk['scores_with_duplicates']['f1']:.3f} "
            f"instead of {t['f1']:.3f}.")

with tab_err:
    st.write(f"{len(errors)} misclassified test messages:")
    st.dataframe(pd.DataFrame(errors), use_container_width=True)
