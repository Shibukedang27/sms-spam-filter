"""Classify SMS messages from the command line.

    python predict.py "WINNER! Claim your free prize now"
    echo "see you at 6?" | python predict.py -
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent / "src"))
from spamfilter import model  # noqa: E402

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"


def load_bundle(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        sys.exit(f"No model at {path}. Run `python train.py` first.")
    return joblib.load(path)


def classify(bundle: dict, text: str) -> dict:
    p = float(bundle["pipeline"].predict_proba([text])[0, 1])
    return {"label": "spam" if p >= bundle["threshold"] else "ham", "spam_probability": round(p, 4),
            "top_terms": model.explain(bundle["pipeline"], text, n=5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("messages", nargs="+", help="message text, or '-' to read one message per line from stdin")
    ap.add_argument("--explain", action="store_true", help="show the words that pushed the score up or down")
    args = ap.parse_args()
    msgs = [l for l in sys.stdin.read().splitlines() if l.strip()] if args.messages == ["-"] else args.messages
    bundle = load_bundle()
    for m in msgs:
        r = classify(bundle, m)
        print(f"{r['label'].upper():4}  p(spam)={r['spam_probability']:.3f}  {m[:70]}")
        if args.explain:
            print("      " + ", ".join(f"{t} ({w:+.2f})" for t, w in r["top_terms"]))


if __name__ == "__main__":
    main()
