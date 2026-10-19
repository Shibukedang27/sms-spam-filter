"""Download, cache and clean the UCI SMS Spam Collection."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_FILE = DATA_DIR / "SMSSpamCollection"


def download(force: bool = False) -> Path:
    """Fetch the dataset zip from UCI once and cache the raw TSV under data/."""
    if RAW_FILE.exists() and not force:
        return RAW_FILE
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        RAW_FILE.write_bytes(zf.read("SMSSpamCollection"))
    return RAW_FILE


def load(path: Path | None = None, dedupe: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns text (str) and label (1 = spam, 0 = ham).

    Exact duplicate messages are dropped by default. The raw file contains
    several hundred repeated texts, and leaving them in lets identical messages
    land in both train and test, which inflates test scores.
    """
    path = path or download()
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        tag, _, text = line.partition("\t")
        rows.append((text.strip(), 1 if tag.strip() == "spam" else 0))
    df = pd.DataFrame(rows, columns=["text", "label"])
    if dedupe:
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df
