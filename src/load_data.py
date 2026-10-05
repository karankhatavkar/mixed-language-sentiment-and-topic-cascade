"""Build data/processed/{dev,test,youtube}.csv.

- dev.csv / test.csv : public-dataset slices (English, Arabic, EESA mixed),
                       sampled DEV_N / TEST_N per slice, 3-way gold labels.
- youtube.csv        : YouTube SHEIN comments, slice assigned per comment,
                       4-way gold labels (adds `mixed` and `gold_topics`).
                       Read from the human-reviewed pre-labelled CSV; no
                       dev/test split — the whole set is `split='test'`.

Deterministic for the public sources (same seed + same raw files -> identical
output). The YouTube step is a pass-through transform of the reviewed labels.

Common schema: id, text, gold_sentiment, slice, split, source
YouTube rows add: gold_topics  (JSON list of topics from the fixed 8)

Labels in public gold: lowercase 3-way {positive, negative, neutral}.
Labels in YouTube gold: 4-way, adds `mixed`.

Run: python -m src.load_data
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DEV_N,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    SENTIMENT_LABELS,
    TEST_N,
    TOPICS,
)

CARDIFF_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
ALLOWED_LABELS = set(SENTIMENT_LABELS)

YOUTUBE_REVIEWED_PATH = PROCESSED_DIR / "youtube_prelabeled.csv"
YOUTUBE_OUT_PATH = PROCESSED_DIR / "youtube.csv"
YOUTUBE_SOURCE = "youtube:shein_clothing_hauls"
YOUTUBE_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
YOUTUBE_SLICES = {"english", "arabic", "mixed"}
YOUTUBE_TOPICS = set(TOPICS)


def _load_cardiff(config: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = RAW_DIR / "tweet_sentiment_multilingual"
    dev = pd.read_csv(base / f"{config}_validation.csv")
    test = pd.read_csv(base / f"{config}_test.csv")
    for d in (dev, test):
        d["gold_sentiment"] = d["label"].map(CARDIFF_LABEL_MAP)
        d.drop(columns=["label"], inplace=True)
    return dev, test


def _load_eesa() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = RAW_DIR / "eesa"
    dev = pd.read_csv(base / "EESA-Dev.csv", header=None, names=["text", "gold_sentiment"])
    test = pd.read_csv(base / "EESA-Test.csv", header=None, names=["text", "gold_sentiment"])
    for d in (dev, test):
        d["gold_sentiment"] = d["gold_sentiment"].astype(str).str.strip().str.lower()
    return dev, test


def _clean_and_sample(df: pd.DataFrame, n: int, slice_name: str, split: str) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    before = len(df)
    df = df[df["text"] != ""]
    dropped = before - len(df)
    if dropped:
        print(f"  [{slice_name}/{split}] dropped {dropped} empty-text rows")

    bad = set(df["gold_sentiment"]) - ALLOWED_LABELS
    if bad:
        raise ValueError(f"[{slice_name}/{split}] unexpected gold labels: {bad}")

    if len(df) < n:
        raise ValueError(f"[{slice_name}/{split}] only {len(df)} rows after cleaning, need {n}")

    sampled, _ = train_test_split(
        df, train_size=n, stratify=df["gold_sentiment"], random_state=SEED
    )
    return sampled.reset_index(drop=True)


def _assemble(df: pd.DataFrame, slice_name: str, split: str, source: str) -> pd.DataFrame:
    ids = [f"{slice_name}_{split}_{i:03d}" for i in range(1, len(df) + 1)]
    return pd.DataFrame(
        {
            "id": ids,
            "text": df["text"].values,
            "gold_sentiment": df["gold_sentiment"].values,
            "slice": slice_name,
            "split": split,
            "source": source,
        }
    )


def _report(df: pd.DataFrame, name: str) -> None:
    counts = df.groupby(["slice", "gold_sentiment"]).size().unstack(fill_value=0)
    print(f"\n{name}  (n={len(df)})")
    print(counts.to_string())


def _load_youtube_reviewed() -> pd.DataFrame:
    """Load the human-reviewed YouTube pre-labels. Fail fast at the boundary."""
    df = pd.read_csv(YOUTUBE_REVIEWED_PATH)

    required = {"id", "slice", "text", "pred_sentiment", "pred_topics"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{YOUTUBE_REVIEWED_PATH.name} missing columns: {missing}")

    bad_sent = set(df["pred_sentiment"]) - YOUTUBE_SENTIMENTS
    if bad_sent:
        raise ValueError(f"unexpected gold_sentiment values: {bad_sent}")

    bad_slice = set(df["slice"]) - YOUTUBE_SLICES
    if bad_slice:
        raise ValueError(f"unexpected slice values: {bad_slice}")

    for i, raw in enumerate(df["pred_topics"]):
        try:
            topics = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"row {i} ({df.iloc[i]['id']}): pred_topics not JSON: {raw!r}") from e
        if not isinstance(topics, list):
            raise ValueError(f"row {i} ({df.iloc[i]['id']}): pred_topics is not a list")
        unknown = set(topics) - YOUTUBE_TOPICS
        if unknown:
            raise ValueError(f"row {i} ({df.iloc[i]['id']}): unknown topics: {unknown}")

    return df


def _assemble_youtube(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": df["id"].values,
            "text": df["text"].values,
            "gold_sentiment": df["pred_sentiment"].values,
            "slice": df["slice"].values,
            "split": "test",
            "source": YOUTUBE_SOURCE,
            "gold_topics": df["pred_topics"].values,
        }
    )


def build_youtube() -> None:
    """Load reviewed pre-labels and emit youtube.csv in the common schema."""
    if not YOUTUBE_REVIEWED_PATH.exists():
        print(f"\n[youtube] skipped — {YOUTUBE_REVIEWED_PATH.name} not found")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    reviewed = _load_youtube_reviewed()
    out = _assemble_youtube(reviewed)
    out.to_csv(YOUTUBE_OUT_PATH, index=False, encoding="utf-8")

    _report(out, "youtube.csv")
    print(f"\nwrote {YOUTUBE_OUT_PATH} ({len(out)} rows)")


def build() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sources = {
        "english": (_load_cardiff("english"), "cardiffnlp/tweet_sentiment_multilingual:english"),
        "arabic":  (_load_cardiff("arabic"),  "cardiffnlp/tweet_sentiment_multilingual:arabic"),
        "mixed":   (_load_eesa(),             "EESA"),
    }

    dev_parts, test_parts = [], []
    for slice_name, ((raw_dev, raw_test), source) in sources.items():
        print(f"\n[{slice_name}] raw dev={len(raw_dev)}  raw test={len(raw_test)}")
        dev = _clean_and_sample(raw_dev, DEV_N, slice_name, "dev")
        test = _clean_and_sample(raw_test, TEST_N, slice_name, "test")
        dev_parts.append(_assemble(dev, slice_name, "dev", source))
        test_parts.append(_assemble(test, slice_name, "test", source))

    dev_df = pd.concat(dev_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    dev_path = PROCESSED_DIR / "dev.csv"
    test_path = PROCESSED_DIR / "test.csv"
    dev_df.to_csv(dev_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    _report(dev_df, "dev.csv")
    _report(test_df, "test.csv")
    print(f"\nwrote {dev_path}  ({len(dev_df)} rows)")
    print(f"wrote {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    build()
    build_youtube()
