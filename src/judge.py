"""Produce topic gold labels for the test posts using Gemini 3.1 Pro.

Reads data/processed/{split}.csv, calls `classify_judge` for every row,
writes results/topic_judge.csv with columns:
  id, text, judge_topics

`judge_topics` is a JSON-serialised list (same convention as `final_topics`
in predictions_*.csv), so `evaluate.py` can `json.loads()` and compare.

The actual Gemini call lives in src.llm; this is just a thin runner that
materialises the labels as a reviewable artifact. All calls are cached in
cache/llm_cache.json — reruns are free.

Run:
  python -m src.judge                 # default: test
  python -m src.judge --split dev     # only if you want dev topics too
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import COST_PER_CALL_JUDGE, PROCESSED_DIR, RESULTS_DIR
from src.llm import classify_judge


def build(split: str) -> None:
    df = pd.read_csv(PROCESSED_DIR / f"{split}.csv")
    print(f"loaded {split}.csv  ({len(df)} rows)")

    rows = []
    for i, (_, src) in enumerate(df.iterrows(), start=1):
        out = classify_judge(src["id"], src["text"])
        rows.append({"id": src["id"], "text": src["text"], "judge_topics": json.dumps(out["topics"])})
        if i % 25 == 0 or i == len(df):
            print(f"  {i}/{len(df)}")

    out_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "topic_judge.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nwrote {out_path}  ({len(out_df)} rows)")
    print(f"max cost of this run (if nothing was cached): ~${len(df) * COST_PER_CALL_JUDGE:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    args = ap.parse_args()
    build(args.split)


if __name__ == "__main__":
    main()
