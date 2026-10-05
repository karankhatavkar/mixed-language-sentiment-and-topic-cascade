"""Pre-label the YouTube pool with Gemini Flash-Lite.

Reads  data/processed/youtube_pool.csv
Writes data/processed/youtube_prelabeled.csv

Each row gets:
  pred_sentiment   positive | negative | neutral | mixed  (4-way; YouTube is
                   the only slice where `mixed` and sarcasm are scored proper)
  pred_sarcasm     bool
  pred_topics      JSON list, multi-label from the fixed 8 topics
  pred_reason      one-line model justification

Uses classify_pipeline (Flash-Lite) rather than Pro: the free-tier Pro quota
is 250 req/day which can't cover the 234-row pool in one go, and every row
is going to human review anyway — the pre-fill quality matters less than
getting unblocked. Cached in cache/llm_cache.json keyed on
`GEMINI_PIPELINE_MODEL::PROMPT_VERSION::post_id`, so when src.pipeline later
runs Setup B or C on these same youtube ids it reuses the cached call for
free (same model, same inputs, same answer).

The output is NOT gold — it is the model's pre-labels. The next step is
human review (see docs/youtube_preprocessing.md) which produces the final
youtube_labeled.csv with gold_sentiment / gold_topics columns.

Cost (fresh run): ~$0.06 for the full 234-row pool at Flash-Lite pricing.
Cached reruns are free.

Run: python -m src.label_youtube
"""
from __future__ import annotations

import csv
import json
import time

from src.config import PROCESSED_DIR
from src.llm import classify_pipeline

POOL_CSV = PROCESSED_DIR / "youtube_pool.csv"
OUT_CSV = PROCESSED_DIR / "youtube_prelabeled.csv"

OUT_COLS = [
    "id", "slice", "video_id", "comment_id", "text",
    "like_count", "reply_count", "published_at",
    "pred_sentiment", "pred_sarcasm", "pred_topics", "pred_reason",
]


def label() -> None:
    with POOL_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"loaded {len(rows)} rows from {POOL_CSV.name}")

    out_rows = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        pred = classify_pipeline(r["id"], r["text"])
        out_rows.append({
            "id": r["id"],
            "slice": r["slice"],
            "video_id": r["video_id"],
            "comment_id": r["comment_id"],
            "text": r["text"],
            "like_count": r["like_count"],
            "reply_count": r["reply_count"],
            "published_at": r["published_at"],
            "pred_sentiment": pred["sentiment"],
            "pred_sarcasm": pred["sarcasm"],
            "pred_topics": json.dumps(pred["topics"]),
            "pred_reason": pred["reason"],
        })
        if i % 10 == 0 or i == len(rows):
            print(f"  {i:>3}/{len(rows)}  ({time.time() - t0:.1f}s elapsed)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {OUT_CSV}  ({len(out_rows)} rows)")

    from collections import Counter
    sc = Counter(r["pred_sentiment"] for r in out_rows)
    sarc = sum(1 for r in out_rows if r["pred_sarcasm"] in (True, "True", "true"))
    print(f"sentiment: {dict(sc)}")
    print(f"sarcasm:   {sarc} / {len(out_rows)}")


if __name__ == "__main__":
    label()
