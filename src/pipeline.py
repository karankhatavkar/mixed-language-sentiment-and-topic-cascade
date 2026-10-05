"""Run the sentiment+topic pipeline on dev, test, or youtube.

Three setups per the spec:
  A - Small only (XLM-R + mDeBERTa, no LLM)
  B - LLM only (Gemini on every post)
  C - Cascade (A, then route hard posts to Gemini)

Output files:
  dev / test  ->  results/predictions_{A,B,C}.csv
  youtube     ->  results/predictions_youtube_{A,B,C}.csv  (keeps the
                  dev/test files intact so you can re-eval them)

Columns match the data contract in CLAUDE.md:
  id, slice, split, text, gold_sentiment,
  xlmr_label, xlmr_conf, topic_scores,
  routed, route_reason,
  final_sentiment, sarcasm, final_topics, llm_reason,
  setup
YouTube rows additionally carry `gold_topics` end-to-end for topic eval.

`topic_scores`, `final_topics`, and `gold_topics` are JSON strings in the
CSV. For Setup B the small-model columns (xlmr_label/xlmr_conf/
topic_scores) are empty.

Run:
  python -m src.pipeline --split dev --only A
  python -m src.pipeline --split test              # all three setups
  python -m src.pipeline --split youtube           # YouTube slice
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

from src.config import (
    MIXED_BOTH_CUTOFF,
    PROCESSED_DIR,
    RESULTS_DIR,
    SENTIMENT_CONF_CUTOFF,
    TOPIC_SCORE_CUTOFF,
)
from src.llm import classify_pipeline
from src.models import predict_sentiment, predict_topics


def _topics_above(scores: dict[str, float], cutoff: float) -> list[str]:
    kept = [t for t, s in scores.items() if s >= cutoff]
    return kept if kept else ["other"]


def _small_row(src, sent, topic) -> dict:
    """Row populated with small-model fields only (defaults for routing)."""
    scores = topic["scores"]
    row = {
        "id": src["id"],
        "slice": src["slice"],
        "split": src["split"],
        "text": src["text"],
        "gold_sentiment": src["gold_sentiment"],
        "xlmr_label": sent["label"],
        "xlmr_conf": sent["conf"],
        "topic_scores": json.dumps(scores),
        "routed": False,
        "route_reason": "",
        "final_sentiment": sent["label"],
        "sarcasm": False,
        "final_topics": json.dumps(_topics_above(scores, TOPIC_SCORE_CUTOFF)),
        "llm_reason": "",
    }
    if "gold_topics" in src:
        row["gold_topics"] = src["gold_topics"]
    return row


def _route_reasons(sent: dict, topic_scores: dict[str, float]) -> list[str]:
    """Return list of triggered routing rules, empty if the post stays small-only."""
    reasons = []
    if sent["conf"] < SENTIMENT_CONF_CUTOFF:
        reasons.append("low_conf")
    pos = sent["scores"]["positive"]
    neg = sent["scores"]["negative"]
    if pos >= MIXED_BOTH_CUTOFF and neg >= MIXED_BOTH_CUTOFF:
        reasons.append("likely_mixed")
    if max(topic_scores.values()) < TOPIC_SCORE_CUTOFF:
        reasons.append("no_topic")
    return reasons


# ----- setups ---------------------------------------------------------------

def run_setup_a(df: pd.DataFrame) -> pd.DataFrame:
    """Small-only: XLM-R + mDeBERTa, no LLM."""
    texts = df["text"].tolist()
    sent_preds = predict_sentiment(texts)
    topic_preds = predict_topics(texts)

    rows = []
    for (_, src), s, t in zip(df.iterrows(), sent_preds, topic_preds):
        row = _small_row(src, s, t)
        row["setup"] = "A"
        rows.append(row)
    return pd.DataFrame(rows)


def run_setup_b(df: pd.DataFrame) -> pd.DataFrame:
    """LLM-only: Gemini on every post. No small models."""
    rows = []
    for _, src in df.iterrows():
        out = classify_pipeline(src["id"], src["text"])
        row = {
            "id": src["id"],
            "slice": src["slice"],
            "split": src["split"],
            "text": src["text"],
            "gold_sentiment": src["gold_sentiment"],
            "xlmr_label": "",
            "xlmr_conf": "",
            "topic_scores": "",
            "routed": True,
            "route_reason": "setup_B",
            "final_sentiment": out["sentiment"],
            "sarcasm": out["sarcasm"],
            "final_topics": json.dumps(out["topics"]),
            "llm_reason": out["reason"],
            "setup": "B",
        }
        if "gold_topics" in src:
            row["gold_topics"] = src["gold_topics"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_setup_c(df: pd.DataFrame) -> pd.DataFrame:
    """Cascade: Setup A, then overwrite routed rows with Gemini's answer."""
    texts = df["text"].tolist()
    sent_preds = predict_sentiment(texts)
    topic_preds = predict_topics(texts)

    rows = []
    reason_counter: Counter[str] = Counter()
    for (_, src), s, t in zip(df.iterrows(), sent_preds, topic_preds):
        row = _small_row(src, s, t)
        reasons = _route_reasons(s, t["scores"])
        if reasons:
            llm = classify_pipeline(src["id"], src["text"])
            row["routed"] = True
            row["route_reason"] = ",".join(reasons)
            row["final_sentiment"] = llm["sentiment"]
            row["sarcasm"] = llm["sarcasm"]
            row["final_topics"] = json.dumps(llm["topics"])
            row["llm_reason"] = llm["reason"]
            for r in reasons:
                reason_counter[r] += 1
        row["setup"] = "C"
        rows.append(row)

    out = pd.DataFrame(rows)
    routed = int(out["routed"].sum())
    print(f"  routed: {routed}/{len(out)}  ({routed / len(out):.0%})")
    print(f"  reasons: {dict(reason_counter)}")
    return out


SETUPS = {"A": run_setup_a, "B": run_setup_b, "C": run_setup_c}


# ----- CLI ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test", "youtube"), required=True)
    ap.add_argument(
        "--only",
        choices=("A", "B", "C"),
        help="Run just one setup. Default: run all three.",
    )
    args = ap.parse_args()

    df = pd.read_csv(PROCESSED_DIR / f"{args.split}.csv")
    print(f"loaded {args.split}.csv  ({len(df)} rows)")
    print(f"cutoffs: sent_conf={SENTIMENT_CONF_CUTOFF}  "
          f"mixed_both={MIXED_BOTH_CUTOFF}  topic_score={TOPIC_SCORE_CUTOFF}")

    setups = (args.only,) if args.only else ("A", "B", "C")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep dev/test filenames stable; prefix youtube so its runs don't clobber.
    prefix = "youtube_" if args.split == "youtube" else ""
    for setup in setups:
        print(f"\n=== Setup {setup} ===")
        out = SETUPS[setup](df)
        path = RESULTS_DIR / f"predictions_{prefix}{setup}.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        print(f"wrote {path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
