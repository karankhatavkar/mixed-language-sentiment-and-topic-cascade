"""Score predictions_{A,B,C}.csv (dev/test) or predictions_youtube_{A,B,C}.csv.

Sentiment metrics:
  macro-F1 (headline), negative precision/recall, confusion matrix.
  - dev/test: gold has 3 labels (positive, negative, neutral); macro-F1 is
    over those 3. A `mixed` prediction counts as wrong but stays visible as
    its own column in the confusion matrix.
  - youtube: gold has 4 labels (adds `mixed`); macro-F1 is over all 4.

Topic metrics:
  Scored against gold topics. Precision/recall/F1 per topic + macro-F1.
  - dev/test: reads results/topic_judge.csv when present (Pro-judge-labelled).
    Skipped silently if the judge CSV is absent.
  - youtube: gold_topics is already in the predictions CSV (from the human
    review); no judge CSV is needed.

LLM call rate = fraction of rows where `routed` is true.
Cost / 1k posts = call_rate * 1000 * COST_PER_CALL_PIPELINE.

Run:
  python -m src.evaluate --split dev
  python -m src.evaluate --split test
  python -m src.evaluate --split youtube
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.config import (
    COST_PER_CALL_PIPELINE,
    RESULTS_DIR,
    TOPICS,
)

SETUPS = ("A", "B", "C")
SLICES = ("english", "arabic", "mixed")
GOLD_LABELS_PUBLIC = ["negative", "neutral", "positive"]            # dev/test
GOLD_LABELS_YOUTUBE = ["negative", "neutral", "positive", "mixed"]  # youtube
PRED_LABELS = ["negative", "neutral", "positive", "mixed"]          # confusion cols


# ----- sentiment metrics ----------------------------------------------------

def _scope_metrics(gold: pd.Series, pred: pd.Series, gold_labels: list[str]) -> dict:
    return {
        "n": len(gold),
        "accuracy": float((pred == gold).mean()) if len(gold) else 0.0,
        "macro_f1": float(f1_score(gold, pred, labels=gold_labels, average="macro", zero_division=0)),
        "neg_precision": float(precision_score(gold, pred, labels=["negative"], average="macro", zero_division=0)),
        "neg_recall": float(recall_score(gold, pred, labels=["negative"], average="macro", zero_division=0)),
    }


def _per_slice_table(df: pd.DataFrame, gold_labels: list[str]) -> pd.DataFrame:
    rows = []
    for slice_name in SLICES:
        sub = df[df["slice"] == slice_name]
        rows.append({"scope": slice_name,
                     **_scope_metrics(sub["gold_sentiment"], sub["final_sentiment"], gold_labels)})
    rows.append({"scope": "overall",
                 **_scope_metrics(df["gold_sentiment"], df["final_sentiment"], gold_labels)})
    return pd.DataFrame(rows).set_index("scope")


def _confusion(df: pd.DataFrame, gold_labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(df["gold_sentiment"], df["final_sentiment"], labels=PRED_LABELS)
    out = pd.DataFrame(cm, index=PRED_LABELS, columns=PRED_LABELS)
    return out.loc[gold_labels]  # keep only valid-gold rows


# ----- topic metrics (optional, needs judge CSV) ---------------------------

def _topic_table(preds: pd.DataFrame, judge: pd.DataFrame) -> pd.DataFrame:
    merged = preds[["id", "final_topics"]].merge(judge[["id", "judge_topics"]], on="id", how="inner")
    pred_sets = merged["final_topics"].apply(lambda s: set(json.loads(s)))
    gold_sets = merged["judge_topics"].apply(lambda s: set(json.loads(s)))

    rows = []
    for t in TOPICS:
        gold_has = gold_sets.apply(lambda s: t in s)
        pred_has = pred_sets.apply(lambda s: t in s)
        tp = int((gold_has & pred_has).sum())
        fp = int((~gold_has & pred_has).sum())
        fn = int((gold_has & ~pred_has).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"topic": t, "P": p, "R": r, "F1": f1, "support_gold": int(gold_has.sum())})

    df = pd.DataFrame(rows).set_index("topic")
    df.loc["macro"] = [df["P"].mean(), df["R"].mean(), df["F1"].mean(), df["support_gold"].sum()]
    return df


# ----- routing / cost -------------------------------------------------------

def _llm_stats(df: pd.DataFrame) -> tuple[float, float]:
    rate = float(df["routed"].astype(bool).mean())
    cost_per_1k = rate * 1000 * COST_PER_CALL_PIPELINE
    return rate, cost_per_1k


# ----- report writer --------------------------------------------------------

def _load_predictions(split: str) -> dict[str, pd.DataFrame]:
    prefix = "youtube_" if split == "youtube" else ""
    out = {}
    for setup in SETUPS:
        path = RESULTS_DIR / f"predictions_{prefix}{setup}.csv"
        if path.exists():
            out[setup] = pd.read_csv(path)
    if not out:
        raise FileNotFoundError(
            f"No predictions_{prefix}{{A,B,C}}.csv found in results/. "
            f"Run `python -m src.pipeline --split {split}` first."
        )
    return out


def _load_judge(split: str, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """For dev/test, read the Pro-judge CSV if it exists.

    For youtube, the gold topics are already in each predictions file (from the
    human-reviewed labels in youtube.csv, carried through by pipeline.py). We
    pull `gold_topics` from the first available predictions frame and alias it
    to `judge_topics` so _topic_table can be reused unchanged.
    """
    if split == "youtube":
        first = next(iter(predictions.values()))
        if "gold_topics" not in first.columns:
            return None
        return first[["id", "gold_topics"]].rename(columns={"gold_topics": "judge_topics"})
    path = RESULTS_DIR / "topic_judge.csv"
    return pd.read_csv(path) if path.exists() else None


def _write_report(predictions: dict[str, pd.DataFrame], judge: pd.DataFrame | None, split: str) -> None:
    first = next(iter(predictions.values()))
    split_label = first["split"].iloc[0]
    gold_labels = GOLD_LABELS_YOUTUBE if split == "youtube" else GOLD_LABELS_PUBLIC

    lines: list[str] = []
    lines.append("# Pipeline Evaluation")
    lines.append("")
    lines.append(f"Date: {date.today().isoformat()}")
    lines.append(f"Split: **{split_label}** ({len(first)} rows)")
    if judge is None:
        lines.append("")
        lines.append("_Topic metrics skipped — no gold topics available._")
    lines.append("")

    # Headline table
    lines.append("## Headline")
    lines.append("")
    lines.append("| Setup | EN F1 | AR F1 | Mixed F1 | Overall F1 | Neg Recall | Topic F1 | LLM Calls | Cost / 1k |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for setup in SETUPS:
        if setup not in predictions:
            continue
        df = predictions[setup]
        m = _per_slice_table(df, gold_labels)
        rate, cost = _llm_stats(df)
        topic_f1 = "—"
        if judge is not None:
            tm = _topic_table(df, judge)
            topic_f1 = f"{tm.loc['macro', 'F1']:.3f}"
        lines.append(
            f"| {setup} | "
            f"{m.loc['english', 'macro_f1']:.3f} | "
            f"{m.loc['arabic', 'macro_f1']:.3f} | "
            f"{m.loc['mixed', 'macro_f1']:.3f} | "
            f"{m.loc['overall', 'macro_f1']:.3f} | "
            f"{m.loc['overall', 'neg_recall']:.3f} | "
            f"{topic_f1} | "
            f"{rate:.0%} | "
            f"${cost:.3f} |"
        )
    lines.append("")

    # Per-setup detail
    for setup in SETUPS:
        if setup not in predictions:
            continue
        df = predictions[setup]
        m = _per_slice_table(df, gold_labels).round(3)
        rate, cost = _llm_stats(df)
        lines.append(f"## Setup {setup}")
        lines.append("")
        lines.append(f"LLM call rate: **{rate:.1%}**  ·  Cost / 1k posts: **${cost:.3f}**")
        lines.append("")
        lines.append("### Sentiment metrics")
        lines.append("")
        lines.append(m.to_markdown())
        lines.append("")
        lines.append("### Confusion (gold rows × pred cols; `mixed` col kept)")
        lines.append("")
        for slice_name in SLICES:
            sub = df[df["slice"] == slice_name]
            lines.append(f"**{slice_name}**")
            lines.append("")
            lines.append(_confusion(sub, gold_labels).to_markdown())
            lines.append("")
        if judge is not None:
            lines.append("### Topic metrics")
            lines.append("")
            lines.append(_topic_table(df, judge).round(3).to_markdown())
            lines.append("")

    out_name = "metrics_youtube.md" if split == "youtube" else "metrics.md"
    out_path = RESULTS_DIR / out_name
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test", "youtube"), default="dev")
    args = ap.parse_args()

    predictions = _load_predictions(args.split)
    judge = _load_judge(args.split, predictions)
    _write_report(predictions, judge, args.split)


if __name__ == "__main__":
    main()
