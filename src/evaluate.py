"""Score predictions_{A,B,C}.csv, write results/metrics.md.

Sentiment metrics (gold has 3 labels, pipeline may predict 4):
  macro-F1 (headline), negative precision/recall, confusion matrix with a
  'mixed' column kept so wrong-but-interesting predictions stay visible.
  Macro-F1 is computed over the 3 gold labels; a 'mixed' prediction counts
  as wrong but does not inflate the denominator.

Topic metrics (optional):
  Scored against results/topic_judge.csv when present. Precision/recall/F1
  per topic + macro-F1. Skipped silently if the judge CSV is absent.

LLM call rate = fraction of rows where `routed` is true.
Cost / 1k posts = call_rate * 1000 * COST_PER_CALL_PIPELINE.

Run: python -m src.evaluate
"""
from __future__ import annotations

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
GOLD_LABELS = ["negative", "neutral", "positive"]           # confusion rows
PRED_LABELS = ["negative", "neutral", "positive", "mixed"]  # confusion cols


# ----- sentiment metrics ----------------------------------------------------

def _scope_metrics(gold: pd.Series, pred: pd.Series) -> dict:
    return {
        "n": len(gold),
        "accuracy": float((pred == gold).mean()) if len(gold) else 0.0,
        "macro_f1": float(f1_score(gold, pred, labels=GOLD_LABELS, average="macro", zero_division=0)),
        "neg_precision": float(precision_score(gold, pred, labels=["negative"], average="macro", zero_division=0)),
        "neg_recall": float(recall_score(gold, pred, labels=["negative"], average="macro", zero_division=0)),
    }


def _per_slice_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slice_name in SLICES:
        sub = df[df["slice"] == slice_name]
        rows.append({"scope": slice_name, **_scope_metrics(sub["gold_sentiment"], sub["final_sentiment"])})
    rows.append({"scope": "overall", **_scope_metrics(df["gold_sentiment"], df["final_sentiment"])})
    return pd.DataFrame(rows).set_index("scope")


def _confusion(df: pd.DataFrame) -> pd.DataFrame:
    cm = confusion_matrix(df["gold_sentiment"], df["final_sentiment"], labels=PRED_LABELS)
    out = pd.DataFrame(cm, index=PRED_LABELS, columns=PRED_LABELS)
    return out.loc[GOLD_LABELS]  # keep only valid-gold rows


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

def _load_predictions() -> dict[str, pd.DataFrame]:
    out = {}
    for setup in SETUPS:
        path = RESULTS_DIR / f"predictions_{setup}.csv"
        if path.exists():
            out[setup] = pd.read_csv(path)
    if not out:
        raise FileNotFoundError("No predictions_{A,B,C}.csv found in results/. Run src.pipeline first.")
    return out


def _load_judge() -> pd.DataFrame | None:
    path = RESULTS_DIR / "topic_judge.csv"
    return pd.read_csv(path) if path.exists() else None


def _write_report(predictions: dict[str, pd.DataFrame], judge: pd.DataFrame | None) -> None:
    first = next(iter(predictions.values()))
    split = first["split"].iloc[0]

    lines: list[str] = []
    lines.append("# Pipeline Evaluation")
    lines.append("")
    lines.append(f"Date: {date.today().isoformat()}")
    lines.append(f"Split: **{split}** ({len(first)} rows)")
    if judge is None:
        lines.append("")
        lines.append("_Topic metrics skipped — `results/topic_judge.csv` not found._")
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
        m = _per_slice_table(df)
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
        m = _per_slice_table(df).round(3)
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
            lines.append(_confusion(sub).to_markdown())
            lines.append("")
        if judge is not None:
            lines.append("### Topic metrics")
            lines.append("")
            lines.append(_topic_table(df, judge).round(3).to_markdown())
            lines.append("")

    out_path = RESULTS_DIR / "metrics.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def main() -> None:
    predictions = _load_predictions()
    judge = _load_judge()
    _write_report(predictions, judge)


if __name__ == "__main__":
    main()
