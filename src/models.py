"""Free HuggingFace models used by the pipeline.

Two models live here:
- XLM-R sentiment  -> `predict_sentiment(texts)`
- mDeBERTa topics  -> `predict_topics(texts)`  (not yet implemented)

Both run on CPU, batched. No disk cache — reruns re-infer. Loaders are
module-level singletons so the model is warm after the first call.

Smoke test: python -m src.models
"""
from __future__ import annotations

from transformers import pipeline

from src.config import (
    MDEBERTA_BATCH_SIZE,
    MDEBERTA_MODEL_ID,
    TOPIC_HYPOTHESIS_TEMPLATE,
    TOPICS,
    XLMR_BATCH_SIZE,
    XLMR_MODEL_ID,
)

MAX_LEN = 512  # XLM-R and mDeBERTa both cap at 512 tokens

_xlmr_pipe = None
_mdeberta_pipe = None


# ----- XLM-R sentiment ------------------------------------------------------

def _get_xlmr():
    global _xlmr_pipe
    if _xlmr_pipe is None:
        _xlmr_pipe = pipeline(
            task="sentiment-analysis",
            model=XLMR_MODEL_ID,
            device=-1,     # CPU
            top_k=None,    # return all 3 class probabilities
        )
    return _xlmr_pipe


def predict_sentiment(texts: list[str], batch_size: int = XLMR_BATCH_SIZE) -> list[dict]:
    """XLM-R 3-way sentiment.

    Returns one dict per input text:
      {
        "label": "positive" | "negative" | "neutral",   # argmax, lowercased
        "conf":  float,                                  # top probability
        "scores": {"positive": float, "negative": float, "neutral": float},
      }
    """
    pipe = _get_xlmr()
    raw = pipe(texts, batch_size=batch_size, truncation=True, max_length=MAX_LEN)

    out = []
    for item in raw:
        scores = {d["label"].lower(): float(d["score"]) for d in item}
        if set(scores) != {"positive", "negative", "neutral"}:
            raise ValueError(f"unexpected XLM-R labels: {set(scores)}")
        label = max(scores, key=scores.get)
        out.append({"label": label, "conf": scores[label], "scores": scores})
    return out


# ----- mDeBERTa zero-shot topics -------------------------------------------

def _get_mdeberta():
    global _mdeberta_pipe
    if _mdeberta_pipe is None:
        _mdeberta_pipe = pipeline(
            task="zero-shot-classification",
            model=MDEBERTA_MODEL_ID,
            device=-1,     # CPU
        )
    return _mdeberta_pipe


def predict_topics(texts: list[str], batch_size: int = MDEBERTA_BATCH_SIZE) -> list[dict]:
    """mDeBERTa zero-shot topic scoring over the fixed 8-topic list.

    `multi_label=True` so each topic gets an independent sigmoid score; the
    scores do not sum to 1. One post can match several topics.

    Returns one dict per input text:
      {
        "scores":    {topic: float for all 8 topics},   # unsorted, keyed by topic
        "top_topic": str,
        "top_score": float,
      }
    """
    pipe = _get_mdeberta()
    raw = pipe(
        texts,
        candidate_labels=list(TOPICS),
        hypothesis_template=TOPIC_HYPOTHESIS_TEMPLATE,
        multi_label=True,
        batch_size=batch_size,
        truncation=True,
        max_length=MAX_LEN,
    )
    # Pipeline returns a dict for a single input; always normalise to list.
    if isinstance(raw, dict):
        raw = [raw]

    out = []
    for item in raw:
        scores = {label: float(score) for label, score in zip(item["labels"], item["scores"])}
        if set(scores) != set(TOPICS):
            raise ValueError(f"unexpected topics: {set(scores)}")
        top_topic = max(scores, key=scores.get)
        out.append({"scores": scores, "top_topic": top_topic, "top_score": scores[top_topic]})
    return out


# ----- smoke test -----------------------------------------------------------

def _smoke_test() -> None:
    import pandas as pd
    from src.config import PROCESSED_DIR

    df = pd.read_csv(PROCESSED_DIR / "dev.csv").groupby("slice").head(2)
    texts = df["text"].tolist()

    print("=== XLM-R sentiment ===")
    for (_, row), pred in zip(df.iterrows(), predict_sentiment(texts)):
        scores = ", ".join(f"{k}={v:.2f}" for k, v in pred["scores"].items())
        print(f"[{row['slice']:7}] gold={row['gold_sentiment']:8} "
              f"pred={pred['label']:8} conf={pred['conf']:.2f}   {scores}")
        print(f"           text: {row['text'][:90]}")

    print("\n=== mDeBERTa topics (top-3 per post) ===")
    for (_, row), pred in zip(df.iterrows(), predict_topics(texts)):
        top3 = sorted(pred["scores"].items(), key=lambda x: x[1], reverse=True)[:3]
        pretty = ", ".join(f"{t}={s:.2f}" for t, s in top3)
        print(f"[{row['slice']:7}] top={pred['top_topic']:16} score={pred['top_score']:.2f}   {pretty}")
        print(f"           text: {row['text'][:90]}")


if __name__ == "__main__":
    _smoke_test()
