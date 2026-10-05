"""Single source of truth for pipeline settings.

Grows as later scripts (models, llm, pipeline, evaluate) land. For now it
only holds what load_data.py needs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
DEV_N = 50
TEST_N = 100

SENTIMENT_LABELS = ("positive", "negative", "neutral")

# Models
XLMR_MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
XLMR_BATCH_SIZE = 16

MDEBERTA_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
MDEBERTA_BATCH_SIZE = 16

# Topic list (same list is used by mDeBERTa, Gemini pipeline, and Gemini judge)
TOPICS = (
    "delivery",
    "sizing",
    "quality",
    "price",
    "returns_refunds",
    "customer_service",
    "product_praise",
    "other",
)
TOPIC_HYPOTHESIS_TEMPLATE = "This post is about {}."
