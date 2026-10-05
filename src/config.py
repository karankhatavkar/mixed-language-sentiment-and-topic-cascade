"""Single source of truth for pipeline settings.

Grows as later scripts (models, llm, pipeline, evaluate) land. For now it
only holds what load_data.py needs.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "cache"
PROMPTS_DIR = ROOT / "prompts"

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

# Routing cutoffs (sweep on dev; set final values here before `--split test`)
SENTIMENT_CONF_CUTOFF = 0.7   # rule 1: route if xlmr_conf below this
MIXED_BOTH_CUTOFF = 0.3       # rule 2: route if pos >= X AND neg >= X
TOPIC_SCORE_CUTOFF = 0.5      # rule 3: route if no topic score >= X
                              # Also used in Setup A to pick final_topics.

# Gemini
GEMINI_PIPELINE_MODEL = "gemini-3.5-flash-lite"
GEMINI_JUDGE_MODEL = "gemini-2.5-pro"  # spec says 3.1-pro-preview; swapped due to free-tier 250/day quota
PROMPT_VERSION = "v2"  # v2: dropped the fashion-brand framing so public-dataset posts aren't shoehorned

# Approx $/call for a 400-in/50-out post, from the pricing in the Part B spec.
# Flash-Lite:  $0.30/M in + $2.50/M out  -> ~0.00012 + ~0.000125 = ~0.00025
# Pro:         $2.00/M in + $12.00/M out -> ~0.00080 + ~0.000600 = ~0.00140
COST_PER_CALL_PIPELINE = 0.00025
COST_PER_CALL_JUDGE = 0.00140
