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
