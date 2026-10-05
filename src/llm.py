"""Gemini wrapper + file cache for pipeline and judge calls.

Two public functions:
  classify_pipeline(post_id, text) -> {sentiment, sarcasm, topics, reason}
  classify_judge(post_id, text)    -> {topics}

Both cache on (model_id + prompt_version + post_id) in cache/llm_cache.json.
Reruns against a warm cache are free and bit-identical. The pipeline and
judge share one file but live in separate key namespaces because the model
id is part of the key.

Response shape is enforced server-side via `response_json_schema` (Pydantic
model's JSON Schema). Gemini cannot return a value that fails the schema, so
there is no defensive parsing — any ValidationError here means the SDK or
the API itself regressed and must be looked at.

System prompts live in prompts/*.txt — edit them there, bump PROMPT_VERSION
in config.py when you do so new rows get fresh cache keys.

Fails fast if GEMINI_API_KEY is missing — do not swallow it, do not fall
back to a 'skip LLM' mode (that would corrupt comparison across setups).
"""
from __future__ import annotations

import json
import os
from typing import Literal, get_args

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.config import (
    CACHE_DIR,
    GEMINI_JUDGE_MODEL,
    GEMINI_PIPELINE_MODEL,
    PROMPT_VERSION,
    PROMPTS_DIR,
    TOPICS,
)

CACHE_PATH = CACHE_DIR / "llm_cache.json"


# ----- schemas --------------------------------------------------------------

Topic = Literal[
    "delivery",
    "sizing",
    "quality",
    "price",
    "returns_refunds",
    "customer_service",
    "product_praise",
    "other",
]
# Fail at import if the Literal drifts from config.TOPICS.
assert set(get_args(Topic)) == set(TOPICS), (
    "Topic literal in llm.py has drifted from config.TOPICS — update both together"
)


class PipelineResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    sarcasm: bool
    topics: list[Topic]
    reason: str


class JudgeResponse(BaseModel):
    topics: list[Topic]


# ----- prompts --------------------------------------------------------------

_TOPICS_STR = ", ".join(TOPICS)
_PIPELINE_SYSTEM = (PROMPTS_DIR / "pipeline_system.txt").read_text(encoding="utf-8").format(topics=_TOPICS_STR)
_JUDGE_SYSTEM = (PROMPTS_DIR / "judge_system.txt").read_text(encoding="utf-8").format(topics=_TOPICS_STR)


# ----- client + cache -------------------------------------------------------

_client = None
_cache: dict | None = None


def _get_client():
    global _client
    if _client is None:
        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY missing. Create .env at project root with:\n"
                "  GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=key)
    return _client


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    return _cache


def _save_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_key(model: str, post_id: str) -> str:
    return f"{model}::{PROMPT_VERSION}::{post_id}"


# ----- call -----------------------------------------------------------------

def _call(
    model: str,
    system_instruction: str,
    user_text: str,
    schema: type[BaseModel],
    thinking_level: types.ThinkingLevel | None = None,
) -> dict:
    cfg_kwargs = dict(
        system_instruction=system_instruction,
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
    )
    # Gemini 2.5 Pro rejects any thinking_level; Gemini 3.x needs it.
    if thinking_level is not None:
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    resp = _get_client().models.generate_content(
        model=model,
        contents=user_text,
        config=types.GenerateContentConfig(**cfg_kwargs),
    )
    return schema.model_validate_json(resp.text).model_dump()


# ----- public API -----------------------------------------------------------

def classify_pipeline(post_id: str, text: str) -> dict:
    """Full 4-way sentiment + sarcasm + topics + reason (Flash-Lite)."""
    cache = _load_cache()
    key = _cache_key(GEMINI_PIPELINE_MODEL, post_id)
    if key in cache:
        return cache[key]
    result = _call(GEMINI_PIPELINE_MODEL, _PIPELINE_SYSTEM, text, PipelineResponse,
                   thinking_level=types.ThinkingLevel.MINIMAL)
    cache[key] = result
    _save_cache()
    return result


def classify_judge(post_id: str, text: str) -> dict:
    """Topics only (Gemini Pro judge).

    No thinking_level is passed — 2.5 Pro rejects the field entirely; the
    model falls back to its default internal thinking budget.
    """
    cache = _load_cache()
    key = _cache_key(GEMINI_JUDGE_MODEL, post_id)
    if key in cache:
        return cache[key]
    result = _call(GEMINI_JUDGE_MODEL, _JUDGE_SYSTEM, text, JudgeResponse)
    cache[key] = result
    _save_cache()
    return result


def classify_full_pro(post_id: str, text: str) -> dict:
    """Full 4-way sentiment + sarcasm + topics + reason using the Pro model.

    Used to pre-label the YouTube pool (sentiment + topics in one call,
    higher-quality than Flash-Lite). Cached under a 'yt::' namespace so
    it never collides with classify_judge, which uses the same model but
    emits a topics-only schema on potentially overlapping post_ids.
    """
    cache = _load_cache()
    key = f"yt::{_cache_key(GEMINI_JUDGE_MODEL, post_id)}"
    if key in cache:
        return cache[key]
    result = _call(GEMINI_JUDGE_MODEL, _PIPELINE_SYSTEM, text, PipelineResponse,
                   thinking_level=types.ThinkingLevel.LOW)
    cache[key] = result
    _save_cache()
    return result


# ----- smoke test -----------------------------------------------------------

def _smoke_test() -> None:
    import pandas as pd
    from src.config import PROCESSED_DIR

    df = pd.read_csv(PROCESSED_DIR / "dev.csv").groupby("slice").head(1)
    print("=== pipeline model (Flash-Lite) ===")
    for _, row in df.iterrows():
        out = classify_pipeline(row["id"], row["text"])
        print(f"[{row['slice']:7}] gold={row['gold_sentiment']:8}  "
              f"pred={out['sentiment']:8}  sarcasm={out['sarcasm']}  "
              f"topics={out['topics']}")
        print(f"           reason: {out['reason']}")
        print(f"           text:   {row['text'][:90]}")


if __name__ == "__main__":
    _smoke_test()
