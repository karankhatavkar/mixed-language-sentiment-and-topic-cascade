"""Boundary tests for src.llm — no live Gemini calls."""
import os
from typing import get_args

import pytest

import src.llm as llm
from src.config import PROMPT_VERSION, TOPICS


def test_topic_literal_matches_config_topics():
    assert set(get_args(llm.Topic)) == set(TOPICS)


def test_pipeline_response_schema_has_required_fields():
    fields = llm.PipelineResponse.model_fields
    assert set(fields) == {"sentiment", "sarcasm", "topics", "reason"}


def test_cache_key_shape():
    key = llm._cache_key("gemini-x", "post-123")
    assert key == f"gemini-x::{PROMPT_VERSION}::post-123"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm._get_client()
