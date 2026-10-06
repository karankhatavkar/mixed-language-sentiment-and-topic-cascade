from src.config import (
    MIXED_BOTH_CUTOFF,
    SENTIMENT_CONF_CUTOFF,
    TOPIC_SCORE_CUTOFF,
)
from src.pipeline import _route_reasons, _topics_above


def test_topics_above_keeps_scores_at_or_over_cutoff():
    scores = {"delivery": 0.9, "sizing": 0.4, "quality": 0.6}
    assert set(_topics_above(scores, 0.5)) == {"delivery", "quality"}


def test_topics_above_falls_back_to_other_when_nothing_passes():
    scores = {"delivery": 0.1, "sizing": 0.2}
    assert _topics_above(scores, 0.5) == ["other"]


def _sent(label, conf, scores):
    return {"label": label, "conf": conf, "scores": scores}


def test_route_reasons_empty_on_confident_single_sided_post():
    sent = _sent("positive", 0.95, {"positive": 0.95, "negative": 0.02, "neutral": 0.03})
    topics = {"delivery": 0.9, "sizing": 0.1}
    assert _route_reasons(sent, topics) == []


def test_route_reasons_low_conf_triggers():
    conf = SENTIMENT_CONF_CUTOFF - 0.1
    sent = _sent("positive", conf, {"positive": conf, "negative": 0.1, "neutral": 0.1})
    topics = {"delivery": 0.9}
    assert "low_conf" in _route_reasons(sent, topics)


def test_route_reasons_likely_mixed_triggers():
    pos = neg = MIXED_BOTH_CUTOFF + 0.05
    sent = _sent("positive", 0.9, {"positive": pos, "negative": neg, "neutral": 0.1})
    topics = {"delivery": 0.9}
    assert "likely_mixed" in _route_reasons(sent, topics)


def test_route_reasons_no_topic_triggers():
    sent = _sent("positive", 0.95, {"positive": 0.95, "negative": 0.02, "neutral": 0.03})
    topics = {t: TOPIC_SCORE_CUTOFF - 0.1 for t in ("delivery", "sizing", "other")}
    assert "no_topic" in _route_reasons(sent, topics)
