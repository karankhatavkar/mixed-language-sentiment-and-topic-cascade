from src import config


def test_topics_are_the_eight_from_the_spec():
    assert len(config.TOPICS) == 8
    assert len(set(config.TOPICS)) == 8
    assert "other" in config.TOPICS


def test_sentiment_labels_are_three_way():
    assert config.SENTIMENT_LABELS == ("positive", "negative", "neutral")


def test_cutoffs_are_in_unit_interval():
    for name in ("SENTIMENT_CONF_CUTOFF", "MIXED_BOTH_CUTOFF", "TOPIC_SCORE_CUTOFF"):
        v = getattr(config, name)
        assert 0.0 < v < 1.0, f"{name}={v} not in (0, 1)"


def test_split_sizes_and_seed():
    assert config.SEED == 42
    assert config.DEV_N == 50
    assert config.TEST_N == 100
