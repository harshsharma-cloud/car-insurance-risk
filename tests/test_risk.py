"""Tests for risk level assignment and verdict generation (src/risk.py)."""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk import (
    assign_risk_level, build_verdict, get_threshold_config,
    get_focus_areas, LOW_THRESHOLD, HIGH_THRESHOLD,
)


# ── assign_risk_level() ───────────────────────────────────────────────────────

def test_low_risk_label():
    """A probability well below LOW_THRESHOLD should return LOW."""
    assert assign_risk_level(0.0) == "LOW"
    assert assign_risk_level(0.01) == "LOW"
    assert assign_risk_level(LOW_THRESHOLD - 0.001) == "LOW"


def test_high_risk_label():
    """A probability at or above HIGH_THRESHOLD should return HIGH."""
    assert assign_risk_level(1.0) == "HIGH"
    assert assign_risk_level(HIGH_THRESHOLD) == "HIGH"
    assert assign_risk_level(HIGH_THRESHOLD + 0.001) == "HIGH"


def test_medium_risk_label():
    """A probability in [LOW_THRESHOLD, HIGH_THRESHOLD) should return MEDIUM."""
    mid = (LOW_THRESHOLD + HIGH_THRESHOLD) / 2
    assert assign_risk_level(LOW_THRESHOLD) == "MEDIUM"
    assert assign_risk_level(mid) == "MEDIUM"
    assert assign_risk_level(HIGH_THRESHOLD - 0.001) == "MEDIUM"


def test_invalid_probability_raises():
    """A probability outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError):
        assign_risk_level(-0.01)
    with pytest.raises(ValueError):
        assign_risk_level(1.001)


def test_risk_level_exhaustive_coverage():
    """Every probability in [0, 1] must map to one of the three labels."""
    import numpy as np
    valid_labels = {"LOW", "MEDIUM", "HIGH"}
    for p in np.linspace(0.0, 1.0, 201):
        label = assign_risk_level(float(p))
        assert label in valid_labels, f"Unexpected label {label!r} for p={p}"


# ── build_verdict() ───────────────────────────────────────────────────────────

def test_build_verdict_keys():
    """build_verdict() must return all four required keys."""
    verdict = build_verdict(0.5)
    for key in ["claim_probability", "claim_probability_pct", "risk_level", "verdict_text"]:
        assert key in verdict, f"Missing key: {key}"


def test_verdict_pct_format():
    """claim_probability_pct must be a string ending in '%'."""
    verdict = build_verdict(0.723)
    assert isinstance(verdict["claim_probability_pct"], str)
    assert verdict["claim_probability_pct"].endswith("%")


def test_verdict_probability_rounded():
    """claim_probability should be rounded to 4 decimal places."""
    verdict = build_verdict(0.123456789)
    assert verdict["claim_probability"] == 0.1235


def test_verdict_risk_level_matches_assign():
    """risk_level in verdict must match assign_risk_level() output."""
    for prob in [0.05, 0.30, 0.80]:
        verdict = build_verdict(prob)
        assert verdict["risk_level"] == assign_risk_level(prob)


def test_verdict_text_low():
    """LOW risk verdict text must not contain causal language."""
    verdict = build_verdict(0.01)
    assert verdict["risk_level"] == "LOW"
    text = verdict["verdict_text"].lower()
    assert "causes" not in text
    assert "guarantee" not in text
    assert "low" in text


def test_verdict_text_high():
    """HIGH risk verdict text must mention HIGH."""
    verdict = build_verdict(0.99)
    assert verdict["risk_level"] == "HIGH"
    assert "high" in verdict["verdict_text"].lower()


# ── get_threshold_config() ────────────────────────────────────────────────────

def test_get_threshold_config_keys():
    """get_threshold_config() must return low_threshold and high_threshold."""
    config = get_threshold_config()
    assert "low_threshold" in config
    assert "high_threshold" in config


def test_get_threshold_config_ordering():
    """low_threshold must be strictly less than high_threshold."""
    config = get_threshold_config()
    assert config["low_threshold"] < config["high_threshold"]


def test_get_threshold_config_valid_range():
    """Both thresholds must be in (0, 1)."""
    config = get_threshold_config()
    assert 0.0 < config["low_threshold"] < 1.0
    assert 0.0 < config["high_threshold"] < 1.0


# ── get_focus_areas() ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def focus_area_df():
    """Minimal DataFrame for focus area testing."""
    import pandas as pd
    import numpy as np
    rng = np.random.default_rng(42)
    n = 500
    # experience: 0-9y has higher claim rate (0.7), others lower (0.2)
    exp = rng.choice(["0-9y", "30y+"], size=n, p=[0.4, 0.6])
    claim = (exp == "0-9y") * rng.binomial(1, 0.7, n) + \
            (exp == "30y+") * rng.binomial(1, 0.2, n)
    prob = claim * 0.8 + (1 - claim) * 0.1
    pred = (prob >= 0.5).astype(int)
    return pd.DataFrame({
        "driving_experience": exp,
        "y_true": claim,
        "y_pred": pred,
        "y_prob": prob,
    })


def test_get_focus_areas_returns_list(focus_area_df):
    """get_focus_areas() must return a list."""
    result = get_focus_areas(focus_area_df, top_n=2)
    assert isinstance(result, list)


def test_get_focus_areas_required_keys(focus_area_df):
    """Each focus area must contain all required keys."""
    required = {
        "segment_feature", "segment_value", "claim_rate",
        "overall_claim_rate", "claim_rate_lift", "segment_size_pct",
        "segment_n", "fn_count", "fn_rate", "avg_pred_prob",
        "score", "headline", "why",
    }
    result = get_focus_areas(focus_area_df, top_n=2)
    for area in result:
        for key in required:
            assert key in area, f"Missing key {key!r} in focus area"


def test_get_focus_areas_headline_format(focus_area_df):
    """Headline must start with Focus area: and not contain causal language."""
    result = get_focus_areas(focus_area_df, top_n=2)
    for area in result:
        assert area["headline"].startswith("Focus area:")
        assert "causes" not in area["headline"].lower()
        assert "causes" not in area["why"].lower()


def test_get_focus_areas_top_n_respected(focus_area_df):
    """get_focus_areas() must return at most top_n results."""
    result = get_focus_areas(focus_area_df, top_n=1)
    assert len(result) <= 1


def test_get_focus_areas_one_per_feature(focus_area_df):
    """No two focus areas should share the same segment_feature."""
    result = get_focus_areas(focus_area_df, top_n=10)
    features = [a["segment_feature"] for a in result]
    assert len(features) == len(set(features)), "Duplicate features in focus areas"


def test_get_focus_areas_scores_sorted(focus_area_df):
    """Focus areas must be sorted by score descending."""
    result = get_focus_areas(focus_area_df, top_n=10)
    scores = [a["score"] for a in result]
    assert scores == sorted(scores, reverse=True)
