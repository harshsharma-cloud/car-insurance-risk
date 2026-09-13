"""Tests for model training, pipeline loading, and prediction (src/train.py, src/predict.py)."""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data, get_feature_target_split
from src.train import make_train_test_split, load_pipeline
from src.predict import predict_single, predict_batch, validate_input, VALID_CATEGORIES


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def split_data():
    df = load_data()
    X, y = get_feature_target_split(df)
    return make_train_test_split(X, y)


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline("final_pipeline.joblib")


# Reference profiles used in multiple tests
NORMAL_PROFILE = {
    "age": 1, "gender": 0, "driving_experience": "10-19y",
    "education": "high school", "income": "middle class",
    "credit_score": 0.52, "vehicle_ownership": 1,
    "vehicle_year": "before 2015", "married": 1, "children": 1,
    "annual_mileage": 12000, "vehicle_type": "sedan",
    "speeding_violations": 0, "duis": 0, "past_accidents": 0,
}

HIGH_RISK_PROFILE = {
    "age": 0, "gender": 1, "driving_experience": "0-9y",
    "education": "none", "income": "poverty",
    "credit_score": 0.10, "vehicle_ownership": 0,
    "vehicle_year": "before 2015", "married": 0, "children": 0,
    "annual_mileage": 20000, "vehicle_type": "sports car",
    "speeding_violations": 5, "duis": 1, "past_accidents": 3,
}

LOW_RISK_PROFILE = {
    "age": 3, "gender": 0, "driving_experience": "30y+",
    "education": "university", "income": "upper class",
    "credit_score": 0.90, "vehicle_ownership": 1,
    "vehicle_year": "after 2015", "married": 1, "children": 1,
    "annual_mileage": 6000, "vehicle_type": "sedan",
    "speeding_violations": 0, "duis": 0, "past_accidents": 0,
}

BOUNDARY_PROFILE = {
    "age": 0, "gender": 0, "driving_experience": "0-9y",
    "education": "high school", "income": "working class",
    "credit_score": 0.0,        # minimum
    "vehicle_ownership": 0,
    "vehicle_year": "before 2015", "married": 0, "children": 0,
    "annual_mileage": 2000,      # minimum
    "vehicle_type": "sedan",
    "speeding_violations": 0, "duis": 0, "past_accidents": 0,
}


# ── train/test split ──────────────────────────────────────────────────────────

def test_train_test_split_proportions(split_data):
    """Split should produce ~80/20 and preserve class balance."""
    X_train, X_test, y_train, y_test = split_data
    total = len(X_train) + len(X_test)
    assert total == 10_000
    assert abs(len(X_test) / total - 0.20) < 0.01

    # Stratification: class balance should be within 1 % of each other
    train_rate = y_train.mean()
    test_rate  = y_test.mean()
    assert abs(train_rate - test_rate) < 0.01, (
        f"Class balance differs: train={train_rate:.3f}, test={test_rate:.3f}"
    )


def test_train_test_split_is_reproducible():
    """The same split must be reproduced with the same seed."""
    df = load_data()
    X, y = get_feature_target_split(df)
    split1 = make_train_test_split(X, y)
    split2 = make_train_test_split(X, y)
    assert list(split1[0].index[:10]) == list(split2[0].index[:10])


# ── Pipeline loading ──────────────────────────────────────────────────────────

def test_pipeline_loads_without_error(pipeline):
    """load_pipeline() should load the saved artifact without raising."""
    assert pipeline is not None


def test_pipeline_has_preprocessor_and_classifier(pipeline):
    """Pipeline must have exactly 2 steps: preprocessor and classifier."""
    step_names = [name for name, _ in pipeline.steps]
    assert "preprocessor" in step_names
    assert "classifier" in step_names


def test_pipeline_missing_artifact_raises():
    """load_pipeline() should raise FileNotFoundError for nonexistent artifact."""
    with pytest.raises(FileNotFoundError):
        load_pipeline("this_does_not_exist.joblib")


# ── predict_single() — normal valid profile ───────────────────────────────────

def test_predict_single_returns_probability():
    """predict_single() should return a float probability between 0 and 1."""
    result = predict_single(NORMAL_PROFILE)
    assert "claim_probability" in result
    prob = result["claim_probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_predict_single_returns_label():
    """predict_single() must include predicted_label (0 or 1)."""
    result = predict_single(NORMAL_PROFILE)
    assert result["predicted_label"] in {0, 1}


def test_predict_single_valid_normal_profile():
    """Normal mid-risk profile should produce a probability in (0, 1)."""
    result = predict_single(NORMAL_PROFILE)
    assert 0.0 < result["claim_probability"] < 1.0


def test_predict_single_high_risk_profile():
    """High-risk profile (young, no experience, violations) should predict high."""
    result = predict_single(HIGH_RISK_PROFILE)
    assert result["claim_probability"] > 0.60, (
        f"Expected high probability, got {result['claim_probability']:.3f}"
    )


def test_predict_single_low_risk_profile():
    """Low-risk profile (experienced, clean record) should predict low."""
    result = predict_single(LOW_RISK_PROFILE)
    assert result["claim_probability"] < 0.20, (
        f"Expected low probability, got {result['claim_probability']:.3f}"
    )


# ── predict_single() — boundary values ───────────────────────────────────────

def test_predict_single_boundary_values():
    """Boundary numerical inputs (min credit_score, min mileage) must not crash."""
    result = predict_single(BOUNDARY_PROFILE)
    prob = result["claim_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_single_max_boundary():
    """Maximum boundary values must not crash."""
    max_profile = NORMAL_PROFILE.copy()
    max_profile["credit_score"] = 1.0
    max_profile["annual_mileage"] = 22000
    max_profile["speeding_violations"] = 22
    max_profile["duis"] = 6
    max_profile["past_accidents"] = 15
    result = predict_single(max_profile)
    assert 0.0 <= result["claim_probability"] <= 1.0


# ── predict_single() — invalid / missing input ────────────────────────────────

def test_predict_single_rejects_missing_features():
    """predict_single() should raise ValueError if required features are missing."""
    incomplete = {"age": 1, "gender": 0}
    with pytest.raises(ValueError, match="Missing required features"):
        predict_single(incomplete)


def test_predict_single_rejects_invalid_category():
    """predict_single() should raise ValueError for an invalid category value."""
    bad = NORMAL_PROFILE.copy()
    bad["driving_experience"] = "100y"
    with pytest.raises(ValueError, match="Invalid value"):
        predict_single(bad)


def test_predict_single_rejects_out_of_range_credit_score():
    """Credit score outside [0, 1] should raise ValueError."""
    bad = NORMAL_PROFILE.copy()
    bad["credit_score"] = 2.5
    with pytest.raises(ValueError, match="out of range"):
        predict_single(bad)


def test_predict_single_rejects_negative_mileage():
    """Negative annual mileage should raise ValueError."""
    bad = NORMAL_PROFILE.copy()
    bad["annual_mileage"] = -1000
    with pytest.raises(ValueError, match="out of range"):
        predict_single(bad)


# ── predict_batch() ───────────────────────────────────────────────────────────

def test_predict_batch_output_shape(split_data):
    """predict_batch() output length must match input DataFrame length."""
    _, X_test, _, _ = split_data
    probs = predict_batch(X_test)
    assert len(probs) == len(X_test)


def test_predict_batch_all_valid_probabilities(split_data):
    """All batch predictions must be valid probabilities in [0, 1]."""
    _, X_test, _, _ = split_data
    probs = predict_batch(X_test)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_predict_batch_roc_auc_above_threshold(split_data):
    """Model ROC-AUC on test set should be >= 0.85 (spec requires >= 0.85)."""
    from sklearn.metrics import roc_auc_score
    _, X_test, _, y_test = split_data
    probs = predict_batch(X_test)
    auc = roc_auc_score(y_test, probs)
    assert auc >= 0.85, f"ROC-AUC {auc:.4f} is below minimum threshold 0.85"
