"""Tests for data loading and validation (src/data.py)."""

import pytest
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_data, validate_schema, get_feature_target_split,
    report_class_balance, EXPECTED_COLUMNS, TARGET_COL,
)


@pytest.fixture(scope="module")
def raw_df():
    return load_data()


# ── load_data() ───────────────────────────────────────────────────────────────

def test_load_data_returns_dataframe(raw_df):
    """load_data() should return a DataFrame with exactly 10,000 rows."""
    assert isinstance(raw_df, pd.DataFrame)
    assert len(raw_df) == 10_000


def test_load_data_expected_columns(raw_df):
    """load_data() should contain all 18 expected columns."""
    for col in EXPECTED_COLUMNS:
        assert col in raw_df.columns, f"Missing column: {col}"
    assert len(raw_df.columns) == len(EXPECTED_COLUMNS)


def test_load_data_missing_file_raises():
    """load_data() should raise FileNotFoundError for a nonexistent path."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent/path/data.csv")


# ── validate_schema() ─────────────────────────────────────────────────────────

def test_validate_schema_passes_on_valid_data(raw_df):
    """validate_schema() should not raise on the real dataset."""
    validate_schema(raw_df)  # must not raise


def test_validate_schema_raises_on_missing_column(raw_df):
    """validate_schema() should raise ValueError if a required column is missing."""
    broken = raw_df.drop(columns=[TARGET_COL])
    with pytest.raises(ValueError, match="missing expected columns"):
        validate_schema(broken)


def test_validate_schema_raises_on_invalid_target(raw_df):
    """validate_schema() should raise ValueError if target has unexpected values."""
    broken = raw_df.copy()
    broken[TARGET_COL] = 99  # not 0 or 1
    with pytest.raises(ValueError, match="unexpected values"):
        validate_schema(broken)


# ── get_feature_target_split() ────────────────────────────────────────────────

def test_get_feature_target_split_drops_id(raw_df):
    """get_feature_target_split() must not include 'id' or 'outcome' in X."""
    X, y = get_feature_target_split(raw_df)
    assert "id" not in X.columns
    assert TARGET_COL not in X.columns


def test_get_feature_target_split_shapes(raw_df):
    """X and y should have matching row counts and correct column counts."""
    X, y = get_feature_target_split(raw_df)
    assert len(X) == len(y) == 10_000
    # 18 total - outcome - id - postal_code = 15 feature columns
    assert X.shape[1] == 15


def test_get_feature_target_split_binary_target(raw_df):
    """y should only contain 0 and 1."""
    _, y = get_feature_target_split(raw_df)
    assert set(y.unique()).issubset({0, 1})


# ── report_class_balance() ────────────────────────────────────────────────────

def test_class_balance_report_keys(raw_df):
    """report_class_balance() must return all required keys."""
    _, y = get_feature_target_split(raw_df)
    result = report_class_balance(y)
    for key in ["n_total", "n_claim", "n_no_claim", "claim_rate"]:
        assert key in result, f"Missing key: {key}"


def test_class_balance_report_values(raw_df):
    """n_claim + n_no_claim == n_total and claim_rate is in [0, 1]."""
    _, y = get_feature_target_split(raw_df)
    result = report_class_balance(y)
    assert result["n_claim"] + result["n_no_claim"] == result["n_total"]
    assert 0.0 <= result["claim_rate"] <= 1.0


def test_class_balance_claim_rate_approx(raw_df):
    """Known claim rate is ~31.3% in this dataset."""
    _, y = get_feature_target_split(raw_df)
    result = report_class_balance(y)
    assert 0.30 < result["claim_rate"] < 0.35, (
        f"Unexpected claim rate: {result['claim_rate']}"
    )
