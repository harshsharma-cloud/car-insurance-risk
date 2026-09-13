"""Inference layer — loads the production pipeline and generates predictions."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from src.train import load_pipeline
from src.preprocessing import ALL_FEATURES


VALID_CATEGORIES = {
    "driving_experience": ["0-9y", "10-19y", "20-29y", "30y+"],
    "education":          ["high school", "none", "university"],
    "income":             ["middle class", "poverty", "upper class", "working class"],
    "vehicle_year":       ["after 2015", "before 2015"],
    "vehicle_type":       ["sedan", "sports car"],
}

VALID_RANGES = {
    "age":                 (0, 3),
    "gender":              (0, 1),
    "vehicle_ownership":   (0, 1),
    "married":             (0, 1),
    "children":            (0, 1),
    "credit_score":        (0.0, 1.0),
    "annual_mileage":      (2000, 22000),
    "speeding_violations": (0, 25),
    "duis":                (0, 10),
    "past_accidents":      (0, 20),
}

# Module-level cache — pipeline is loaded once at app startup
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Return the cached production pipeline, loading it on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline("final_pipeline.joblib")
    return _pipeline


def validate_input(profile: dict) -> None:
    """Check that all required features are present and within valid bounds."""
    missing = [f for f in ALL_FEATURES if f not in profile]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    for feat, valid_vals in VALID_CATEGORIES.items():
        val = profile.get(feat)
        if val is not None and val not in valid_vals:
            raise ValueError(
                f"Invalid value for '{feat}': '{val}'. Must be one of: {valid_vals}"
            )

    for feat, (lo, hi) in VALID_RANGES.items():
        val = profile.get(feat)
        if val is not None and not (lo <= float(val) <= hi):
            raise ValueError(
                f"Value for '{feat}' = {val} is out of range [{lo}, {hi}]."
            )


def predict_single(profile: dict) -> dict[str, float | int]:
    """Return claim probability and binary label for a single customer profile.

    Raises ValueError if any feature is missing or out of bounds.
    """
    validate_input(profile)
    pipeline = get_pipeline()
    input_df = pd.DataFrame([profile])[ALL_FEATURES]
    prob = pipeline.predict_proba(input_df)[:, 1][0]
    return {
        "claim_probability": float(prob),
        "predicted_label":   int(prob >= 0.5),
    }


def predict_batch(df: pd.DataFrame) -> np.ndarray:
    """Return a 1-D array of claim probabilities for a DataFrame of profiles."""
    pipeline = get_pipeline()
    return pipeline.predict_proba(df[ALL_FEATURES])[:, 1]
