"""Model training, cross-validation, and pipeline persistence."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


RANDOM_SEED = 42
TEST_SIZE = 0.20
MODELS_DIR = Path(__file__).parent.parent / "models"


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train-test split with a fixed random seed."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def train_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit and return a pipeline on training data."""
    return pipeline.fit(X_train, y_train)


def cross_validate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: list[str] | None = None,
) -> dict[str, Any]:
    """Stratified k-fold cross-validation on training data."""
    from sklearn.model_selection import cross_validate, StratifiedKFold
    import numpy as np

    if scoring is None:
        scoring = ["roc_auc", "f1", "precision", "recall", "accuracy"]

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    raw = cross_validate(pipeline, X_train, y_train, cv=cv_strategy,
                         scoring=scoring, return_train_score=False)
    results = {}
    for metric in scoring:
        key = f"test_{metric}"
        results[f"{metric}_mean"] = float(np.mean(raw[key]))
        results[f"{metric}_std"]  = float(np.std(raw[key]))
    return results


def save_pipeline(pipeline: Pipeline, filename: str = "final_pipeline.joblib") -> Path:
    """Save a fitted pipeline to the models/ directory."""
    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(pipeline, path)
    return path


def load_pipeline(filename: str = "final_pipeline.joblib") -> Pipeline:
    """Load a pipeline from the models/ directory."""
    import joblib
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Make sure models/final_pipeline.joblib is present in the repository."
        )
    return joblib.load(path)
