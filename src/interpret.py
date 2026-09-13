"""Feature importance and model interpretation utilities."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance as sklearn_permutation_importance


def _get_column_transformer(pipeline: Pipeline) -> ColumnTransformer:
    for _, step in pipeline.steps:
        if isinstance(step, ColumnTransformer):
            return step
    raise ValueError("No ColumnTransformer found in pipeline steps.")


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract clean feature names from a fitted Pipeline (post-OHE expansion)."""
    ct = _get_column_transformer(pipeline)
    raw_names = ct.get_feature_names_out()
    return [name.split("__", 1)[1] if "__" in name else name for name in raw_names]


def feature_importance_df(pipeline: Pipeline) -> pd.DataFrame:
    """Return a DataFrame of feature importances sorted by absolute magnitude.

    Works for both linear models (uses coefficients) and tree models (uses feature_importances_).
    """
    feature_names = get_feature_names(pipeline)
    estimator = pipeline.steps[-1][1]

    if hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if coef.ndim == 2:
            coef = coef[0]
        df = pd.DataFrame({
            "feature":     feature_names,
            "coefficient": coef,
            "importance":  np.abs(coef),
        })
    elif hasattr(estimator, "feature_importances_"):
        df = pd.DataFrame({
            "feature":     feature_names,
            "coefficient": np.nan,
            "importance":  estimator.feature_importances_,
        })
    else:
        raise TypeError(
            f"Estimator {type(estimator).__name__} has neither coef_ nor feature_importances_."
        )

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def permutation_importance_df(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance on held-out test data."""
    result = sklearn_permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )
    df = pd.DataFrame({
        "feature":         X_test.columns.tolist(),
        "importance_mean": result.importances_mean,
        "importance_std":  result.importances_std,
    })
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
