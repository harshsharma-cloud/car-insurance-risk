"""Classification metrics, calibration diagnostics, and segment error analysis."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.pipeline import Pipeline


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label: str = "Model",
    verbose: bool = True,
) -> dict[str, float]:
    """Compute the full classification metric suite and optionally print a summary table."""
    acc    = accuracy_score(y_true, y_pred)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    auc    = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    if verbose:
        print(f"\n  {label} — Test Set Metrics  (threshold={threshold})")
        print(f"  {'Metric':<22} {'Value':>8}")
        print(f"  {'-'*32}")
        print(f"  {'Accuracy':<22} {acc:>8.4f}  ({acc*100:.2f}%)")
        print(f"  {'Precision':<22} {prec:>8.4f}")
        print(f"  {'Recall':<22} {rec:>8.4f}")
        print(f"  {'F1 Score':<22} {f1:>8.4f}")
        print(f"  {'ROC-AUC':<22} {auc:>8.4f}")
        print(f"  {'PR-AUC':<22} {pr_auc:>8.4f}")

    return dict(model=label, accuracy=acc, precision=prec, recall=rec,
                f1=f1, roc_auc=auc, pr_auc=pr_auc, threshold=threshold)


def confusion_matrix_df(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a labelled 2×2 confusion matrix as a DataFrame."""
    cm = sk_confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual: No Claim", "Actual: Claim"],
        columns=["Predicted: No Claim", "Predicted: Claim"],
    )


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a model comparison table sorted by ROC-AUC descending."""
    df = pd.DataFrame(results)
    if "model" in df.columns:
        df = df.set_index("model")
    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
                   if c in df.columns]
    return df[metric_cols].sort_values("roc_auc", ascending=False).round(4)


def calibration_summary(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return reliability diagram data (mean predicted prob vs observed fraction positive)."""
    from sklearn.calibration import calibration_curve

    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
    counts      = np.bincount(bin_indices, minlength=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    non_empty   = counts > 0

    return pd.DataFrame({
        "bin_center":            bin_centers[non_empty][:len(fraction_pos)],
        "mean_predicted_prob":   mean_pred,
        "fraction_of_positives": fraction_pos,
        "count":                 counts[non_empty][:len(fraction_pos)],
    })


def segment_error_analysis(
    X_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    segment_col: str,
) -> pd.DataFrame:
    """Break down prediction errors by a given feature segment."""
    tmp = X_test[[segment_col]].copy()
    tmp["y_true"] = y_true.values
    tmp["y_pred"] = y_pred
    tmp["tp"] = ((tmp["y_true"] == 1) & (tmp["y_pred"] == 1)).astype(int)
    tmp["fp"] = ((tmp["y_true"] == 0) & (tmp["y_pred"] == 1)).astype(int)
    tmp["fn"] = ((tmp["y_true"] == 1) & (tmp["y_pred"] == 0)).astype(int)
    tmp["tn"] = ((tmp["y_true"] == 0) & (tmp["y_pred"] == 0)).astype(int)

    grouped = tmp.groupby(segment_col).agg(
        n=("y_true", "count"), tp=("tp", "sum"), fp=("fp", "sum"),
        fn=("fn", "sum"), tn=("tn", "sum"),
    ).reset_index().rename(columns={segment_col: "segment"})

    grouped["accuracy"]   = (grouped["tp"] + grouped["tn"]) / grouped["n"]
    grouped["error_rate"] = 1 - grouped["accuracy"]
    grouped["precision"]  = grouped["tp"] / (grouped["tp"] + grouped["fp"]).replace(0, np.nan)
    grouped["recall"]     = grouped["tp"] / (grouped["tp"] + grouped["fn"]).replace(0, np.nan)

    return grouped.sort_values("error_rate", ascending=False).reset_index(drop=True)
