"""Risk tier classification and focus area analysis."""

# Thresholds derived from calibration analysis on the held-out test set (n=2,000).
# Below 0.15 → observed claim rate was 5.4% (LOW).
# At or above 0.50 → observed claim rate was 73.2% (HIGH).
LOW_THRESHOLD: float = 0.15
HIGH_THRESHOLD: float = 0.50


def assign_risk_level(probability: float) -> str:
    """Map a predicted probability to LOW, MEDIUM, or HIGH."""
    if not (0.0 <= probability <= 1.0):
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    if probability < LOW_THRESHOLD:
        return "LOW"
    if probability >= HIGH_THRESHOLD:
        return "HIGH"
    return "MEDIUM"


def build_verdict(probability: float) -> dict:
    """Return a formatted verdict dict for display in the app."""
    risk = assign_risk_level(probability)

    messages = {
        "LOW":    "This customer profile is associated with a LOW predicted claim probability.",
        "MEDIUM": "This customer profile is associated with a MODERATE predicted claim probability. Consider additional review.",
        "HIGH":   "This customer profile is associated with a HIGH predicted claim probability. Recommend elevated underwriting scrutiny.",
    }

    return {
        "claim_probability":     round(probability, 4),
        "claim_probability_pct": f"{probability * 100:.1f}%",
        "risk_level":            risk,
        "verdict_text":          messages[risk],
    }


def get_threshold_config() -> dict[str, float]:
    """Return the current risk threshold configuration."""
    return {
        "low_threshold":  LOW_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD,
    }


def get_focus_areas(
    df_with_preds,
    y_true_col="y_true",
    y_pred_col="y_pred",
    y_prob_col="y_prob",
    overall_claim_rate=None,
    top_n=4,
):
    """Identify segments with meaningfully different claim rates from the overall population.

    Scoring logic: |claim_rate_lift| × log(segment_size).
    One result is returned per feature, sorted by score descending.
    """
    import numpy as np

    if overall_claim_rate is None:
        overall_claim_rate = float(df_with_preds[y_true_col].mean())

    skip_cols = {y_true_col, y_pred_col, y_prob_col,
                 "risk_level", "age_group", "id", "postal_code"}
    scan_features = [c for c in df_with_preds.columns if c not in skip_cols]

    findings = []
    for feat in scan_features:
        try:
            unique_vals = df_with_preds[feat].dropna().unique()
            if len(unique_vals) < 2 or len(unique_vals) > 12:
                continue
            for val in unique_vals:
                mask = df_with_preds[feat] == val
                if mask.sum() < 30:
                    continue
                sub          = df_with_preds[mask]
                claim_rate   = float(sub[y_true_col].mean())
                lift         = claim_rate - overall_claim_rate
                fn_count     = int(((sub[y_true_col] == 1) & (sub[y_pred_col] == 0)).sum())
                actual_claims = int(sub[y_true_col].sum())
                findings.append({
                    "segment_feature":    feat,
                    "segment_value":      str(val),
                    "claim_rate":         round(claim_rate, 4),
                    "overall_claim_rate": round(overall_claim_rate, 4),
                    "claim_rate_lift":    round(lift, 4),
                    "segment_size_pct":   round(float(mask.mean()), 4),
                    "segment_n":          int(mask.sum()),
                    "fn_count":           fn_count,
                    "fn_rate":            round(fn_count / max(actual_claims, 1), 4),
                    "avg_pred_prob":      round(float(sub[y_prob_col].mean()), 4),
                    "score":              abs(lift) * np.log1p(int(mask.sum())),
                })
        except Exception:
            continue

    findings.sort(key=lambda x: x["score"], reverse=True)

    seen: set = set()
    top = []
    for f in findings:
        if f["segment_feature"] not in seen:
            seen.add(f["segment_feature"])
            top.append(f)
        if len(top) >= top_n:
            break

    for f in top:
        direction = "higher" if f["claim_rate_lift"] > 0 else "lower"
        feat_nice = f["segment_feature"].replace("_", " ").title()
        f["headline"] = f"Focus area: {feat_nice} = '{f['segment_value']}'"
        f["why"] = (
            f"This segment has a {direction} observed claim rate than the overall population "
            f"({f['claim_rate']:.1%} vs {f['overall_claim_rate']:.1%}, "
            f"a difference of {abs(f['claim_rate_lift']):.1%}). "
            f"It represents {f['segment_size_pct']:.0%} of customers ({f['segment_n']:,} records)."
        )
        if f["fn_count"] > 0:
            f["why"] += (
                f" The model misses {f['fn_count']} actual claims in this segment "
                f"(false negative rate: {f['fn_rate']:.1%}), making it worth further investigation."
            )
    return top
