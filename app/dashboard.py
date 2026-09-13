"""
dashboard.py -- Stakeholder Risk Intelligence Dashboard 

7-section interactive analytics dashboard. Answers:
  "What is happening?"        -> Executive Overview
  "Which segments stand out?" -> Claim Pattern Explorer + Risk Segment Analysis
  "How reliable is the model?"-> Model Performance + Calibration
  "Where should I focus?"     -> Error Analysis + Individual Risk Explorer

Run with:  streamlit run app/dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc as sklearn_auc,
    ConfusionMatrixDisplay, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from src.data import load_data, get_feature_target_split
from src.train import make_train_test_split
from src.predict import predict_batch, get_pipeline, VALID_CATEGORIES
from src.risk import assign_risk_level, get_threshold_config, get_focus_areas
from src.interpret import feature_importance_df
from src.evaluate import compute_metrics, segment_error_analysis, calibration_summary
from src.predict import predict_single

# Page configuration
st.set_page_config(
    page_title="Risk Intelligence Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# Cached loaders 
@st.cache_resource
def load_everything():
    pipeline = get_pipeline()
    df = load_data()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)
    y_prob  = predict_batch(X_test)
    y_pred  = (y_prob >= 0.5).astype(int)
    risk_levels = [assign_risk_level(p) for p in y_prob]
    # Full dataset predictions for explorer
    X_all, y_all = get_feature_target_split(df)
    y_prob_all = predict_batch(X_all)
    risk_all   = [assign_risk_level(p) for p in y_prob_all]
    return (pipeline, df, X_train, X_test, y_train, y_test,
            y_prob, y_pred, risk_levels,
            X_all, y_all, y_prob_all, risk_all)

(pipeline, df, X_train, X_test, y_train, y_test,
 y_prob, y_pred, risk_levels,
 X_all, y_all, y_prob_all, risk_all) = load_everything()

thresholds = get_threshold_config()
LOW_T  = thresholds["low_threshold"]
HIGH_T = thresholds["high_threshold"]

# Enrich test set with predictions
test_df = X_test.copy()
test_df["y_true"]     = y_test.values
test_df["y_pred"]     = y_pred
test_df["y_prob"]     = y_prob
test_df["risk_level"] = risk_levels

# Enrich full dataset
full_df = X_all.copy()
full_df["y_true"]     = y_all.values
full_df["y_prob"]     = y_prob_all
full_df["risk_level"] = risk_all

AGE_LABELS = {0: "16-25", 1: "26-39", 2: "40-64", 3: "65+"}
full_df["age_group"] = full_df["age"].map(AGE_LABELS)
test_df["age_group"] = test_df["age"].map(AGE_LABELS)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to section:",
    [
        "0. Focus Areas",
        "1. Executive Overview",
        "2. Claim Pattern Explorer",
        "3. Risk Segment Analysis",
        "4. Feature Importance",
        "5. Model Performance",
        "6. Error Analysis",
        "7. Individual Risk Explorer",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.caption(
    f"**Risk Thresholds:**\n\n"
    f"LOW < {LOW_T:.0%}  |  "
    f"MEDIUM [{LOW_T:.0%}, {HIGH_T:.0%})  |  "
    f"HIGH >= {HIGH_T:.0%}"
)
st.sidebar.caption("Model: Logistic Regression | ROC-AUC: 0.8858 | Test set: 2,000 records")

# ============================================================
# SECTION 0 - FOCUS AREAS 
# ============================================================
if section == "0. Focus Areas":
    st.title("Focus Areas")
    st.markdown(
        "Evidence-backed segments that deserve closer attention. "
        "Recommendations are based on **transparent signals only** — "
        "observed claim rates, segment size, and model error patterns. "
        "No causal claims are made."
    )
    st.info(
        "**How to read this:** Each focus area is flagged because it shows a "
        "meaningful difference in observed claim rate compared to the overall "
        "population. The supporting evidence is shown alongside every recommendation."
    )
    st.divider()

    @st.cache_data
    def compute_focus_areas():
        return get_focus_areas(test_df, top_n=4)

    areas = compute_focus_areas()

    ICON_MAP = {True: "rotating_light", False: "arrow_down_small"}
    COLOR_FN = {True: "error", False: "success"}  # high lift = warning

    for i, area in enumerate(areas):
        is_high = area["claim_rate_lift"] > 0
        lift_pct = f"{abs(area['claim_rate_lift']):.1%}"
        direction_word = "ABOVE" if is_high else "BELOW"
        icon = "rotating_light" if is_high else "white_check_mark"

        # Headline box
        if is_high:
            st.error(f":{icon}: **{area['headline']}**")
        else:
            st.success(f":{icon}: **{area['headline']}**")

        col_why, col_stats = st.columns([1.5, 1])

        with col_why:
            st.markdown(f"**Why this matters:**")
            st.markdown(area["why"])
            st.markdown(
                f"**Scoring logic:** |Claim rate lift| × log(segment size) = "
                f"{abs(area['claim_rate_lift']):.3f} × log({area['segment_n']}) "
                f"= **{area['score']:.2f}**"
            )
            st.caption(
                "This score is used only to rank segments for display. "
                "It does not represent risk severity or a guarantee of future claims."
            )

        with col_stats:
            st.markdown("**Supporting evidence:**")
            stats_df = pd.DataFrame({
                "Signal": [
                    "Observed claim rate",
                    "Overall claim rate",
                    "Claim rate lift",
                    "Segment size",
                    "Missed claims (FN)",
                    "False negative rate",
                    "Avg predicted prob",
                ],
                "Value": [
                    f"{area['claim_rate']:.1%}",
                    f"{area['overall_claim_rate']:.1%}",
                    f"{area['claim_rate_lift']:+.1%} ({direction_word} average)",
                    f"{area['segment_n']:,} customers ({area['segment_size_pct']:.0%})",
                    f"{area['fn_count']} claims missed",
                    f"{area['fn_rate']:.1%}",
                    f"{area['avg_pred_prob']:.3f}",
                ],
            })
            st.table(stats_df)

        st.divider()

    st.subheader("All Findings — Ranked Table")
    all_areas = get_focus_areas(test_df, top_n=20)
    ranked = pd.DataFrame({
        "Rank": [str(i+1) for i in range(len(all_areas))],
        "Feature": [a["segment_feature"].replace("_"," ").title() for a in all_areas],
        "Segment Value": [a["segment_value"] for a in all_areas],
        "Claim Rate": [f"{a['claim_rate']:.1%}" for a in all_areas],
        "Lift vs Overall": [f"{a['claim_rate_lift']:+.1%}" for a in all_areas],
        "Segment N": [str(a["segment_n"]) for a in all_areas],
        "FN Count": [str(a["fn_count"]) for a in all_areas],
        "Score": [f"{a['score']:.2f}" for a in all_areas],
    })
    st.table(ranked)
    st.caption(
        "Score = |Claim rate lift| x log(segment size). "
        "Higher score = stronger and larger deviation from the overall population. "
        "All findings describe statistical associations in this dataset only."
    )

# ============================================================
# SECTION 1 - EXECUTIVE OVERVIEW
# ============================================================
if section == "1. Executive Overview":
    st.title("Executive Overview")
    st.markdown("High-level summary of the dataset, model performance, and risk portfolio.")
    st.divider()

    total_customers = len(df)
    claim_rate = df["outcome"].mean()
    n_claim    = df["outcome"].sum()
    high_risk_n = sum(r == "HIGH" for r in risk_all)
    high_risk_pct = high_risk_n / total_customers

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Observed Claim Rate", f"{claim_rate:.1%}")
    c3.metric("Predicted High-Risk", f"{high_risk_n:,} ({high_risk_pct:.1%})")
    c4.metric("Model ROC-AUC", "0.8858")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Tier Distribution")
        risk_counts = pd.Series(risk_all).value_counts().reindex(["LOW","MEDIUM","HIGH"]).fillna(0)
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = ["#27ae60", "#f39c12", "#e74c3c"]
        bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="white", width=0.5)
        for bar, val in zip(bars, risk_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                    f"{int(val):,}\n({val/total_customers:.0%})",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Customer Count")
        ax.set_title("Predicted Risk Distribution\n(Full Dataset)", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("Observed Claim Rate by Risk Tier")
        tier_data = []
        for tier in ["LOW","MEDIUM","HIGH"]:
            mask = full_df["risk_level"] == tier
            rate = full_df.loc[mask, "y_true"].mean() if mask.sum() > 0 else 0
            tier_data.append({"Tier": tier, "Claim Rate": rate, "Count": int(mask.sum())})
        tier_df = pd.DataFrame(tier_data)

        fig, ax = plt.subplots(figsize=(5, 4))
        bar_colors = ["#27ae60", "#f39c12", "#e74c3c"]
        bars = ax.bar(tier_df["Tier"], tier_df["Claim Rate"], color=bar_colors, edgecolor="white", width=0.5)
        for bar, rate in zip(bars, tier_df["Claim Rate"]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{rate:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual Claim Rate")
        ax.set_ylim([0, 1.0])
        ax.set_title("Observed Claim Rate per Risk Tier\n(Validates Threshold Logic)", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Model Performance Summary")
    perf_data = {
        "Metric": ["Accuracy","Precision","Recall","F1 Score","ROC-AUC","PR-AUC","Brier Score"],
        "Value":  ["82.95%","73.21%","71.93%","72.57%","0.8858","0.7914","0.1216"],
        "Interpretation": [
            "Of all test customers, 83% were classified correctly",
            "73% of customers flagged as HIGH risk actually claimed",
            "72% of actual claimants were correctly identified",
            "Harmonic mean of precision and recall",
            "Excellent discrimination (1.0 = perfect, 0.5 = random)",
            "Strong performance on imbalanced classes",
            "Well-calibrated probabilities (0 = perfect)"
        ],
    }
    st.table(pd.DataFrame(perf_data))

# ============================================================
# SECTION 2 — CLAIM PATTERN EXPLORER
# ============================================================
elif section == "2. Claim Pattern Explorer":
    st.title("Claim Pattern Explorer")
    st.markdown("Explore how claim rates vary across customer segments.")
    st.divider()

    seg_options = {
        "Driving Experience": "driving_experience",
        "Age Group": "age_group",
        "Income Class": "income",
        "Vehicle Year": "vehicle_year",
        "Vehicle Type": "vehicle_type",
        "Education": "education",
        "Gender": "gender",
        "Married": "married",
    }

    selected_seg = st.selectbox("Select segment to explore:", list(seg_options.keys()))
    seg_col = seg_options[selected_seg]

    # Build stats table
    grp = full_df.groupby(seg_col).agg(
        Count=("y_true", "count"),
        Claims=("y_true", "sum"),
        Claim_Rate=("y_true", "mean"),
        Avg_Pred_Prob=("y_prob", "mean"),
    ).reset_index().rename(columns={seg_col: selected_seg})
    grp["Claim_Rate_Pct"] = grp["Claim_Rate"].apply(lambda x: f"{x:.1%}")
    grp["Avg_Pred_Prob_Fmt"] = grp["Avg_Pred_Prob"].apply(lambda x: f"{x:.3f}")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader(f"Claim Rate by {selected_seg}")
        fig, ax = plt.subplots(figsize=(7, 4))
        palette = sns.color_palette("RdYlGn_r", n_colors=len(grp))
        order = grp.sort_values("Claim_Rate", ascending=False)[selected_seg].tolist()
        grp_sorted = grp.set_index(selected_seg).loc[order].reset_index()
        bars = ax.barh(grp_sorted[selected_seg].astype(str)[::-1],
                       grp_sorted["Claim_Rate"][::-1],
                       color=list(reversed(palette)), edgecolor="white")
        for bar, rate in zip(bars, grp_sorted["Claim_Rate"][::-1]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{rate:.1%}", va="center", fontsize=9)
        ax.set_xlabel("Observed Claim Rate")
        ax.axvline(full_df["y_true"].mean(), color="navy", linestyle="--",
                   linewidth=1.2, label=f"Overall ({full_df['y_true'].mean():.1%})")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Segment Statistics")
        display_table = pd.DataFrame({
            selected_seg: grp_sorted[selected_seg].astype(str).tolist(),
            "Count": grp_sorted["Count"].astype(int).astype(str).tolist(),
            "Claims": grp_sorted["Claims"].astype(int).astype(str).tolist(),
            "Claim Rate": grp_sorted["Claim_Rate_Pct"].tolist(),
            "Avg Predicted": grp_sorted["Avg_Pred_Prob_Fmt"].tolist(),
        })
        st.table(display_table)
        st.caption("Avg Predicted = average model claim probability for this segment.")

# ============================================================
# SECTION 3 — RISK SEGMENT ANALYSIS
# ============================================================
elif section == "3. Risk Segment Analysis":
    st.title("Risk Segment Analysis")
    st.markdown("Predicted risk tier breakdown and validation against observed outcomes.")
    st.divider()

    tier_summary = []
    for tier in ["LOW","MEDIUM","HIGH"]:
        mask = test_df["risk_level"] == tier
        sub = test_df[mask]
        tier_summary.append({
            "Risk Tier": tier,
            "Count": int(mask.sum()),
            "Pct of Test Set": f"{mask.mean():.1%}",
            "Avg Pred Prob": f"{sub['y_prob'].mean():.3f}",
            "Observed Claim Rate": f"{sub['y_true'].mean():.1%}",
            "Correctly Classified": f"{((sub['y_pred'] == sub['y_true']).mean()):.1%}",
        })
    st.subheader("Risk Tier Summary (Test Set)")
    st.table(pd.DataFrame(tier_summary))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Probability Distribution per Tier")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_map = {"LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
        for tier in ["LOW","MEDIUM","HIGH"]:
            mask = test_df["risk_level"] == tier
            ax.hist(test_df.loc[mask, "y_prob"], bins=30, alpha=0.6,
                    label=tier, color=colors_map[tier], edgecolor="white")
        ax.axvline(LOW_T, color="green", linestyle="--", linewidth=1.5, label=f"Low threshold ({LOW_T})")
        ax.axvline(HIGH_T, color="red", linestyle="--", linewidth=1.5, label=f"High threshold ({HIGH_T})")
        ax.set_xlabel("Predicted Claim Probability")
        ax.set_ylabel("Count")
        ax.set_title("Probability Distribution by Risk Tier")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Risk Tier by Driving Experience")
        cross = pd.crosstab(test_df["driving_experience"], test_df["risk_level"])
        cross_pct = cross.div(cross.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        cross_pct[["LOW","MEDIUM","HIGH"]].plot(
            kind="bar", stacked=True, ax=ax,
            color=["#27ae60","#f39c12","#e74c3c"], edgecolor="white"
        )
        ax.set_ylabel("Proportion")
        ax.set_title("Risk Tier Composition\nby Driving Experience")
        ax.set_xlabel("Driving Experience")
        ax.legend(title="Risk Tier", bbox_to_anchor=(1.01,1), fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# SECTION 4 — FEATURE IMPORTANCE
# ============================================================
elif section == "4. Feature Importance":
    st.title("Feature Importance")
    st.warning(
        "Feature importance indicates **predictive association** only. "
        "It does NOT imply causation. Do not interpret these values as "
        "'Feature X causes claims.'"
    )
    st.divider()

    imp_df = feature_importance_df(pipeline)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 — Signed Coefficients")
        top15 = imp_df.head(15)
        fig, ax = plt.subplots(figsize=(7, 6))
        colors = ["#e74c3c" if c > 0 else "#2980b9" for c in top15["coefficient"]]
        ax.barh(top15["feature"][::-1], top15["coefficient"][::-1],
                color=colors[::-1], edgecolor="white", height=0.7)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("LR Coefficient (standardised features)")
        ax.set_title("Top 15 Feature Coefficients\nRed=higher risk | Blue=lower risk")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Grouped by Original Feature")
        from src.preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, ORDINAL_FEATURES, COUNT_FEATURES
        def map_to_orig(name):
            for orig in CATEGORICAL_FEATURES:
                if name.startswith(orig + "_") or name == orig:
                    return orig
            return name
        imp_df["original"] = imp_df["feature"].apply(map_to_orig)
        grouped = (imp_df.groupby("original")["importance"].sum()
                   .sort_values(ascending=False).reset_index())
        fig, ax = plt.subplots(figsize=(7, 6))
        palette = sns.color_palette("Blues_r", n_colors=len(grouped))
        ax.barh(grouped["original"][::-1], grouped["importance"][::-1],
                color=palette[::-1], edgecolor="white")
        ax.set_xlabel("Total |Coefficient| (OHE categories summed)")
        ax.set_title("Importance by Original Feature\n(Accounts for one-hot encoding)")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Full Coefficient Table")
    coeff_table = pd.DataFrame({
        "Rank": imp_df["rank"].astype(int).astype(str).tolist(),
        "Feature": imp_df["feature"].tolist(),
        "Coefficient": imp_df["coefficient"].apply(lambda x: f"{x:+.4f}").tolist(),
        "Abs Coeff": imp_df["importance"].apply(lambda x: f"{x:.4f}").tolist(),
        "Direction": imp_df["coefficient"].apply(
            lambda c: "Higher claim risk" if c > 0 else "Lower claim risk"
        ).tolist(),
    })
    st.table(coeff_table)

# ============================================================
# SECTION 5 — MODEL PERFORMANCE
# ============================================================
elif section == "5. Model Performance":
    st.title("Model Performance")
    st.markdown("Evaluation on the held-out test set (n=2,000). Test set was never seen during training.")
    st.divider()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy",  "82.95%")
    col2.metric("Precision", "73.21%")
    col3.metric("Recall",    "71.93%")
    col4.metric("F1",        "0.7257")
    col5.metric("ROC-AUC",   "0.8858")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "PR Curve", "Calibration"])

    with tab1:
        col_a, col_b = st.columns([1, 1.5])
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=["No Claim","Claim"],
                cmap="Blues", ax=ax, colorbar=False
            )
            ax.set_title("Confusion Matrix\n(threshold = 0.5)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with col_b:
            tn = int(((y_test==0)&(y_pred==0)).sum())
            tp = int(((y_test==1)&(y_pred==1)).sum())
            fp = int(((y_test==0)&(y_pred==1)).sum())
            fn = int(((y_test==1)&(y_pred==0)).sum())
            st.markdown(f"""
**Reading the matrix:**

| | |
|---|---|
| **True Negatives (TN) = {tn}** | Correctly predicted *No Claim* |
| **True Positives (TP) = {tp}** | Correctly predicted *Claim* |
| **False Positives (FP) = {fp}** | Flagged as Claim — actually No Claim |
| **False Negatives (FN) = {fn}** | Missed Claims — predicted No Claim |

**False Positives** cause unnecessary scrutiny of good customers.
**False Negatives** are missed claims — the costlier business error.

At threshold 0.5, the model has balanced FP/FN. Lower the threshold to catch more claims (higher recall, lower precision).
""")

    with tab2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = sklearn_auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"LR (AUC = {roc_auc:.4f})")
        ax.plot([0,1],[0,1],"k--", alpha=0.6, label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("AUC = 0.8858 — excellent discrimination. A random model would score 0.5.")

    with tab3:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = sklearn_auc(rec_arr, prec_arr)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec_arr, prec_arr, color="#2980b9", lw=2, label=f"LR (PR-AUC = {pr_auc:.4f})")
        baseline = y_test.mean()
        ax.axhline(baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")
        ax.fill_between(rec_arr, prec_arr, alpha=0.08, color="#2980b9")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("PR-AUC = 0.7914 — strong performance even on an imbalanced dataset (31% claim rate).")

    with tab4:
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_test, y_prob)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0,1],[0,1],"k--", alpha=0.6, label="Perfectly calibrated")
        ax.plot(mean_pred, frac_pos, "s-", color="#e74c3c", markersize=8,
                label=f"LR (Brier = {brier:.4f})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Fraction of Positives")
        ax.set_title("Reliability Diagram (Calibration)\nPoints near diagonal = well calibrated")
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Brier score = {brier:.4f}. A perfect model scores 0.0. A random model scores ~0.25.")

# ============================================================
# SECTION 6 — ERROR ANALYSIS
# ============================================================
elif section == "6. Error Analysis":
    st.title("Error Analysis")
    st.markdown("Understand **where** the model fails, and **who** is being misclassified.")
    st.divider()

    fp_mask = (test_df["y_true"]==0) & (test_df["y_pred"]==1)
    fn_mask = (test_df["y_true"]==1) & (test_df["y_pred"]==0)
    fp_df = test_df[fp_mask]
    fn_df = test_df[fn_mask]

    c1, c2, c3 = st.columns(3)
    c1.metric("False Positives", f"{int(fp_mask.sum())}",
              help="Predicted claim — actually no claim")
    c2.metric("False Negatives", f"{int(fn_mask.sum())}",
              help="Predicted no claim — actually claimed")
    c3.metric("Overall Error Rate", f"{1 - (y_pred==y_test.values).mean():.1%}")

    st.divider()
    st.subheader("Segment Error Rates")

    seg_for_error = st.selectbox(
        "Segment by:",
        ["driving_experience", "age_group", "vehicle_year", "gender", "income"],
        index=0,
    )

    # Rebuild seg col if needed (age_group is derived)
    if seg_for_error == "age_group":
        seg_df = segment_error_analysis(
            test_df[["age_group"]], y_test, y_pred, "age_group"
        )
    else:
        seg_df = segment_error_analysis(X_test, y_test, y_pred, seg_for_error)

    col_chart, col_table = st.columns([1.2, 1])
    with col_chart:
        fig, ax = plt.subplots(figsize=(6, 4))
        palette = sns.color_palette("RdYlGn_r", n_colors=len(seg_df))
        ax.barh(seg_df["segment"].astype(str)[::-1],
                seg_df["error_rate"][::-1],
                color=palette[::-1], edgecolor="white")
        ax.set_xlabel("Error Rate (1 - accuracy)")
        ax.set_title(f"Error Rate by {seg_for_error}")
        ax.axvline(1-(y_pred==y_test.values).mean(), color="navy",
                   linestyle="--", linewidth=1.2, label="Overall")
        ax.legend(fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        err_display = pd.DataFrame({
            "Segment": seg_df["segment"].astype(str).tolist(),
            "N": seg_df["n"].astype(int).astype(str).tolist(),
            "Error Rate": seg_df["error_rate"].apply(lambda x: f"{x:.1%}").tolist(),
            "Precision": seg_df["precision"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            ).tolist(),
            "Recall": seg_df["recall"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
            ).tolist(),
        })
        st.table(err_display)

    st.divider()
    col_fp, col_fn = st.columns(2)
    with col_fp:
        st.subheader(f"False Positive Profile (n={len(fp_df)})")
        st.caption("Customers flagged as claimants who did NOT claim.")
        fp_exp = pd.DataFrame({
            "Feature": ["Driving Experience (0-9y)", "Mean Predicted Prob", "Mean Speeding Viol."],
            "Value": [
                f"{(fp_df['driving_experience']=='0-9y').mean():.0%}",
                f"{fp_df['y_prob'].mean():.3f}",
                f"{fp_df['speeding_violations'].mean():.2f}",
            ]
        })
        st.table(fp_exp)

    with col_fn:
        st.subheader(f"False Negative Profile (n={len(fn_df)})")
        st.caption("Claimants the model MISSED — predicted as no-claim.")
        fn_exp = pd.DataFrame({
            "Feature": ["Experience 10-19y", "Mean Predicted Prob", "Mean Past Accidents"],
            "Value": [
                f"{(fn_df['driving_experience']=='10-19y').mean():.0%}",
                f"{fn_df['y_prob'].mean():.3f}",
                f"{fn_df['past_accidents'].mean():.2f}",
            ]
        })
        st.table(fn_exp)

# ============================================================
# SECTION 7 — INDIVIDUAL RISK EXPLORER
# ============================================================
elif section == "7. Individual Risk Explorer":
    st.title("Individual Risk Explorer")
    st.markdown("Enter or adjust a customer profile and see the predicted risk in real time.")
    st.divider()

    col1, col2 = st.columns(2)
    age_labels = {0:"16-25", 1:"26-39", 2:"40-64", 3:"65+"}
    with col1:
        age_d = st.selectbox("Age Group", list(age_labels.values()), index=1, key="ire_age")
        age_v = [k for k,v in age_labels.items() if v==age_d][0]
        gender_d = st.selectbox("Gender", ["Female","Male"], key="ire_gender")
        gender_v = 0 if gender_d=="Female" else 1
        driv_exp = st.selectbox("Driving Experience", VALID_CATEGORIES["driving_experience"], key="ire_exp")
        education = st.selectbox("Education", VALID_CATEGORIES["education"], key="ire_edu")
        income = st.selectbox("Income", VALID_CATEGORIES["income"], key="ire_inc")
        married_d = st.selectbox("Married", ["No","Yes"], key="ire_mar")
        married_v = 0 if married_d=="No" else 1

    with col2:
        children_d = st.selectbox("Children", ["No","Yes"], key="ire_child")
        children_v = 0 if children_d=="No" else 1
        veh_own_d = st.selectbox("Vehicle Ownership", ["Does not own","Owns vehicle"], key="ire_vown")
        veh_own_v = 0 if veh_own_d=="Does not own" else 1
        veh_year = st.selectbox("Vehicle Year", VALID_CATEGORIES["vehicle_year"], key="ire_yr")
        veh_type = st.selectbox("Vehicle Type", VALID_CATEGORIES["vehicle_type"], key="ire_type")
        credit_score = st.slider("Credit Score", 0.0, 1.0, 0.52, 0.01, key="ire_cs")
        annual_mileage = st.slider("Annual Mileage", 2000, 22000, 12000, 500, key="ire_mi")

    spd = st.number_input("Speeding Violations", 0, 25, 0, key="ire_spd")
    dui = st.number_input("DUIs", 0, 10, 0, key="ire_dui")
    acc = st.number_input("Past Accidents", 0, 20, 0, key="ire_acc")

    profile = {
        "age": age_v, "gender": gender_v, "driving_experience": driv_exp,
        "education": education, "income": income,
        "credit_score": credit_score, "vehicle_ownership": veh_own_v,
        "vehicle_year": veh_year, "married": married_v, "children": children_v,
        "annual_mileage": annual_mileage, "vehicle_type": veh_type,
        "speeding_violations": spd, "duis": dui, "past_accidents": acc,
    }

    if st.button("Get Risk Assessment", type="primary"):
        try:
            from src.risk import build_verdict
            result  = predict_single(profile)
            verdict = build_verdict(result["claim_probability"])
            prob  = verdict["claim_probability"]
            pct   = verdict["claim_probability_pct"]
            risk  = verdict["risk_level"]

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Claim Probability", pct)
            m2.metric("Risk Level", risk)
            m3.metric("Confidence", f"{max(prob,1-prob)*100:.0f}%")

            emojis = {"LOW":"white_check_mark","MEDIUM":"warning","HIGH":"rotating_light"}
            if risk == "LOW":
                st.success(f":{emojis[risk]}: **{risk} RISK** — {verdict['verdict_text']}")
            elif risk == "MEDIUM":
                st.warning(f":{emojis[risk]}: **{risk} RISK** — {verdict['verdict_text']}")
            else:
                st.error(f":{emojis[risk]}: **{risk} RISK** — {verdict['verdict_text']}")

            # Probability gauge
            st.subheader("Probability Gauge")
            fig, ax = plt.subplots(figsize=(8, 1.2))
            ax.barh([""], [LOW_T], color="#27ae60", height=0.4)
            ax.barh([""], [HIGH_T-LOW_T], left=[LOW_T], color="#f39c12", height=0.4)
            ax.barh([""], [1-HIGH_T], left=[HIGH_T], color="#e74c3c", height=0.4)
            ax.axvline(prob, color="black", linewidth=3, label=f"Predicted: {pct}")
            ax.set_xlim([0,1])
            ax.set_xlabel("Predicted Claim Probability")
            ax.set_title(f"Risk Gauge — Predicted probability: {pct}", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)
            ax.set_yticks([])
            ax.spines[["top","right","left"]].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption(
    "Car Insurance Risk Intelligence Dashboard | "
    "Model: Logistic Regression | "
    "Data: 10,000 customers | "
    "Predictions are statistical associations, not causal findings."
)
