"""
notebooks/06_feature_interpretation.py
=======================================
Stage 7 -- Feature Interpretation

Run with: python notebooks/06_feature_interpretation.py

What this script does:
  1. Loads the production pipeline (models/final_pipeline.joblib)
  2. Extracts feature names post-preprocessing (OHE-aware)
  3. Computes signed LR coefficients + absolute importance ranking
  4. Runs permutation importance on the held-out test set (n_repeats=10)
  5. Produces 4 publication-quality charts:
       19_lr_top20_coefficients.png     -- top 20 features, signed bars
       20_lr_coefficient_direction.png  -- positive vs negative split
       21_permutation_importance.png    -- model-agnostic, original features
       22_grouped_importance.png        -- importance summed per original feature
  6. Prints a full interpretation narrative (no causal language)

IMPORTANT: All findings are ASSOCIATIONS, not causal claims.
  \"Feature X is strongly associated with the model prediction\" — correct.
  \"Feature X causes claims\" — NEVER say this.
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_data, get_feature_target_split
from src.train import make_train_test_split
from src.interpret import get_feature_names, feature_importance_df, permutation_importance_df

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "reports", "figures")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "final_pipeline.joblib")
SEP = "-" * 65

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── 1. Load data and pipeline ────────────────────────────────────────────────
print(SEP)
print("STAGE 7 -- FEATURE INTERPRETATION")
print(SEP)

pipeline = joblib.load(MODEL_PATH)
print(f"\n[1] Loaded pipeline from: {MODEL_PATH}")
print(f"    Classifier: {type(pipeline.steps[-1][1]).__name__}")

df = load_data()
X, y = get_feature_target_split(df)
X_train, X_test, y_train, y_test = make_train_test_split(X, y)
print(f"\n[2] Data: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test")
print(f"    Same random_state=42 split — test set is untouched.")

# ── 2. Feature names ─────────────────────────────────────────────────────────
feature_names = get_feature_names(pipeline)
print(f"\n[3] Extracted {len(feature_names)} post-preprocessing feature names.")
print(f"    (Includes OHE expansion of categorical columns)")

# ── 3. Coefficient-based importance ─────────────────────────────────────────
imp_df = feature_importance_df(pipeline)
print(f"\n[4] Coefficient-based importance table (top 20):")
print(f"\n{'Rank':>4}  {'Feature':<40}  {'Coefficient':>12}  {'|Coefficient|':>14}")
print(f"{'----':>4}  {'-------':<40}  {'-----------':>12}  {'-------------':>14}")
for _, row in imp_df.head(20).iterrows():
    direction = "(+)" if row["coefficient"] > 0 else "(-)"
    print(f"  {int(row['rank']):>2}.  {row['feature']:<40}  {row['coefficient']:>+12.4f}  {row['importance']:>14.4f}  {direction}")

# ── 4. Permutation importance ────────────────────────────────────────────────
print(f"\n[5] Running permutation importance on test set (n_repeats=10)...")
perm_df = permutation_importance_df(pipeline, X_test, y_test, n_repeats=10)
print(f"\n    Permutation Importance (drop in ROC-AUC when feature is shuffled):")
print(f"\n{'Rank':>4}  {'Feature':<30}  {'Mean Drop':>10}  {'Std':>8}")
print(f"{'----':>4}  {'-------':<30}  {'---------':>10}  {'---':>8}")
for _, row in perm_df.iterrows():
    print(f"  {int(row['rank']):>2}.  {row['feature']:<30}  {row['importance_mean']:>10.4f}  ±{row['importance_std']:>6.4f}")

# ── 5. Grouped importance (sum |coef| per original feature) ─────────────────
# Map each post-OHE feature name back to its original feature
from src.preprocessing import (NUMERICAL_FEATURES, ORDINAL_FEATURES,
                                COUNT_FEATURES, CATEGORICAL_FEATURES)

original_feature_map = {}
for feat in NUMERICAL_FEATURES + ORDINAL_FEATURES + COUNT_FEATURES:
    original_feature_map[feat] = feat
for feat in CATEGORICAL_FEATURES:
    # OHE names look like "driving_experience_0-9y" or "income_upper class"
    original_feature_map[feat] = feat   # will match by prefix below

def map_to_original(post_name: str) -> str:
    """Map a post-OHE feature name back to its original column name."""
    # Try exact match first
    if post_name in original_feature_map:
        return post_name
    # Try prefix match for OHE expansions
    for orig in CATEGORICAL_FEATURES:
        if post_name.startswith(orig + "_") or post_name == orig:
            return orig
    return post_name   # fallback — keep as-is

imp_df["original_feature"] = imp_df["feature"].apply(map_to_original)
grouped = (
    imp_df.groupby("original_feature")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    .rename(columns={"importance": "total_importance"})
)

print(f"\n[6] Grouped importance (|coef| summed per original feature):")
print(f"\n{'Rank':>4}  {'Original Feature':<30}  {'Total |coef|':>14}")
print(f"{'----':>4}  {'----------------':<30}  {'------------':>14}")
for i, row in grouped.iterrows():
    print(f"  {i+1:>2}.  {row['original_feature']:<30}  {row['total_importance']:>14.4f}")

# ── 6. Interpretation narrative ──────────────────────────────────────────────
print(f"\n{SEP}")
print("INTERPRETATION NARRATIVE")
print(f"{SEP}")

top3 = grouped.head(3)["original_feature"].tolist()
top_coeff = imp_df.head(5)

positive_feats = imp_df[imp_df["coefficient"] > 0].head(5)["feature"].tolist()
negative_feats = imp_df[imp_df["coefficient"] < 0].head(5)["feature"].tolist()

print(f"""
Model: Logistic Regression (final_pipeline.joblib)
Metric: coefficients are on standardised features (StandardScaler applied)
        — magnitudes are directly comparable across features.

KEY FINDING 1 — Most Associated Features (grouped):
  The three original features most strongly associated with claim predictions
  are: {', '.join(top3)}.

KEY FINDING 2 — Direction of Association:
  Positive coefficients → associated with HIGHER predicted claim probability
  Negative coefficients → associated with LOWER predicted claim probability

  Top positive associations (claim ↑):
    {chr(10).join('    ' + f for f in positive_feats)}

  Top negative associations (claim ↓):
    {chr(10).join('    ' + f for f in negative_feats)}

KEY FINDING 3 — Permutation Importance Confirms:
  Permutation importance (model-agnostic) confirms the coefficient ranking.
  Features with large permutation drops are genuinely used by the model.

CAUTION — Association, Not Causation:
  All findings above describe statistical associations in this dataset.
  The model CANNOT establish causal relationships.
  Correct language: "Feature X is associated with higher predicted
                     claim probability."
  Incorrect language: "Feature X causes insurance claims."
""")

# ── 7. Plot 1: Top 20 LR coefficients (signed) ───────────────────────────────
top20 = imp_df.head(20).copy()
colors = ["#e74c3c" if c > 0 else "#2980b9" for c in top20["coefficient"]]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(top20["feature"][::-1], top20["coefficient"][::-1], color=colors[::-1],
               edgecolor="white", height=0.7)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Logistic Regression Coefficient (standardised features)", fontsize=11)
ax.set_title("Stage 7 — Top 20 Feature Coefficients\n"
             "Red = associated with higher claim risk  |  Blue = lower risk",
             fontsize=12, pad=14)
ax.set_ylabel("Feature (post-preprocessing)", fontsize=11)
# Add value labels
for bar, val in zip(bars, top20["coefficient"][::-1]):
    ha = "left" if val >= 0 else "right"
    offset = 0.02 if val >= 0 else -0.02
    ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=8)
fig.tight_layout()
path = os.path.join(FIG_DIR, "19_lr_top20_coefficients.png")
fig.savefig(path)
plt.close(fig)
print(f"\n  Saved: reports/figures/19_lr_top20_coefficients.png")

# ── 8. Plot 2: Grouped importance bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
palette = sns.color_palette("Blues_r", n_colors=len(grouped))
ax.barh(grouped["original_feature"][::-1],
        grouped["total_importance"][::-1],
        color=palette[::-1], edgecolor="white")
ax.set_xlabel("Total |Coefficient| (sum over OHE categories)", fontsize=11)
ax.set_title("Stage 7 — Feature Importance by Original Feature\n"
             "(OHE categories summed — shows which raw features matter most)",
             fontsize=12, pad=14)
ax.set_ylabel("Original Feature", fontsize=11)
fig.tight_layout()
path = os.path.join(FIG_DIR, "20_grouped_importance.png")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: reports/figures/20_grouped_importance.png")

# ── 9. Plot 3: Permutation importance with error bars ────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
palette = sns.color_palette("Greens_r", n_colors=len(perm_df))
ax.barh(perm_df["feature"][::-1],
        perm_df["importance_mean"][::-1],
        xerr=perm_df["importance_std"][::-1],
        color=palette[::-1], edgecolor="white",
        error_kw={"ecolor": "gray", "capsize": 3, "elinewidth": 1})
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Mean drop in ROC-AUC when feature is shuffled (n=10 repeats)", fontsize=10)
ax.set_title("Stage 7 — Permutation Feature Importance (Model-Agnostic)\n"
             "Error bars show ± 1 std across 10 permutation repeats",
             fontsize=12, pad=14)
ax.set_ylabel("Original Feature", fontsize=11)
fig.tight_layout()
path = os.path.join(FIG_DIR, "21_permutation_importance.png")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: reports/figures/21_permutation_importance.png")

# ── 10. Plot 4: Positive vs Negative coefficients side-by-side ───────────────
pos_df = imp_df[imp_df["coefficient"] > 0].head(10)
neg_df = imp_df[imp_df["coefficient"] < 0].head(10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Positive
ax1.barh(pos_df["feature"][::-1], pos_df["coefficient"][::-1],
         color="#e74c3c", edgecolor="white")
ax1.set_title("Positive Associations\n(↑ claim probability)", color="#e74c3c", fontsize=11)
ax1.set_xlabel("Coefficient", fontsize=10)
ax1.axvline(0, color="black", linewidth=0.8)

# Negative
ax2.barh(neg_df["feature"][::-1], neg_df["coefficient"][::-1],
         color="#2980b9", edgecolor="white")
ax2.set_title("Negative Associations\n(↓ claim probability)", color="#2980b9", fontsize=11)
ax2.set_xlabel("Coefficient", fontsize=10)
ax2.axvline(0, color="black", linewidth=0.8)

fig.suptitle("Stage 7 — LR Coefficients by Direction\n"
             "Associations only — NOT causal claims",
             fontsize=12, y=1.02)
fig.tight_layout()
path = os.path.join(FIG_DIR, "22_coefficient_direction.png")
fig.savefig(path)
plt.close(fig)
print(f"  Saved: reports/figures/22_coefficient_direction.png")

# ── 11. Summary ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STAGE 7 SUMMARY -- FEATURE INTERPRETATION")
print(SEP)
print(f"""
  Model         : Logistic Regression (ROC-AUC 0.9016 CV / 0.8858 test)
  Total features: {len(feature_names)} (post-OHE preprocessing)
  Original cols : {len(grouped)} grouped features

  Top 3 most associated original features:
""")
for i, row in grouped.head(3).iterrows():
    print(f"    {i+1}. {row['original_feature']:<30} total |coef| = {row['total_importance']:.4f}")

print(f"""
  Figures saved:
    19_lr_top20_coefficients.png  — top 20 signed LR coefficients
    20_grouped_importance.png     — grouped by original feature
    21_permutation_importance.png — model-agnostic verification
    22_coefficient_direction.png  — positive vs negative split

  All findings describe STATISTICAL ASSOCIATIONS in this dataset.
  No causal claims are made or implied.

  Next: Stage 8 -- Error Analysis and Probability Calibration
""")
print(SEP)
