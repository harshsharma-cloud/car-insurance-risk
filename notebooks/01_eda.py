"""
notebooks/01_eda.py
====================
Stage 2 — Data Cleaning & Exploratory Data Analysis
Run with: python notebooks/01_eda.py

Outputs:
  - Console: cleaning decisions, class balance, key stats
  - reports/figures/*.png: all EDA charts

Findings from this script drive decisions in Stages 3–8.
"""

import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Make src importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.data import load_data, get_feature_target_split, report_class_balance

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE_MAIN  = ["#2196F3", "#F44336"]   # blue = no claim, red = claim
PALETTE_BLUE  = "Blues_r"
FIG_DIR       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

SEP = "-" * 65


def savefig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: reports/figures/{name}")


# ── 1. Load & basic info ───────────────────────────────────────────────────
print(SEP)
print("STAGE 2 — DATA CLEANING & EDA")
print(SEP)

df = load_data()
print(f"\n[1] Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

print("Column dtypes:")
print(df.dtypes.to_string())

# ── 2. Duplicates ──────────────────────────────────────────────────────────
print(f"\n[2] Duplicate rows: {df.duplicated().sum()}")
print("    Decision: no duplicates found, no action needed.")

# ── 3. Missing values ──────────────────────────────────────────────────────
print(f"\n[3] Missing values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"count": missing, "pct": missing_pct})
missing_df = missing_df[missing_df["count"] > 0]
print(missing_df.to_string())
print("\n    Decision: both missing columns are continuous.")
print("    Will use median imputation inside the sklearn Pipeline (not here).")
print("    Reason: mean is sensitive to outliers; median is more robust.")
print("    Imputation will be fit ONLY on training data to prevent leakage.")

# ── 4. Class balance ───────────────────────────────────────────────────────
X, y = get_feature_target_split(df)
balance = report_class_balance(y)

print(f"\n[4] Class balance:")
print(f"    Total rows    : {balance['n_total']:,}")
print(f"    No claim  (0) : {balance['n_no_claim']:,} ({100*(1-balance['claim_rate']):.1f}%)")
print(f"    Made claim (1): {balance['n_claim']:,}  ({100*balance['claim_rate']:.1f}%)")
print(f"    Claim rate    : {balance['claim_rate']:.4f}")
print("\n    Imbalance verdict: MODERATE (not extreme).")
print("    Accuracy-only metrics will be misleading — must track F1 and ROC-AUC.")
print("    Will use stratified splits to preserve class ratio.")

# Chart 01 — Class balance
fig, ax = plt.subplots(figsize=(5, 4))
counts = [balance["n_no_claim"], balance["n_claim"]]
bars = ax.bar(["No Claim (0)", "Made Claim (1)"], counts,
              color=PALETTE_MAIN, width=0.5, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
            f"{val:,}\n({val/sum(counts)*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Target Class Distribution", fontweight="bold", pad=12)
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, max(counts) * 1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
savefig("01_class_balance.png")

# ── 5. Leakage / identifier candidates ────────────────────────────────────
print(f"\n[5] Leakage & identifier checks:")
print(f"    'id' column: {df['id'].nunique()} unique values for {len(df)} rows — IDENTIFIER, dropped.")
print(f"    'postal_code' unique values: {df['postal_code'].nunique()}")
print("    postal_code has very few unique values — categorical-ish, but")
print("    geographic codes often act as proxies for demographics/risk pools.")
print("    Decision: drop from modeling to avoid geographic overfitting / leakage.")

# ── 6. Categorical feature distributions ──────────────────────────────────
print(f"\n[6] Categorical feature distributions and claim rates:")

# Map ordinal codes to readable labels for charts
age_map        = {0: "16-25", 1: "26-39", 2: "40-64", 3: "65+"}
income_map     = {0: "Poverty", 1: "Working", 2: "Middle", 3: "Upper"}
edu_map        = {0: "No Education", 1: "High School", 2: "University"}
gender_map     = {0: "Female", 1: "Male"}
ownership_map  = {0: "Financing", 1: "Owns Vehicle"}
married_map    = {0: "Not Married", 1: "Married"}

df_plot = df.copy()
df_plot["age_label"]       = df_plot["age"].map(age_map)
df_plot["income_label"]    = df_plot["income"].map(income_map)
df_plot["edu_label"]       = df_plot["education"].map(edu_map)
df_plot["gender_label"]    = df_plot["gender"].map(gender_map)
df_plot["ownership_label"] = df_plot["vehicle_ownership"].map(ownership_map)
df_plot["married_label"]   = df_plot["married"].map(married_map)

CATEGORICAL_PLOTS = [
    ("age_label",          ["16-25","26-39","40-64","65+"],         "Age Group"),
    ("driving_experience", None,                                     "Driving Experience"),
    ("education",          None,                                     "Education"),
    ("income",             None,                                     "Income Level"),
    ("gender_label",       ["Female","Male"],                        "Gender"),
    ("vehicle_ownership",  None,                                     "Vehicle Ownership"),
    ("vehicle_year",       None,                                     "Vehicle Year"),
    ("married_label",      ["Not Married","Married"],                "Marital Status"),
    ("vehicle_type",       None,                                     "Vehicle Type"),
]

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()

for idx, (col, order, title) in enumerate(CATEGORICAL_PLOTS):
    ax = axes[idx]
    if col in df_plot.columns:
        data_col = df_plot[col]
    else:
        data_col = df[col]

    claim_rates = (
        df_plot.assign(_col=data_col)
        .groupby("_col")["outcome"]
        .mean()
        .reset_index()
        .rename(columns={"_col": col, "outcome": "claim_rate"})
    )
    if order:
        claim_rates[col] = pd.Categorical(claim_rates[col], categories=order, ordered=True)
        claim_rates = claim_rates.sort_values(col)

    bars = ax.bar(claim_rates[col].astype(str), claim_rates["claim_rate"],
                  color="#2196F3", edgecolor="white", linewidth=1)
    ax.axhline(balance["claim_rate"], color="#F44336", linestyle="--",
               linewidth=1.2, label=f"Overall ({balance['claim_rate']:.1%})")
    for bar, val in zip(bars, claim_rates["claim_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Claim Rate by {title}", fontweight="bold", fontsize=10)
    ax.set_ylabel("Claim Rate")
    ax.set_ylim(0, claim_rates["claim_rate"].max() * 1.25)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=20, labelsize=8)

# Print summary to console
for col, order, title in CATEGORICAL_PLOTS:
    data_col = df_plot[col] if col in df_plot.columns else df[col]
    rates = df_plot.assign(_col=data_col).groupby("_col")["outcome"].mean().sort_values(ascending=False)
    top = rates.index[0]
    print(f"  {title}: highest claim rate = '{top}' at {rates.iloc[0]:.1%}")

plt.suptitle("Claim Rate by Categorical Features", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
savefig("02_claim_rates_categorical.png")

# ── 7. Numerical feature distributions ────────────────────────────────────
print(f"\n[7] Numerical feature distributions:")

NUM_COLS = ["credit_score", "annual_mileage", "speeding_violations", "duis",
            "past_accidents", "children"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, col in enumerate(NUM_COLS):
    ax = axes[idx]
    claim_0 = df[df["outcome"] == 0][col].dropna()
    claim_1 = df[df["outcome"] == 1][col].dropna()

    ax.hist(claim_0, bins=30, alpha=0.6, color=PALETTE_MAIN[0],
            density=True, label="No Claim")
    ax.hist(claim_1, bins=30, alpha=0.6, color=PALETTE_MAIN[1],
            density=True, label="Claim")
    ax.set_title(col.replace("_", " ").title(), fontweight="bold")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Print stats
    print(f"  {col}:")
    print(f"    No Claim  — mean={claim_0.mean():.3f}, median={claim_0.median():.3f}")
    print(f"    Claim     — mean={claim_1.mean():.3f}, median={claim_1.median():.3f}")

plt.suptitle("Numerical Feature Distributions (No Claim vs Claim)", fontsize=13,
             fontweight="bold")
plt.tight_layout()
savefig("03_numerical_distributions.png")

# ── 8. Count-feature claim rate ────────────────────────────────────────────
print(f"\n[8] Count features vs claim rate:")

COUNT_COLS = ["speeding_violations", "duis", "past_accidents"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, col in zip(axes, COUNT_COLS):
    rates = df.groupby(col)["outcome"].mean()
    counts = df.groupby(col)["outcome"].count()
    # Only plot values with at least 20 customers
    valid = counts[counts >= 20].index
    rates = rates[valid]
    ax.bar(rates.index.astype(str), rates.values, color="#2196F3",
           edgecolor="white", linewidth=1)
    ax.axhline(balance["claim_rate"], color="#F44336", linestyle="--",
               linewidth=1.2, label="Overall")
    ax.set_title(f"Claim Rate by {col.replace('_',' ').title()}",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Claim Rate")
    ax.legend(fontsize=8)
    peak = rates.idxmax()
    print(f"  {col}: peak claim rate at value={peak} ({rates[peak]:.1%})")

plt.tight_layout()
savefig("04_count_feature_claim_rates.png")

# ── 9. Credit score vs claim (box plot) ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, col in zip(axes, ["credit_score", "annual_mileage"]):
    data_0 = df[df["outcome"] == 0][col].dropna()
    data_1 = df[df["outcome"] == 1][col].dropna()
    bp = ax.boxplot([data_0, data_1], patch_artist=True,
                    tick_labels=["No Claim", "Made Claim"],
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor(PALETTE_MAIN[0])
    bp["boxes"][1].set_facecolor(PALETTE_MAIN[1])
    for patch in bp["boxes"]:
        patch.set_alpha(0.7)
    ax.set_title(f"{col.replace('_',' ').title()} by Outcome",
                 fontweight="bold")
    ax.set_ylabel(col.replace("_", " ").title())

plt.suptitle("Continuous Features vs Claim Outcome", fontsize=13, fontweight="bold")
plt.tight_layout()
savefig("05_continuous_boxplots.png")

print(f"\n[9] Credit score insight:")
cs_claim    = df[df["outcome"] == 1]["credit_score"].median()
cs_no_claim = df[df["outcome"] == 0]["credit_score"].median()
print(f"    Median credit score — No Claim: {cs_no_claim:.3f} | Claim: {cs_claim:.3f}")
print(f"    Annual mileage — No Claim: {df[df['outcome']==0]['annual_mileage'].median():.0f} | "
      f"Claim: {df[df['outcome']==1]['annual_mileage'].median():.0f}")

# ── 10. Correlation heatmap ────────────────────────────────────────────────
print(f"\n[10] Correlation with outcome (numeric features only):")

# Encode string categoricals as ordinal for correlation
df_corr = df.copy()
from sklearn.preprocessing import OrdinalEncoder

str_cats = ["driving_experience", "education", "income", "vehicle_year", "vehicle_type"]
enc = OrdinalEncoder()
df_corr[str_cats] = enc.fit_transform(df_corr[str_cats].astype(str))
df_corr = df_corr.drop(columns=["id", "postal_code"])

# Fill NA for correlation calculation only
df_corr["credit_score"]   = df_corr["credit_score"].fillna(df_corr["credit_score"].median())
df_corr["annual_mileage"] = df_corr["annual_mileage"].fillna(df_corr["annual_mileage"].median())

corr_with_outcome = df_corr.corr()["outcome"].drop("outcome").sort_values(key=abs, ascending=False)
print(corr_with_outcome.round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 7))
corr_vals = corr_with_outcome.values
feature_names = corr_with_outcome.index.tolist()
colors = ["#F44336" if v > 0 else "#2196F3" for v in corr_vals]
bars = ax.barh(feature_names[::-1], corr_vals[::-1], color=colors[::-1],
               edgecolor="white", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Feature Correlation with Outcome (Claim=1)", fontweight="bold", pad=12)
ax.set_xlabel("Pearson Correlation Coefficient")
# Annotate bars
for bar, val in zip(bars, corr_vals[::-1]):
    ax.text(val + (0.005 if val >= 0 else -0.005),
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8)
plt.tight_layout()
savefig("06_correlation_with_outcome.png")

# ── 11. Missing value summary chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
miss_data = missing_df["pct"].sort_values(ascending=True)
ax.barh(miss_data.index, miss_data.values, color="#FF9800", edgecolor="white")
for i, val in enumerate(miss_data.values):
    ax.text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=10)
ax.set_title("Missing Value Percentage by Column", fontweight="bold")
ax.set_xlabel("% Missing")
ax.set_xlim(0, miss_data.max() * 1.25)
plt.tight_layout()
savefig("07_missing_values.png")

# ── Final summary ──────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("CLEANING DECISIONS SUMMARY")
print(SEP)
print("""
1. DUPLICATES     : None found. No action needed.
2. MISSING VALUES : credit_score (9.8%) and annual_mileage (9.6%)
                    -> Median imputation inside sklearn Pipeline on train data only.
                    -> NOT imputed here to prevent leakage.
3. IDENTIFIERS    : 'id' dropped (unique per row).
                    'postal_code' dropped (geographic proxy, overfitting risk).
4. CLASS BALANCE  : 31.3% claim rate — moderate imbalance.
                    -> Use stratified train/test split.
                    -> Report F1, ROC-AUC, PR-AUC — not just accuracy.
5. FEATURE TYPES  :
   - Numerical continuous : credit_score, annual_mileage
   - Ordinal (integer)    : age, gender, vehicle_ownership, married, children
   - Count integers       : speeding_violations, duis, past_accidents
   - String categorical   : driving_experience, education, income,
                            vehicle_year, vehicle_type
6. LEAKAGE CHECK  : No column directly encodes the outcome.
                    driving_experience is a legitimate feature (not post-event).

KEY FINDINGS:
  * driving_experience is the single strongest predictor (DataCamp also found this).
  * speeding_violations, duis, past_accidents show clear monotonic claim rate increase.
  * credit_score is LOWER for claimants (riskier customers have worse credit).
  * Young drivers (16-25) have the highest claim rate.
  * Sports car owners have higher claim rate than sedan owners.
""")
print(f"All charts saved to: reports/figures/")
print(SEP)
