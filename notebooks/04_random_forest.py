"""
notebooks/04_random_forest.py
=====================================
Stage 5 — Tree-Based Model (Random Forest)

Run with: python notebooks/04_random_forest.py

What this script does:
  1. Builds a full sklearn Pipeline: ColumnTransformer -> RandomForestClassifier
     (Uses ordinal encoding and skips scaling, as trees don't need it)
  2. Fits ONLY on training data
  3. Evaluates on the held-out test set
  4. Saves the trained pipeline to models/rf_pipeline.joblib
  5. Plots ROC curve, confusion matrix, and probability distribution
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

from src.data import load_data, get_feature_target_split
from src.train import make_train_test_split, train_pipeline, save_pipeline
from src.preprocessing import build_pipeline
from src.evaluate import compute_metrics

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "reports", "figures")
SEP = "-" * 65

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


# ── 1. Load and split ──────────────────────────────────────────────────────
print(SEP)
print("STAGE 5 — RANDOM FOREST PIPELINE")
print(SEP)

df = load_data()
X, y = get_feature_target_split(df)
X_train, X_test, y_train, y_test = make_train_test_split(X, y)

# ── 2. Build pipeline ─────────────────────────────────────────────────────
print("[1] Building pipeline: ColumnTransformer -> RandomForestClassifier")
print("""
    Preprocessing (all fit on train data only):
      - model_type='tree' means NO StandardScaler is used.
      - String categoricals use OrdinalEncoder instead of OneHotEncoder.
      - Missing values are still imputed with median/most_frequent.

    Classifier:
      RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5)
      - max_depth=10 prevents the tree from growing too deep (overfitting).
      - min_samples_leaf=5 forces the tree to generalize at the leaves.
""")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

# Crucial: Use model_type="tree" to get the right preprocessing
pipeline_rf = build_pipeline(rf, model_type="tree")

# ── 3. Train ───────────────────────────────────────────────────────────────
print("[2] Fitting pipeline on training data only...")
pipeline_rf = train_pipeline(pipeline_rf, X_train, y_train)
print("    Done.")

# ── 4. Predict on test set ─────────────────────────────────────────────────
print("\n[3] Evaluating on held-out test set:")
y_pred_rf = pipeline_rf.predict(X_test)
y_prob_rf  = pipeline_rf.predict_proba(X_test)[:, 1]

metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf,
                              label="Random Forest")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_rf,
                              target_names=["No Claim", "Claim"], zero_division=0))

# ── 5. Probability distribution ───────────────────────────────────────────
print(f"\n[4] Probability distribution analysis:")
prob_claim    = y_prob_rf[y_test == 1]
prob_no_claim = y_prob_rf[y_test == 0]
print(f"    Actual claimants    - mean prob: {prob_claim.mean():.3f}, "
      f"median: {np.median(prob_claim):.3f}")
print(f"    Actual non-claimants - mean prob: {prob_no_claim.mean():.3f}, "
      f"median: {np.median(prob_no_claim):.3f}")
print(f"    Separation: {prob_claim.mean() - prob_no_claim.mean():.3f}")

# ── 6. Charts ─────────────────────────────────────────────────────────────

# Chart: Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Claim", "Claim"])
disp.plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title("Random Forest - Confusion Matrix", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "14_rf_confusion_matrix.png"))
plt.close()
print(f"\n  Saved: reports/figures/14_rf_confusion_matrix.png")

# Chart: ROC Curve
# thresholds retained here — will be used in Stage 8 for evidence-based
# risk threshold selection (Low / Medium / High cutpoints)
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_rf)
auc_val = metrics_rf["roc_auc"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#4CAF50", lw=2,
        label=f"Random Forest (AUC = {auc_val:.4f})")
ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=1.5,
        linestyle="--", label="Random Baseline (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.07, color="#4CAF50")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Random Forest", fontweight="bold", pad=12)
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "15_rf_roc_curve.png"))
plt.close()
print(f"  Saved: reports/figures/15_rf_roc_curve.png")

# Chart: Predicted probability distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(prob_no_claim, bins=40, alpha=0.6, color="#2196F3",
        density=True, label="Actual No Claim")
ax.hist(prob_claim,    bins=40, alpha=0.6, color="#F44336",
        density=True, label="Actual Claim")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2,
           label="Default threshold (0.50)")
ax.set_xlabel("Predicted Claim Probability")
ax.set_ylabel("Density")
ax.set_title("Random Forest Probability Distribution by Actual Outcome",
             fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "16_rf_probability_distribution.png"))
plt.close()
print(f"  Saved: reports/figures/16_rf_probability_distribution.png")

# ── 7. Save pipeline ───────────────────────────────────────────────────────
print(f"\n[5] Saving trained pipeline...")
saved_path = save_pipeline(pipeline_rf, filename="rf_pipeline.joblib")
print(f"    Saved: {saved_path}")

# ── 8. Stage summary ──────────────────────────────────────────────────────
baseline_acc = 0.6865
baseline_auc = 0.5000
baseline_f1  = 0.0000

print(f"\n{SEP}")
print("STAGE 5 SUMMARY - RANDOM FOREST")
print(SEP)
print(f"""
  Accuracy  : {metrics_rf['accuracy']:.4f}  ({metrics_rf['accuracy']*100:.2f}%)
  Precision : {metrics_rf['precision']:.4f}
  Recall    : {metrics_rf['recall']:.4f}
  F1 Score  : {metrics_rf['f1']:.4f}
  ROC-AUC   : {metrics_rf['roc_auc']:.4f}
  PR-AUC    : {metrics_rf['pr_auc']:.4f}

  vs Baseline A (Always No-Claim):
    ROC-AUC : {baseline_auc:.4f} -> {metrics_rf['roc_auc']:.4f}
    F1      : {baseline_f1:.4f} -> {metrics_rf['f1']:.4f}
    Recall  : {baseline_f1:.4f} -> {metrics_rf['recall']:.4f}

  Hyperparameters used (not tuned via CV — validated in Stage 6):
    n_estimators=200, max_depth=10, min_samples_leaf=5

  Pipeline saved to: models/rf_pipeline.joblib
  Next: Stage 6 — Head-to-head model comparison + CV validation
""")
print(SEP)
