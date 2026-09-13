"""
notebooks/03_logistic_regression.py
=====================================
Stage 4  Logistic Regression Pipeline

Run with: python notebooks/03_logistic_regression.py

What this script does:
  1. Builds a full sklearn Pipeline: ColumnTransformer â†’ LogisticRegression
  2. Fits ONLY on training data (test set stays untouched)
  3. Evaluates on the held-out test set with the full metric suite
  4. Saves the trained pipeline to models/logistic_pipeline.joblib
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

from sklearn.linear_model import LogisticRegression
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


#1. Load and split -------------------------------------------
print(SEP)
print("STAGE 4 â€” LOGISTIC REGRESSION PIPELINE")
print(SEP)

df = load_data()
X, y = get_feature_target_split(df)
X_train, X_test, y_train, y_test = make_train_test_split(X, y)

print(f"\n[1] Data loaded: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test rows")
print(f"    Features used: {X_train.shape[1]} columns")
print(f"    Features dropped: id, postal_code (identifier / geographic proxy)")

#2. Why Logistic Regression first 
print(f"""
[2] Why Logistic Regression?
    - Industry standard for binary classification in regulated domains (insurance, credit)
    - Produces well-calibrated probabilities directly essential for our risk thresholds
    - Fully interpretable via coefficients: we can explain every prediction
    - Fast to train and cheap to serve in production
    - Forces us to think about feature scaling and encoding explicitly
    - Acts as a strong linear baseline before introducing tree complexity
""")

#3. Build pipeline 
print("[3] Building pipeline: ColumnTransformer LogisticRegression")
print("""
    Preprocessing (all fit on train data only):
      Numerical (credit_score, annual_mileage):
        SimpleImputer(median) â†’ StandardScaler
      Ordinal integers (age, gender, ownership, married, children):
        SimpleImputer(most_frequent) â†’ StandardScaler
      Count integers (speeding_violations, duis, past_accidents):
        SimpleImputer(most_frequent) â†’ StandardScaler
      String categoricals (driving_experience, education, income,
                           vehicle_year, vehicle_type):
        SimpleImputer(most_frequent) â†’ OneHotEncoder(handle_unknown=ignore)

    Classifier:
      LogisticRegression(max_iter=1000, C=1.0, class_weight=None)
      - C=1.0 is the default regularisation strength (L2)
      - max_iter=1000 ensures convergence on this dataset
      - class_weight=None initially â€” revisit if recall is too low
""")

lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    solver="lbfgs",
    class_weight=None,
)

pipeline_lr = build_pipeline(lr, model_type="linear")

# 4. Train 
print("[4] Fitting pipeline on training data only...")
pipeline_lr = train_pipeline(pipeline_lr, X_train, y_train)
print("    Done.")

# 5. Predict on test set 
print("\n[5] Evaluating on held-out test set (first time test data is used):")
y_pred_lr = pipeline_lr.predict(X_test)
y_prob_lr  = pipeline_lr.predict_proba(X_test)[:, 1]

metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr,
                              label="Logistic Regression")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_lr,
                              target_names=["No Claim", "Claim"], zero_division=0))

# 6. Compare vs baseline 
baseline_auc  = 0.5000
baseline_f1   = 0.0000
baseline_acc  = 0.6865

print(f"\n[6] Improvement over Baseline A (Always No-Claim):")
print(f"    Accuracy  : {baseline_acc:.4f} â†’ {metrics_lr['accuracy']:.4f}  "
      f"(+{(metrics_lr['accuracy']-baseline_acc)*100:.2f}pp)")
print(f"    ROC-AUC   : {baseline_auc:.4f} â†’ {metrics_lr['roc_auc']:.4f}  "
      f"(+{metrics_lr['roc_auc']-baseline_auc:.4f})")
print(f"    F1 Score  : {baseline_f1:.4f} â†’ {metrics_lr['f1']:.4f}  "
      f"(+{metrics_lr['f1']-baseline_f1:.4f})")

# 7. Probability distribution 
print(f"\n[7] Probability distribution analysis:")
prob_claim    = y_prob_lr[y_test == 1]
prob_no_claim = y_prob_lr[y_test == 0]
print(f"    Actual claimants    â€” mean prob: {prob_claim.mean():.3f}, "
      f"median: {np.median(prob_claim):.3f}")
print(f"    Actual non-claimants â€” mean prob: {prob_no_claim.mean():.3f}, "
      f"median: {np.median(prob_no_claim):.3f}")
print(f"    Separation: {prob_claim.mean() - prob_no_claim.mean():.3f} "
      f"(higher = better probability calibration)")

# 8. Charts 

# Chart: Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred_lr)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Claim", "Claim"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Logistic Regression â€” Confusion Matrix", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "11_lr_confusion_matrix.png"))
plt.close()
print(f"\n  Saved: reports/figures/11_lr_confusion_matrix.png")

# Chart: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)
auc_val = metrics_lr["roc_auc"]

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="#2196F3", lw=2,
        label=f"Logistic Regression (AUC = {auc_val:.4f})")
ax.plot([0, 1], [0, 1], color="#BDBDBD", lw=1.5,
        linestyle="--", label="Random Baseline (AUC = 0.50)")
ax.fill_between(fpr, tpr, alpha=0.07, color="#2196F3")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve â€” Logistic Regression", fontweight="bold", pad=12)
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "12_lr_roc_curve.png"))
plt.close()
print(f"  Saved: reports/figures/12_lr_roc_curve.png")

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
ax.set_title("Predicted Probability Distribution by Actual Outcome",
             fontweight="bold", pad=12)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "13_lr_probability_distribution.png"))
plt.close()
print(f"  Saved: reports/figures/13_lr_probability_distribution.png")

# 9. Save pipeline 
print(f"\n[8] Saving trained pipeline...")
saved_path = save_pipeline(pipeline_lr, filename="logistic_pipeline.joblib")
print(f"    Saved: {saved_path}")
print(f"    Pipeline contains: ColumnTransformer â†’ LogisticRegression")
print(f"    Load with: from src.train import load_pipeline; p = load_pipeline('logistic_pipeline.joblib')")

# 10. Stage summary 
print(f"\n{SEP}")
print("STAGE 4 SUMMARY â€” LOGISTIC REGRESSION")
print(SEP)
print(f"""
  Accuracy  : {metrics_lr['accuracy']:.4f}  ({metrics_lr['accuracy']*100:.2f}%)
  Precision : {metrics_lr['precision']:.4f}
  Recall    : {metrics_lr['recall']:.4f}
  F1 Score  : {metrics_lr['f1']:.4f}
  ROC-AUC   : {metrics_lr['roc_auc']:.4f}
  PR-AUC    : {metrics_lr['pr_auc']:.4f}

  vs Baseline A (Always No-Claim):
    ROC-AUC : 0.5000 â†’ {metrics_lr['roc_auc']:.4f}  âœ“
    F1      : 0.0000 â†’ {metrics_lr['f1']:.4f}  âœ“
    Recall  : 0.0000 â†’ {metrics_lr['recall']:.4f}  âœ“

  Verdict: Logistic Regression comfortably beats the dummy baseline.
  The pipeline correctly handles:
    - Median imputation (only on train data)
    - Standard scaling of all numeric/ordinal features
    - One-hot encoding of string categoricals
    - No preprocessing leakage

  Next: Stage 5 â€” Random Forest (tree-based model, no scaling needed)
  We will compare the two models head-to-head in Stage 6.
""")
print(SEP)

