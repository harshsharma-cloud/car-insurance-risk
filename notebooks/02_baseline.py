"""
notebooks/02_baseline.py
=========================
Stage 3 — Train/Test Split & Baseline

Run with: python notebooks/02_baseline.py

What this script does:
  1. Loads data via src/data.py
  2. Creates a reproducible 80/20 stratified split (locked for the entire project)
  3. Trains a DummyClassifier baseline on train data
  4. Evaluates it on the test set using the full metric suite
  5. Explains why accuracy alone fails for imbalanced classification
  6. Saves split indices and confusion matrix chart

The test set is NEVER used for model selection — only for final evaluation.
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

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)

from src.data import load_data, get_feature_target_split, report_class_balance
from src.train import make_train_test_split, RANDOM_SEED, TEST_SIZE

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "reports", "figures")
SEP = "-" * 65

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight"})


def compute_metrics(y_true, y_pred, y_prob, model_name="Model"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    print(f"\n  {model_name} — Test Set Metrics")
    print(f"  {'Accuracy':<20}: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  {'Precision':<20}: {prec:.4f}")
    print(f"  {'Recall':<20}: {rec:.4f}")
    print(f"  {'F1 Score':<20}: {f1:.4f}")
    print(f"  {'ROC-AUC':<20}: {auc:.4f}")
    return {"model": model_name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}


def save_confusion_matrix(y_true, y_pred, model_name, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No Claim (0)", "Claim (1)"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"  Saved: reports/figures/{filename}")


# ── 1. Load & Split ────────────────────────────────────────────────────────
print(SEP)
print("STAGE 3 — TRAIN/TEST SPLIT & BASELINE")
print(SEP)

df = load_data()
X, y = get_feature_target_split(df)
balance = report_class_balance(y)

X_train, X_test, y_train, y_test = make_train_test_split(X, y)

print(f"\n[1] Data split (random_state={RANDOM_SEED}, test_size={TEST_SIZE})")
print(f"    Train : {len(X_train):,} rows  |  claim rate = {y_train.mean():.4f}")
print(f"    Test  : {len(X_test):,}  rows  |  claim rate = {y_test.mean():.4f}")
print(f"\n    Stratification check: train={y_train.mean():.3f} vs test={y_test.mean():.3f}")
diff = abs(y_train.mean() - y_test.mean())
if diff < 0.005:
    print(f"    PASS — class ratio preserved (diff={diff:.5f} < 0.005 threshold)")
else:
    print(f"    WARN — class ratio differs by {diff:.5f}")

print(f"\n    Test set is now LOCKED. It will not be used again until final evaluation.")

# ── 2. Why accuracy alone fails ───────────────────────────────────────────
print(f"\n[2] Why accuracy alone is misleading here:")
print(f"""
    The dataset has 31.3% claim rate — it is imbalanced.
    A model that ALWAYS predicts "No Claim" (never claims) would score:

      Accuracy = {balance['n_no_claim'] / balance['n_total']:.1%}

    That sounds decent! But it would:
      - Have ZERO recall for the claim class
      - Catch ZERO actual claimants
      - Be completely useless to an insurance company

    This is the "accuracy paradox" for imbalanced datasets.
    We MUST track:
      * F1 Score  — balances precision and recall
      * ROC-AUC   — how well the model separates classes at any threshold
      * Recall    — critical: how many real claimants do we catch?
      * Precision — when we predict a claim, how often are we right?
""")

# ── 3. Baseline 1: Always predict majority class ──────────────────────────
print(f"\n[3] Baseline A — DummyClassifier (always predicts 'No Claim')")
print(f"    Strategy: most_frequent (predicts class 0 for every customer)")

dummy_majority = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
dummy_majority.fit(X_train, y_train)
y_pred_maj  = dummy_majority.predict(X_test)
y_prob_maj  = np.full(len(y_test), 1 - balance["claim_rate"])  # constant prob

metrics_majority = compute_metrics(y_test, y_pred_maj, y_prob_maj,
                                    "Baseline A (Always No-Claim)")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_maj,
                              target_names=["No Claim", "Claim"], zero_division=0))

save_confusion_matrix(y_test, y_pred_maj,
                       "Baseline A — Always No-Claim",
                       "08_baseline_A_confusion_matrix.png")

# ── 4. Baseline 2: Stratified random ─────────────────────────────────────
print(f"\n[4] Baseline B — DummyClassifier (stratified random predictions)")
print(f"    Strategy: stratified (random predictions respecting class distribution)")

dummy_strat = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
dummy_strat.fit(X_train, y_train)
y_pred_strat = dummy_strat.predict(X_test)
y_prob_strat = dummy_strat.predict_proba(X_test)[:, 1]

metrics_stratified = compute_metrics(y_test, y_pred_strat, y_prob_strat,
                                      "Baseline B (Stratified Random)")

save_confusion_matrix(y_test, y_pred_strat,
                       "Baseline B — Stratified Random",
                       "09_baseline_B_confusion_matrix.png")

# ── 5. Comparison table ────────────────────────────────────────────────────
print(f"\n[5] Baseline Comparison Table")
comparison = pd.DataFrame([metrics_majority, metrics_stratified])
comparison = comparison.set_index("model")
comparison = comparison.round(4)
print(f"\n{comparison.to_string()}")

print(f"""
  What these numbers tell us:
  - Baseline A has {metrics_majority['accuracy']*100:.1f}% accuracy but ZERO F1 and ZERO ROC-AUC uplift
  - Baseline B is essentially random noise — ROC-AUC close to 0.50

  Our real models (Logistic Regression and Random Forest) must:
    * Beat {metrics_majority['accuracy']*100:.1f}% accuracy (easy to do, but not our goal)
    * Beat ROC-AUC of ~0.50 (the real challenge — ideally 0.80+)
    * Have meaningful Recall for the claim class
    * Have meaningful F1 Score
""")

# ── 6. Side-by-side confusion matrix visual ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, (y_pred, title) in zip(axes, [
    (y_pred_maj,   "Baseline A\nAlways Predicts No-Claim"),
    (y_pred_strat, "Baseline B\nStratified Random"),
]):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["No Claim", "Claim"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontweight="bold", fontsize=10)

plt.suptitle("Baseline Models — Confusion Matrices on Test Set",
             fontsize=12, fontweight="bold")
plt.tight_layout()
path = os.path.join(FIG_DIR, "10_baselines_side_by_side.png")
plt.savefig(path)
plt.close()
print(f"  Saved: reports/figures/10_baselines_side_by_side.png")

# ── 7. Final summary ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STAGE 3 SUMMARY")
print(SEP)
print(f"""
Train size  : {len(X_train):,} rows
Test size   : {len(X_test):,} rows
Stratified  : YES — class ratio preserved in both halves
Random seed : {RANDOM_SEED} (fixed for the entire project)

Baseline targets our real models must beat:
  Accuracy  > {metrics_majority['accuracy']:.4f}  (trivially easy)
  ROC-AUC   > 0.5500  (meaningful threshold)
  F1 Score  > 0.3500  (must catch real claimants)
  Recall    > 0.4000  (insurance needs to find claimants)

Next: Stage 4 — Logistic Regression Pipeline
""")
print(SEP)
